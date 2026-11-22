# -*- coding: utf-8 -*-
"""
Module: paper_2_phantom_fabrication.attacks.yolo_attacks
Mục đích: Cài đặt các đòn tấn công đối kháng chế tạo đối tượng ma (Targeted Object Fabrication)
trực tiếp trên kiến trúc CNN dense anchor-grid (YOLOv8) theo mô hình White-Box:
  - Khai thác trực tiếp gradient từ output tensor của YOLOv8 (8,400 anchors đa tỷ lệ: P3 80x80, P4 40x40, P5 20x20).
  - Hỗ trợ 2 chế độ hình học không gian:
    1. Localized Fabrication: Tập trung đẩy các anchor lân cận quanh 1 vùng nền (để kiểm tra xem NMS có tự nhiên gộp các box trùng lặp không).
    2. Dispersed Fabrication (TOG-style): Ép các phantom phân tán cách xa nhau trên lưới ảnh nhằm chủ động vượt qua bộ lọc NMS.
  - Hỗ trợ biến thể bảo toàn vật thể thật (Foreground-Preserving Fabrication).
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False


def generate_yolo_adversarial_batch(
    yolo_model: Any,
    bx: torch.Tensor,
    attack_mode: str = "localized",
    target_class_idx: int = 1,
    epsilon: float = 16.0 / 255.0,
    alpha: float = 2.0 / 255.0,
    steps: int = 8,
    k_fake_anchors: int = 30,
    preserve_foreground: bool = False,
    lambda_fg: float = 2.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Sinh batch ảnh đối kháng đối với YOLOv8 theo chế độ white-box thuần túy.
    bx: Tensor [B, 3, 640, 640] trong đoạn [0.0, 1.0].
    attack_mode:
        - "localized": Gom cụm anchors tại 1 vùng nền.
        - "dispersed": Phân tán anchors xa nhau (vượt NMS).
        - "untargeted": Phá hủy toàn cục dự đoán của YOLO.
    """
    if yolo_model is None:
        return bx

    if device is None:
        device = bx.device

    # Lấy mô hình PyTorch nội bộ của Ultralytics (yolo_model.model)
    raw_pt_model = getattr(yolo_model, "model", yolo_model)
    raw_pt_model.eval()

    # Đảm bảo gradient không tính vào trọng số của model
    for p in raw_pt_model.parameters():
        p.requires_grad = False

    # 1. Forward pass ảnh sạch để xác định mask nền (background mask) và vật thể thật (foreground mask)
    with torch.no_grad():
        clean_out = raw_pt_model(bx)
        # Trong YOLOv8 eval mode, output thường có shape [B, 84, 8400] hoặc tuple
        if isinstance(clean_out, (list, tuple)):
            clean_preds = clean_out[0]
        else:
            clean_preds = clean_out

        # clean_preds: [B, 84, 8400]
        # 4 kênh đầu: cx, cy, w, h; 80 kênh sau: class confidence scores
        clean_scores = clean_preds[:, 4:, :]  # [B, 80, 8400]
        clean_max_p, _ = clean_scores.max(dim=1)  # [B, 8400]
        
        bg_mask = (clean_max_p < 0.15)  # [B, 8400]
        fg_mask = (clean_max_p >= 0.35)  # [B, 8400]

    # 2. Vòng lặp tối ưu hóa PGD White-Box
    x_adv = bx.clone().detach().requires_grad_(True)
    b_size = bx.shape[0]

    for _ in range(steps):
        out = raw_pt_model(x_adv)
        if isinstance(out, (list, tuple)):
            preds = out[0]
        else:
            preds = out

        # Trích xuất class scores và coordinates
        coords = preds[:, :4, :]  # [B, 4, 8400]
        cls_scores = preds[:, 4:, :]  # [B, 80, 8400]

        target_scores = cls_scores[:, target_class_idx, :]  # [B, 8400]
        loss_batch = []

        for b in range(b_size):
            b_bg = bg_mask[b]
            b_target = target_scores[b]

            if attack_mode == "untargeted":
                loss_b = -cls_scores[b].max(dim=0)[0].mean()
            elif attack_mode == "localized":
                # Chế độ Localized: Chọn top K anchors nền có điểm target cao nhất hoặc gần 1 cụm
                masked_scores = b_target.masked_fill(~b_bg, -1e4)
                k_sel = min(k_fake_anchors, int(b_bg.sum().item()))
                if k_sel > 0:
                    top_k_scores = masked_scores.topk(k_sel)[0]
                    loss_push = -top_k_scores.mean()
                else:
                    loss_push = torch.tensor(0.0, device=device)

                loss_b = loss_push
            elif attack_mode == "dispersed":
                # Chế độ Dispersed: Tối đa hóa score đồng thời phạt khoảng cách gần nhau
                masked_scores = b_target.masked_fill(~b_bg, -1e4)
                k_sel = min(k_fake_anchors, int(b_bg.sum().item()))
                if k_sel > 1:
                    top_indices = masked_scores.topk(k_sel)[1]
                    top_k_scores = masked_scores[top_indices]
                    loss_push = -top_k_scores.mean()

                    # Lấy tâm tọa độ [k_sel, 2] chuẩn hóa về [0, 1]
                    b_centers = coords[b, :2, top_indices].T / 640.0
                    diff = b_centers.unsqueeze(0) - b_centers.unsqueeze(1)
                    dist = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-6)
                    triu_m = torch.triu(torch.ones(k_sel, k_sel, dtype=torch.bool, device=device), diagonal=1)
                    pairwise_dists = dist[triu_m]
                    
                    # Ngưỡng phân tán d_min = 0.15 (~96 pixels trên ảnh 640)
                    repulsion = torch.relu(0.15 - pairwise_dists) ** 2
                    loss_disp = repulsion.mean() if len(repulsion) > 0 else torch.tensor(0.0, device=device)
                    loss_b = loss_push + 2.0 * loss_disp
                else:
                    loss_b = torch.tensor(0.0, device=device)
            else:
                loss_b = -b_target.mean()

            # Bổ sung thành phần bảo toàn Foreground (nếu được yêu cầu)
            if preserve_foreground and fg_mask[b].sum() > 0:
                clean_fg = clean_preds[b, :, fg_mask[b]]
                adv_fg = preds[b, :, fg_mask[b]]
                loss_preserve = F.mse_loss(adv_fg, clean_fg)
                loss_b = loss_b + lambda_fg * loss_preserve

            loss_batch.append(loss_b)

        total_loss = torch.stack(loss_batch).mean()
        raw_pt_model.zero_grad()
        total_loss.backward()

        with torch.no_grad():
            if x_adv.grad is not None:
                grad_sign = x_adv.grad.sign()
            else:
                grad_sign = torch.zeros_like(x_adv)
            x_adv = x_adv - alpha * grad_sign
            delta = torch.clamp(x_adv - bx, -epsilon, epsilon)
            x_adv = torch.clamp(bx + delta, 0.0, 1.0).requires_grad_(True)

    return x_adv.detach()
