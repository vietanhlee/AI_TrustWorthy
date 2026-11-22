# -*- coding: utf-8 -*-
"""
Module: paper_2_phantom_fabrication.attacks.yolo_attacks
Mục đích: Cài đặt các đòn tấn công đối kháng chế tạo đối tượng ma (Targeted Object Fabrication)
trực tiếp trên kiến trúc CNN dense anchor-grid (YOLOv8) theo mô hình White-Box:
  - Khai thác trực tiếp gradient từ output tensor của YOLOv8 (8,400 anchors đa tỷ lệ: P3 80x80, P4 40x40, P5 20x20).
  - Triển khai hàm mất mát Task-Aligned L_YOLO theo Mục 4.5 của bản thảo main_phantom.tex:
      L_YOLO(X + delta) = - (1 / |C_grid|) sum_{j in C_grid} L_hat_{j, c_target}(X + delta)
    với C_grid gồm 30 background anchor-free grid cells trên các tầng P3, P4, P5 có confidence sạch < 0.10.
  - Hỗ trợ 2 chế độ hình học không gian:
    1. Localized Fabrication: Gom cụm 30 ô lưới nền trong một vùng cục bộ (Localized Grid Excitation)
       để kiểm tra cơ chế tự nhiên gộp/loại bỏ box của NMS (88.91% reduction).
    2. Dispersed Fabrication (TOG-style): Ép các phantom phân tán cách xa nhau trên lưới ảnh nhằm chủ động vượt qua bộ lọc NMS.
  - Hỗ trợ bảo toàn vật thể thật (Foreground-Preserving Fabrication) qua hàm phạt MSE trên các ô foreground (conf >= 0.35).
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
    Sinh batch ảnh đối kháng đối với YOLOv8 theo chế độ white-box thuần túy (Mục 4.5 main_phantom.tex).
    
    Tham số:
        yolo_model: Thể hiện YOLOv8 (Ultralytics YOLO hoặc torch.nn.Module nội bộ).
        bx: Tensor ảnh đầu vào [B, 3, 640, 640] trong đoạn [0.0, 1.0].
        attack_mode:
            - "localized": Gom cụm 30 ô lưới nền trong vùng cục bộ để kích thích receptive field liên kề (Mục 4.5).
            - "dispersed": Phân tán 30 ô lưới nền cách xa nhau với khoảng cách d_min >= 0.15 (vượt NMS).
            - "untargeted": Phá hủy toàn cục dự đoán của mô hình YOLOv8.
        target_class_idx: Lớp đối tượng mục tiêu cần chế tạo (mặc định 1: bicycle).
        epsilon: Ngân sách nhiễu tối đa L_inf (16/255).
        alpha: Kích thước bước lặp PGD (2/255).
        steps: Số bước lặp tối ưu hóa (T=8).
        k_fake_anchors: Số ô lưới nền mục tiêu (|C_grid| = 30).
        preserve_foreground: Bật cơ chế giữ nguyên độ tin cậy của các vật thể thật (clean conf >= 0.35).
        lambda_fg: Trọng số cân bằng bảo toàn foreground.
        device: Thiết bị tính toán (CUDA/CPU).
    """
    if yolo_model is None:
        return bx

    if device is None:
        device = bx.device

    # Lấy mô hình PyTorch nội bộ của Ultralytics (yolo_model.model)
    raw_pt_model = getattr(yolo_model, "model", yolo_model)
    if hasattr(raw_pt_model, "to"):
        raw_pt_model = raw_pt_model.to(device)
    raw_pt_model.eval()

    # Đảm bảo gradient không cập nhật vào trọng số của model
    for p in raw_pt_model.parameters():
        p.requires_grad = False

    # 1. Forward pass ảnh sạch để xác định mask nền (background mask, conf < 0.10)
    # và mask vật thể thật (foreground mask, conf >= 0.35) chuẩn Section 4.5
    with torch.no_grad():
        clean_out = raw_pt_model(bx)
        # Trong YOLOv8 eval mode, output thường có shape [B, 84, 8400] hoặc tuple
        if isinstance(clean_out, (list, tuple)):
            clean_preds = clean_out[0]
        else:
            clean_preds = clean_out

        # clean_preds: [B, 84, 8400]
        # 4 kênh đầu: cx, cy, w, h; 80 kênh sau: class confidence scores sigma(L_hat)
        clean_coords = clean_preds[:, :4, :]  # [B, 4, 8400]
        clean_scores = clean_preds[:, 4:, :]  # [B, 80, 8400]
        clean_max_p, _ = clean_scores.max(dim=1)  # [B, 8400]
        
        # Ngưỡng nền chuẩn xác: tau_bg = 0.10 theo công thức Section 4.5 main_phantom.tex
        bg_mask = (clean_max_p < 0.10)  # [B, 8400]
        fg_mask = (clean_max_p >= 0.35)  # [B, 8400]

    # 2. Xác định danh sách chỉ số 30 background cells cho từng ảnh trong batch
    b_size = bx.shape[0]
    selected_indices_per_batch = []
    for b in range(b_size):
        b_bg = bg_mask[b]
        b_bg_indices = torch.nonzero(b_bg, as_tuple=False).squeeze(-1)
        num_bg = b_bg_indices.numel()

        if num_bg == 0:
            # Fallback nếu toàn bộ ảnh không có ô nền < 0.10: chọn các ô có conf nhỏ nhất
            _, sorted_idx = torch.sort(clean_max_p[b])
            sel_idx = sorted_idx[:k_fake_anchors]
        elif attack_mode == "localized":
            # Chế độ Localized Grid Excitation (Mục 4.5):
            # Chọn ô nền có target score cao nhất làm tâm cụm, sau đó gom 30 ô nền lân cận
            # gần nhất trong không gian tọa độ (cx, cy) để kích thích receptive field liền kề
            b_target_clean = clean_scores[b, target_class_idx, b_bg_indices]
            best_local_sub_idx = torch.argmax(b_target_clean)
            center_cell_idx = b_bg_indices[best_local_sub_idx]
            center_xy = clean_coords[b, :2, center_cell_idx].unsqueeze(0)  # [1, 2]

            bg_xy = clean_coords[b, :2, b_bg_indices].T  # [num_bg, 2]
            dists = torch.norm(bg_xy - center_xy, dim=-1)  # [num_bg]
            k_sel = min(k_fake_anchors, num_bg)
            _, nearest_sub_idx = torch.topk(dists, k=k_sel, largest=False)
            sel_idx = b_bg_indices[nearest_sub_idx]
        elif attack_mode == "dispersed":
            # Chế độ Dispersed: chọn top K anchors có target score cao nhất trong nền
            b_target_clean = clean_scores[b, target_class_idx, :]
            masked_scores = b_target_clean.masked_fill(~b_bg, -1e4)
            k_sel = min(k_fake_anchors, num_bg)
            sel_idx = torch.topk(masked_scores, k=k_sel)[1]
        else:
            k_sel = min(k_fake_anchors, num_bg)
            sel_idx = b_bg_indices[:k_sel]

        selected_indices_per_batch.append(sel_idx)

    # 3. Vòng lặp tối ưu hóa PGD White-Box
    x_adv = bx.clone().detach().requires_grad_(True)

    for _ in range(steps):
        out = raw_pt_model(x_adv)
        if isinstance(out, (list, tuple)):
            preds = out[0]
        else:
            preds = out

        coords = preds[:, :4, :]  # [B, 4, 8400]
        cls_scores = preds[:, 4:, :]  # [B, 80, 8400]
        target_scores = cls_scores[:, target_class_idx, :]  # [B, 8400]
        loss_batch = []

        for b in range(b_size):
            sel_idx = selected_indices_per_batch[b]
            if sel_idx.numel() == 0:
                loss_batch.append(torch.tensor(0.0, device=device))
                continue

            # Lấy xác suất lớp mục tiêu trên 30 ô nền được chọn
            b_sel_probs = target_scores[b, sel_idx]

            # Biến đổi sang logit không gian để tránh bão hòa gradient (non-saturating logit loss)
            # tương ứng L_hat_{j, c_target} trong phương trình Eq. (6) Mục 4.5
            eps_clip = 1e-6
            clamped_probs = torch.clamp(b_sel_probs, eps_clip, 1.0 - eps_clip)
            b_sel_logits = torch.logit(clamped_probs)

            if attack_mode == "untargeted":
                loss_b = -cls_scores[b].max(dim=0)[0].mean()
            elif attack_mode == "localized":
                # Hàm mất mát L_YOLO theo Section 4.5 Eq. (6):
                # Tối đa hóa logits của target class trên 30 ô nền cục bộ
                loss_b = -b_sel_logits.mean()
            elif attack_mode == "dispersed":
                loss_push = -b_sel_logits.mean()
                k_sel = sel_idx.numel()
                if k_sel > 1:
                    # Phạt khoảng cách gần nhau để phân tán đều trên lưới
                    b_centers = coords[b, :2, sel_idx].T / 640.0
                    diff = b_centers.unsqueeze(0) - b_centers.unsqueeze(1)
                    dist = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-6)
                    triu_m = torch.triu(torch.ones(k_sel, k_sel, dtype=torch.bool, device=device), diagonal=1)
                    pairwise_dists = dist[triu_m]
                    repulsion = torch.relu(0.15 - pairwise_dists) ** 2
                    loss_disp = repulsion.mean() if len(repulsion) > 0 else torch.tensor(0.0, device=device)
                    loss_b = loss_push + 2.0 * loss_disp
                else:
                    loss_b = loss_push
            else:
                loss_b = -b_sel_logits.mean()

            # Bổ sung thành phần bảo toàn Foreground (nếu được kích hoạt)
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
