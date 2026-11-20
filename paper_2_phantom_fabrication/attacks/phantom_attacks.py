# -*- coding: utf-8 -*-
"""
Module: paper_2_phantom_fabrication.attacks.phantom_attacks
Mục đích: Cài đặt các thuật toán tấn công đối kháng chế tạo đối tượng ma (Phantom Queries)
trên kiến trúc RT-DETR theo từng lô ảnh (Batch-level Adversarial Perturbation Generation).

Các đòn tấn công:
  1. Clean: Giữ nguyên ảnh gốc không nhiễu.
  2. Gaussian_Noise: Nhiễu Gauss ngẫu nhiên chuẩn hóa cùng ngân sách L_inf (baseline).
  3. EQF (Encoder Query Fabrication): Tối ưu hóa phân hạng để đẩy các query nền (background queries)
     có điểm số phân loại thuộc lớp mục tiêu vượt ngưỡng chọn lọc S=300 của Deformable Encoder.
  4. DQF (Decoder Query Fabrication): Khai thác không gian Deformable Cross-Attention tầng Decoder
     để kích hoạt cực đại xác suất dự đoán của các query nhàn rỗi (inactive queries).
  5. Untargeted_PGD: Tối đa hóa xác suất cực đại tổng quát trên mọi lớp danh định.
"""

from typing import Optional
import torch


def generate_batch_adversarial(
    model: torch.nn.Module,
    bx: torch.Tensor,
    attack_type: str = "Clean",
    target_class_idx: int = 1,
    epsilon: float = 16.0 / 255.0,
    alpha: float = 2.0 / 255.0,
    steps: int = 8,
    k_fake_eqf: int = 50,
    k_fake_dqf: int = 15,
) -> torch.Tensor:
    """
    Sinh batch ảnh đối kháng trực tiếp trên GPU/CPU theo lô hiện thời.
    Bảo toàn tính liên tục trong không gian ảnh [0.0, 1.0] và ngân sách chuẩn L_inf (epsilon).
    Chuẩn hóa tên tấn công không phân biệt hoa thường (Case-insensitive normalization).
    """
    norm_type = str(attack_type).strip().lower()
    if norm_type in ("clean", "none", "identity", ""):
        return bx

    if norm_type in ("gaussian_noise", "gaussian", "noise"):
        noise = torch.randn_like(bx) * epsilon
        noise = torch.clamp(noise, -epsilon, epsilon)
        return torch.clamp(bx + noise, 0.0, 1.0)

    # 1. Forward pass trên ảnh sạch để lấy background / foreground / inactive mask
    with torch.no_grad():
        clean_out = model(pixel_values=bx)
        if norm_type in ("eqf", "standard_eqf", "rab_eqf", "rab", "rank_aware_eqf"):
            clean_enc = clean_out.enc_outputs_class
            clean_enc_p = torch.sigmoid(clean_enc).max(dim=-1)[0]
            bg_mask = clean_enc_p < 0.10
            fg_mask = clean_enc_p >= 0.35
        elif norm_type in ("dqf", "output_logit", "output_logit_attack", "dc_eqf", "dc", "dispersion_constrained_eqf"):
            clean_dec = clean_out.logits
            clean_dec_p = torch.sigmoid(clean_dec).max(dim=-1)[0]
            inactive_dec_mask = clean_dec_p < 0.15
        elif norm_type not in ("untargeted_pgd", "untargeted", "pgd"):
            raise ValueError(f"Dạng tấn công không hợp lệ: {attack_type}")

    # 2. Vòng lặp tối ưu hóa Projected Gradient Descent (PGD)
    x_adv = bx.clone().detach().requires_grad_(True)
    for _ in range(steps):
        out = model(pixel_values=x_adv)
        if norm_type in ("eqf", "standard_eqf"):
            target_logits = out.enc_outputs_class[..., target_class_idx]
            masked_target = target_logits.masked_fill(~bg_mask, -1e4)
            k_sel = min(k_fake_eqf, masked_target.shape[1])
            top_fake = masked_target.topk(k_sel, dim=-1)[0]
            loss = -top_fake.mean()

        elif norm_type in ("rab_eqf", "rab", "rank_aware_eqf"):
            # Rank-Aware Boundary EQF: Đẩy background vượt ngưỡng và triệt tiêu query vật thể thật
            target_logits = out.enc_outputs_class[..., target_class_idx]
            masked_target = target_logits.masked_fill(~bg_mask, -1e4)
            k_sel = min(k_fake_eqf, masked_target.shape[1])
            top_fake = masked_target.topk(k_sel, dim=-1)[0]
            loss_push = -top_fake.mean()

            # Thành phần triệt tiêu vật thể thật (Foreground eviction)
            if fg_mask.sum() > 0:
                fg_logits = out.enc_outputs_class[fg_mask]
                loss_evict = fg_logits.max(dim=-1)[0].mean()
            else:
                loss_evict = torch.tensor(0.0, device=bx.device)

            loss = loss_push + 0.5 * loss_evict

        elif norm_type in ("dc_eqf", "dc", "dispersion_constrained_eqf"):
            # Dispersion-Constrained EQF: Ràng buộc phân tán tâm hình học để vượt qua SQD-Filter
            target_logits = out.logits[..., target_class_idx]
            masked_target = target_logits.masked_fill(~inactive_dec_mask, -1e4)
            k_sel = min(k_fake_dqf, masked_target.shape[1])
            top_indices = masked_target.topk(k_sel, dim=-1)[1][0]
            top_fake = masked_target[0, top_indices]
            loss_push = -top_fake.mean()

            # Phạt khoảng cách tâm hình học nhỏ hơn d_min = 0.15
            centers = out.pred_boxes[0, top_indices, :2]  # [k_sel, 2]
            diff = centers.unsqueeze(0) - centers.unsqueeze(1)
            dist = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-6)
            triu_m = torch.triu(torch.ones(k_sel, k_sel, dtype=torch.bool, device=bx.device), diagonal=1)
            pairwise_dists = dist[triu_m]
            repulsion = torch.relu(0.15 - pairwise_dists) ** 2
            loss_disp = repulsion.mean() if len(repulsion) > 0 else torch.tensor(0.0, device=bx.device)

            loss = loss_push + 2.0 * loss_disp

        elif norm_type in ("dqf", "output_logit", "output_logit_attack"):
            target_logits = out.logits[..., target_class_idx]
            masked_target = target_logits.masked_fill(~inactive_dec_mask, -1e4)
            k_sel = min(k_fake_dqf, masked_target.shape[1])
            top_fake = masked_target.topk(k_sel, dim=-1)[0]
            loss = -top_fake.mean()

        elif norm_type in ("untargeted_pgd", "untargeted", "pgd"):
            probs = torch.sigmoid(out.logits)
            loss = -probs.max(dim=-1)[0].mean()
        else:
            raise ValueError(f"Dạng tấn công không hợp lệ: {attack_type}")

        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            if x_adv.grad is not None:
                grad_sign = x_adv.grad.sign()
            else:
                grad_sign = torch.zeros_like(x_adv)
            x_adv = x_adv - alpha * grad_sign
            delta = torch.clamp(x_adv - bx, -epsilon, epsilon)
            x_adv = torch.clamp(bx + delta, 0.0, 1.0).requires_grad_(True)

    return x_adv.detach()
