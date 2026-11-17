"""
Module: query_attention_features.py
Mục đích: Trích xuất các đặc trưng an toàn tầng mô hình (Model-Layer Safety Features)
từ trạng thái nội tại của Transformer Object Detector (RT-DETR & DETR).

Không sử dụng bất kỳ tín hiệu phần cứng hay thời gian nào.
Bao gồm 4 nhóm đặc trưng cốt lõi (16 chiều):
1. Query Confidence & Margin Dynamics (5 chiều)
2. Cross-Attention Entropy & Spatial Focus (4 chiều) - tương thích Deformable Attention & Dense Attention
3. Decoder Refinement Drift qua các tầng (5 chiều)
4. Query Representation Diversity & Collapse (2 chiều)
"""

from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn.functional as F

from edge_sec_pro.transformer_monitors.detr_wrapper import DETRInternalState


class TransformerFeatureExtractor:
    """
    Bộ trích xuất đặc trưng tầng mô hình cho Transformer Object Detectors.
    Tối ưu hóa tính toán ma trận để độ trễ overhead <= 1ms trên CPU/Edge.
    """

    FEATURE_NAMES = [
        # Nhóm 1: Query Confidence & Phân phối (5)
        "query_top1_mean",
        "query_margin_mean",
        "query_entropy_mean",
        "active_query_ratio",
        "bg_prob_mean",
        # Nhóm 2: Cross-Attention Dynamics (4)
        "attn_entropy_mean",
        "attn_peak_mean",
        "attn_head_diversity",
        "attn_spatial_dispersion",
        # Nhóm 3: Decoder Refinement Drift (5)
        "refinement_delta_l1",
        "refinement_delta_l2",
        "refinement_delta_l3",
        "refinement_delta_l4",
        "refinement_delta_l5",
        # Nhóm 4: Query Diversity & Collapse (2)
        "query_cosine_sim_mean",
        "query_norm_std",
    ]

    def __init__(self, active_threshold: float = 0.3, epsilon: float = 1e-8) -> None:
        """
        Khởi tạo bộ trích xuất đặc trưng.

        Tham số:
            active_threshold: Ngưỡng xác suất coi một query là kích hoạt đối tượng thực.
            epsilon: Giá trị số nhỏ tránh chia cho 0 hoặc log(0).
        """
        self.active_threshold = active_threshold
        self.epsilon = epsilon

    @torch.no_grad()
    def extract_from_state(self, state: DETRInternalState) -> np.ndarray:
        """
        Trích xuất vector đặc trưng từ đối tượng DETRInternalState.

        Tham số:
            state: DETRInternalState chứa logits, decoder_hidden_states, cross_attentions.

        Trả về:
            numpy.ndarray: Ma trận đặc trưng [B, 16].
        """
        batch_size = state.logits.shape[0]
        device = state.logits.device

        # --- Nhóm 1: Query Confidence & Margin Dynamics ---
        if state.is_sigmoid:
            # RT-DETR: Độc lập nhãn qua Sigmoid (Focal Loss)
            probs = torch.sigmoid(state.logits)  # [B, N, C]
            sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
            top1_prob = sorted_probs[..., 0]  # [B, N]
            top2_prob = sorted_probs[..., 1] if sorted_probs.shape[-1] > 1 else torch.zeros_like(top1_prob)

            query_top1_mean = top1_prob.mean(dim=-1, keepdim=True)  # [B, 1]
            query_margin_mean = (top1_prob - top2_prob).mean(dim=-1, keepdim=True)  # [B, 1]

            # Binary entropy trên các nhãn dự đoán
            clamped_probs = torch.clamp(probs, self.epsilon, 1.0 - self.epsilon)
            binary_ent = -(clamped_probs * torch.log(clamped_probs) + (1.0 - clamped_probs) * torch.log(1.0 - clamped_probs))
            query_entropy_mean = binary_ent.mean(dim=[-1, -2]).view(batch_size, 1)  # [B, 1]

            active_mask = (top1_prob > self.active_threshold).float()
            active_query_ratio = active_mask.mean(dim=-1, keepdim=True)  # [B, 1]

            # Background probability: Không phát hiện nhãn nào => 1 - top1_prob
            bg_prob_mean = (1.0 - top1_prob).mean(dim=-1, keepdim=True)  # [B, 1]
        else:
            # Vanilla DETR: Softmax với lớp cuối là background / no-object
            probs = torch.softmax(state.logits, dim=-1)  # [B, N, C+1]
            object_probs = probs[..., :-1]  # [B, N, C]
            bg_probs = probs[..., -1]  # [B, N]

            sorted_obj_probs, _ = torch.sort(object_probs, dim=-1, descending=True)
            top1_prob = sorted_obj_probs[..., 0]  # [B, N]
            top2_prob = sorted_obj_probs[..., 1] if sorted_obj_probs.shape[-1] > 1 else torch.zeros_like(top1_prob)

            query_top1_mean = top1_prob.mean(dim=-1, keepdim=True)  # [B, 1]
            query_margin_mean = (top1_prob - top2_prob).mean(dim=-1, keepdim=True)  # [B, 1]

            entropy = -torch.sum(probs * torch.log(probs + self.epsilon), dim=-1)  # [B, N]
            query_entropy_mean = entropy.mean(dim=-1, keepdim=True)  # [B, 1]

            active_mask = (top1_prob > self.active_threshold).float()
            active_query_ratio = active_mask.mean(dim=-1, keepdim=True)  # [B, 1]

            bg_prob_mean = bg_probs.mean(dim=-1, keepdim=True)  # [B, 1]

        # --- Nhóm 2: Cross-Attention Dynamics ---
        # Hỗ trợ cả RT-DETR Deformable Attention [B, N, H, L, P] lẫn Vanilla DETR Dense Attention [B, H, N, S]
        if state.cross_attentions is not None and len(state.cross_attentions) > 0:
            last_attn = state.cross_attentions[-1]
            if last_attn.ndim == 5:
                # [B, N, H, L, P] -> permute -> [B, H, N, L*P]
                last_attn = last_attn.permute(0, 2, 1, 3, 4).flatten(3, 4)

            B, H, N, S = last_attn.shape

            # Attention entropy: -sum(A * log(A)) trên trục spatial S
            attn_ent = -torch.sum(last_attn * torch.log(last_attn + self.epsilon), dim=-1)  # [B, H, N]
            attn_entropy_mean = attn_ent.mean(dim=[1, 2], keepdim=True).view(B, 1)  # [B, 1]

            # Peak attention intensity
            attn_peak = torch.max(last_attn, dim=-1)[0]  # [B, H, N]
            attn_peak_mean = attn_peak.mean(dim=[1, 2], keepdim=True).view(B, 1)  # [B, 1]

            # Head diversity: Độ lệch chuẩn giữa các attention patterns của các heads
            head_mean_attn = last_attn.mean(dim=2)  # [B, H, S] trung bình qua các queries
            head_std = torch.std(head_mean_attn, dim=1).mean(dim=-1, keepdim=True)  # [B, 1]
            attn_head_diversity = head_std

            # Spatial dispersion: Phương sai vị trí chú ý
            spatial_indices = torch.linspace(0, 1, steps=S, device=device).view(1, 1, 1, S)
            mean_pos = torch.sum(last_attn * spatial_indices, dim=-1, keepdim=True)  # [B, H, N, 1]
            var_pos = torch.sum(last_attn * (spatial_indices - mean_pos) ** 2, dim=-1)  # [B, H, N]
            attn_spatial_dispersion = var_pos.mean(dim=[1, 2], keepdim=True).view(B, 1)  # [B, 1]
        else:
            attn_entropy_mean = torch.zeros((batch_size, 1), device=device)
            attn_peak_mean = torch.zeros((batch_size, 1), device=device)
            attn_head_diversity = torch.zeros((batch_size, 1), device=device)
            attn_spatial_dispersion = torch.zeros((batch_size, 1), device=device)

        # --- Nhóm 3: Decoder Refinement Drift ---
        refinement_deltas: List[torch.Tensor] = []
        num_states = len(state.decoder_hidden_states) if state.decoder_hidden_states is not None else 0

        if num_states >= 2:
            for l in range(1, num_states):
                prev_h = state.decoder_hidden_states[l - 1]  # [B, N, D]
                curr_h = state.decoder_hidden_states[l]      # [B, N, D]
                # Tính độ lệch chuẩn hóa ổn định số học: tỷ số giữa tổng norm chênh lệch và tổng norm vector
                diff_sum = torch.norm(curr_h - prev_h, p=2, dim=-1).sum(dim=-1, keepdim=True)  # [B, 1]
                prev_sum = torch.norm(prev_h, p=2, dim=-1).sum(dim=-1, keepdim=True) + 1e-4  # [B, 1]
                rel_delta = diff_sum / prev_sum  # [B, 1]
                refinement_deltas.append(rel_delta)

        # Chuẩn hóa về đúng 5 giá trị delta (cố định kích thước vector)
        target_layers = 5
        if len(refinement_deltas) == 0:
            padded_deltas = [torch.zeros((batch_size, 1), device=device) for _ in range(target_layers)]
        elif len(refinement_deltas) >= target_layers:
            padded_deltas = refinement_deltas[:target_layers]
        else:
            padded_deltas = list(refinement_deltas)
            last_val = refinement_deltas[-1]
            while len(padded_deltas) < target_layers:
                padded_deltas.append(last_val)

        # --- Nhóm 4: Query Representation Diversity & Collapse ---
        if num_states > 0:
            final_h = state.decoder_hidden_states[-1]  # [B, N, D]
            normed_h = F.normalize(final_h, p=2, dim=-1)  # [B, N, D]
            sim_matrix = torch.bmm(normed_h, normed_h.transpose(1, 2))  # [B, N, N]
            N_q = final_h.shape[1]
            eye_mask = torch.eye(N_q, device=device).unsqueeze(0).bool()
            off_diagonal_sim = sim_matrix.masked_select(~eye_mask).view(batch_size, N_q * (N_q - 1))
            query_cosine_sim_mean = off_diagonal_sim.mean(dim=-1, keepdim=True)  # [B, 1]

            h_norms = torch.norm(final_h, p=2, dim=-1)  # [B, N]
            query_norm_std = torch.std(h_norms, dim=-1, keepdim=True)  # [B, 1]
        else:
            query_cosine_sim_mean = torch.zeros((batch_size, 1), device=device)
            query_norm_std = torch.zeros((batch_size, 1), device=device)

        all_features = torch.cat(
            [
                query_top1_mean,
                query_margin_mean,
                query_entropy_mean,
                active_query_ratio,
                bg_prob_mean,
                attn_entropy_mean,
                attn_peak_mean,
                attn_head_diversity,
                attn_spatial_dispersion,
                *padded_deltas,
                query_cosine_sim_mean,
                query_norm_std,
            ],
            dim=-1,
        )

        return all_features.detach().cpu().numpy()

    def get_feature_dict(self, feature_vector: np.ndarray) -> Dict[str, float]:
        """
        Chuyển đổi vector đặc trưng 1 chiều thành dictionary để tiện phân tích.
        """
        vec = feature_vector.squeeze()
        return {name: float(vec[i]) for i, name in enumerate(self.FEATURE_NAMES)}
