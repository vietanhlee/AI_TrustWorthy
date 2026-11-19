"""
Module: rtdetr_advanced_features.py
Mục đích: Trích xuất các đặc trưng phân tích sâu nội tại (Deep Internal Representation) của RT-DETR:
1. Layer-wise Hidden Embeddings: Biểu diễn ẩn tại từng tầng Decoder (Layer 0, 1, 2, 3) theo Lee et al. (NeurIPS 2018).
2. Prediction Inconsistency Metrics: Độ đứt gãy và bất đồng dự đoán giữa các tầng Decoder
   (Cross-layer KL Divergence, Box Drift, Rank Turbulence) lấy cảm hứng từ Yun et al. (2025) và Prediction Inconsistency (2025).
3. 1D Baseline Metrics: active_query_ratio, query_top1_mean, refinement_delta.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F

from paper_3_defense_guardrail.transformer_monitors.detr_wrapper import DETRInternalState


@dataclass
class RTDETRDeepFeatures:
    """
    Cấu trúc dữ liệu chứa toàn bộ đặc trưng sâu từ một mẫu suy luận RT-DETR.
    """
    # 1. Layer-wise Pooled Vectors cho Mahalanobis (Tuple 4 vector d_dim cho 4 tầng)
    layer_vectors: List[np.ndarray]  # Danh sách L vectors [D] (ví dụ 4 vectors 64 chiều)

    # 2. Prediction Inconsistency Metrics
    kl_mean_consecutive: float       # KL divergence trung bình giữa các tầng kế tiếp
    kl_max_consecutive: float        # KL divergence cực đại giữa các tầng kế tiếp
    kl_penultimate_final: float      # KL divergence giữa tầng áp chót và tầng chót
    box_drift_mean: float            # Độ dịch chuyển bounding box L1 trung bình
    box_drift_max: float             # Độ dịch chuyển bounding box L1 cực đại
    box_iou_min: float               # IoU tối thiểu giữa các hộp dự đoán qua các tầng
    rank_turbulence: float           # Độ xáo trộn thứ hạng của top-k queries (1.0 - Spearman Corr)

    # 3. Baseline 1D Metrics
    active_query_ratio: float = 0.0        # Tỷ lệ query có P >= conf_threshold (Discrete count k / 300)
    active_query_soft: float = 0.0         # Tỷ lệ query kích hoạt mềm (Continuous Soft Activation k_soft / 300)
    query_top1_mean: float = 0.0           # Xác suất top-1 trung bình của các queries (Liên tục)
    query_margin_mean: float = 0.0         # Margin giữa lớp cao nhất và lớp thứ hai
    refinement_delta_l1: float = 0.0       # Độ dịch chuyển biểu diễn ẩn giữa tầng đầu và tầng cuối

    # 4. Spatial Geometry & Dispersion Metrics (Chuyên biệt phân tích Localized Patch)
    box_area_mean: float = 0.0             # Diện tích trung bình (w*h) của top-k bounding boxes
    spatial_dispersion: float = 0.0        # Độ phân tán tâm hộp sqrt(var(cx) + var(cy)) của top-k queries

    # 5. Encoder Candidate Selection Metrics (Chuyên biệt bắt đòn AQSM / Selection Starvation)
    enc_top300_mean: float = 0.0     # Điểm phân loại trung bình của Top-300 candidates từ Encoder
    enc_margin: float = 0.0          # Margin giữa Top-300 và 1800 candidates còn lại
    enc_dropoff: float = 0.0         # Tỷ số suy giảm điểm từ candidate #1 đến candidate #300
    enc_bg_mean: float = 0.0         # Điểm trung bình của 1800 candidates nền phía dưới

    def to_feature_vector(self) -> np.ndarray:
        """Chuyển đổi các chỉ số thống kê thành vector 1D cho phân tích."""
        stats = [
            self.kl_mean_consecutive,
            self.kl_max_consecutive,
            self.kl_penultimate_final,
            self.box_drift_mean,
            self.box_drift_max,
            self.box_iou_min,
            self.rank_turbulence,
            self.active_query_ratio,
            self.query_top1_mean,
            self.query_margin_mean,
            self.refinement_delta_l1,
            self.box_area_mean,
            self.spatial_dispersion,
        ]
        return np.array(stats, dtype=np.float32)


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Chuyển đổi bounding boxes từ định dạng [cx, cy, w, h] sang [x1, y1, x2, y2]."""
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def compute_box_iou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Tính IoU giữa 2 tập hộp tương ứng [N, 4] dạng [cx, cy, w, h].
    """
    b1 = box_cxcywh_to_xyxy(box1)
    b2 = box_cxcywh_to_xyxy(box2)

    lt = torch.max(b1[..., :2], b2[..., :2])
    rb = torch.min(b1[..., 2:], b2[..., 2:])

    wh = (rb - lt).clamp(min=0.0)
    inter = wh[..., 0] * wh[..., 1]

    area1 = (b1[..., 2] - b1[..., 0]).clamp(min=0.0) * (b1[..., 3] - b1[..., 1]).clamp(min=0.0)
    area2 = (b2[..., 2] - b2[..., 0]).clamp(min=0.0) * (b2[..., 3] - b2[..., 1]).clamp(min=0.0)

    union = area1 + area2 - inter + eps
    return inter / union


class RTDETRDeepFeatureExtractor:
    """
    Bộ trích xuất đặc trưng sâu RT-DETR tối ưu hóa cho Edge Computing.
    """

    def __init__(
        self,
        conf_threshold: float = 0.35,
        top_k_queries: int = 10,
        projection_dim: int = 64,
        random_seed: int = 42,
    ) -> None:
        self.conf_threshold = conf_threshold
        self.top_k_queries = top_k_queries
        self.projection_dim = projection_dim
        self.projection_matrix: Optional[torch.Tensor] = None
        self.random_seed = random_seed

    def _get_projection_matrix(self, d_model: int, device: torch.device) -> torch.Tensor:
        """Tạo ma trận chiếu ngẫu nhiên trực giao (Orthogonal Projection) để giảm số chiều ổn định."""
        if self.projection_matrix is None or self.projection_matrix.shape[0] != d_model:
            rng = torch.Generator(device="cpu").manual_seed(self.random_seed)
            proj = torch.randn(d_model, self.projection_dim, generator=rng)
            q, _ = torch.linalg.qr(proj)
            self.projection_matrix = q.to(device)
        return self.projection_matrix

    def extract_from_state(self, state: DETRInternalState) -> List[RTDETRDeepFeatures]:
        """
        Trích xuất đặc trưng sâu cho toàn bộ batch từ DETRInternalState.
        """
        device = state.logits.device
        B, N, C = state.logits.shape

        # 1. Thu thập chuỗi Logits qua các tầng
        # intermediate_logits: [B, L_inter, N, C]
        if state.intermediate_logits is not None:
            L_inter = state.intermediate_logits.shape[1]
            all_logits = torch.cat([state.intermediate_logits, state.logits.unsqueeze(1)], dim=1) # [B, L_total, N, C]
        else:
            all_logits = state.logits.unsqueeze(1) # [B, 1, N, C]

        L_total = all_logits.shape[1]
        all_probs = torch.sigmoid(all_logits) # [B, L_total, N, C]

        # 2. Thu thập chuỗi Bounding Boxes qua các tầng
        if state.intermediate_pred_boxes is not None:
            all_boxes = torch.cat([state.intermediate_pred_boxes, state.pred_boxes.unsqueeze(1)], dim=1) # [B, L_total, N, 4]
        else:
            all_boxes = state.pred_boxes.unsqueeze(1)

        # 3. Chuỗi Hidden States qua các tầng
        # decoder_hidden_states: Tuple of (L+1) tensors [B, N, D]
        hidden_states = state.decoder_hidden_states
        d_model = hidden_states[-1].shape[-1]
        proj_matrix = self._get_projection_matrix(d_model, device)

        batch_features: List[RTDETRDeepFeatures] = []

        for b in range(B):
            # Xác định top-k queries dựa trên tầng cuối cùng
            final_probs = all_probs[b, -1] # [N, C]
            max_p_final, _ = final_probs.max(dim=-1) # [N]
            k_val = min(self.top_k_queries, N)
            topk_vals, topk_indices = torch.topk(max_p_final, k=k_val)

            # --- A. Layer-wise Pooled Vectors (cho Mahalanobis) ---
            layer_vecs = []
            for l_idx, h_layer in enumerate(hidden_states):
                # Lấy trung bình embedding của top-k queries tại tầng này
                topk_h = h_layer[b, topk_indices] # [k_val, D]
                mean_h = topk_h.mean(dim=0, keepdim=True) # [1, D]
                # Chiếu xuống projection_dim (ví dụ 64) để tránh quá khớp hiệp phương sai
                reduced_h = torch.mm(mean_h, proj_matrix).squeeze(0) # [projection_dim]
                layer_vecs.append(reduced_h.detach().cpu().numpy())

            # --- B. Prediction Inconsistency Metrics ---
            if L_total > 1:
                consecutive_kls = []
                consecutive_box_drifts = []
                consecutive_ious = []
                rank_turbulences = []

                for l in range(1, L_total):
                    p_prev = all_probs[b, l - 1, topk_indices].detach().clamp(min=1e-6, max=1.0 - 1e-6)
                    p_curr = all_probs[b, l, topk_indices].detach().clamp(min=1e-6, max=1.0 - 1e-6)

                    # Binary KL divergence cho Sigmoid multi-label:
                    kl = p_curr * torch.log(p_curr / p_prev) + (1.0 - p_curr) * torch.log((1.0 - p_curr) / (1.0 - p_prev))
                    kl_val = float(kl.sum(dim=-1).mean().item())
                    consecutive_kls.append(kl_val)

                    # BBox drift
                    b_prev = all_boxes[b, l - 1, topk_indices].detach()
                    b_curr = all_boxes[b, l, topk_indices].detach()
                    drift = float(torch.abs(b_curr - b_prev).sum(dim=-1).mean().item())
                    consecutive_box_drifts.append(drift)

                    # IoU
                    ious = compute_box_iou(b_curr, b_prev)
                    consecutive_ious.append(float(ious.mean().item()))

                    # Rank correlation giữa 2 tầng
                    p_prev_max, _ = all_probs[b, l - 1].detach().max(dim=-1)
                    p_curr_max, _ = all_probs[b, l].detach().max(dim=-1)
                    # Tính rank xáo trộn trên top-20 queries
                    top20 = min(20, N)
                    _, r_prev = torch.topk(p_prev_max, k=top20)
                    _, r_curr = torch.topk(p_curr_max, k=top20)
                    # Tỷ lệ query khác biệt giữa top-20 của 2 tầng
                    intersection = len(set(r_prev.tolist()).intersection(set(r_curr.tolist())))
                    turbulence = 1.0 - (intersection / float(top20))
                    rank_turbulences.append(turbulence)

                kl_mean = float(np.mean(consecutive_kls))
                kl_max = float(np.max(consecutive_kls))
                kl_penult = float(consecutive_kls[-1])
                box_drift_mean = float(np.mean(consecutive_box_drifts))
                box_drift_max = float(np.max(consecutive_box_drifts))
                box_iou_min = float(np.min(consecutive_ious))
                rank_turb = float(np.mean(rank_turbulences))
            else:
                kl_mean = kl_max = kl_penult = 0.0
                box_drift_mean = box_drift_max = 0.0
                box_iou_min = 1.0
                rank_turb = 0.0

            # --- C. Baseline 1D Metrics ---
            max_p_final_d = max_p_final.detach()
            active_ratio = float((max_p_final_d >= self.conf_threshold).float().mean().item())
            # Continuous Soft Active Query Ratio (loại bỏ hoàn toàn Discrete Tie Artifact)
            active_soft = float(torch.sigmoid(20.0 * (max_p_final_d - self.conf_threshold)).mean().item())
            top1_mean = float(max_p_final_d.mean().item())

            # Margin giữa top-1 và top-2 lớp
            sorted_probs, _ = final_probs.detach().sort(dim=-1, descending=True)
            margin = float((sorted_probs[:, 0] - sorted_probs[:, 1]).mean().item())

            # Refinement delta L1 giữa tầng decoder đầu và cuối
            h_init = hidden_states[0][b].detach()
            h_final = hidden_states[-1][b].detach()
            refinement_delta = float(torch.norm(h_final - h_init, p=1, dim=-1).mean().item())

            # --- D. Spatial Geometry & Dispersion (cho Patch Analysis) ---
            b_final = all_boxes[b, -1, topk_indices].detach()  # [k_val, 4] dạng [cx, cy, w, h]
            w_box = b_final[:, 2].clamp(min=0.0, max=1.0)
            h_box = b_final[:, 3].clamp(min=0.0, max=1.0)
            area = w_box * h_box
            box_area_mean = float(area.mean().item())

            cx_box = b_final[:, 0]
            cy_box = b_final[:, 1]
            if k_val > 1:
                cx_var = float(cx_box.var(unbiased=False).item())
                cy_var = float(cy_box.var(unbiased=False).item())
                spatial_dispersion = float(np.sqrt(cx_var + cy_var))
            else:
                spatial_dispersion = 0.0

            # --- E. Encoder Candidate Selection Metrics (cho AQSM Analysis) ---
            # NGHIÊM CẤM FALLBACK HẰNG SỐ GIẢ ĐỊNH: Bắt buộc phải có tensor enc_outputs_class từ forward pass
            if state.enc_outputs_class is None:
                raise RuntimeError(
                    "LỖI TOÀN VẸN MÔ HÌNH: Không tìm thấy 'enc_outputs_class' trong DETRInternalState! "
                    "Mô hình RT-DETR bắt buộc phải trích xuất được logits của encoder phục vụ phân tích ESM. "
                    "Nghiêm cấm fallback giá trị giả lập cố định để bảo đảm tính trung thực khoa học 100%."
                )

            enc_probs = torch.sigmoid(state.enc_outputs_class[b]).detach()
            max_enc_p, _ = enc_probs.max(dim=-1)
            sorted_enc_p, _ = torch.sort(max_enc_p, descending=True)
            enc_top300 = float(sorted_enc_p[:300].mean().item())
            enc_bg = float(sorted_enc_p[300:].mean().item())
            enc_margin = enc_top300 - enc_bg
            top1_val = float(sorted_enc_p[0].item())
            top300_val = float(sorted_enc_p[299].item()) if len(sorted_enc_p) >= 300 else 1e-4
            enc_dropoff = top1_val / max(1e-4, top300_val)

            feat = RTDETRDeepFeatures(
                layer_vectors=layer_vecs,
                kl_mean_consecutive=kl_mean,
                kl_max_consecutive=kl_max,
                kl_penultimate_final=kl_penult,
                box_drift_mean=box_drift_mean,
                box_drift_max=box_drift_max,
                box_iou_min=box_iou_min,
                rank_turbulence=rank_turb,
                active_query_ratio=active_ratio,
                active_query_soft=active_soft,
                query_top1_mean=top1_mean,
                query_margin_mean=margin,
                refinement_delta_l1=refinement_delta,
                box_area_mean=box_area_mean,
                spatial_dispersion=spatial_dispersion,
                enc_top300_mean=enc_top300,
                enc_margin=enc_margin,
                enc_dropoff=enc_dropoff,
                enc_bg_mean=enc_bg,
            )
            batch_features.append(feat)

        return batch_features
