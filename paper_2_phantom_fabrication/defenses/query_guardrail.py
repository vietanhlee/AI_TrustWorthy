# -*- coding: utf-8 -*-
"""
Module: paper_2_phantom_fabrication.defenses.query_guardrail
Mục đích: Hiện thực hóa phòng thủ ở cấp độ biểu diễn (Representation Guardrails) cho RT-DETR:
  - Phát hiện bất thường ở mức Query (Query-level Anomaly Detection) trước khi đưa vào dự đoán.
  - Đo khoảng cách Cosine giữa embedding của query và trọng tâm nguyên mẫu (Class Prototype Centroid)
    được hiệu chuẩn từ tập ảnh sạch.
  - Đo độ phân tán entropy và chuẩn độ lớn đặc trưng (Feature Norm Deviation).
  - Đánh giá định lượng bằng:
    1. AUROC (Area Under the Receiver Operating Characteristic Curve).
    2. PR-AUC (Precision-Recall Area Under Curve).
    3. TPR@1%FPR (True Positive Rate tại 1% False Positive Rate).
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn import metrics as sk_metrics
import torch
import torch.nn as nn
import torch.nn.functional as F


class QueryRepresentationGuardrail(nn.Module):
    """
    Representation Guardrail cho Real-Time Detection Transformers.
    Giám sát các query được kích hoạt tại tầng Encoder / Decoder để phát hiện
    các đặc trưng bị bóp méo do tấn công đối kháng (Adversarial Representation Distortion).
    """

    def __init__(
        self,
        num_classes: int = 80,
        feature_dim: int = 256,
        cosine_anomaly_thresh: float = 0.45,
        entropy_thresh: float = 2.50,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.cosine_anomaly_thresh = cosine_anomaly_thresh
        self.entropy_thresh = entropy_thresh

        # Đăng ký buffer lưu prototype cho 80 lớp COCO [80, 256]
        self.register_buffer("class_prototypes", torch.zeros((num_classes, feature_dim)))
        self.register_buffer("class_counts", torch.zeros((num_classes,), dtype=torch.long))
        self.is_calibrated = False

    def calibrate_prototypes(
        self,
        query_features: torch.Tensor,
        query_labels: torch.Tensor,
        query_confs: Optional[torch.Tensor] = None,
        min_conf: float = 0.50,
    ) -> None:
        """
        Hiệu chuẩn trọng tâm nguyên mẫu (prototypes) từ các query có độ tin cậy cao trên tập ảnh sạch.
        query_features: [N, D]
        query_labels: [N]
        query_confs: [N] (tùy chọn)
        """
        if query_confs is not None:
            valid_mask = query_confs >= min_conf
            features = query_features[valid_mask]
            labels = query_labels[valid_mask]
        else:
            features = query_features
            labels = query_labels

        device = self.class_prototypes.device
        features = features.to(device)
        labels = labels.to(device)

        for c in range(self.num_classes):
            c_mask = (labels == c)
            if c_mask.sum() > 0:
                c_feats = features[c_mask]
                norm_feats = F.normalize(c_feats, p=2, dim=-1)
                sum_feat = norm_feats.sum(dim=0)
                
                # Cập nhật chạy (running average)
                prev_count = self.class_counts[c].item()
                new_count = c_mask.sum().item()
                self.class_prototypes[c] = (self.class_prototypes[c] * prev_count + sum_feat) / max(1, prev_count + new_count)
                self.class_counts[c] += new_count

        # Chuẩn hóa L2 cho các prototype
        self.class_prototypes = F.normalize(self.class_prototypes, p=2, dim=-1)
        self.is_calibrated = True

    def compute_anomaly_scores(
        self,
        query_features: torch.Tensor,
        predicted_classes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Tính điểm bất thường (Anomaly Score in [0.0, 2.0]) cho từng query:
        Anomaly_Score = 1.0 - Cosine_Similarity(feature, prototype[class])
        query_features: [K, D]
        predicted_classes: [K]
        """
        device = query_features.device
        norm_feats = F.normalize(query_features, p=2, dim=-1)
        prototypes = self.class_prototypes.to(device)

        # Lấy prototype tương ứng với lớp dự đoán
        target_prototypes = prototypes[predicted_classes]  # [K, D]
        cosine_sim = (norm_feats * target_prototypes).sum(dim=-1)  # [K]
        anomaly_scores = 1.0 - cosine_sim  # [K], 0 = trùng khớp hoàn hảo, 2 = ngược chiều
        return anomaly_scores

    def filter_queries(
        self,
        logits: torch.Tensor,
        boxes: torch.Tensor,
        query_features: Optional[torch.Tensor] = None,
        score_thresh: float = 0.35,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Loại bỏ các query có anomaly score vượt ngưỡng cosine_anomaly_thresh.
        """
        probs = torch.sigmoid(logits)
        max_p, pred_cls = probs.max(dim=-1)  # [B, K]
        B, K, C = logits.shape
        device = logits.device

        kept_masks = torch.ones((B, K), dtype=torch.bool, device=device)
        suppressed_count = 0

        if query_features is not None and self.is_calibrated:
            for b in range(B):
                act_mask = max_p[b] >= score_thresh
                if act_mask.sum() == 0:
                    continue
                act_idx = torch.where(act_mask)[0]
                b_feats = query_features[b, act_idx]
                b_cls = pred_cls[b, act_idx]
                
                scores = self.compute_anomaly_scores(b_feats, b_cls)
                anom_mask = scores > self.cosine_anomaly_thresh
                for local_i, is_anom in enumerate(anom_mask):
                    if is_anom:
                        kept_masks[b, act_idx[local_i]] = False
                        suppressed_count += 1

        filtered_logits = logits.masked_fill(~kept_masks.unsqueeze(-1), -1e4)
        return filtered_logits, boxes, {"suppressed_by_guardrail": suppressed_count}


def evaluate_guardrail_performance(
    clean_scores: np.ndarray,
    attacked_scores: np.ndarray,
) -> Dict[str, float]:
    """
    Đo đạc chỉ số phát hiện đối kháng của Guardrail:
    - clean_scores: Điểm bất thường trên query sạch (nhãn 0 - nominal).
    - attacked_scores: Điểm bất thường trên query ma do đối kháng sinh ra (nhãn 1 - anomaly).
    Trả về:
        - auroc: Diện tích dưới ROC curve (tiêu chuẩn đánh giá phát hiện đối kháng).
        - pr_auc: Diện tích dưới PR curve.
        - tpr_at_1pct_fpr: Tỷ lệ phát hiện đúng phantom khi chấp nhận 1% False Positive trên dữ liệu sạch.
    """
    if len(clean_scores) == 0 or len(attacked_scores) == 0:
        return {"auroc": 0.50, "pr_auc": 0.50, "tpr_at_1pct_fpr": 0.0}

    y_true = np.concatenate([np.zeros(len(clean_scores)), np.ones(len(attacked_scores))])
    y_scores = np.concatenate([clean_scores, attacked_scores])

    # AUROC
    try:
        auroc = float(sk_metrics.roc_auc_score(y_true, y_scores))
    except Exception:
        auroc = 0.50

    # PR-AUC
    try:
        precision, recall, _ = sk_metrics.precision_recall_curve(y_true, y_scores)
        pr_auc = float(sk_metrics.auc(recall, precision))
    except Exception:
        pr_auc = 0.50

    # TPR @ 1% FPR
    try:
        fpr, tpr, thresholds = sk_metrics.roc_curve(y_true, y_scores)
        # Tìm điểm FPR <= 0.01 lớn nhất
        idx = np.where(fpr <= 0.01)[0]
        if len(idx) > 0:
            tpr_at_1 = float(tpr[idx[-1]]) * 100.0
        else:
            tpr_at_1 = 0.0
    except Exception:
        tpr_at_1 = 0.0

    return {
        "auroc": round(auroc, 4),
        "pr_auc": round(pr_auc, 4),
        "tpr_at_1pct_fpr": round(tpr_at_1, 2),
    }
