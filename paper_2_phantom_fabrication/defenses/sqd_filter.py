# -*- coding: utf-8 -*-
"""
Module: paper_2_phantom_fabrication.defenses.sqd_filter
Paper 2: "Phantom Queries: Exploiting NMS-Free Set Prediction for Adversarial Object Fabrication in Real-Time Transformers"

Spatial Query Deduplication and Dispersion Filter (SQD-Filter)
Ultra-lightweight post-processing filter designed for NMS-Free Detection Transformers.
Suppresses adversarially fabricated phantom queries based on spatial proximity clusters
and dispersion entropy without invoking heavy quadratic IoU clustering loops.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from paper_2_phantom_fabrication.core.coco_gt import box_cxcywh_to_xyxy


class SpatialQueryDeduplicationFilter(nn.Module):
    """
    Lightweight, vectorized spatial filter for filtering fabricated queries in RT-DETR.
    """

    def __init__(
        self,
        distance_threshold: float = 0.08,  # normalized coordinate distance
        conf_threshold: float = 0.35,
        max_active_queries: int = 300,
    ):
        super().__init__()
        self.distance_threshold = distance_threshold
        self.conf_threshold = conf_threshold
        self.max_active_queries = max_active_queries

    @torch.no_grad()
    def forward(
        self,
        boxes: torch.Tensor,       # [B, K, 4] (cx, cy, w, h) in normalized [0, 1]
        logits: torch.Tensor,      # [B, K, C] raw classification logits
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Filters out spatially clustered fabricated queries.
        Returns:
            filtered_boxes: [B, K, 4]
            filtered_probs: [B, K, C]
            telemetry: metrics on suppressed queries
        """
        device = boxes.device
        B, K, C = logits.shape
        probs = torch.sigmoid(logits)
        max_p, pred_class = probs.max(dim=-1)  # [B, K], [B, K]

        kept_masks = torch.ones((B, K), dtype=torch.bool, device=device)
        suppressed_count = 0

        for b in range(B):
            active_mask = (max_p[b] >= self.conf_threshold)
            active_indices = torch.where(active_mask)[0]
            n_active = len(active_indices)

            if n_active <= 1:
                continue

            active_boxes = boxes[b, active_indices]  # [N_act, 4]
            active_classes = pred_class[b, active_indices]  # [N_act]
            active_conf = max_p[b, active_indices]  # [N_act]

            # Vectorized pairwise center distances: ||(cx_i, cy_i) - (cx_j, cy_j)||_2
            centers = active_boxes[:, :2]  # [N_act, 2]
            diff = centers.unsqueeze(0) - centers.unsqueeze(1)  # [N_act, N_act, 2]
            dist_matrix = torch.norm(diff, dim=-1)  # [N_act, N_act]

            # Same-class mask
            same_class_mask = (active_classes.unsqueeze(0) == active_classes.unsqueeze(1))
            # Proximity mask
            clustered = (dist_matrix < self.distance_threshold) & same_class_mask

            # Sort by confidence descending
            sorted_order = torch.argsort(active_conf, descending=True)
            suppressed = torch.zeros(n_active, dtype=torch.bool, device=device)

            for idx in sorted_order:
                if suppressed[idx]:
                    continue
                # Suppress other clustered queries of lower confidence
                neighbors = torch.where(clustered[idx])[0]
                for n_idx in neighbors:
                    if n_idx != idx and not suppressed[n_idx]:
                        suppressed[n_idx] = True
                        kept_masks[b, active_indices[n_idx]] = False
                        suppressed_count += 1

        filtered_boxes = boxes * kept_masks.unsqueeze(-1)
        filtered_probs = probs * kept_masks.unsqueeze(-1)

        telemetry = {
            "suppressed_queries": suppressed_count,
            "suppression_rate": float(suppressed_count / max(1, (max_p >= self.conf_threshold).sum().item())),
        }

        return filtered_boxes, filtered_probs, telemetry


def apply_sqd_filter(
    logits: torch.Tensor,
    pred_boxes: torch.Tensor,
    score_thresh: float = 0.35,
    dist_thresh: float = 0.15,
) -> List[Dict[str, np.ndarray]]:
    """
    Hàm tiện ích chạy Spatial Query Deduplication Filter trên dự đoán RT-DETR.
    Nhận:
        logits: [B, K, C] hoặc tensor logits
        pred_boxes: [B, K, 4] dạng cxcywh [0, 1]
    Trả về:
        List[Dict[str, np.ndarray]] chứa boxes (xyxy [0, 1]), scores, labels
    """
    probs = torch.sigmoid(logits.detach().cpu())
    max_p, pred_cls = probs.max(dim=-1)
    boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes.detach().cpu()).clamp(0.0, 1.0)

    bsz = len(logits)
    results = []

    for b in range(bsz):
        act_mask = max_p[b] >= score_thresh
        b_boxes = boxes_xyxy[b, act_mask].numpy()
        b_scores = max_p[b, act_mask].numpy()
        b_labels = pred_cls[b, act_mask].numpy()

        if len(b_boxes) <= 1:
            results.append({"boxes": b_boxes, "scores": b_scores, "labels": b_labels})
            continue

        centers = np.stack([(b_boxes[:, 0] + b_boxes[:, 2]) / 2.0, (b_boxes[:, 1] + b_boxes[:, 3]) / 2.0], axis=1)
        diff = centers[:, None, :] - centers[None, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=-1))

        order = np.argsort(-b_scores)
        suppressed = np.zeros(len(b_boxes), dtype=bool)
        keep = []

        for idx in order:
            if suppressed[idx]:
                continue
            keep.append(idx)
            same_cls = (b_labels == b_labels[idx])
            within_dist = (dists[idx] < dist_thresh)
            suppressed |= (same_cls & within_dist)

        keep = sorted(keep)
        results.append({
            "boxes": b_boxes[keep],
            "scores": b_scores[keep],
            "labels": b_labels[keep],
        })

    return results


# Bí danh tương thích ngược và tiện ích đánh giá
SQDFilter = SpatialQueryDeduplicationFilter


def evaluate_sqd_filter_on_detections(
    detections: List[Dict[str, np.ndarray]],
    divergence_threshold: float = 0.15,
    iou_threshold: float = 0.35,
    enable_cross_class_collision: bool = True,
) -> List[Dict[str, np.ndarray]]:
    """
    Áp dụng lọc trùng lặp không gian (SQD Filter) trực tiếp trên danh sách detection thô.
    Hỗ trợ cả khử trùng lặp đồng lớp (same-class deduplication) và khử xung đột không gian
    dị lớp do ảo giác đối kháng (cross-class collision suppression).
    detections: List các dict {"boxes": ndarray[N, 4], "scores": ndarray[N], "labels": ndarray[N]}
    """
    filtered_list = []
    for det in detections:
        boxes = det.get("boxes", np.zeros((0, 4), dtype=np.float32))
        scores = det.get("scores", np.zeros((0,), dtype=np.float32))
        labels = det.get("labels", np.zeros((0,), dtype=np.int64))

        if len(boxes) <= 1:
            filtered_list.append({"boxes": boxes, "scores": scores, "labels": labels})
            continue

        centers = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2.0, (boxes[:, 1] + boxes[:, 3]) / 2.0], axis=1)
        diff = centers[:, None, :] - centers[None, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=-1))

        # Tính ma trận IoU
        x1 = np.maximum(boxes[:, None, 0], boxes[None, :, 0])
        y1 = np.maximum(boxes[:, None, 1], boxes[None, :, 1])
        x2 = np.minimum(boxes[:, None, 2], boxes[None, :, 2])
        y2 = np.minimum(boxes[:, None, 3], boxes[None, :, 3])
        inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = area[:, None] + area[None, :] - inter
        ious = inter / np.maximum(union, 1e-6)

        order = np.argsort(-scores)
        suppressed = np.zeros(len(boxes), dtype=bool)
        keep = []

        for idx in order:
            if suppressed[idx]:
                continue
            keep.append(idx)
            for other in range(len(boxes)):
                if other == idx or suppressed[other]:
                    continue
                # Điều kiện 1: Khử trùng lặp đồng lớp theo khoảng cách tâm hoặc IoU
                if labels[idx] == labels[other]:
                    if dists[idx, other] < divergence_threshold or ious[idx, other] > iou_threshold:
                        suppressed[other] = True
                # Điều kiện 2: Khử xung đột không gian dị lớp (hai vật thể khác loại đè lên cùng 1 vị trí)
                elif enable_cross_class_collision:
                    if ious[idx, other] > iou_threshold or dists[idx, other] < 0.08:
                        suppressed[other] = True

        keep = sorted(keep)
        filtered_list.append({
            "boxes": boxes[keep],
            "scores": scores[keep],
            "labels": labels[keep],
        })

    return filtered_list

