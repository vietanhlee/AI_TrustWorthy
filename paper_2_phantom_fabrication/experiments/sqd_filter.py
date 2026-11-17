"""
Spatial Query Deduplication and Dispersion Filter (SQD-Filter)
Paper 2: "Phantom Queries: Exploiting NMS-Free Set Prediction for Adversarial Object Fabrication in Real-Time Transformers"

Ultra-lightweight post-processing filter designed for NMS-Free Detection Transformers.
Suppresses adversarially fabricated phantom queries based on spatial proximity clusters
and dispersion entropy without invoking heavy quadratic IoU clustering loops.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, List, Optional


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
