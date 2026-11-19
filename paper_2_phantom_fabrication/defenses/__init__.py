# -*- coding: utf-8 -*-
"""
Package: paper_2_phantom_fabrication.defenses
Hệ thống các cơ chế phòng thủ và giảm thiểu rủi ro đối kháng cho RT-DETR:
  1. SQD-Filter (Spatial Query Deduplication & Dispersion Filter): Đóng góp cốt lõi của Paper 2
  2. Baseline Defenses: Raw Model, Standard NMS, Threshold Elevation, Spatial DBSCAN
"""

from .sqd_filter import (
    SpatialQueryDeduplicationFilter,
    apply_sqd_filter,
)
from .baseline_defenses import (
    apply_defense_to_predictions,
    evaluate_defenses_suite,
)

__all__ = [
    "SpatialQueryDeduplicationFilter",
    "apply_sqd_filter",
    "apply_defense_to_predictions",
    "evaluate_defenses_suite",
]
