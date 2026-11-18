# -*- coding: utf-8 -*-
"""
Package: paper_2_phantom_fabrication.core
Mô-đun lõi phục vụ nạp dữ liệu COCO, tính toán độ đo mAP / Hungarian Matching
và quản lý cấu hình thực thi phần cứng cho Paper 2 (Phantom Queries Fabrication).
"""

from .coco_gt import (
    load_flexible_coco_cohort,
    evaluate_detection_metrics,
    discover_coco_annotations,
    calculate_iou_matrix,
    hungarian_match_predictions,
    compute_ap_voc2012
)
from .config import (
    get_device,
    configure_runtime_threads,
    MODEL_NAME,
    IMG_SIZE,
    NUM_QUERIES
)

__all__ = [
    "load_flexible_coco_cohort",
    "evaluate_detection_metrics",
    "discover_coco_annotations",
    "calculate_iou_matrix",
    "hungarian_match_predictions",
    "compute_ap_voc2012",
    "get_device",
    "configure_runtime_threads",
    "MODEL_NAME",
    "IMG_SIZE",
    "NUM_QUERIES"
]
