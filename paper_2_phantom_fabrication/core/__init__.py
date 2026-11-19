# -*- coding: utf-8 -*-
"""
Package: paper_2_phantom_fabrication.core
Mô-đun lõi phục vụ nạp dữ liệu COCO, tính toán độ đo mAP / Hungarian Matching
và quản lý cấu hình thực thi phần cứng cho Paper 2 (Phantom Queries Fabrication).
"""

from .coco_gt import (
    CocoCohortDataset,
    load_flexible_coco_cohort,
    discover_coco_annotations,
    load_coco_val2017_gt_cohort,
    box_cxcywh_to_xyxy,
    compute_iou_matrix,
    build_coco_category_mapping,
    postprocess_rtdetr_predictions,
    evaluate_coco_predictions,
    compute_per_image_coco_aps,
    bootstrap_paired_diff,
    repeated_measures_2x2_anova,
)
from .config import (
    get_device,
    configure_runtime_threads,
    MODEL_NAME,
    IMG_SIZE,
    NUM_QUERIES,
)

__all__ = [
    "CocoCohortDataset",
    "load_flexible_coco_cohort",
    "discover_coco_annotations",
    "load_coco_val2017_gt_cohort",
    "box_cxcywh_to_xyxy",
    "compute_iou_matrix",
    "build_coco_category_mapping",
    "postprocess_rtdetr_predictions",
    "evaluate_coco_predictions",
    "compute_per_image_coco_aps",
    "bootstrap_paired_diff",
    "repeated_measures_2x2_anova",
    "get_device",
    "configure_runtime_threads",
    "MODEL_NAME",
    "IMG_SIZE",
    "NUM_QUERIES",
]
