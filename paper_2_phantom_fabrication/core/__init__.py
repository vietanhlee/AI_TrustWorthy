# -*- coding: utf-8 -*-
"""
Package: paper_2_phantom_fabrication.core
Mô-đun lõi phục vụ nạp dữ liệu COCO, tính toán độ đo mAP / Hungarian Matching,
nhận diện phần cứng thực thi và quản lý bộ nhớ đệm đối kháng cho Paper 2.
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
from .adv_cache import (
    find_cache_file,
    load_cached_adv_delta,
    save_adv_cache,
    eps_tag,
)
from .config import (
    get_device,
    configure_runtime_threads,
    get_adv_cache_dir,
    set_adv_cache_dir,
    ADV_CACHE_DIR,
    MODEL_NAME,
    IMG_SIZE,
    NUM_QUERIES,
)
from .hardware import (
    detect_hardware,
    benchmark_model_latency,
)
from .matcher import (
    match_detections_against_gt,
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
    "find_cache_file",
    "load_cached_adv_delta",
    "save_adv_cache",
    "eps_tag",
    "get_device",
    "configure_runtime_threads",
    "get_adv_cache_dir",
    "set_adv_cache_dir",
    "ADV_CACHE_DIR",
    "MODEL_NAME",
    "IMG_SIZE",
    "NUM_QUERIES",
    "detect_hardware",
    "benchmark_model_latency",
    "match_detections_against_gt",
]
