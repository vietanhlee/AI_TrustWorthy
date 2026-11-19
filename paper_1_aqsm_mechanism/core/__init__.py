# -*- coding: utf-8 -*-
"""
Package: paper_1_aqsm_mechanism.core
Mục đích: Module nền tảng tự chứa (self-contained) phục vụ nghiên cứu và thực nghiệm
bài báo 1 (Mechanistic Breakdown of Dynamic Query Selection trong RT-DETR).

Bao gồm:
  - detr_wrapper: Lớp bao bọc RTDETRModelWrapper trích xuất toàn bộ trạng thái nội tại.
  - dataset_loader: Bộ nạp dataset COCO linh hoạt (FlexibleCOCODataset, LazyImageBatchProxy).
  - adv_cache: Quản lý nạp cache mẫu đối kháng định dạng .npz (chuẩn float32).
  - coco_gt: Nạp ground truth COCO val2017, tính toán mAP chuẩn COCO.
  - mechanism_analyzer: Phẫu thuật hoán đổi thành phần nhân quả (Causal Surgery).
  - attack_generator: Bộ sinh tấn công đối kháng có kiểm soát (DETRAttackGenerator).
  - defense_engine: Bộ giám sát phổ năng lượng Encoder (ESM Monitor) và đánh giá phòng thủ.
  - metrics: Tính khoảng tin cậy nhị thức Clopper-Pearson, AUROC, và chuyển đổi JSON.
"""

from .adv_cache import find_cache_file, get_adv_cache_dir, load_cached_adv_images, load_cached_adv_proxy
from .attack_generator import DETRAttackGenerator
from .coco_gt import (
    bootstrap_paired_diff,
    box_cxcywh_to_xyxy,
    compute_iou_matrix,
    evaluate_coco_predictions,
    repeated_measures_2x2_anova,
)
from .dataset_loader import FlexibleCOCODataset, LazyAdvImageBatchProxy, LazyImageBatchProxy
from .defense_engine import EncoderSelectionMonitor, evaluate_defense_cohort
from .detr_wrapper import RTDETRModelWrapper
from .mechanism_analyzer import (
    analyze_deformable_attention_batch,
    analyze_encoder_memory_batch,
    analyze_query_selection_batch,
    extract_encoder_components,
    run_decoder_from_components,
)
from .metrics import (
    clopper_pearson_interval,
    compute_auroc_and_bootstrap_ci,
    compute_tpr_at_fixed_fpr,
)

__all__ = [
    # Adv Cache
    "load_cached_adv_images",
    "load_cached_adv_proxy",
    "find_cache_file",
    "get_adv_cache_dir",
    # Dataset Loader
    "FlexibleCOCODataset",
    "LazyImageBatchProxy",
    "LazyAdvImageBatchProxy",
    # DETR Model Wrapper
    "RTDETRModelWrapper",
    # Attack Generator
    "DETRAttackGenerator",
    # COCO Ground Truth & Evaluation
    "compute_iou_matrix",
    "box_cxcywh_to_xyxy",
    "evaluate_coco_predictions",
    "bootstrap_paired_diff",
    "repeated_measures_2x2_anova",
    # Mechanism Analyzer
    "extract_encoder_components",
    "run_decoder_from_components",
    "analyze_encoder_memory_batch",
    "analyze_query_selection_batch",
    "analyze_deformable_attention_batch",
    # Defense Engine
    "EncoderSelectionMonitor",
    "evaluate_defense_cohort",
    # Metrics
    "clopper_pearson_interval",
    "compute_auroc_and_bootstrap_ci",
    "compute_tpr_at_fixed_fpr",
]
