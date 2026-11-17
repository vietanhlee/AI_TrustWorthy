# -*- coding: utf-8 -*-
"""
Module: edge_sec_pro.experiments.common_exp
Mục đích: Chứa các tiện ích, hàm thống kê và lớp dùng chung cho toàn bộ pipeline thực nghiệm
độc lập của RT-DETR Streamlined Security Evaluation.

Tái sử dụng các thành phần từ `edge_sec_pro.core` và `edge_sec_pro.transformer_monitors`
nhằm đảm bảo tính module hóa, loại bỏ code trùng lặp và giữ nguyên 100% tính tương thích ngược.
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
from pathlib import Path
import pickle
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from tqdm import tqdm

# Thiết lập số luồng CPU mặc định cho worker processes
torch.set_num_threads(4)

# Định vị package và workspace
PACKAGE_DIR = Path(__file__).resolve().parent.parent
WORKSPACE_DIR = PACKAGE_DIR.parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

# Tái xuất các đường dẫn chuẩn từ core nội bộ Paper 3
try:
    from paper_3_defense_guardrail.core.config import (
        CHECKPOINTS_DIR,
        COCO_VAL_DIR,
        DATA_DIR,
        RESULTS_DIR,
        SPLIT_FILE,
    )
    from paper_3_defense_guardrail.core.dataset import (
        COCOValSubsetDataset,
        apply_natural_corruption,
        load_coco_images,
    )
    from paper_3_defense_guardrail.core.coco_gt import (
        box_cxcywh_to_xyxy,
        compute_map_coco,
        evaluate_coco_predictions,
        load_coco_val2017_gt_cohort,
    )
    from paper_3_defense_guardrail.core.metrics import (
        calculate_task_level_asr,
        clopper_pearson_interval,
        compute_auroc_and_bootstrap_ci,
        compute_tpr_at_fixed_fpr,
        make_serializable,
    )
    from paper_3_defense_guardrail.transformer_monitors.detr_attack_generator import DETRAttackGenerator
    from paper_3_defense_guardrail.transformer_monitors.detr_wrapper import (
        DETRInternalState,
        RTDETRModelWrapper,
    )
    from paper_3_defense_guardrail.transformer_monitors.rtdetr_advanced_features import (
        RTDETRDeepFeatureExtractor,
        RTDETRDeepFeatures,
    )
    from paper_3_defense_guardrail.transformer_monitors.rtdetr_advanced_monitor import (
        CalibratedMaxOREnsemble,
        EncoderSelectionMonitor,
        LayerwiseMahalanobisDetector,
        SingleFeatureBaseline,
    )
except ImportError:
    from core.config import (
        CHECKPOINTS_DIR,
        COCO_VAL_DIR,
        DATA_DIR,
        RESULTS_DIR,
        SPLIT_FILE,
    )
    from core.dataset import (
        COCOValSubsetDataset,
        apply_natural_corruption,
        load_coco_images,
    )
    from core.coco_gt import (
        box_cxcywh_to_xyxy,
        compute_map_coco,
        evaluate_coco_predictions,
        load_coco_val2017_gt_cohort,
    )
    from core.metrics import (
        calculate_task_level_asr,
        clopper_pearson_interval,
        compute_auroc_and_bootstrap_ci,
        compute_tpr_at_fixed_fpr,
        make_serializable,
    )
    from transformer_monitors.detr_attack_generator import DETRAttackGenerator
    from transformer_monitors.detr_wrapper import (
        DETRInternalState,
        RTDETRModelWrapper,
    )
    from transformer_monitors.rtdetr_advanced_features import (
        RTDETRDeepFeatureExtractor,
        RTDETRDeepFeatures,
    )
    from transformer_monitors.rtdetr_advanced_monitor import (
        CalibratedMaxOREnsemble,
        EncoderSelectionMonitor,
        LayerwiseMahalanobisDetector,
        SingleFeatureBaseline,
    )

__all__ = [
    "PACKAGE_DIR",
    "WORKSPACE_DIR",
    "RESULTS_DIR",
    "CHECKPOINTS_DIR",
    "DATA_DIR",
    "COCO_VAL_DIR",
    "SPLIT_FILE",
    "clopper_pearson_interval",
    "compute_auroc_and_bootstrap_ci",
    "calculate_task_level_asr",
    "compute_tpr_at_fixed_fpr",
    "load_coco_images",
    "apply_natural_corruption",
    "make_serializable",
    "COCOValSubsetDataset",
    "RTDETRModelWrapper",
    "DETRInternalState",
    "RTDETRDeepFeatureExtractor",
    "RTDETRDeepFeatures",
    "SingleFeatureBaseline",
    "LayerwiseMahalanobisDetector",
    "CalibratedMaxOREnsemble",
    "EncoderSelectionMonitor",
    "DETRAttackGenerator",
    "box_cxcywh_to_xyxy",
    "compute_map_coco",
    "load_coco_val2017_gt_cohort",
    "evaluate_coco_predictions",
]
