# -*- coding: utf-8 -*-
"""
Package: edge_sec_pro.core
Mục đích: Nền tảng lõi (Core Framework) quản lý cấu hình, tập dữ liệu COCO,
các thước đo thống kê chuẩn hóa quốc tế, và bộ định hình hiệu năng phần cứng biên.
"""

from edge_sec_pro.core.config import (
    ADAPTIVE_LAMBDA_1D,
    ADAPTIVE_LAMBDA_ALL,
    ADAPTIVE_LAMBDA_ESM,
    ADAPTIVE_LAMBDA_JOINT,
    AQSM_LAMBDA_SUPP,
    AQSM_RANKING_MARGIN,
    CHECKPOINTS_DIR,
    COCO_VAL_DIR,
    CORE_DIR,
    DATA_DIR,
    DETECTION_CONF_THRESHOLD,
    EPSILON_L_INF,
    HIDDEN_DIM,
    IMG_SIZE,
    MODEL_NAME,
    N_CALIBRATION_SAMPLES,
    N_TEST_CLEAN_SAMPLES,
    NUM_CANDIDATES,
    NUM_DECODER_LAYERS,
    NUM_QUERIES,
    PACKAGE_DIR,
    PATCH_SIZE,
    PGD_NUM_STEPS,
    PGD_STEP_SIZE,
    PROJECTION_DIM,
    RESULTS_DIR,
    SHRINKAGE_BETA,
    SOFT_ACTIVATION_TAU,
    SOFT_ACTIVATION_TEMP,
    SPLIT_FILE,
    TARGET_FPR,
    WORKSPACE_DIR,
    configure_runtime_threads,
    get_device,
)
from edge_sec_pro.core.dataset import (
    COCOValSubsetDataset,
    apply_natural_corruption,
    load_coco_images,
)
from edge_sec_pro.core.detector import (
    DETRInternalState,
    DETRModelWrapper,
    RTDETRModelWrapper,
    load_edge_detector,
)
from edge_sec_pro.core.features import (
    RTDETRDeepFeatureExtractor,
    RTDETRDeepFeatures,
)
from edge_sec_pro.core.metrics import (
    calculate_task_level_asr,
    clopper_pearson_interval,
    compute_auroc_and_bootstrap_ci,
    compute_tpr_at_fixed_fpr,
    make_serializable,
)
from edge_sec_pro.core.profiler import (
    LatencyReport,
    SystemProfiler,
)

__all__ = [
    # Config
    "CORE_DIR",
    "PACKAGE_DIR",
    "WORKSPACE_DIR",
    "DATA_DIR",
    "COCO_VAL_DIR",
    "SPLIT_FILE",
    "RESULTS_DIR",
    "CHECKPOINTS_DIR",
    "MODEL_NAME",
    "IMG_SIZE",
    "NUM_CANDIDATES",
    "NUM_QUERIES",
    "HIDDEN_DIM",
    "PROJECTION_DIM",
    "NUM_DECODER_LAYERS",
    "EPSILON_L_INF",
    "PGD_STEP_SIZE",
    "PGD_NUM_STEPS",
    "PATCH_SIZE",
    "AQSM_RANKING_MARGIN",
    "AQSM_LAMBDA_SUPP",
    "ADAPTIVE_LAMBDA_1D",
    "ADAPTIVE_LAMBDA_JOINT",
    "ADAPTIVE_LAMBDA_ESM",
    "ADAPTIVE_LAMBDA_ALL",
    "TARGET_FPR",
    "N_CALIBRATION_SAMPLES",
    "N_TEST_CLEAN_SAMPLES",
    "SOFT_ACTIVATION_TEMP",
    "SOFT_ACTIVATION_TAU",
    "SHRINKAGE_BETA",
    "DETECTION_CONF_THRESHOLD",
    "get_device",
    "configure_runtime_threads",
    # Dataset
    "COCOValSubsetDataset",
    "load_coco_images",
    "apply_natural_corruption",
    # Detector
    "DETRInternalState",
    "DETRModelWrapper",
    "RTDETRModelWrapper",
    "load_edge_detector",
    # Features
    "RTDETRDeepFeatureExtractor",
    "RTDETRDeepFeatures",
    # Metrics
    "clopper_pearson_interval",
    "compute_auroc_and_bootstrap_ci",
    "calculate_task_level_asr",
    "compute_tpr_at_fixed_fpr",
    "make_serializable",
    # Profiler
    "LatencyReport",
    "SystemProfiler",
]
