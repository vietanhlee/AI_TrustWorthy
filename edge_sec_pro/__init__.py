# -*- coding: utf-8 -*-
"""
Package: edge_sec_pro
Mục đích: Hệ thống nghiên cứu và thực thi bảo mật toàn diện cho Real-Time Detection Transformers
trên phần cứng tính toán biên (Edge AI Security Framework).

Cung cấp:
- Lớp bao bọc mô hình RT-DETR khai thác Deformable Cross-Attention (`RTDETRModelWrapper`).
- Bộ sinh tấn công đối kháng 4 tiers chuẩn mực (`DETRAttackGenerator`).
- Bộ trích xuất đặc trưng sâu nội tại (`RTDETRDeepFeatureExtractor`, `RTDETRDeepFeatures`).
- Các bộ giám sát suy luận nhẹ và cổng dung hợp phân vị cực trị (`CalibratedMaxOREnsemble`,
  `LayerwiseMahalanobisDetector`, `EncoderSelectionMonitor`).
- Bộ công cụ đo đạc hiệu năng và phân tích thống kê (`SystemProfiler`, `clopper_pearson_interval`).
"""

__version__ = "1.0.0"
__author__ = "Viet-Anh Le, Dung Hoang-Viet, Khanh Nguyen-Trong (PTIT)"

from edge_sec_pro.core.config import (
    IMG_SIZE,
    MODEL_NAME,
    NUM_CANDIDATES,
    NUM_QUERIES,
    TARGET_FPR,
)
from edge_sec_pro.core.dataset import (
    COCOValSubsetDataset,
    apply_natural_corruption,
    load_coco_images,
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
from edge_sec_pro.transformer_monitors.detr_attack_generator import DETRAttackGenerator
from edge_sec_pro.transformer_monitors.detr_wrapper import (
    DETRInternalState,
    RTDETRModelWrapper,
)
from edge_sec_pro.transformer_monitors.rtdetr_advanced_features import (
    RTDETRDeepFeatureExtractor,
    RTDETRDeepFeatures,
)
from edge_sec_pro.transformer_monitors.rtdetr_advanced_monitor import (
    CalibratedMaxOREnsemble,
    EncoderSelectionMonitor,
    LayerwiseMahalanobisDetector,
    SingleFeatureBaseline,
)

__all__ = [
    "__version__",
    "__author__",
    # Core
    "MODEL_NAME",
    "IMG_SIZE",
    "NUM_CANDIDATES",
    "NUM_QUERIES",
    "TARGET_FPR",
    "COCOValSubsetDataset",
    "load_coco_images",
    "apply_natural_corruption",
    "clopper_pearson_interval",
    "compute_auroc_and_bootstrap_ci",
    "calculate_task_level_asr",
    "compute_tpr_at_fixed_fpr",
    "make_serializable",
    "LatencyReport",
    "SystemProfiler",
    # Wrapper
    "DETRInternalState",
    "RTDETRModelWrapper",
    # Features & Attacks
    "RTDETRDeepFeatureExtractor",
    "RTDETRDeepFeatures",
    "DETRAttackGenerator",
    # Monitors
    "SingleFeatureBaseline",
    "LayerwiseMahalanobisDetector",
    "CalibratedMaxOREnsemble",
    "EncoderSelectionMonitor",
]
