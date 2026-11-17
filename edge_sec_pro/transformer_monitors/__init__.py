# -*- coding: utf-8 -*-
"""
Package: edge_sec_pro.transformer_monitors
Mục đích: Bộ công cụ giám sát suy luận nội tại (Model-Layer Inference Monitoring)
và sinh tấn công đối kháng chuyên biệt cho Real-Time Detection Transformers (RT-DETR).

Bao gồm:
1. detr_wrapper: Lớp bọc mô hình RT-DETR khai thác Deformable Cross-Attention và Sigmoid Head.
2. detr_attack_generator: Bộ sinh các đòn tấn công đối kháng 4 tiers (Patch, AQSM, EQF, Vanish, DQF, Adaptive).
3. rtdetr_advanced_features: Trích xuất đặc trưng sâu, độ lệch dự đoán và động học candidate selection.
4. rtdetr_advanced_monitor: Các bộ giám sát 4 tiers, Mahalanobis đa tầng và cổng dung hợp Calibrated Max-OR.
"""

from edge_sec_pro.transformer_monitors.detr_attack_generator import DETRAttackGenerator
from edge_sec_pro.transformer_monitors.detr_query_monitor import DETRQueryMonitor
from edge_sec_pro.transformer_monitors.detr_wrapper import (
    DETRInternalState,
    DETRModelWrapper,
    RTDETRModelWrapper,
)
from edge_sec_pro.transformer_monitors.query_attention_features import TransformerFeatureExtractor
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
    # Wrappers
    "DETRInternalState",
    "DETRModelWrapper",
    "RTDETRModelWrapper",
    # Feature Extractors
    "TransformerFeatureExtractor",
    "RTDETRDeepFeatureExtractor",
    "RTDETRDeepFeatures",
    # Attack Suite
    "DETRAttackGenerator",
    # Detection & Guardrail Suite
    "DETRQueryMonitor",
    "SingleFeatureBaseline",
    "LayerwiseMahalanobisDetector",
    "CalibratedMaxOREnsemble",
    "EncoderSelectionMonitor",
]
