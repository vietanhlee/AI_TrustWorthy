# -*- coding: utf-8 -*-
"""
Module: edge_sec_pro.core.features
Mục đích: Facade module cung cấp bộ trích xuất đặc trưng phân tích sâu (Deep Features)
cho Real-Time DETR.

Tái xuất và đóng gói RTDETRDeepFeatureExtractor và cấu trúc dữ liệu RTDETRDeepFeatures.
"""

from paper_3_defense_guardrail.transformer_monitors.rtdetr_advanced_features import (
    RTDETRDeepFeatureExtractor,
    RTDETRDeepFeatures,
)

__all__ = [
    "RTDETRDeepFeatureExtractor",
    "RTDETRDeepFeatures",
]
