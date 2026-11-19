# -*- coding: utf-8 -*-
"""
Facade Module: paper_1_aqsm_mechanism.experiments.telemetry_analyzer
Tương thích ngược (Backward Compatibility): Re-export từ experiments.engines.telemetry_analyzer.
"""

from paper_1_aqsm_mechanism.experiments.engines.telemetry_analyzer import (
    compute_attention_entropy,
    compute_foreground_spatial_hit_rate,
    compute_reference_coordinate_drift,
    run_mechanism_telemetry_cohort,
)

__all__ = [
    "compute_attention_entropy",
    "compute_foreground_spatial_hit_rate",
    "compute_reference_coordinate_drift",
    "run_mechanism_telemetry_cohort",
]
