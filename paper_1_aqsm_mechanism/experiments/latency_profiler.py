# -*- coding: utf-8 -*-
"""
Facade Module: paper_1_aqsm_mechanism.experiments.latency_profiler
Tương thích ngược (Backward Compatibility): Re-export từ experiments.engines.latency_profiler.
"""

from paper_1_aqsm_mechanism.experiments.engines.latency_profiler import (
    profile_pipeline_latency,
)

__all__ = ["profile_pipeline_latency"]
