# -*- coding: utf-8 -*-
"""
Module: paper_2_phantom_fabrication.experiments.sqd_filter
Tương thích ngược (Backward Compatibility): Re-export từ paper_2_phantom_fabrication.defenses.sqd_filter.
"""

from paper_2_phantom_fabrication.defenses.sqd_filter import (
    SpatialQueryDeduplicationFilter,
    apply_sqd_filter,
)

__all__ = [
    "SpatialQueryDeduplicationFilter",
    "apply_sqd_filter",
]
