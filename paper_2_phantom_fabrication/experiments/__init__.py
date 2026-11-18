# -*- coding: utf-8 -*-
"""
Package: paper_2_phantom_fabrication.experiments
Bộ công cụ thực nghiệm cho Paper 2: Phantom Queries Fabrication.

Gồm các thành phần chính:
- run_full_paper2_suite: Bộ chạy thực nghiệm toàn diện 3 Stage (Benchmark, Ablations, Figures).
- plot_paper_figures: Sinh đồ thị vector (Figures 1-4) chất lượng xuất bản khoa học.
- sync_telemetry_to_paper: Tự động trích xuất kết quả JSON, điền số liệu vào Tables 1-4 và biên dịch PDF.
- sqd_filter: Bộ lọc phòng thủ phân tán không gian (Spatial Query Dispersion).
"""

from .sqd_filter import apply_sqd_filter

__all__ = [
    "apply_sqd_filter"
]
