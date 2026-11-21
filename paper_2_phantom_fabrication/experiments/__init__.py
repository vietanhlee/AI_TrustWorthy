# -*- coding: utf-8 -*-
"""
Package: paper_2_phantom_fabrication.experiments
Bộ công cụ thực nghiệm cho Paper 2: Phantom Queries Fabrication.

Gồm các thành phần chính:
- run_full_paper2_suite: Master CLI runner điều phối các giai đoạn thực nghiệm.
- stage1_benchmark: Đánh giá đa hạt giống Multi-Seed Benchmark và 5 cơ chế phòng thủ (Tables 1 & 4).
- stage2_ablations: Phân tích độ nhạy (Sweeps) và Cyber-Physical Downstream Impact (Tables 2 & 3).
- streaming_evaluator: Trình thực thi suy luận đối kháng theo luồng dữ liệu phẳng.
- plot_paper_figures: Sinh đồ thị vector (Figures 1-4) chất lượng xuất bản khoa học.
- sync_telemetry_to_paper: Tự động trích xuất kết quả JSON, điền số liệu vào Tables 1-4 và biên dịch PDF.
- sqd_filter: Bộ lọc phòng thủ phân tán không gian (Spatial Query Dispersion).
"""

try:
    from .sqd_filter import apply_sqd_filter
    from .streaming_evaluator import (
        get_cohort_dataloader,
        run_streaming_condition_eval,
    )
    from .stage1_benchmark import run_stage_1_benchmark
    from .stage2_ablations import run_stage_2_ablations
except ImportError:
    apply_sqd_filter = None
    get_cohort_dataloader = None
    run_streaming_condition_eval = None
    run_stage_1_benchmark = None
    run_stage_2_ablations = None

__all__ = [
    "apply_sqd_filter",
    "get_cohort_dataloader",
    "run_streaming_condition_eval",
    "run_stage_1_benchmark",
    "run_stage_2_ablations",
]
