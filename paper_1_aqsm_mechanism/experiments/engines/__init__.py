# -*- coding: utf-8 -*-
"""
Package: paper_1_aqsm_mechanism.experiments.engines
Mục đích: Chứa các bộ động cơ tính toán và đánh giá thực nghiệm cốt lõi của Paper 1:
  - causal_surgery_engine: Phẫu thuật hoán đổi nhân quả 2x2 Factorial Causal Tensor-Swap.
  - encoder_dynamics_engine: Phân tích động lực học truy vấn Encoder (Cosine drift, JS div, margin).
  - telemetry_analyzer: Viễn trắc Deformable Attention (Shannon entropy & spatial hit rate).
  - misrouting_engine: Kiểm chứng Spatial Specificity & bác bỏ Tautology.
  - matched_budget_sweep: Quét ngân sách nhiễu L_inf qua eps in {4, 8, 16}/255.
  - latency_profiler: Đo đạc và phân rã độ trễ phần cứng CPU/GPU.
  - generate_figures_and_tables: Tự động xuất 5 biểu đồ vector PDF/PNG chuẩn Elsevier EAAI.
  - defense_evaluation_engine: Đánh giá cơ chế phòng thủ ESM & kiểm thử tấn công thích ứng.
"""

from paper_1_aqsm_mechanism.experiments.engines.causal_surgery_engine import (
    run_causal_surgery,
)
from paper_1_aqsm_mechanism.experiments.engines.defense_evaluation_engine import (
    run_adaptive_evasion_test,
    run_defense_evaluation,
)
from paper_1_aqsm_mechanism.experiments.engines.encoder_dynamics_engine import (
    run_encoder_query_dynamics_evaluation,
)
from paper_1_aqsm_mechanism.experiments.engines.generate_figures_and_tables import (
    generate_all_figures_and_tables,
    plot_attention_telemetry_figure,
    plot_causal_swap_figure,
    plot_defense_evaluation_figure,
    plot_matched_budget_figure,
    plot_score_spectrum_figure,
)
from paper_1_aqsm_mechanism.experiments.engines.latency_profiler import (
    profile_pipeline_latency,
)
from paper_1_aqsm_mechanism.experiments.engines.matched_budget_sweep import (
    run_matched_budget_sweep,
)
from paper_1_aqsm_mechanism.experiments.engines.misrouting_engine import (
    compute_object_recall,
    compute_ref_fg_hit,
    run_controlled_misrouting_evaluation,
)
from paper_1_aqsm_mechanism.experiments.engines.telemetry_analyzer import (
    compute_attention_entropy,
    compute_foreground_spatial_hit_rate,
    compute_reference_coordinate_drift,
    run_mechanism_telemetry_cohort,
)

__all__ = [
    # Causal surgery
    "run_causal_surgery",
    # Encoder dynamics
    "run_encoder_query_dynamics_evaluation",
    # Telemetry
    "compute_attention_entropy",
    "compute_foreground_spatial_hit_rate",
    "compute_reference_coordinate_drift",
    "run_mechanism_telemetry_cohort",
    # Spatial misrouting
    "compute_ref_fg_hit",
    "compute_object_recall",
    "run_controlled_misrouting_evaluation",
    # Matched budget sweep
    "run_matched_budget_sweep",
    # Latency profiler
    "profile_pipeline_latency",
    # Figures and tables
    "generate_all_figures_and_tables",
    "plot_score_spectrum_figure",
    "plot_causal_swap_figure",
    "plot_attention_telemetry_figure",
    "plot_matched_budget_figure",
    "plot_defense_evaluation_figure",
    # Defense evaluation
    "run_defense_evaluation",
    "run_adaptive_evasion_test",
]
