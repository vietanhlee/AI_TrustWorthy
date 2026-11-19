# -*- coding: utf-8 -*-
"""
Facade Module: paper_1_aqsm_mechanism.experiments.generate_figures_and_tables
Tương thích ngược (Backward Compatibility): Re-export từ experiments.engines.generate_figures_and_tables.
"""

from paper_1_aqsm_mechanism.experiments.engines.generate_figures_and_tables import (
    generate_all_figures_and_tables,
    plot_attention_telemetry_figure,
    plot_causal_swap_figure,
    plot_defense_evaluation_figure,
    plot_matched_budget_figure,
    plot_score_spectrum_figure,
)

__all__ = [
    "generate_all_figures_and_tables",
    "plot_score_spectrum_figure",
    "plot_causal_swap_figure",
    "plot_attention_telemetry_figure",
    "plot_matched_budget_figure",
    "plot_defense_evaluation_figure",
]

if __name__ == "__main__":
    generate_all_figures_and_tables()
