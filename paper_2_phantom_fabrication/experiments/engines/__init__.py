# -*- coding: utf-8 -*-
"""
Package: paper_2_phantom_fabrication.experiments.engines
Mục đích: Chứa các động cơ tính toán và đánh giá thực nghiệm cốt lõi của Paper 2:
  - cache_engine: Sinh bộ nhớ đệm đối kháng float32 chống OOM (-9).
  - benchmark_engine: Multi-seed benchmark cho RT-DETR và YOLOv8 (Table 1).
  - ablation_engine: Sensitivity sweeps ngân sách và số vòng lặp PGD (Table 2).
  - downstream_engine: Chuyển giao liên lớp và an toàn Cyber-Physical AEB/ByteTrack (Table 3).
  - defense_engine: Đánh giá bộ lọc SQD Filter và các baselines (Table 4).
  - figures_engine: Tự động kết xuất 4 biểu đồ vector xuất bản chuẩn Elsevier EAAI.
"""

from paper_2_phantom_fabrication.experiments.engines.ablation_engine import (
    run_stage_2_sensitivity_sweeps,
)
from paper_2_phantom_fabrication.experiments.engines.benchmark_engine import (
    run_stage_1_benchmark,
)
from paper_2_phantom_fabrication.experiments.engines.cache_engine import (
    generate_single_attack_cache,
)
from paper_2_phantom_fabrication.experiments.engines.defense_engine import (
    run_comprehensive_defense_evaluation,
)
from paper_2_phantom_fabrication.experiments.engines.downstream_engine import (
    run_stage_2_downstream_eval,
)
from paper_2_phantom_fabrication.experiments.engines.figures_engine import (
    generate_all_paper2_figures,
)

__all__ = [
    "generate_single_attack_cache",
    "run_stage_1_benchmark",
    "run_stage_2_sensitivity_sweeps",
    "run_stage_2_downstream_eval",
    "run_comprehensive_defense_evaluation",
    "generate_all_paper2_figures",
]
