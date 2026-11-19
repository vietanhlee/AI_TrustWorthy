# -*- coding: utf-8 -*-
"""
Package: paper_2_phantom_fabrication.experiments.tasks
Mục đích: Chứa các CLI tasks độc lập đo đạc từng phần của Paper 2 (Task 0 -> Task 5).
Mỗi task có thể chạy riêng biệt hoặc thông qua Orchestrator run_all_experiments.py.
"""

from .task0_generate_adv_cache import main as run_task0
from .task1_multiseed_benchmark import main as run_task1
from .task2_sensitivity_sweeps import main as run_task2
from .task3_downstream_impact import main as run_task3
from .task4_defense_evaluation import main as run_task4
from .task5_generate_figures import main as run_task5

__all__ = [
    "run_task0",
    "run_task1",
    "run_task2",
    "run_task3",
    "run_task4",
    "run_task5",
]
