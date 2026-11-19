# -*- coding: utf-8 -*-
"""
Module: paper_2_phantom_fabrication.experiments.engines.benchmark_engine
Mục đích: Động cơ đo đạc thực nghiệm Stage 1 - Multi-Seed Benchmark (3 seeds: 42, 123, 456)
Ủy quyền (Delegate) trực tiếp sang pipeline chuẩn đã được kiểm chứng của stage1_benchmark:
  - RT-DETR vs YOLOv8n trên Clean, Untargeted PGD, EQF, DQF.
  - Đo lường: mAP@50, mAP@75, mAP@50:95, False Positive Count, Latency (FPS), GPU VRAM.
  - Phân tích thống kê: Paired Wilcoxon p-value, Wilson Score Interval, Bootstrap 95% CI.
  - Đánh giá bộ 5 giải pháp phòng thủ (Table 4).
"""

from paper_2_phantom_fabrication.experiments.stage1_benchmark import run_stage_1_benchmark

__all__ = ["run_stage_1_benchmark"]
