# -*- coding: utf-8 -*-
"""
Package: paper_1_aqsm_mechanism
Mục đích: Dự án nghiên cứu thực nghiệm và bản thảo bài báo khoa học:
"Mechanistic Breakdown of Dynamic Query Selection in Real-Time Detection Transformers"
(Gửi Elsevier Engineering Applications of Artificial Intelligence - EAAI).

Kiến trúc gói:
  - core: Các module nền tảng trích xuất mô hình RT-DETR, dataset loader chống tràn RAM, cache đối kháng, và đo đạc mAP/ANOVA.
  - experiments: Bộ cấu hình, logger kiểm toán, các engines thực nghiệm chuyên sâu và các tasks độc lập (Tasks 1-8).
"""

__version__ = "1.0.0"
__author__ = "Edge AI Security Research Team"
