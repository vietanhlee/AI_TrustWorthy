# -*- coding: utf-8 -*-
"""
Package: paper_2_phantom_fabrication.attacks
Mô-đun tấn công đối kháng chuyên biệt cho Real-Time DETR:
  - EQF: Encoder Query Fabrication (Targeted Misrouting ở giai đoạn Encoder Selection)
  - DQF: Decoder Query Fabrication (Dynamic Query Misrouting ở giai đoạn Decoder Cross-Attention)
  - Untargeted_PGD: Tấn công PGD không mục tiêu chuẩn làm baseline so sánh
  - Gaussian_Noise: Baseline nhiễu ngẫu nhiên
"""

from .phantom_attacks import generate_batch_adversarial

__all__ = ["generate_batch_adversarial"]
