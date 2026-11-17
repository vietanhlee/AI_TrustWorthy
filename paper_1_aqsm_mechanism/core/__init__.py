# -*- coding: utf-8 -*-
"""
Package: paper_1_aqsm_mechanism.core
Mục đích: Module nền tảng tự chứa (self-contained) phục vụ nghiên cứu và thực nghiệm
bài báo 1 (Mechanistic Breakdown of Dynamic Query Selection trong RT-DETR).

Bao gồm:
- detr_wrapper: Lớp bao bọc RT-DETR trích xuất toàn bộ trạng thái nội tại (tokens, queries, attention).
- coco_gt: Nạp ground truth COCO val2017, tính toán mAP chuẩn COCO, Repeated-Measures 2x2 ANOVA và Bootstrap CI.
- mechanism_analyzer: Phẫu thuật hoán đổi thành phần nhân quả (Causal Surgery), phân tích Encoder Memory và Deformable Attention.
- attack_generator: Bộ sinh tấn công đối kháng có kiểm soát (AQSM, Vanishing, Task PGD).
- metrics: Tính khoảng tin cậy nhị thức Clopper-Pearson, AUROC, và chuyển đổi JSON.
"""

__all__ = [
    "detr_wrapper",
    "coco_gt",
    "mechanism_analyzer",
    "attack_generator",
    "metrics",
]
