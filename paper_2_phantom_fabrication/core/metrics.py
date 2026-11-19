# -*- coding: utf-8 -*-
"""
Module: paper_2_phantom_fabrication.core.metrics
Mục đích: Định nghĩa các hàm đo lường hiệu năng và tính toán chỉ số khoa học cho Paper 2:
  - Global COCO mAP@50, mAP@75, mAP@50:95.
  - False Positive Count (FP) & False Positive Rate per Image.
  - Phantom Query Fabrication Score.
  - Mean Displaced Object Localization Error.
"""

from typing import Any, Dict, List, Sequence, Tuple
import numpy as np


def compute_phantom_metrics(
    matched_results: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Tính toán các chỉ số đặc trưng của hiện tượng Phantom Queries:
    - total_predictions: Tổng số dự đoán
    - total_tp: Số True Positive
    - total_fp: Số False Positive (Phantom Objects do tấn công chế tạo ra)
    - fp_per_image: Số lượng vật thể ma trung bình trên mỗi ảnh
    - phantom_query_rate: Tỷ lệ phần trăm các query bị chế tạo thành công
    """
    if not matched_results:
        return {
            "total_predictions": 0,
            "total_tp": 0,
            "total_fp": 0,
            "fp_per_image": 0.0,
            "phantom_query_rate": 0.0,
        }

    n_images = len(matched_results)
    tot_preds = sum(m.get("num_predictions", 0) for m in matched_results)
    tot_tp = sum(m.get("tp_count", 0) for m in matched_results)
    tot_fp = sum(m.get("fp_count", 0) for m in matched_results)

    fp_per_img = tot_fp / max(1, n_images)
    pqr = (tot_fp / max(1, tot_preds)) * 100.0 if tot_preds > 0 else 0.0

    return {
        "total_predictions": tot_preds,
        "total_tp": tot_tp,
        "total_fp": tot_fp,
        "fp_per_image": round(float(fp_per_img), 3),
        "phantom_query_rate": round(float(pqr), 2),
    }
