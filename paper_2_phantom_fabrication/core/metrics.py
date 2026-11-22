# -*- coding: utf-8 -*-
"""
Module: paper_2_phantom_fabrication.core.metrics
Mục đích: Định nghĩa các hàm đo lường hiệu năng và tính toán chỉ số khoa học chuẩn xác:
  - Bóc tách phân loại lỗi: True Positive (TP), Target False Positive (FP_target / Phantoms), Other False Positive (FP_other).
  - Multi-level Attack Success Rate (ASR >= 1, ASR >= 3, ASR >= 5).
  - Chi phí tấn công tối thiểu (Attack Cost: minimum epsilon đạt ASR >= 50%).
  - Global COCO mAP@50, mAP@75, mAP@50:95.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np


def compute_phantom_metrics(
    matched_results: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Tính toán các chỉ số đặc trưng của hiện tượng Phantom Queries:
    - total_predictions: Tổng số dự đoán
    - total_tp: Số True Positive
    - total_fp: Tổng số False Positive
    - total_fp_target: Số FP thuộc lớp mục tiêu (Phantoms)
    - total_fp_other: Số FP thuộc các lớp khác
    - fp_per_image: Số lượng vật thể ma trung bình trên mỗi ảnh
    - asr_1_pct: Tỷ lệ ảnh có >= 1 phantom
    - asr_3_pct: Tỷ lệ ảnh có >= 3 phantoms
    - asr_5_pct: Tỷ lệ ảnh có >= 5 phantoms
    """
    if not matched_results:
        return {
            "total_predictions": 0,
            "total_tp": 0,
            "total_fp": 0,
            "total_fp_target": 0,
            "total_fp_other": 0,
            "fp_per_image": 0.0,
            "phantom_query_rate": 0.0,
            "asr_1_pct": 0.0,
            "asr_3_pct": 0.0,
            "asr_5_pct": 0.0,
        }

    n_images = len(matched_results)
    tot_preds = sum(m.get("active", m.get("num_predictions", 0)) for m in matched_results)
    tot_tp = sum(m.get("tp", m.get("tp_count", 0)) for m in matched_results)
    tot_fp = sum(m.get("fp", m.get("fp_count", 0)) for m in matched_results)
    tot_fp_target = sum(m.get("fp_target", m.get("phantoms", 0)) for m in matched_results)
    tot_fp_other = sum(m.get("fp_other", max(0, m.get("fp", 0) - m.get("phantoms", 0))) for m in matched_results)
    tot_phantoms = sum(m.get("phantoms", 0) for m in matched_results)

    asr_1_cnt = sum(1 for m in matched_results if m.get("phantoms", 0) >= 1)
    asr_3_cnt = sum(1 for m in matched_results if m.get("phantoms", 0) >= 3)
    asr_5_cnt = sum(1 for m in matched_results if m.get("phantoms", 0) >= 5)

    fp_per_img = tot_phantoms / max(1, n_images)
    pqr = (tot_phantoms / max(1, tot_preds)) * 100.0 if tot_preds > 0 else 0.0

    return {
        "total_predictions": tot_preds,
        "total_tp": tot_tp,
        "total_fp": tot_fp,
        "total_fp_target": tot_fp_target,
        "total_fp_other": tot_fp_other,
        "total_phantoms": tot_phantoms,
        "fp_per_image": round(float(fp_per_img), 3),
        "phantom_query_rate": round(float(pqr), 2),
        "asr_1_pct": round(float(asr_1_cnt / n_images * 100.0), 2),
        "asr_3_pct": round(float(asr_3_cnt / n_images * 100.0), 2),
        "asr_5_pct": round(float(asr_5_cnt / n_images * 100.0), 2),
    }


def compute_disentangled_precision_breakdown(
    tp: int,
    fp_target: int,
    fp_other: int,
) -> Dict[str, float]:
    """
    Bóc tách tỷ lệ suy giảm Precision:
    Precision = TP / (TP + FP_target + FP_other)
    Xác định chính xác lượng suy giảm là do phantoms (FP_target) hay do nhận diện sai lớp (FP_other).
    """
    denom = max(1, tp + fp_target + fp_other)
    precision_total = (tp / denom) * 100.0

    # Giả định nếu không có phantom (FP_target = 0):
    precision_counterfactual_no_phantom = (tp / max(1, tp + fp_other)) * 100.0
    phantom_impact_pp = precision_counterfactual_no_phantom - precision_total

    return {
        "precision_total": round(float(precision_total), 2),
        "precision_without_phantoms": round(float(precision_counterfactual_no_phantom), 2),
        "phantom_precision_drop_pp": round(float(phantom_impact_pp), 2),
        "tp": tp,
        "fp_target": fp_target,
        "fp_other": fp_other,
    }


def compute_minimum_attack_cost(
    epsilon_results: Dict[float, float],
    target_asr_thresh: float = 50.0,
) -> Optional[float]:
    """
    Tính chi phí tấn công tối thiểu (Attack Cost):
    Giá trị epsilon nhỏ nhất (hoặc nội suy) mà cuộc tấn công đạt ASR >= target_asr_thresh (50%).
    epsilon_results: dict dạng {epsilon_val: asr_pct}
    """
    sorted_eps = sorted(epsilon_results.keys())
    for eps in sorted_eps:
        if epsilon_results[eps] >= target_asr_thresh:
            return float(eps)
    return None
