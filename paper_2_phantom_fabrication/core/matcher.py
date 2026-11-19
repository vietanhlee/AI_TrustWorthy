# -*- coding: utf-8 -*-
"""
Module: paper_2_phantom_fabrication.core.matcher
Mục đích: Đối sánh các dự đoán (Predictions) của mô hình RT-DETR với nhãn thật (Ground Truth).
Phân loại chính xác:
  - True Positive (TP): Dự đoán khớp nhãn và IoU >= 0.50 với Ground Truth.
  - False Positive (FP): Dự đoán dư thừa không khớp Ground Truth.
  - Fabricated Phantom Queries: Các FP thuộc lớp mục tiêu (Target Class) và hoàn toàn
    không trùng lặp với bất kỳ vật thể thật nào (IoU_max < 0.30), xác định đây là vật thể ma nhân tạo.
"""

from typing import Any, Dict
import numpy as np

from paper_2_phantom_fabrication.core.coco_gt import compute_iou_matrix


def match_detections_against_gt(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    pred_labels: np.ndarray,
    gt_boxes: np.ndarray,
    gt_labels: np.ndarray,
    target_cls: int = 1,
    iou_thresh: float = 0.50,
    phantom_iou_max: float = 0.30,
) -> Dict[str, Any]:
    """
    Khớp các hộp dự đoán với ground truth theo thuật toán Greedy Bipartite Matching:
    - Trả về số lượng TP, FP, Phantoms, Precision (%), Recall (%), và cờ ASR (tấn công thành công nếu có ít nhất 1 phantom).
    """
    n_pred = len(pred_boxes)
    n_gt = len(gt_boxes)

    if n_pred == 0:
        return {
            "tp": 0,
            "fp": 0,
            "precision": 100.0,
            "recall": 0.0 if n_gt > 0 else 100.0,
            "phantoms": 0,
            "active": 0,
            "is_asr": 0,
        }

    if n_gt == 0:
        phantoms = int(np.sum(pred_labels == target_cls))
        return {
            "tp": 0,
            "fp": n_pred,
            "precision": 0.0,
            "recall": 100.0,
            "phantoms": phantoms,
            "active": n_pred,
            "is_asr": 1 if phantoms > 0 else 0,
        }

    iou_mat = compute_iou_matrix(pred_boxes, gt_boxes)
    matched_gt = set()
    tp, fp, phantoms = 0, 0, 0

    sorted_idx = np.argsort(-pred_scores)
    for p_idx in sorted_idx:
        p_lbl = pred_labels[p_idx]
        best_gt_idx = -1
        best_iou = -1.0
        for g_idx in range(n_gt):
            if g_idx in matched_gt:
                continue
            if gt_labels[g_idx] == p_lbl and iou_mat[p_idx, g_idx] >= iou_thresh:
                if iou_mat[p_idx, g_idx] > best_iou:
                    best_iou = iou_mat[p_idx, g_idx]
                    best_gt_idx = g_idx

        if best_gt_idx >= 0:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1
            # Kiểm tra xem FP này có phải là Phantom Query thuộc lớp mục tiêu không
            if p_lbl == target_cls:
                max_any_gt_iou = np.max(iou_mat[p_idx]) if n_gt > 0 else 0.0
                if max_any_gt_iou < phantom_iou_max:
                    phantoms += 1

    precision = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
    recall = (tp / n_gt * 100.0) if n_gt > 0 else 100.0
    is_asr = 1 if phantoms > 0 else 0

    return {
        "tp": tp,
        "fp": fp,
        "precision": precision,
        "recall": recall,
        "phantoms": phantoms,
        "active": n_pred,
        "is_asr": is_asr,
    }
