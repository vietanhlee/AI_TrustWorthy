# -*- coding: utf-8 -*-
"""
Module: paper_2_phantom_fabrication.core.matcher
Mục đích: Đối sánh các dự đoán (Predictions) của mô hình RT-DETR với nhãn thật (Ground Truth).
Phân loại chính xác:
  - True Positive (TP): Dự đoán khớp nhãn và IoU >= 0.50 với Ground Truth.
  - False Positive (FP): Dự đoán dư thừa không khớp Ground Truth.
  - Fabricated Phantom Queries: Các FP thuộc lớp mục tiêu (Target Class) và hoàn toàn
    không trùng lặp với bất kỳ vật thể thật nào (IoU_max < 0.30), xác định đây là vật thể ma nhân tạo.
Hỗ trợ cả 2 chế độ:
  1. Single-image: (pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, ...)
  2. Dataset-level: (detections_list, gt_list, ...)
"""

from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np

from paper_2_phantom_fabrication.core.coco_gt import compute_iou_matrix


def _match_single_image(
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
    Khớp các hộp dự đoán với ground truth của 1 ảnh theo Greedy Bipartite Matching.
    Đảm bảo định nghĩa khoa học nghiêm ngặt:
    - Phantom: Bounding box của lớp mục tiêu (target_cls) có IoU < 0.30 với MỌI GT cùng lớp mục tiêu.
    - Phân tách rõ ràng: True Positives (TP), False Positives lớp mục tiêu (FP_target), False Positives lớp khác (FP_other).
    - Multi-level ASR: is_asr_1 (>=1 phantom), is_asr_3 (>=3 phantoms), is_asr_5 (>=5 phantoms).
    """
    n_pred = len(pred_boxes)
    n_gt = len(gt_boxes)

    if n_pred == 0:
        return {
            "tp": 0,
            "fp": 0,
            "fp_target": 0,
            "fp_other": 0,
            "precision": 100.0,
            "recall": 0.0 if n_gt > 0 else 100.0,
            "phantoms": 0,
            "active": 0,
            "is_asr": 0,
            "is_asr_1": 0,
            "is_asr_3": 0,
            "is_asr_5": 0,
        }

    # Lấy các chỉ số GT thuộc lớp mục tiêu
    target_gt_idx = np.where(gt_labels == target_cls)[0] if n_gt > 0 else np.array([], dtype=int)

    if n_gt == 0:
        phantoms = int(np.sum(pred_labels == target_cls))
        fp_target = phantoms
        fp_other = n_pred - fp_target
        return {
            "tp": 0,
            "fp": n_pred,
            "fp_target": fp_target,
            "fp_other": fp_other,
            "precision": 0.0,
            "recall": 100.0,
            "phantoms": phantoms,
            "active": n_pred,
            "is_asr": 1 if phantoms >= 1 else 0,
            "is_asr_1": 1 if phantoms >= 1 else 0,
            "is_asr_3": 1 if phantoms >= 3 else 0,
            "is_asr_5": 1 if phantoms >= 5 else 0,
        }

    iou_mat = compute_iou_matrix(pred_boxes, gt_boxes)
    matched_gt = set()
    tp, fp = 0, 0
    fp_target, fp_other = 0, 0
    phantoms = 0

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
            if p_lbl == target_cls:
                fp_target += 1
                # Kiểm tra nghiêm ngặt: IoU < 0.30 với MỌI GT cùng lớp mục tiêu
                if len(target_gt_idx) > 0:
                    max_target_iou = np.max(iou_mat[p_idx, target_gt_idx])
                else:
                    max_target_iou = 0.0
                
                if max_target_iou < phantom_iou_max:
                    phantoms += 1
            else:
                fp_other += 1

    precision = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
    recall = (tp / n_gt * 100.0) if n_gt > 0 else 100.0
    is_asr_1 = 1 if phantoms >= 1 else 0
    is_asr_3 = 1 if phantoms >= 3 else 0
    is_asr_5 = 1 if phantoms >= 5 else 0

    return {
        "tp": tp,
        "fp": fp,
        "fp_target": fp_target,
        "fp_other": fp_other,
        "precision": precision,
        "recall": recall,
        "phantoms": phantoms,
        "active": n_pred,
        "is_asr": is_asr_1,
        "is_asr_1": is_asr_1,
        "is_asr_3": is_asr_3,
        "is_asr_5": is_asr_5,
    }


def match_detections_against_gt(
    pred_boxes: Any,
    pred_scores: Any = None,
    pred_labels: Any = None,
    gt_boxes: Any = None,
    gt_labels: Any = None,
    target_cls: int = 1,
    iou_thresh: float = 0.50,
    phantom_iou_max: float = 0.30,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Hàm đối sánh linh hoạt hỗ trợ cả 2 dạng lời gọi:
    1. Chế độ đơn ảnh (Single Image):
       match_detections_against_gt(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, ...)
    2. Chế độ danh sách tập dữ liệu (Dataset List):
       match_detections_against_gt(detections_list, gt_list, iou_threshold=0.50, target_class=1, ...)
    """
    # Xử lý alias tham số trong kwargs
    iou_thresh = kwargs.get("iou_threshold", kwargs.get("iou_thresh", iou_thresh))
    target_cls = kwargs.get("target_class", kwargs.get("target_cls", target_cls))
    phantom_iou_max = kwargs.get("phantom_iou_max", phantom_iou_max)

    # Kiểm tra xem có đang được gọi ở chế độ danh sách (List of detections, List of GTs)
    if isinstance(pred_boxes, (list, tuple)) and (isinstance(pred_scores, (list, tuple)) or pred_scores is None):
        dets_list = pred_boxes
        gt_list = pred_scores if pred_scores is not None else []

        total_tp = 0
        total_fp = 0
        total_phantoms = 0
        total_active = 0
        total_gt = 0
        per_image_results: List[Dict[str, Any]] = []

        n_items = min(len(dets_list), len(gt_list)) if gt_list else len(dets_list)
        for i in range(n_items):
            det = dets_list[i]
            gt = gt_list[i] if i < len(gt_list) else {}

            if isinstance(det, dict):
                p_b = det.get("boxes", np.zeros((0, 4), dtype=np.float64))
                p_s = det.get("scores", np.zeros((0,), dtype=np.float64))
                p_l = det.get("labels", np.zeros((0,), dtype=int))
            elif isinstance(det, (list, tuple)) and len(det) >= 3:
                p_b, p_s, p_l = det[0], det[1], det[2]
            else:
                p_b, p_s, p_l = np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), dtype=int)

            if isinstance(gt, dict):
                g_b = gt.get("boxes_xyxy", gt.get("boxes", np.zeros((0, 4), dtype=np.float64)))
                g_l = gt.get("labels", np.zeros((0,), dtype=int))
            elif isinstance(gt, (list, tuple)) and len(gt) >= 2:
                g_b, g_l = gt[0], gt[1]
            else:
                g_b, g_l = np.zeros((0, 4)), np.zeros((0,), dtype=int)

            res_i = _match_single_image(
                pred_boxes=p_b,
                pred_scores=p_s,
                pred_labels=p_l,
                gt_boxes=g_b,
                gt_labels=g_l,
                target_cls=target_cls,
                iou_thresh=iou_thresh,
                phantom_iou_max=phantom_iou_max,
            )
            total_tp += res_i["tp"]
            total_fp += res_i["fp"]
            total_phantoms += res_i["phantoms"]
            total_active += res_i["active"]
            total_gt += len(g_b)
            per_image_results.append(res_i)

        n_valid_images = max(1, len(per_image_results))
        total_fp_target = sum(r.get("fp_target", 0) for r in per_image_results)
        total_fp_other = sum(r.get("fp_other", 0) for r in per_image_results)
        asr_1_count = sum(r.get("is_asr_1", 0) for r in per_image_results)
        asr_3_count = sum(r.get("is_asr_3", 0) for r in per_image_results)
        asr_5_count = sum(r.get("is_asr_5", 0) for r in per_image_results)

        precision = (total_tp / (total_tp + total_fp) * 100.0) if (total_tp + total_fp) > 0 else 0.0
        recall = (total_tp / total_gt * 100.0) if total_gt > 0 else 100.0
        f1 = (2.0 * precision * recall / (precision + recall)) / 100.0 if (precision + recall) > 0 else 0.0

        return {
            "total_tp": total_tp,
            "total_fp": total_fp,
            "total_fp_target": total_fp_target,
            "total_fp_other": total_fp_other,
            "total_phantoms": total_phantoms,
            "total_active": total_active,
            "precision": round(precision, 2),
            "recall": round(recall, 2),
            "f1": round(f1, 4),
            "asr_1_pct": round(asr_1_count / n_valid_images * 100.0, 2),
            "asr_3_pct": round(asr_3_count / n_valid_images * 100.0, 2),
            "asr_5_pct": round(asr_5_count / n_valid_images * 100.0, 2),
            "asr_pct": round(asr_1_count / n_valid_images * 100.0, 2),
            "tp": total_tp,
            "fp": total_fp,
            "phantoms": total_phantoms,
            "active": total_active,
            "per_image": per_image_results,
        }

    # Chế độ đơn ảnh thông thường
    p_b = pred_boxes if pred_boxes is not None else np.zeros((0, 4))
    p_s = pred_scores if pred_scores is not None else np.zeros((0,))
    p_l = pred_labels if pred_labels is not None else np.zeros((0,), dtype=int)
    g_b = gt_boxes if gt_boxes is not None else np.zeros((0, 4))
    g_l = gt_labels if gt_labels is not None else np.zeros((0,), dtype=int)

    return _match_single_image(
        pred_boxes=p_b,
        pred_scores=p_s,
        pred_labels=p_l,
        gt_boxes=g_b,
        gt_labels=g_l,
        target_cls=target_cls,
        iou_thresh=iou_thresh,
        phantom_iou_max=phantom_iou_max,
    )
