# -*- coding: utf-8 -*-
"""
Module: paper_2_phantom_fabrication.downstream.bytetrack_sim
Mục đích: Mô phỏng phân tích proxy (Analytical Proxy Simulation) đo độ trễ và sự bùng nổ kích thước
ma trận gán cặp Bipartite Matching (Kuhn-Munkres Hungarian Algorithm O(N^3)) của thuật toán ByteTrack
khi bị ngập lụt bởi Phantom Queries trên tập ảnh tĩnh COCO.
Lưu ý phương pháp luận: Đây là proxy độ phức tạp thuật toán gán cặp tổ hợp trên từng khung hình,
không phải là benchmark tracking video theo chuỗi thời gian (MOTA/IDF1).
"""

import time
from typing import Any, Dict, List, Optional, Sequence
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

from paper_2_phantom_fabrication.core.coco_gt import (
    compute_iou_matrix,
    postprocess_rtdetr_predictions,
)


def simulate_bytetrack_complexity(
    clean_post: Sequence[Dict[str, Any]],
    eqf_post: Sequence[Dict[str, Any]],
    dqf_logits: torch.Tensor,
    dqf_boxes: torch.Tensor,
    gt_list: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Mô phỏng bước Hungarian Association trong ByteTrack trên từng khung hình.
    Đo thời gian thực thi thuật toán Khớp cực tiểu Kuhn-Munkres O(N^3) với ground truth xác thực.
    """
    if not gt_list or len(gt_list) == 0:
        raise ValueError("Yêu cầu danh sách nhãn ground truth (gt_list) xác thực từ COCO cohort.")

    track_times_clean: List[float] = []
    track_times_eqf: List[float] = []
    track_times_dqf: List[float] = []
    clean_dims: List[int] = []
    adv_dims: List[int] = []

    n_samples = min(len(gt_list), len(clean_post), len(eqf_post))
    dqf_post = postprocess_rtdetr_predictions(dqf_logits, dqf_boxes, score_thresh=0.35)

    for i in range(n_samples):
        cb = clean_post[i].get("boxes", np.zeros((0, 4)))
        eb = eqf_post[i].get("boxes", np.zeros((0, 4)))
        db = dqf_post[i].get("boxes", np.zeros((0, 4)))

        gt_boxes = gt_list[i].get("boxes_xyxy", gt_list[i].get("boxes", np.zeros((0, 4))))
        n_trk = min(len(gt_boxes), 12)
        if n_trk == 0:
            n_trk = 4

        for b_set, time_acc in [
            (cb, track_times_clean),
            (eb, track_times_eqf),
            (db, track_times_dqf),
        ]:
            n_det = max(len(b_set), 1)
            cost = (
                1.0 - compute_iou_matrix(b_set[:n_det], b_set[:n_trk])
                if len(b_set) > 0
                else np.ones((n_trk, n_det))
            )
            if cost.shape[0] == 0 or cost.shape[1] == 0:
                cost = np.ones((n_trk, n_det))

            t0 = time.perf_counter()
            _ = linear_sum_assignment(cost)
            t1 = time.perf_counter()
            time_acc.append(float(round((t1 - t0) * 1000.0, 4)))

        clean_dims.append(len(cb))
        adv_dims.append(len(eb))

    c_mean = float(np.mean(clean_dims)) if clean_dims else 1.0
    a_mean = float(np.mean(adv_dims)) if adv_dims else 1.0
    c_dim_str = f"{int(round(c_mean))} x {int(round(c_mean))}"
    a_dim_str = f"{int(round(c_mean))} x {int(round(a_mean))}"
    exp_factor = f"{a_mean / max(c_mean, 1e-4):.1f}x"

    return {
        "clean_matrix_dim": c_dim_str,
        "adv_matrix_dim": a_dim_str,
        "expansion_factor": exp_factor,
        "clean_latency_ms": float(np.median(track_times_clean)) if track_times_clean else 0.0,
        "adv_latency_ms": float(np.median(track_times_eqf)) if track_times_eqf else 0.0,
        "dqf_latency_ms": float(np.median(track_times_dqf)) if track_times_dqf else 0.0,
        "track_times_clean": track_times_clean,
        "track_times_eqf": track_times_eqf,
        "track_times_dqf": track_times_dqf,
        "proxy_note": "Analytical proxy evaluating Kuhn-Munkres Hungarian association complexity under phantom query floods.",
    }


def simulate_bytetrack_tracking_failures(
    clean_detections: Sequence[Dict[str, Any]],
    adv_detections: Sequence[Dict[str, Any]],
    gt_list: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Tiện ích đo độ phức tạp gán cặp Hungarian của ByteTrack giữa tập sạch và đối kháng.
    Bắt buộc nhận danh sách gt_list xác thực, không dùng pseudo-GT giả định.
    """
    if gt_list is None or len(gt_list) == 0:
        raise ValueError("simulate_bytetrack_tracking_failures yêu cầu nhãn ground truth (gt_list) thực tế từ dataset.")

    n_samples = min(len(clean_detections), len(adv_detections), len(gt_list))

    track_times_clean: List[float] = []
    track_times_adv: List[float] = []
    clean_dims: List[int] = []
    adv_dims: List[int] = []

    for i in range(n_samples):
        cb = clean_detections[i].get("boxes", np.zeros((0, 4)))
        ab = adv_detections[i].get("boxes", np.zeros((0, 4)))

        gt_boxes = gt_list[i].get("boxes_xyxy", gt_list[i].get("boxes", np.zeros((0, 4))))
        n_trk = min(len(gt_boxes), 12)
        if n_trk == 0:
            n_trk = 4

        for b_set, acc in [(cb, track_times_clean), (ab, track_times_adv)]:
            n_det = max(len(b_set), 1)
            cost = 1.0 - compute_iou_matrix(b_set[:n_det], b_set[:n_trk]) if len(b_set) > 0 else np.ones((n_trk, n_det))
            if cost.shape[0] == 0 or cost.shape[1] == 0:
                cost = np.ones((n_trk, n_det))
            t0 = time.perf_counter()
            _ = linear_sum_assignment(cost)
            t1 = time.perf_counter()
            acc.append(float(round((t1 - t0) * 1000.0, 4)))

        clean_dims.append(len(cb))
        adv_dims.append(len(ab))

    c_mean = float(np.mean(clean_dims)) if clean_dims else 1.0
    a_mean = float(np.mean(adv_dims)) if adv_dims else 1.0
    c_dim_str = f"{int(round(c_mean))} x {int(round(c_mean))}"
    a_dim_str = f"{int(round(c_mean))} x {int(round(a_mean))}"
    exp_factor = f"{a_mean / max(c_mean, 1e-4):.1f}x"

    return {
        "clean_matrix_dim": c_dim_str,
        "adv_matrix_dim": a_dim_str,
        "clean_latency_ms": float(np.median(track_times_clean)) if track_times_clean else 0.0,
        "adv_latency_ms": float(np.median(track_times_adv)) if track_times_adv else 0.0,
        "expansion_factor": exp_factor,
        "clean_dims_mean": c_mean,
        "adv_dims_mean": a_mean,
        "track_times_clean": track_times_clean,
        "track_times_adv": track_times_adv,
        "proxy_note": "Analytical proxy evaluating Kuhn-Munkres Hungarian association complexity under phantom query floods.",
    }
