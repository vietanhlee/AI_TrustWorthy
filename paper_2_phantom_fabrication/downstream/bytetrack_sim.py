# -*- coding: utf-8 -*-
"""
Module: paper_2_phantom_fabrication.downstream.bytetrack_sim
Mục đích: Đo lường độ trễ và sự bùng nổ kích thước ma trận gán cặp Bipartite Matching
của thuật toán theo dõi đa đối tượng ByteTrack khi bị ngập lụt bởi Phantom Queries.
"""

import time
from typing import Any, Dict, List
import numpy as np
from scipy.optimize import linear_sum_assignment

from paper_2_phantom_fabrication.core.coco_gt import (
    compute_iou_matrix,
    postprocess_rtdetr_predictions,
)
import torch


def simulate_bytetrack_complexity(
    clean_post: List[Dict[str, Any]],
    eqf_post: List[Dict[str, Any]],
    dqf_logits: torch.Tensor,
    dqf_boxes: torch.Tensor,
    gt_list: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Mô phỏng bước Hungarian Association trong ByteTrack trên từng khung hình.
    Đo thời gian thực thi thuật toán Khớp cực tiểu Kuhn-Munkres O(N^3).
    """
    track_times_clean: List[float] = []
    track_times_eqf: List[float] = []
    track_times_dqf: List[float] = []
    clean_dims: List[int] = []
    adv_dims: List[int] = []

    n_samples = len(gt_list)
    dqf_post = postprocess_rtdetr_predictions(dqf_logits, dqf_boxes, score_thresh=0.35)

    for i in range(n_samples):
        cb = clean_post[i]["boxes"]
        eb = eqf_post[i]["boxes"]
        db = dqf_post[i]["boxes"]

        for b_set, time_acc in [
            (cb, track_times_clean),
            (eb, track_times_eqf),
            (db, track_times_dqf),
        ]:
            n_det = max(len(b_set), 1)
            n_trk = min(len(gt_list[i]["boxes_xyxy"]), 12)
            if n_trk == 0:
                n_trk = 4

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

    c_dim_str = f"{int(np.mean(clean_dims))} x {int(np.mean(clean_dims))}"
    a_dim_str = f"{int(np.mean(clean_dims))} x {int(np.mean(adv_dims))}"
    exp_factor = f"{np.mean(adv_dims) / max(np.mean(clean_dims), 1):.1f}x"

    return {
        "clean_matrix_dim": c_dim_str,
        "adv_matrix_dim": a_dim_str,
        "expansion_factor": exp_factor,
        "clean_latency_ms": float(np.median(track_times_clean)),
        "adv_latency_ms": float(np.median(track_times_eqf)),
        "track_times_clean": track_times_clean,
        "track_times_eqf": track_times_eqf,
        "track_times_dqf": track_times_dqf,
    }
