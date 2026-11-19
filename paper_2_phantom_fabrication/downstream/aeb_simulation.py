# -*- coding: utf-8 -*-
"""
Module: paper_2_phantom_fabrication.downstream.aeb_simulation
Mục đích: Mô phỏng hình học chiếu 3D Pinhole Camera và tính toán đường cong rủi ro phanh khẩn cấp
tự động (Automatic Emergency Braking - AEB) khi xuất hiện các chướng ngại vật ảo (Phantom Detections).
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch

from paper_2_phantom_fabrication.core.coco_gt import postprocess_rtdetr_predictions


def project_boxes_to_ground_plane(
    boxes_xyxy: np.ndarray,
    h_cam: float = 1.3,
    f_y: float = 700.0,
    v_0: float = 0.5,
    img_size: float = 640.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Chiếu tọa độ bounding boxes từ không gian ảnh 2D (xyxy chuẩn hóa [0, 1])
    xuống mặt phẳng đường (Ground Plane 3D) theo mô hình Pinhole Camera:
      - Khoảng cách dọc (Longitudinal Distance): d_long = (h_cam * f_y) / ((v_b - v_0) * img_size)
      - Khoảng cách ngang (Lateral Offset): y_lat = d_long * ((u_b - 0.5) * img_size) / f_y

    Parameters:
        boxes_xyxy: Ma trận (N, 4) chứa [x1, y1, x2, y2] trong [0, 1].
        h_cam: Chiều cao đặt camera tính bằng mét (mặc định 1.3m).
        f_y: Tiêu cự camera theo trục y tính bằng pixel (mặc định 700.0px).
        v_0: Tọa độ điểm chân trời (vanishing line y-intercept) chuẩn hóa [0, 1].
        img_size: Kích thước cạnh ảnh (pixel).

    Returns:
        (d_long, y_lat): Hai mảng 1D chứa khoảng cách dọc và lệch ngang (mét).
    """
    if len(boxes_xyxy) == 0:
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)

    # Đáy bounding box (tiếp xúc mặt đường)
    u_b = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2.0
    v_b = boxes_xyxy[:, 3]

    valid = v_b > (v_0 + 0.01)
    if not np.any(valid):
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)

    v_valid = v_b[valid]
    u_valid = u_b[valid]

    d_long = (h_cam * f_y) / ((v_valid - v_0) * img_size)
    y_lat = d_long * ((u_valid - 0.5) * img_size) / f_y

    return d_long, y_lat


def compute_aeb_curve(
    postprocessed_list: Sequence[Dict[str, Any]],
    dists_grid: np.ndarray,
    h_cam: float = 1.3,
    f_y: float = 700.0,
    v_0: float = 0.5,
    img_size: float = 640.0,
    corridor_half_width: float = 1.8,
) -> List[float]:
    """
    Tính toán tỷ lệ kích hoạt phanh AEB (% số khung hình có vật cản ảo trong hành lang di chuyển)
    theo từng cự ly kích hoạt phanh (dists_grid).

    Parameters:
        postprocessed_list: Danh sách kết quả dự đoán sau ngưỡng cho từng ảnh.
        dists_grid: Mảng các cự ly phanh (mét), e.g., np.linspace(10, 80, 36).
        corridor_half_width: Bán kính làn xe (m), e.g., 1.8m (tương ứng làn đường rộng 3.6m).

    Returns:
        hazard_percentages: Danh sách tỷ lệ kích hoạt (%) ứng với từng mốc khoảng cách.
    """
    n_frames = max(len(postprocessed_list), 1)
    hazard_counts = np.zeros(len(dists_grid), dtype=np.float64)

    for p in postprocessed_list:
        boxes = p["boxes"]
        if len(boxes) == 0:
            continue

        d_long, y_lat = project_boxes_to_ground_plane(
            boxes, h_cam=h_cam, f_y=f_y, v_0=v_0, img_size=img_size
        )
        if len(d_long) == 0:
            continue

        in_corridor = np.abs(y_lat) <= corridor_half_width
        if not np.any(in_corridor):
            continue

        d_corridor = d_long[in_corridor]
        for d_idx, d_max in enumerate(dists_grid):
            if np.any(d_corridor <= d_max):
                hazard_counts[d_idx] += 1

    return (hazard_counts / n_frames * 100.0).tolist()


def simulate_aeb_impact(
    clean_post: Sequence[Dict[str, Any]],
    eqf_post: Sequence[Dict[str, Any]],
    dqf_logits: torch.Tensor,
    dqf_boxes: torch.Tensor,
    dists_grid: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Thực thi mô phỏng toàn diện tác động AEB cho cả 3 điều kiện:
    Clean (bình thường), EQF (tấn công mã hóa), DQF (tấn công giải mã).
    """
    if dists_grid is None:
        dists_grid = np.linspace(10, 80, 36)

    dqf_post = postprocess_rtdetr_predictions(dqf_logits, dqf_boxes, score_thresh=0.35)

    aeb_clean = compute_aeb_curve(clean_post, dists_grid)
    aeb_eqf = compute_aeb_curve(eqf_post, dists_grid)
    aeb_dqf = compute_aeb_curve(dqf_post, dists_grid)

    idx_28m = 12 if len(dists_grid) > 12 else (len(dists_grid) - 1)
    clean_28m = float(aeb_clean[idx_28m]) if len(aeb_clean) > idx_28m else 0.0
    eqf_28m = float(aeb_eqf[idx_28m]) if len(aeb_eqf) > idx_28m else 100.0
    dqf_28m = float(aeb_dqf[idx_28m]) if len(aeb_dqf) > idx_28m else 100.0

    return {
        "clean_risk_28m": clean_28m,
        "eqf_risk_28m": eqf_28m,
        "dqf_risk_28m": dqf_28m,
        "distance_grid_m": dists_grid.tolist(),
        "aeb_curve_clean": aeb_clean,
        "aeb_curve_eqf": aeb_eqf,
        "aeb_curve_dqf": aeb_dqf,
    }


def simulate_aeb_braking_events(
    clean_detections: Sequence[Dict[str, Any]],
    adv_detections: Sequence[Dict[str, Any]],
    dists_grid: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Tiện ích mô phỏng phanh AEB giữa kết quả detect sạch và đối kháng.
    """
    if dists_grid is None:
        dists_grid = np.linspace(10, 80, 36)

    clean_curve = compute_aeb_curve(clean_detections, dists_grid)
    adv_curve = compute_aeb_curve(adv_detections, dists_grid)

    idx_28m = 12 if len(dists_grid) > 12 else (len(dists_grid) - 1)
    clean_28m = float(clean_curve[idx_28m]) if len(clean_curve) > idx_28m else 0.0
    adv_28m = float(adv_curve[idx_28m]) if len(adv_curve) > idx_28m else 100.0

    return {
        "clean_risk_28m": clean_28m,
        "adv_risk_28m": adv_28m,
        "distance_grid_m": dists_grid.tolist(),
        "aeb_curve_clean": clean_curve,
        "aeb_curve_adv": adv_curve,
    }

