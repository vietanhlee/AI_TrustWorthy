# -*- coding: utf-8 -*-
"""
Module: paper_2_phantom_fabrication.downstream.aeb_simulation
Mục đích: Mô phỏng phân tích proxy hình học chiếu 3D Pinhole Camera (Analytical Inverse Perspective Projection Proxy)
và tính toán đường cong rủi ro phanh khẩn cấp tự động (Automatic Emergency Braking - AEB)
khi xuất hiện các chướng ngại vật ảo (Phantom Detections) trong kịch bản động học ADAS đường cao tốc tiêu chuẩn.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch

from paper_2_phantom_fabrication.core.coco_gt import postprocess_rtdetr_predictions


@dataclass
class AEBGeometryConfig:
    """
    Cấu hình hình học camera Pinhole và kịch bản động học ADAS đường cao tốc (Analytical Proxy Model):
      - h_cam: Chiều cao đặt camera tính bằng mét (mặc định 1.3m).
      - f_y: Tiêu cự camera trục y tính bằng pixel (mặc định 700.0px cho ảnh 640x640).
      - v_0: Tọa độ điểm chân trời (vanishing line y-intercept) chuẩn hóa [0, 1] (mặc định 0.5).
      - img_size: Kích thước cạnh ảnh chuẩn (640.0 px).
      - corridor_half_width: Bán kính hành lang an toàn làn xe tính bằng mét (1.8m, tương ứng làn đường 3.6m).
      - v_ego: Vận tốc di chuyển của xe tự hành (20.0 m/s = 72 km/h).
      - critical_ttc: Ngưỡng thời gian tới va chạm nguy cấp (1.4s).
      - critical_distance_m: Khoảng cách phanh khẩn cấp = v_ego * critical_ttc = 28.0m.
    """
    h_cam: float = 1.3
    f_y: float = 700.0
    v_0: float = 0.5
    img_size: float = 640.0
    corridor_half_width: float = 1.8
    v_ego: float = 20.0
    critical_ttc: float = 1.4

    @property
    def critical_distance_m(self) -> float:
        return self.v_ego * self.critical_ttc  # 20.0 m/s * 1.4 s = 28.0 m


def project_boxes_to_ground_plane(
    boxes_xyxy: np.ndarray,
    config: Optional[AEBGeometryConfig] = None,
    h_cam: Optional[float] = None,
    f_y: Optional[float] = None,
    v_0: Optional[float] = None,
    img_size: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Chiếu tọa độ bounding boxes từ không gian ảnh 2D (xyxy chuẩn hóa [0, 1])
    xuống mặt phẳng đường (Ground Plane 3D) theo mô hình Pinhole Camera:
      - Khoảng cách dọc (Longitudinal Distance): d_long = (h_cam * f_y) / ((v_b - v_0) * img_size)
      - Khoảng cách ngang (Lateral Offset): y_lat = d_long * ((u_b - 0.5) * img_size) / f_y
    """
    cfg = config if config is not None else AEBGeometryConfig()
    h = h_cam if h_cam is not None else cfg.h_cam
    fy = f_y if f_y is not None else cfg.f_y
    v0 = v_0 if v_0 is not None else cfg.v_0
    sz = img_size if img_size is not None else cfg.img_size

    if len(boxes_xyxy) == 0:
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)

    # Đáy bounding box (tiếp xúc mặt đường)
    u_b = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2.0
    v_b = boxes_xyxy[:, 3]

    valid = v_b > (v0 + 0.01)
    if not np.any(valid):
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)

    v_valid = v_b[valid]
    u_valid = u_b[valid]

    d_long = (h * fy) / ((v_valid - v0) * sz)
    y_lat = d_long * ((u_valid - 0.5) * sz) / fy

    return d_long, y_lat


def compute_aeb_curve(
    postprocessed_list: Sequence[Dict[str, Any]],
    dists_grid: np.ndarray,
    config: Optional[AEBGeometryConfig] = None,
    h_cam: Optional[float] = None,
    f_y: Optional[float] = None,
    v_0: Optional[float] = None,
    img_size: Optional[float] = None,
    corridor_half_width: Optional[float] = None,
) -> List[float]:
    """
    Tính toán tỷ lệ kích hoạt phanh AEB (% số khung hình có vật cản ảo trong hành lang di chuyển)
    theo từng cự ly kích hoạt phanh (dists_grid).
    """
    cfg = config if config is not None else AEBGeometryConfig()
    c_width = corridor_half_width if corridor_half_width is not None else cfg.corridor_half_width

    n_frames = max(len(postprocessed_list), 1)
    hazard_counts = np.zeros(len(dists_grid), dtype=np.float64)

    for p in postprocessed_list:
        boxes = p.get("boxes", np.zeros((0, 4)))
        if len(boxes) == 0:
            continue

        d_long, y_lat = project_boxes_to_ground_plane(
            boxes, config=cfg, h_cam=h_cam, f_y=f_y, v_0=v_0, img_size=img_size
        )
        if len(d_long) == 0:
            continue

        in_corridor = np.abs(y_lat) <= c_width
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
    config: Optional[AEBGeometryConfig] = None,
) -> Dict[str, Any]:
    """
    Thực thi mô phỏng toàn diện tác động AEB cho cả 3 điều kiện:
    Clean (bình thường), EQF (tấn công mã hóa), DQF (tấn công giải mã).
    Khóa chuẩn xác mốc 28.0m (TTC = 1.4s tại 72 km/h).
    """
    cfg = config if config is not None else AEBGeometryConfig()
    if dists_grid is None:
        dists_grid = np.linspace(10, 80, 36)

    dqf_post = postprocess_rtdetr_predictions(dqf_logits, dqf_boxes, score_thresh=0.35)

    aeb_clean = compute_aeb_curve(clean_post, dists_grid, config=cfg)
    aeb_eqf = compute_aeb_curve(eqf_post, dists_grid, config=cfg)
    aeb_dqf = compute_aeb_curve(dqf_post, dists_grid, config=cfg)

    # Khóa chính xác mốc 28.0 mét bằng phép tìm kiếm khoảng cách nhỏ nhất
    dists_arr = np.asarray(dists_grid, dtype=np.float64)
    idx_28m = int(np.argmin(np.abs(dists_arr - cfg.critical_distance_m)))
    actual_eval_dist = float(dists_arr[idx_28m])

    clean_28m = float(aeb_clean[idx_28m]) if len(aeb_clean) > idx_28m else 0.0
    eqf_28m = float(aeb_eqf[idx_28m]) if len(aeb_eqf) > idx_28m else 100.0
    dqf_28m = float(aeb_dqf[idx_28m]) if len(aeb_dqf) > idx_28m else 100.0

    return {
        "clean_risk_28m": clean_28m,
        "eqf_risk_28m": eqf_28m,
        "dqf_risk_28m": dqf_28m,
        "evaluated_stopping_distance_m": actual_eval_dist,
        "target_critical_distance_m": cfg.critical_distance_m,
        "distance_grid_m": dists_grid.tolist(),
        "aeb_curve_clean": aeb_clean,
        "aeb_curve_eqf": aeb_eqf,
        "aeb_curve_dqf": aeb_dqf,
        "geometry_config": asdict(cfg),
        "proxy_note": "Analytical inverse perspective projection proxy under highway kinematic parameters.",
    }


def simulate_aeb_braking_events(
    clean_detections: Sequence[Dict[str, Any]],
    adv_detections: Sequence[Dict[str, Any]],
    dists_grid: Optional[np.ndarray] = None,
    config: Optional[AEBGeometryConfig] = None,
) -> Dict[str, Any]:
    """
    Tiện ích mô phỏng phanh AEB giữa kết quả detect sạch và đối kháng.
    Khóa chuẩn xác mốc khoảng cách 28.0m (TTC = 1.4s).
    """
    cfg = config if config is not None else AEBGeometryConfig()
    if dists_grid is None:
        dists_grid = np.linspace(10, 80, 36)

    clean_curve = compute_aeb_curve(clean_detections, dists_grid, config=cfg)
    adv_curve = compute_aeb_curve(adv_detections, dists_grid, config=cfg)

    dists_arr = np.asarray(dists_grid, dtype=np.float64)
    idx_28m = int(np.argmin(np.abs(dists_arr - cfg.critical_distance_m)))
    actual_eval_dist = float(dists_arr[idx_28m])

    clean_28m = float(clean_curve[idx_28m]) if len(clean_curve) > idx_28m else 0.0
    adv_28m = float(adv_curve[idx_28m]) if len(adv_curve) > idx_28m else 100.0

    return {
        "clean_risk_28m": clean_28m,
        "adv_risk_28m": adv_28m,
        "dqf_risk_28m": adv_28m,
        "evaluated_stopping_distance_m": actual_eval_dist,
        "target_critical_distance_m": cfg.critical_distance_m,
        "distance_grid_m": dists_grid.tolist(),
        "aeb_curve_clean": clean_curve,
        "aeb_curve_adv": adv_curve,
        "geometry_config": asdict(cfg),
        "proxy_note": "Analytical inverse perspective projection proxy under highway kinematic parameters.",
    }
