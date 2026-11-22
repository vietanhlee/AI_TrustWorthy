# -*- coding: utf-8 -*-
"""
Module: paper_2_phantom_fabrication.defenses.baseline_defenses
Mục đích: Cài đặt và đánh giá toàn diện các cơ chế phòng thủ cho Paper 2 (Table 4):
  1. No Defense (Raw RT-DETR Model)
  2. Standard Post-hoc NMS (IoU = 0.50)
  3. DIoU-NMS (Distance-IoU NMS, IoU = 0.50)
  4. Soft-NMS (Gaussian Decay, sigma = 0.50)
  5. Threshold Elevation (Confidence Threshold tau = 0.50)
  6. Spatial Clustering DBSCAN (eps = 0.15)
  7. Proposed SQD-Filter (Distance Threshold r = 0.15)
  8. Input Pre-Processing Defenses (Gaussian Blur, Median Filter, JPEG Compression, Bit-Depth Reduction)
"""

import io
import time
from typing import Any, Dict, List, Tuple
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision.ops import nms

from paper_2_phantom_fabrication.core.coco_gt import (
    evaluate_coco_predictions,
    postprocess_rtdetr_predictions,
)
from paper_2_phantom_fabrication.core.matcher import match_detections_against_gt
from paper_2_phantom_fabrication.defenses.sqd_filter import apply_sqd_filter


class JPEGCompressionDefense(nn.Module):
    """Phòng thủ tiền xử lý ảnh bằng nén JPEG chuẩn hóa không suy giảm không gian màu."""
    def __init__(self, quality: int = 75):
        super().__init__()
        self.quality = quality

    def forward(self, img_tensor: torch.Tensor) -> torch.Tensor:
        device = img_tensor.device
        dtype = img_tensor.dtype
        b_size = img_tensor.shape[0]
        out_list = []
        for i in range(b_size):
            img_clamped = img_tensor[i].detach().cpu().clamp(0.0, 1.0)
            pil_img = TF.to_pil_image(img_clamped).convert("RGB")
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=self.quality)
            buf.seek(0)
            with Image.open(buf) as rec_img:
                rec_rgb = rec_img.convert("RGB")
                out_list.append(TF.to_tensor(rec_rgb))
        return torch.stack(out_list, dim=0).to(device=device, dtype=dtype)


class GaussianBlurDefense(nn.Module):
    """Phòng thủ tiền xử lý ảnh bằng bộ lọc làm mờ Gaussian."""
    def __init__(self, kernel_size: int = 5, sigma: float = 1.0):
        super().__init__()
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        self.sigma = sigma

    def forward(self, img_tensor: torch.Tensor) -> torch.Tensor:
        return TF.gaussian_blur(img_tensor, kernel_size=self.kernel_size, sigma=[self.sigma, self.sigma])


class MedianFilterDefense(nn.Module):
    """Phòng thủ tiền xử lý ảnh bằng bộ lọc trung vị."""
    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, img_tensor: torch.Tensor) -> torch.Tensor:
        device = img_tensor.device
        dtype = img_tensor.dtype
        pad = self.kernel_size // 2
        unfolded = torch.nn.functional.unfold(img_tensor, kernel_size=self.kernel_size, padding=pad)
        B, C_K2, L = unfolded.shape
        C = img_tensor.shape[1]
        unfolded = unfolded.view(B, C, self.kernel_size * self.kernel_size, L)
        median_val = unfolded.median(dim=2).values
        H, W = img_tensor.shape[2], img_tensor.shape[3]
        return median_val.view(B, C, H, W).to(device=device, dtype=dtype)


class BitDepthReductionDefense(nn.Module):
    """Phòng thủ tiền xử lý ảnh bằng giảm độ sâu bit (Lượng tử hóa giá trị pixel)."""
    def __init__(self, bits: int = 4):
        super().__init__()
        self.levels = 2 ** bits

    def forward(self, img_tensor: torch.Tensor) -> torch.Tensor:
        quantized = torch.floor(img_tensor * (self.levels - 1) + 0.5) / (self.levels - 1)
        return quantized.clamp(0.0, 1.0)


def diou_nms_single_image(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_thresh: float = 0.50,
) -> List[int]:
    """
    Distance-IoU NMS (DIoU-NMS):
    DIoU = IoU - (rho^2(b1, b2) / c^2)
    boxes: [N, 4] dạng xyxy
    scores: [N]
    """
    if len(boxes) == 0:
        return []
    
    order = np.argsort(-scores)
    keep = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break

        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = areas[i] + areas[rest] - inter
        iou = inter / np.maximum(union, 1e-12)

        # Khoảng cách tâm
        d2 = (cx[i] - cx[rest]) ** 2 + (cy[i] - cy[rest]) ** 2
        # Đường chéo nhỏ nhất bao quanh cả 2 hộp
        c_x1 = np.minimum(x1[i], x1[rest])
        c_y1 = np.minimum(y1[i], y1[rest])
        c_x2 = np.maximum(x2[i], x2[rest])
        c_y2 = np.maximum(y2[i], y2[rest])
        c2 = (c_x2 - c_x1) ** 2 + (c_y2 - c_y1) ** 2 + 1e-12

        diou = iou - (d2 / c2)
        inds = np.where(diou <= iou_thresh)[0]
        order = rest[inds]

    return keep


def soft_nms_single_image(
    boxes: np.ndarray,
    scores: np.ndarray,
    sigma: float = 0.50,
    score_thresh: float = 0.35,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Gaussian Soft-NMS:
    Giảm điểm số tin cậy s_j = s_j * exp(-IoU^2 / sigma) thay vì xóa cứng.
    """
    if len(boxes) == 0:
        return boxes, scores, []

    b_boxes = boxes.copy()
    b_scores = scores.copy()
    n = len(b_boxes)
    keep = []

    for i in range(n):
        max_idx = i + np.argmax(b_scores[i:])
        # Hoán vị
        b_boxes[[i, max_idx]] = b_boxes[[max_idx, i]]
        b_scores[[i, max_idx]] = b_scores[[max_idx, i]]

        if b_scores[i] < score_thresh:
            continue
        keep.append(i)

        # Tính IoU giữa hộp i và các hộp còn lại
        xx1 = np.maximum(b_boxes[i, 0], b_boxes[i + 1:, 0])
        yy1 = np.maximum(b_boxes[i, 1], b_boxes[i + 1:, 1])
        xx2 = np.minimum(b_boxes[i, 2], b_boxes[i + 1:, 2])
        yy2 = np.minimum(b_boxes[i, 3], b_boxes[i + 1:, 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        area_i = (b_boxes[i, 2] - b_boxes[i, 0]) * (b_boxes[i, 3] - b_boxes[i, 1])
        area_rest = (b_boxes[i + 1:, 2] - b_boxes[i + 1:, 0]) * (b_boxes[i + 1:, 3] - b_boxes[i + 1:, 1])
        union = area_i + area_rest - inter
        iou = inter / np.maximum(union, 1e-12)

        # Gaussian decay
        weight = np.exp(-(iou ** 2) / sigma)
        b_scores[i + 1:] *= weight

    valid_mask = b_scores[keep] >= score_thresh
    final_keep = [keep[idx] for idx in np.where(valid_mask)[0]]
    return b_boxes[final_keep], b_scores[final_keep], final_keep


def apply_defense_to_predictions(
    logits_in: torch.Tensor,
    boxes_in: torch.Tensor,
    defense_type: str = "raw",
    score_th: float = 0.35,
    kwargs: Dict[str, Any] = None,
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Áp dụng một cơ chế phòng thủ lên logits và boxes dự đoán của mô hình.
    Trả về danh sách detections sau xử lý cùng độ trễ thực thi (milliseconds/ảnh).
    """
    kw = kwargs or {}
    t_start = time.perf_counter()

    if defense_type == "raw":
        res = postprocess_rtdetr_predictions(logits_in, boxes_in, score_thresh=score_th)
    elif defense_type == "sqd":
        dist_th = kw.get("dist_thresh", 0.15)
        res = apply_sqd_filter(logits_in, boxes_in, score_thresh=score_th, dist_thresh=dist_th)
    elif defense_type == "thresh":
        elevated_th = kw.get("thresh", 0.50)
        res = postprocess_rtdetr_predictions(logits_in, boxes_in, score_thresh=elevated_th)
    elif defense_type == "nms":
        raw = postprocess_rtdetr_predictions(logits_in, boxes_in, score_thresh=score_th)
        iou_th = kw.get("iou_thresh", 0.50)
        res = []
        for r in raw:
            if len(r["boxes"]) == 0:
                res.append(r)
                continue
            keep = []
            boxes_t = torch.tensor(r["boxes"])
            scores_t = torch.tensor(r["scores"])
            for cls_id in np.unique(r["labels"]):
                c_mask = torch.tensor(r["labels"] == cls_id)
                idx = torch.nonzero(c_mask).squeeze(1)
                b_c = boxes_t[idx]
                s_c = scores_t[idx]
                k_c = nms(b_c, s_c, iou_th)
                keep.extend(idx[k_c].tolist())
            keep = sorted(keep)
            res.append({
                "boxes": r["boxes"][keep],
                "scores": r["scores"][keep],
                "labels": r["labels"][keep],
            })
    elif defense_type == "diou":
        raw = postprocess_rtdetr_predictions(logits_in, boxes_in, score_thresh=score_th)
        iou_th = kw.get("iou_thresh", 0.50)
        res = []
        for r in raw:
            if len(r["boxes"]) <= 1:
                res.append(r)
                continue
            keep = []
            for cls_id in np.unique(r["labels"]):
                c_mask = (r["labels"] == cls_id)
                idx = np.where(c_mask)[0]
                k_c = diou_nms_single_image(r["boxes"][idx], r["scores"][idx], iou_thresh=iou_th)
                keep.extend(idx[k_c].tolist())
            keep = sorted(keep)
            res.append({
                "boxes": r["boxes"][keep],
                "scores": r["scores"][keep],
                "labels": r["labels"][keep],
            })
    elif defense_type == "soft_nms":
        raw = postprocess_rtdetr_predictions(logits_in, boxes_in, score_thresh=score_th)
        sigma = kw.get("sigma", 0.50)
        res = []
        for r in raw:
            if len(r["boxes"]) <= 1:
                res.append(r)
                continue
            final_b, final_s, final_l = [], [], []
            for cls_id in np.unique(r["labels"]):
                c_mask = (r["labels"] == cls_id)
                idx = np.where(c_mask)[0]
                sb, ss, _ = soft_nms_single_image(r["boxes"][idx], r["scores"][idx], sigma=sigma, score_thresh=score_th)
                if len(sb) > 0:
                    final_b.append(sb)
                    final_s.append(ss)
                    final_l.append(np.full(len(sb), cls_id, dtype=int))
            if final_b:
                res.append({
                    "boxes": np.concatenate(final_b, axis=0),
                    "scores": np.concatenate(final_s, axis=0),
                    "labels": np.concatenate(final_l, axis=0),
                })
            else:
                res.append({"boxes": np.zeros((0, 4)), "scores": np.zeros((0,)), "labels": np.zeros((0,), dtype=int)})
    elif defense_type == "dbscan":
        raw = postprocess_rtdetr_predictions(logits_in, boxes_in, score_thresh=score_th)
        eps_val = kw.get("eps", 0.15)
        res = []
        from sklearn.cluster import DBSCAN
        for r in raw:
            if len(r["boxes"]) <= 1:
                res.append(r)
                continue
            centers = np.stack(
                [(r["boxes"][:, 0] + r["boxes"][:, 2]) / 2.0, (r["boxes"][:, 1] + r["boxes"][:, 3]) / 2.0],
                axis=-1,
            )
            db = DBSCAN(eps=eps_val, min_samples=2).fit(centers)
            labels = db.labels_
            keep = []
            for cid in np.unique(labels):
                if cid == -1:
                    keep.extend(np.where(labels == -1)[0].tolist())
                else:
                    cluster_idx = np.where(labels == cid)[0]
                    best_in_c = cluster_idx[np.argmax(r["scores"][cluster_idx])]
                    keep.append(best_in_c)
            keep = sorted(keep)
            res.append({
                "boxes": r["boxes"][keep],
                "scores": r["scores"][keep],
                "labels": r["labels"][keep],
            })
    else:
        raise ValueError(f"Kiểu phòng thủ không được hỗ trợ: {defense_type}")

    elapsed_ms = (time.perf_counter() - t_start) * 1000.0 / max(len(logits_in), 1)
    return res, float(elapsed_ms)


def evaluate_defenses_suite(
    clean_logits: torch.Tensor,
    clean_boxes: torch.Tensor,
    eqf_logits: torch.Tensor,
    eqf_boxes: torch.Tensor,
    dqf_logits: torch.Tensor,
    dqf_boxes: torch.Tensor,
    gt_list: List[Dict[str, Any]],
    nominal_clean_active_mean: float,
    nominal_clean_map50: float,
    raw_eqf_phantoms_mean: float,
    raw_dqf_phantoms_mean: float,
    target_cls: int = 1,
) -> List[Dict[str, Any]]:
    """
    Thực hiện đánh giá đối chứng toàn diện các cơ chế phòng thủ cho Bảng 4.
    Bao gồm No Defense, NMS, DIoU-NMS, Soft-NMS, Threshold Elevation, DBSCAN, SQD-Filter.
    """
    def_configs = [
        {"name": "No Defense (Raw Model)", "type": "raw", "kwargs": {}},
        {"name": "Standard NMS (IoU=0.50)", "type": "nms", "kwargs": {"iou_thresh": 0.50}},
        {"name": "DIoU-NMS (IoU=0.50)", "type": "diou", "kwargs": {"iou_thresh": 0.50}},
        {"name": "Soft-NMS (sigma=0.50)", "type": "soft_nms", "kwargs": {"sigma": 0.50}},
        {"name": "Threshold Elevation (tau=0.50)", "type": "thresh", "kwargs": {"thresh": 0.50}},
        {"name": "Spatial DBSCAN (eps=0.15)", "type": "dbscan", "kwargs": {"eps": 0.15}},
        {"name": "SQD-Filter (dist=0.15)", "type": "sqd", "kwargs": {"dist_thresh": 0.15}},
    ]

    defenses_results = []
    # Đo latency của raw baseline để tính overhead chính xác
    _, raw_lat = apply_defense_to_predictions(clean_logits, clean_boxes, defense_type="raw", score_th=0.35)

    for cfg in def_configs:
        d_type = cfg["type"]
        kw = cfg["kwargs"]

        preds_clean, def_lat = apply_defense_to_predictions(clean_logits, clean_boxes, defense_type=d_type, score_th=0.35, kwargs=kw)
        preds_eqf, _ = apply_defense_to_predictions(eqf_logits, eqf_boxes, defense_type=d_type, score_th=0.35, kwargs=kw)
        preds_dqf, _ = apply_defense_to_predictions(dqf_logits, dqf_boxes, defense_type=d_type, score_th=0.35, kwargs=kw)

        # 1. Đánh giá trên dữ liệu Clean
        cur_clean_cnt = np.mean([len(p["boxes"]) for p in preds_clean])
        clean_ret = (cur_clean_cnt / max(nominal_clean_active_mean, 1e-6)) * 100.0

        clean_coco = evaluate_coco_predictions(preds_clean, gt_list)
        clean_map_shift = clean_coco["mAP@50"] - nominal_clean_map50

        # 2. Đánh giá trên tấn công EQF & DQF
        eqf_m = [
            match_detections_against_gt(p["boxes"], p["scores"], p["labels"], gt_list[i]["boxes_xyxy"], gt_list[i]["labels"], target_cls=target_cls)
            for i, p in enumerate(preds_eqf)
        ]
        dqf_m = [
            match_detections_against_gt(p["boxes"], p["scores"], p["labels"], gt_list[i]["boxes_xyxy"], gt_list[i]["labels"], target_cls=target_cls)
            for i, p in enumerate(preds_dqf)
        ]

        surv_eqf = np.mean([m["phantoms"] for m in eqf_m])
        surv_dqf = np.mean([m["phantoms"] for m in dqf_m])

        eqf_rej = (1.0 - surv_eqf / max(raw_eqf_phantoms_mean, 1e-6)) * 100.0 if d_type != "raw" else 0.0
        dqf_rej = (1.0 - surv_dqf / max(raw_dqf_phantoms_mean, 1e-6)) * 100.0 if d_type != "raw" else 0.0

        # Overhead thực thi so với raw baseline
        overhead = max(0.0, def_lat - raw_lat) if d_type != "raw" else 0.0

        defenses_results.append({
            "name": cfg["name"],
            "eqf_rejection_pct": float(np.clip(eqf_rej, 0.0, 100.0)),
            "dqf_rejection_pct": float(np.clip(dqf_rej, 0.0, 100.0)),
            "surviving_eqf": float(surv_eqf),
            "surviving_dqf": float(surv_dqf),
            "post_defense_asr_eqf": float(np.mean([m["is_asr"] for m in eqf_m]) * 100.0),
            "post_defense_asr_dqf": float(np.mean([m["is_asr"] for m in dqf_m]) * 100.0),
            "clean_retention_pct": float(np.clip(clean_ret, 0.0, 100.0)),
            "clean_map50": float(clean_coco["mAP@50"]),
            "clean_map50_shift": float(clean_map_shift),
            "cpu_latency_ms": float(def_lat),
            "overhead_ms": float(overhead),
        })

    return defenses_results
