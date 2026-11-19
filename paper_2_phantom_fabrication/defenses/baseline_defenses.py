# -*- coding: utf-8 -*-
"""
Module: paper_2_phantom_fabrication.defenses.baseline_defenses
Mục đích: Cài đặt và đánh giá toàn diện 5 cơ chế phòng thủ (Bảng 4 trong bài báo Elsevier):
  1. No Defense (Raw RT-DETR Model)
  2. Standard Post-hoc NMS (IoU = 0.50)
  3. Threshold Elevation (Confidence Threshold tau = 0.50)
  4. Spatial Clustering DBSCAN (eps = 0.15)
  5. Proposed SQD-Filter (Distance Threshold r = 0.15)
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
    """Phòng thủ tiền xử lý ảnh bằng nén JPEG có tổn hao chất lượng."""
    def __init__(self, quality: int = 75):
        super().__init__()
        self.quality = quality

    def forward(self, img_tensor: torch.Tensor) -> torch.Tensor:
        # img_tensor: [B, C, H, W] in [0, 1]
        device = img_tensor.device
        dtype = img_tensor.dtype
        out_list = []
        for img in img_tensor:
            pil_img = TF.to_pil_image(img.cpu())
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=self.quality)
            buf.seek(0)
            rec_img = Image.open(buf)
            out_list.append(TF.to_tensor(rec_img))
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
        # Xấp xỉ trung vị hoặc dùng pooling/Pillow
        device = img_tensor.device
        dtype = img_tensor.dtype
        pad = self.kernel_size // 2
        # unfolded: [B, C, H, W, K*K]
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
        # img_tensor in [0, 1]
        quantized = torch.floor(img_tensor * (self.levels - 1) + 0.5) / (self.levels - 1)
        return quantized.clamp(0.0, 1.0)



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
    Thực hiện đánh giá đối chứng toàn diện 5 cơ chế phòng thủ cho Bảng 4.
    """
    def_configs = [
        {"name": "No Defense (Raw Model)", "type": "raw", "kwargs": {}},
        {"name": "Standard NMS (IoU=0.50)", "type": "nms", "kwargs": {"iou_thresh": 0.50}},
        {"name": "Threshold Elevation (tau=0.50)", "type": "thresh", "kwargs": {"thresh": 0.50}},
        {"name": "Spatial DBSCAN (eps=0.15)", "type": "dbscan", "kwargs": {"eps": 0.15}},
        {"name": "SQD-Filter (dist=0.15)", "type": "sqd", "kwargs": {"dist_thresh": 0.15}},
    ]

    defenses_results = []

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
        })

    return defenses_results
