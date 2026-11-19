# -*- coding: utf-8 -*-
"""
Module: paper_1_aqsm_mechanism.core.coco_gt
Mục đích: Nạp nhãn thật (Ground Truth) COCO val2017 từ `instances_val2017.json`,
ánh xạ chuẩn xác 80 lớp COCO sang chỉ số nhãn [0..79] của RT-DETR, và cung cấp
bộ đánh giá COCO mAP@50, mAP@50:95 (toàn cục + theo từng ảnh + Paired Bootstrap CI)
cũng như phân tích phương sai Repeated-Measures 2x2 ANOVA đầy đủ.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image
from scipy import stats
import torch
import torchvision.transforms.functional as TF

from paper_1_aqsm_mechanism.experiments.config import IMG_SIZE

def get_default_coco_val_dir() -> Path:
    """Lấy đường dẫn thư mục ảnh COCO val2017 động từ cấu hình."""
    from paper_1_aqsm_mechanism.experiments.config import get_coco_val_dir
    return get_coco_val_dir()


def get_default_coco_ann_file() -> Path:
    """Lấy đường dẫn file chú thích instances_val2017.json động từ cấu hình."""
    from paper_1_aqsm_mechanism.experiments.config import get_coco_ann_file
    return get_coco_ann_file()


N_CALIBRATION_SAMPLES = 1200
N_TEST_CLEAN_SAMPLES = 600

# Bảng alias tên lớp giữa COCO JSON chuẩn (80 lớp) và config.id2label của PekingU/rtdetr_r18vd
_COCO_NAME_ALIASES: Dict[str, str] = {
    "motorbike": "motorcycle",
    "aeroplane": "airplane",
    "sofa": "couch",
    "pottedplant": "potted plant",
    "diningtable": "dining table",
    "tvmonitor": "tv",
}


def box_cxcywh_to_xyxy(boxes: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Chuyển hộp từ [cx, cy, w, h] sang [x1, y1, x2, y2]."""
    if isinstance(boxes, torch.Tensor):
        cx, cy, w, h = boxes.unbind(-1)
        return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)
    cx = boxes[..., 0]
    cy = boxes[..., 1]
    w = boxes[..., 2]
    h = boxes[..., 3]
    return np.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], axis=-1)


def compute_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Tính ma trận IoU giữa hai tập hộp [N, 4] và [M, 4] dạng xyxy."""
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float64)

    area1 = np.maximum(0.0, boxes1[:, 2] - boxes1[:, 0]) * np.maximum(0.0, boxes1[:, 3] - boxes1[:, 1])
    area2 = np.maximum(0.0, boxes2[:, 2] - boxes2[:, 0]) * np.maximum(0.0, boxes2[:, 3] - boxes2[:, 1])

    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = np.clip(rb - lt, a_min=0.0, a_max=None)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = np.maximum(area1[:, None] + area2[None, :] - inter, 1e-12)
    return inter / union


def build_coco_category_mapping(
    coco_categories: Sequence[Dict[str, Any]],
    id2label: Optional[Dict[Union[int, str], str]] = None,
) -> Dict[int, int]:
    """
    Xây dựng ánh xạ chuẩn xác từ COCO category_id (không liên tục 1..90)
    sang chỉ số lớp liên tục 0..79 của RT-DETR.
    """
    cats_sorted = sorted(coco_categories, key=lambda c: int(c["id"]))
    if len(cats_sorted) != 80:
        raise ValueError(f"COCO categories phải có đúng 80 lớp, nhận được {len(cats_sorted)}")

    mapping: Dict[int, int] = {int(cat["id"]): idx for idx, cat in enumerate(cats_sorted)}

    if id2label is not None:
        norm_id2label = {int(k): str(v).strip().lower() for k, v in id2label.items()}
        for idx, cat in enumerate(cats_sorted):
            coco_name = str(cat["name"]).strip().lower()
            model_name = norm_id2label.get(idx, "")
            model_canonical = _COCO_NAME_ALIASES.get(model_name, model_name)
            if coco_name != model_name and coco_name != model_canonical:
                raise ValueError(
                    f"Lệch ánh xạ lớp tại index {idx}: COCO='{coco_name}', model id2label='{model_name}'"
                )

    return mapping


def get_partition_paths(
    coco_dir: Optional[Union[str, Path]] = None,
    n_cal: int = N_CALIBRATION_SAMPLES,
    n_test: int = N_TEST_CLEAN_SAMPLES,
    seed: int = 42,
) -> Tuple[List[Path], List[Path]]:
    """
    Tái lập chính xác danh sách đường dẫn ảnh của `core.dataset.load_coco_images`
    theo giao thức Zero Data Leakage (seed=42, N_cal=1200, N_test=600).
    """
    data_dir = Path(coco_dir) if coco_dir is not None else get_default_coco_val_dir()
    all_files = sorted(list(data_dir.glob("*.jpg")))
    total_needed = n_cal + n_test
    if len(all_files) < total_needed:
        raise FileNotFoundError(
            f"Thư mục {data_dir} chỉ có {len(all_files)} ảnh, thiếu so với {total_needed}."
        )
    rng = np.random.RandomState(seed)
    chosen_indices = rng.choice(len(all_files), size=total_needed, replace=False)
    chosen_files = [all_files[idx] for idx in chosen_indices]
    return chosen_files[:n_cal], chosen_files[n_cal:total_needed]


def load_coco_val2017_gt_cohort(
    coco_dir: Optional[Union[str, Path]] = None,
    ann_path: Optional[Union[str, Path]] = None,
    n_cal: int = N_CALIBRATION_SAMPLES,
    n_test: int = N_TEST_CLEAN_SAMPLES,
    eval_start: int = 0,
    n_eval: int = 300,
    img_size: int = IMG_SIZE,
    seed: int = 42,
    id2label: Optional[Dict[Union[int, str], str]] = None,
    load_tensors: bool = True,
) -> Dict[str, Any]:
    """
    Nạp tập ảnh test cùng Ground Truth COCO val2017 thật cho từng ảnh.

    Returns:
        Dict chứa:
        - `tensors`: Tensor [n_eval, 3, img_size, img_size] (nếu load_tensors=True)
        - `paths`: List[Path]
        - `image_ids`: List[int]
        - `gt_list`: List[Dict] mỗi ảnh gồm `boxes_xyxy`, `boxes_cxcywh`, `labels`, `iscrowd`, `orig_size`, `image_id`
        - `hf_labels`: List[Dict] tương thích `RTDetrForObjectDetection(..., labels=...)` (loại bỏ iscrowd)
        - `cat_id_to_idx`: Dict[int, int]
    """
    ann_file = Path(ann_path) if ann_path is not None else get_default_coco_ann_file()
    if not ann_file.exists():
        raise FileNotFoundError(f"Chưa có file COCO Ground Truth: {ann_file}")

    with open(ann_file, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    cat_map = build_coco_category_mapping(coco_data["categories"], id2label=id2label)
    img_meta: Dict[int, Dict[str, Any]] = {int(im["id"]): im for im in coco_data["images"]}
    anns_by_img: Dict[int, List[Dict[str, Any]]] = {}
    for ann in coco_data["annotations"]:
        img_id = int(ann["image_id"])
        anns_by_img.setdefault(img_id, []).append(ann)

    _, test_paths = get_partition_paths(
        coco_dir=coco_dir,
        n_cal=n_cal,
        n_test=n_test,
        seed=seed,
    )
    selected_paths = test_paths[eval_start : eval_start + n_eval]

    gt_list: List[Dict[str, Any]] = []
    hf_labels: List[Dict[str, torch.Tensor]] = []
    image_ids: List[int] = []
    tensor_list: List[torch.Tensor] = []

    for fpath in selected_paths:
        img_id = int(fpath.stem)
        image_ids.append(img_id)
        if img_id not in img_meta:
            raise KeyError(f"image_id={img_id} ({fpath.name}) không tồn tại trong {ann_file}")

        meta = img_meta[img_id]
        orig_w = float(meta["width"])
        orig_h = float(meta["height"])

        boxes_xyxy: List[List[float]] = []
        boxes_cxcywh: List[List[float]] = []
        labels: List[int] = []
        iscrowd: List[bool] = []

        for ann in anns_by_img.get(img_id, []):
            cid = int(ann["category_id"])
            if cid not in cat_map:
                continue
            x, y, w, h = [float(v) for v in ann["bbox"]]
            if w <= 1e-3 or h <= 1e-3:
                continue

            x1 = max(0.0, min(1.0, x / orig_w))
            y1 = max(0.0, min(1.0, y / orig_h))
            x2 = max(x1 + 1e-4, min(1.0, (x + w) / orig_w))
            y2 = max(y1 + 1e-4, min(1.0, (y + h) / orig_h))
            bw = max(1e-4, x2 - x1)
            bh = max(1e-4, y2 - y1)
            cx = x1 + 0.5 * bw
            cy = y1 + 0.5 * bh

            boxes_xyxy.append([x1, y1, x2, y2])
            boxes_cxcywh.append([cx, cy, bw, bh])
            labels.append(cat_map[cid])
            iscrowd.append(bool(ann.get("iscrowd", 0) == 1))

        arr_xyxy = np.array(boxes_xyxy, dtype=np.float32) if boxes_xyxy else np.zeros((0, 4), dtype=np.float32)
        arr_cxcywh = np.array(boxes_cxcywh, dtype=np.float32) if boxes_cxcywh else np.zeros((0, 4), dtype=np.float32)
        arr_lbl = np.array(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64)
        arr_crowd = np.array(iscrowd, dtype=bool) if iscrowd else np.zeros((0,), dtype=bool)

        gt_list.append({
            "image_id": img_id,
            "file_name": fpath.name,
            "orig_size": (int(orig_w), int(orig_h)),
            "boxes_xyxy": arr_xyxy,
            "boxes_cxcywh": arr_cxcywh,
            "labels": arr_lbl,
            "iscrowd": arr_crowd,
        })

        non_crowd_mask = ~arr_crowd
        hf_labels.append({
            "class_labels": torch.from_numpy(arr_lbl[non_crowd_mask].copy()),
            "boxes": torch.from_numpy(arr_cxcywh[non_crowd_mask].copy()),
        })

        if load_tensors:
            with Image.open(fpath) as img:
                rgb = img.convert("RGB").resize((img_size, img_size))
                tensor_list.append(TF.to_tensor(rgb))

    tensors = torch.stack(tensor_list, dim=0) if load_tensors else None
    return {
        "tensors": tensors,
        "paths": selected_paths,
        "image_ids": image_ids,
        "gt_list": gt_list,
        "hf_labels": hf_labels,
        "cat_id_to_idx": cat_map,
    }


def postprocess_rtdetr_predictions(
    logits: torch.Tensor,
    pred_boxes: torch.Tensor,
    score_thresh: float = 0.0,
    max_dets: int = 100,
    mode: str = "hf_multilabel",
) -> List[Dict[str, np.ndarray]]:
    """
    Chuyển logits [B, Q, C] và pred_boxes [B, Q, 4] (cxcywh [0,1]) sang dự đoán xyxy [0,1].
    - `hf_multilabel`: khớp `RTDetrImageProcessor.post_process_object_detection`
      (chọn top-k cặp (query, class) trên ma trận sigmoid [Q*C], mặc định cắt top `max_dets`).
    - `per_query_max`: lấy lớp có xác suất cao nhất cho mỗi query [Q].
    """
    logits_c = logits.detach().cpu()
    boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes.detach().cpu()).clamp(0.0, 1.0)
    probs = torch.sigmoid(logits_c)
    bsz, num_q, num_c = probs.shape

    results: List[Dict[str, np.ndarray]] = []
    for b in range(bsz):
        if mode == "hf_multilabel":
            k_sel = min(max_dets, num_q * num_c)
            scores, flat_idx = torch.topk(probs[b].reshape(-1), k=k_sel, dim=-1)
            labels = flat_idx % num_c
            box_idx = torch.div(flat_idx, num_c, rounding_mode="floor")
            b_boxes = boxes_xyxy[b, box_idx]
        elif mode == "per_query_max":
            scores, labels = probs[b].max(dim=-1)
            order = torch.argsort(scores, descending=True)[:max_dets]
            scores = scores[order]
            labels = labels[order]
            b_boxes = boxes_xyxy[b, order]
        else:
            raise ValueError(f"mode không hợp lệ: {mode}")

        if score_thresh > 0.0:
            keep = scores >= score_thresh
            scores = scores[keep]
            labels = labels[keep]
            b_boxes = b_boxes[keep]

        results.append({
            "boxes": b_boxes.numpy().astype(np.float64),
            "scores": scores.numpy().astype(np.float64),
            "labels": labels.numpy().astype(np.int64),
        })
    return results


def _evaluate_class_ap_for_subset(
    preds: Sequence[Dict[str, np.ndarray]],
    gt_list: Sequence[Dict[str, Any]],
    indices: Sequence[int],
    class_id: int,
    iou_thresholds: np.ndarray,
) -> np.ndarray:
    """
    Tính AP cho một lớp trên tập chỉ số `indices` qua danh sách `iou_thresholds`
    theo quy chuẩn COCO 101-point (có xử lý iscrowd ignore).
    Trả về vector shape [len(iou_thresholds)] (NaN nếu n_gt == 0).
    """
    all_scores: List[float] = []
    # Với mỗi detection matched: lưu best IoU với unmatched non-crowd GT hoặc -1 nếu bị crowd ignore
    det_iou_matched: List[np.ndarray] = []
    n_gt_total = 0

    for idx in indices:
        gt = gt_list[idx]
        gt_mask_c = (gt["labels"] == class_id)
        gt_boxes_c = gt["boxes_xyxy"][gt_mask_c]
        gt_crowd_c = gt["iscrowd"][gt_mask_c] if "iscrowd" in gt else np.zeros(len(gt_boxes_c), dtype=bool)

        non_crowd_boxes = gt_boxes_c[~gt_crowd_c]
        crowd_boxes = gt_boxes_c[gt_crowd_c]
        n_gt_total += len(non_crowd_boxes)

        pr = preds[idx]
        pr_mask_c = (pr["labels"] == class_id)
        pr_boxes_c = pr["boxes"][pr_mask_c]
        pr_scores_c = pr["scores"][pr_mask_c]

        if len(pr_scores_c) == 0:
            continue

        order = np.argsort(-pr_scores_c)
        pr_boxes_c = pr_boxes_c[order]
        pr_scores_c = pr_scores_c[order]

        ious_reg = compute_iou_matrix(pr_boxes_c, non_crowd_boxes) if len(non_crowd_boxes) > 0 else np.zeros((len(pr_boxes_c), 0))
        ious_crowd = compute_iou_matrix(pr_boxes_c, crowd_boxes) if len(crowd_boxes) > 0 else np.zeros((len(pr_boxes_c), 0))

        # Theo COCOeval: ghép greedy trên từng ngưỡng IoU
        # Lưu trạng thái per-threshold: 1=TP, 0=FP, -1=ignore(crowd)
        matched_status = np.zeros((len(pr_scores_c), len(iou_thresholds)), dtype=np.int8)

        for t_idx, iou_th in enumerate(iou_thresholds):
            used_gt = np.zeros(len(non_crowd_boxes), dtype=bool)
            for p_i in range(len(pr_scores_c)):
                best_iou = 0.0
                best_g = -1
                if len(non_crowd_boxes) > 0:
                    # Ưu tiên GT chưa khớp
                    cand_ious = np.where(~used_gt, ious_reg[p_i], -1.0)
                    best_g = int(np.argmax(cand_ious))
                    best_iou = float(cand_ious[best_g])

                if best_iou >= iou_th and best_g >= 0:
                    matched_status[p_i, t_idx] = 1
                    used_gt[best_g] = True
                elif len(crowd_boxes) > 0 and float(np.max(ious_crowd[p_i])) >= iou_th:
                    matched_status[p_i, t_idx] = -1
                else:
                    matched_status[p_i, t_idx] = 0

        all_scores.extend(pr_scores_c.tolist())
        det_iou_matched.append(matched_status)

    if n_gt_total == 0:
        return np.full(len(iou_thresholds), np.nan, dtype=np.float64)
    if len(all_scores) == 0:
        return np.zeros(len(iou_thresholds), dtype=np.float64)

    score_arr = np.asarray(all_scores, dtype=np.float64)
    match_mat = np.concatenate(det_iou_matched, axis=0)  # [N_det, N_thresh]
    global_order = np.argsort(-score_arr, kind="mergesort")
    match_mat = match_mat[global_order]

    recall_levels = np.linspace(0.0, 1.0, 101)
    aps = np.zeros(len(iou_thresholds), dtype=np.float64)

    for t_idx in range(len(iou_thresholds)):
        col = match_mat[:, t_idx]
        valid = (col != -1)
        col_v = col[valid]
        if len(col_v) == 0:
            aps[t_idx] = 0.0
            continue
        tp = (col_v == 1).astype(np.float64)
        fp = (col_v == 0).astype(np.float64)
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
        recalls = tp_cum / float(n_gt_total)

        # Envelope giảm đơn điệu từ phải sang trái (tương đương np.max(precisions[recalls >= r]))
        prec_interp = np.zeros(len(recall_levels), dtype=np.float64)
        for r_i, r_val in enumerate(recall_levels):
            mask_r = recalls >= r_val
            prec_interp[r_i] = np.max(precisions[mask_r]) if np.any(mask_r) else 0.0
        aps[t_idx] = float(np.mean(prec_interp))

    return aps


def evaluate_coco_predictions(
    preds: Sequence[Dict[str, np.ndarray]],
    gt_list: Sequence[Dict[str, Any]],
    indices: Optional[Sequence[int]] = None,
    num_classes: int = 80,
) -> Dict[str, Any]:
    """
    Tính COCO mAP@50, mAP@75 và mAP@50:95 (thang %) chuẩn xác trên tập ảnh.
    """
    idx_list = list(range(len(gt_list))) if indices is None else list(indices)
    iou_thresholds = np.linspace(0.50, 0.95, 10)
    class_ap_mat = np.full((num_classes, len(iou_thresholds)), np.nan, dtype=np.float64)

    for c in range(num_classes):
        class_ap_mat[c] = _evaluate_class_ap_for_subset(
            preds=preds,
            gt_list=gt_list,
            indices=idx_list,
            class_id=c,
            iou_thresholds=iou_thresholds,
        )

    valid_classes = ~np.isnan(class_ap_mat[:, 0])
    n_active_classes = int(np.sum(valid_classes))
    if n_active_classes == 0:
        return {
            "mAP@50": 0.0,
            "mAP@75": 0.0,
            "mAP@50:95": 0.0,
            "active_classes": 0,
        }

    ap_per_iou = np.mean(class_ap_mat[valid_classes], axis=0) * 100.0
    return {
        "mAP@50": float(ap_per_iou[0]),
        "mAP@75": float(ap_per_iou[5]),
        "mAP@50:95": float(np.mean(ap_per_iou)),
        "active_classes": n_active_classes,
    }


def precompute_coco_match_cache(
    preds: Sequence[Dict[str, np.ndarray]],
    gt_list: Sequence[Dict[str, Any]],
    num_classes: int = 80,
) -> Dict[str, Any]:
    """
    Tiền tính toán sẵn trạng thái khớp TP/FP/ignore từng cặp (ảnh, lớp) qua 10 ngưỡng IoU.
    Vì việc khớp prediction với GT diễn ra độc lập trên từng ảnh, kết quả bootstrap
    trên mọi `indices` bất kỳ dùng cache này hoàn toàn tương đương bit-for-bit với
    `evaluate_coco_predictions` nhưng nhanh hơn >100 lần.
    """
    iou_thresholds = np.linspace(0.50, 0.95, 10)
    n_img = len(gt_list)
    per_class: List[Dict[str, Any]] = []

    for c in range(num_classes):
        n_gt_vec = np.zeros(n_img, dtype=np.int64)
        scores_list: List[np.ndarray] = []
        matches_list: List[np.ndarray] = []

        for idx in range(n_img):
            gt = gt_list[idx]
            gt_mask_c = (gt["labels"] == c)
            gt_boxes_c = gt["boxes_xyxy"][gt_mask_c]
            gt_crowd_c = gt["iscrowd"][gt_mask_c] if "iscrowd" in gt else np.zeros(len(gt_boxes_c), dtype=bool)
            non_crowd_boxes = gt_boxes_c[~gt_crowd_c]
            crowd_boxes = gt_boxes_c[gt_crowd_c]
            n_gt_vec[idx] = len(non_crowd_boxes)

            pr = preds[idx]
            pr_mask_c = (pr["labels"] == c)
            pr_boxes_c = pr["boxes"][pr_mask_c]
            pr_scores_c = pr["scores"][pr_mask_c]
            if len(pr_scores_c) == 0:
                scores_list.append(np.zeros((0,), dtype=np.float64))
                matches_list.append(np.zeros((0, len(iou_thresholds)), dtype=np.int8))
                continue

            order = np.argsort(-pr_scores_c)
            pr_boxes_c = pr_boxes_c[order]
            pr_scores_c = pr_scores_c[order]

            ious_reg = compute_iou_matrix(pr_boxes_c, non_crowd_boxes) if len(non_crowd_boxes) > 0 else np.zeros((len(pr_boxes_c), 0))
            ious_crowd = compute_iou_matrix(pr_boxes_c, crowd_boxes) if len(crowd_boxes) > 0 else np.zeros((len(pr_boxes_c), 0))
            matched_status = np.zeros((len(pr_scores_c), len(iou_thresholds)), dtype=np.int8)

            for t_idx, iou_th in enumerate(iou_thresholds):
                used_gt = np.zeros(len(non_crowd_boxes), dtype=bool)
                for p_i in range(len(pr_scores_c)):
                    best_iou = 0.0
                    best_g = -1
                    if len(non_crowd_boxes) > 0:
                        cand_ious = np.where(~used_gt, ious_reg[p_i], -1.0)
                        best_g = int(np.argmax(cand_ious))
                        best_iou = float(cand_ious[best_g])
                    if best_iou >= iou_th and best_g >= 0:
                        matched_status[p_i, t_idx] = 1
                        used_gt[best_g] = True
                    elif len(crowd_boxes) > 0 and float(np.max(ious_crowd[p_i])) >= iou_th:
                        matched_status[p_i, t_idx] = -1

            scores_list.append(pr_scores_c.astype(np.float64))
            matches_list.append(matched_status)

        per_class.append({
            "n_gt_vec": n_gt_vec,
            "scores": scores_list,
            "matches": matches_list,
            "has_any_det": bool(any(len(s) > 0 for s in scores_list)),
        })

    return {
        "n_img": n_img,
        "num_classes": num_classes,
        "iou_thresholds": iou_thresholds,
        "per_class": per_class,
    }


def evaluate_coco_from_match_cache(
    cache: Dict[str, Any],
    indices: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Tính nhanh COCO mAP@50, mAP@75 và mAP@50:95 từ cache đã precompute."""
    idx_arr = np.arange(cache["n_img"], dtype=np.int64) if indices is None else np.asarray(indices, dtype=np.int64)
    n_thr = len(cache["iou_thresholds"])
    num_classes = cache["num_classes"]
    recall_levels = np.linspace(0.0, 1.0, 101)
    class_ap_mat = np.full((num_classes, n_thr), np.nan, dtype=np.float64)

    for c in range(num_classes):
        cd = cache["per_class"][c]
        n_gt_total = int(np.sum(cd["n_gt_vec"][idx_arr]))
        if n_gt_total == 0:
            continue
        if not cd["has_any_det"]:
            class_ap_mat[c] = 0.0
            continue

        sc_parts = [cd["scores"][i] for i in idx_arr if len(cd["scores"][i]) > 0]
        if not sc_parts:
            class_ap_mat[c] = 0.0
            continue
        mt_parts = [cd["matches"][i] for i in idx_arr if len(cd["scores"][i]) > 0]

        score_arr = np.concatenate(sc_parts, axis=0)
        match_mat = np.concatenate(mt_parts, axis=0)
        global_order = np.argsort(-score_arr, kind="mergesort")
        match_mat = match_mat[global_order]

        aps = np.zeros(n_thr, dtype=np.float64)
        for t_idx in range(n_thr):
            col = match_mat[:, t_idx]
            col_v = col[col != -1]
            if len(col_v) == 0:
                continue
            tp_cum = np.cumsum((col_v == 1).astype(np.float64))
            fp_cum = np.cumsum((col_v == 0).astype(np.float64))
            precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
            recalls = tp_cum / float(n_gt_total)
            prec_env = np.maximum.accumulate(precisions[::-1])[::-1]
            pos = np.searchsorted(recalls, recall_levels, side="left")
            valid_r = pos < len(recalls)
            prec_interp = np.zeros(len(recall_levels), dtype=np.float64)
            if np.any(valid_r):
                prec_interp[valid_r] = prec_env[pos[valid_r]]
            aps[t_idx] = float(np.mean(prec_interp))
        class_ap_mat[c] = aps

    valid_classes = ~np.isnan(class_ap_mat[:, 0])
    n_active_classes = int(np.sum(valid_classes))
    if n_active_classes == 0:
        return {"mAP@50": 0.0, "mAP@75": 0.0, "mAP@50:95": 0.0, "active_classes": 0}

    ap_per_iou = np.mean(class_ap_mat[valid_classes], axis=0) * 100.0
    return {
        "mAP@50": float(ap_per_iou[0]),
        "mAP@75": float(ap_per_iou[5]),
        "mAP@50:95": float(np.mean(ap_per_iou)),
        "active_classes": n_active_classes,
    }


def compute_per_image_coco_aps(
    preds: Sequence[Dict[str, np.ndarray]],
    gt_list: Sequence[Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tính AP@50 và AP@50:95 (thang 0..100%) trên từng ảnh độc lập cho các lớp GT có trong ảnh.
    Trả về (ap50_vec, ap50_95_vec); ảnh không có non-crowd GT có giá trị NaN.
    """
    iou_thresholds = np.linspace(0.50, 0.95, 10)
    ap50 = np.full(len(gt_list), np.nan, dtype=np.float64)
    ap50_95 = np.full(len(gt_list), np.nan, dtype=np.float64)

    for i, gt in enumerate(gt_list):
        crowd = gt["iscrowd"] if "iscrowd" in gt else np.zeros(len(gt["labels"]), dtype=bool)
        reg_labels = gt["labels"][~crowd]
        present_classes = np.unique(reg_labels)
        if len(present_classes) == 0:
            continue

        aps_list = []
        for c in present_classes:
            ap_vec = _evaluate_class_ap_for_subset(
                preds=preds,
                gt_list=gt_list,
                indices=[i],
                class_id=int(c),
                iou_thresholds=iou_thresholds,
            )
            if not np.isnan(ap_vec[0]):
                aps_list.append(ap_vec)

        if aps_list:
            mat = np.stack(aps_list, axis=0) * 100.0
            ap50[i] = float(np.mean(mat[:, 0]))
            ap50_95[i] = float(np.mean(mat))

    return ap50, ap50_95


# -----------------------------------------------------------------------------
# Re-export các thuật toán thống kê suy luận từ package utils.stats
# nhằm đảm bảo tuân thủ nguyên tắc Single Responsibility (SRP) và giữ nguyên
# 100% tính tương thích ngược cho các module đang import từ coco_gt.
# -----------------------------------------------------------------------------
from paper_1_aqsm_mechanism.utils.stats import (
    bootstrap_paired_diff,
    repeated_measures_2x2_anova,
)

