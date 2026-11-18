# -*- coding: utf-8 -*-
"""
Module: edge_sec_pro.core.coco_gt
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

try:
    from paper_2_phantom_fabrication.core.config import (
        COCO_VAL_DIR,
        DATA_DIR,
        IMG_SIZE,
        N_CALIBRATION_SAMPLES,
        N_TEST_CLEAN_SAMPLES,
    )
except ImportError:
    from paper_2_phantom_fabrication.core.config import (
        COCO_VAL_DIR,
        DATA_DIR,
        IMG_SIZE,
        N_CALIBRATION_SAMPLES,
        N_TEST_CLEAN_SAMPLES,
    )

CORE_DIR: Path = Path(__file__).resolve().parent
PACKAGE_DIR: Path = CORE_DIR.parent
PROJECT_ROOT: Path = PACKAGE_DIR.parent

COCO_ANN_PATH: Path = DATA_DIR / "coco5k" / "annotations" / "instances_val2017.json"

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


def discover_coco_annotations(image_folder: Union[str, Path], custom_ann: Optional[Union[str, Path]] = None) -> Path:
    """Tự động tìm kiếm file chú thích instances_val2017.json."""
    if custom_ann:
        p = Path(custom_ann).resolve()
        if p.exists():
            return p

    img_folder = Path(image_folder).resolve()
    potential_paths = [
        img_folder / "annotations" / "instances_val2017.json",
        img_folder.parent / "annotations" / "instances_val2017.json",
        img_folder.parent.parent / "annotations" / "instances_val2017.json",
        PROJECT_ROOT / "coco5k" / "annotations" / "instances_val2017.json",
        PROJECT_ROOT / "edge_sec_pro" / "data" / "coco5k" / "annotations" / "instances_val2017.json",
        PROJECT_ROOT / "data" / "coco5k" / "annotations" / "instances_val2017.json",
        PROJECT_ROOT / "paper_2_phantom_fabrication" / "data" / "coco5k" / "annotations" / "instances_val2017.json",
    ]
    for p in potential_paths:
        if p.exists():
            return p.resolve()

    # Thử tìm kiếm đệ quy trong thư mục gốc
    for candidate in PROJECT_ROOT.glob("**/instances_val2017.json"):
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Không tìm thấy file annotations `instances_val2017.json`. "
        f"Vui lòng truyền đường dẫn chính xác qua `--annotations <path>`."
    )


def load_flexible_coco_cohort(
    image_folder: Union[str, Path],
    annotations_path: Optional[Union[str, Path]] = None,
    n_samples: int = 0,
    seed: int = 42,
    img_size: int = 640,
    id2label: Optional[Dict[int, str]] = None,
    as_uint8: bool = True,
) -> Dict[str, Any]:
    """
    Nạp tập cohort ảnh COCO linh hoạt từ bất kỳ thư mục nào (từ vài ảnh đến 5,000 ảnh).
    - Nếu n_samples <= 0 hoặc None: Tự động chạy toàn bộ ảnh có trong thư mục (full dataset).
    - as_uint8=True: Lưu tensor dưới dạng uint8 trên CPU (tiết kiệm 75% RAM, chống OOM khi chạy 5,000 ảnh).
    """
    img_dir = Path(image_folder).resolve()
    if not img_dir.exists():
        raise FileNotFoundError(f"Thư mục ảnh không tồn tại: {img_dir}")

    ann_file = discover_coco_annotations(img_dir, annotations_path)

    with open(ann_file, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    cat_map = build_coco_category_mapping(coco_data["categories"], id2label=id2label)
    filename_to_meta = {str(im["file_name"]): im for im in coco_data["images"]}
    anns_by_img = {}
    for ann in coco_data["annotations"]:
        anns_by_img.setdefault(int(ann["image_id"]), []).append(ann)

    all_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.jpeg")) + list(img_dir.glob("*.png")))
    if not all_files:
        raise FileNotFoundError(f"Không tìm thấy ảnh nào trong thư mục: {img_dir}")

    # Lọc những ảnh có mặt trong file annotations
    valid_paths = [f for f in all_files if f.name in filename_to_meta]
    if not valid_paths:
        valid_paths = all_files  # fallback nếu tên file khớp dạng id.jpg

    # Nếu n_samples <= 0: Chạy toàn bộ ảnh có sẵn trong thư mục
    if n_samples is None or n_samples <= 0 or n_samples >= len(valid_paths):
        selected_paths = valid_paths
    else:
        rng = np.random.RandomState(seed)
        chosen_indices = rng.choice(len(valid_paths), size=n_samples, replace=False)
        selected_paths = [valid_paths[idx] for idx in chosen_indices]

    gt_list = []
    hf_labels = []
    image_ids = []
    tensors = []

    for fpath in selected_paths:
        meta = filename_to_meta.get(fpath.name)
        if meta is None:
            try:
                img_id = int(fpath.stem)
            except ValueError:
                img_id = len(image_ids) + 1
            orig_w, orig_h = float(img_size), float(img_size)
        else:
            img_id = int(meta["id"])
            orig_w = float(meta["width"])
            orig_h = float(meta["height"])

        image_ids.append(img_id)
        anns = anns_by_img.get(img_id, [])

        boxes_xyxy, boxes_cxcywh, labels, iscrowd = [], [], [], []
        for ann in anns:
            cid = int(ann["category_id"])
            if cid not in cat_map:
                continue
            x, y, w, h = ann["bbox"]
            if w <= 1e-3 or h <= 1e-3:
                continue
            cx = (x + w / 2.0) / orig_w
            cy = (y + h / 2.0) / orig_h
            nw = w / orig_w
            nh = h / orig_h
            boxes_cxcywh.append([cx, cy, nw, nh])

            x1 = x / orig_w
            y1 = y / orig_h
            x2 = (x + w) / orig_w
            y2 = (y + h) / orig_h
            boxes_xyxy.append([x1, y1, x2, y2])

            labels.append(cat_map[cid])
            iscrowd.append(int(ann.get("iscrowd", 0)))

        gt_list.append({
            "image_id": img_id,
            "orig_size": [orig_h, orig_w],
            "boxes_xyxy": np.array(boxes_xyxy, dtype=np.float32) if boxes_xyxy else np.zeros((0, 4), dtype=np.float32),
            "boxes_cxcywh": np.array(boxes_cxcywh, dtype=np.float32) if boxes_cxcywh else np.zeros((0, 4), dtype=np.float32),
            "labels": np.array(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64),
            "iscrowd": np.array(iscrowd, dtype=np.int64) if iscrowd else np.zeros((0,), dtype=np.int64),
            "path": fpath,
        })

        valid_mask = [ic == 0 for ic in iscrowd]
        hf_labels.append({
            "class_labels": torch.tensor([labels[i] for i, v in enumerate(valid_mask) if v], dtype=torch.int64),
            "boxes": torch.tensor([boxes_cxcywh[i] for i, v in enumerate(valid_mask) if v], dtype=torch.float32),
        })

        try:
            with Image.open(fpath) as pil_img:
                img_rgb = pil_img.convert("RGB")
                img_resized = img_rgb.resize((img_size, img_size), Image.BILINEAR)
                t_flt = TF.to_tensor(img_resized)  # float in [0.0, 1.0]
                if as_uint8:
                    tensors.append((t_flt * 255.0).to(torch.uint8))
                else:
                    tensors.append(t_flt)
        except Exception:
            if as_uint8:
                tensors.append(torch.zeros(3, img_size, img_size, dtype=torch.uint8))
            else:
                tensors.append(torch.zeros(3, img_size, img_size, dtype=torch.float32))

    stacked_tensors = torch.stack(tensors, dim=0)

    return {
        "tensors": stacked_tensors,
        "paths": selected_paths,
        "image_ids": image_ids,
        "gt_list": gt_list,
        "hf_labels": hf_labels,
        "cat_id_to_idx": cat_map,
        "ann_file": str(ann_file),
    }


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
    data_dir = Path(coco_dir) if coco_dir is not None else COCO_VAL_DIR
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
    ann_file = Path(ann_path) if ann_path is not None else COCO_ANN_PATH
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


def bootstrap_paired_diff(
    vals_a: np.ndarray,
    vals_b: np.ndarray,
    n_boot: int = 5000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Tính trung bình chênh lệch ghép cặp (vals_a - vals_b), độ lệch chuẩn và 95% Bootstrap CI.
    """
    mask = (~np.isnan(vals_a)) & (~np.isnan(vals_b))
    diff = np.asarray(vals_a[mask] - vals_b[mask], dtype=np.float64)
    n = len(diff)
    if n == 0:
        return {"n": 0, "mean_diff": 0.0, "std_diff": 0.0, "ci_95": [0.0, 0.0]}

    mean_d = float(np.mean(diff))
    std_d = float(np.std(diff, ddof=1)) if n > 1 else 0.0
    rng = np.random.RandomState(seed)
    idx_mat = rng.randint(0, n, size=(n_boot, n))
    boot_means = diff[idx_mat].mean(axis=1)
    ci = np.percentile(boot_means, [2.5, 97.5]).tolist()
    _, p_wilcoxon = stats.wilcoxon(diff) if np.any(diff != 0) else (0.0, 1.0)
    return {
        "n": int(n),
        "mean_diff": mean_d,
        "std_diff": std_d,
        "ci_95": [float(ci[0]), float(ci[1])],
        "wilcoxon_p": float(p_wilcoxon),
    }


def repeated_measures_2x2_anova(
    y_00: np.ndarray,
    y_11: np.ndarray,
    y_10: np.ndarray,
    y_01: np.ndarray,
    n_boot: int = 10000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Thực hiện phân tích phương sai Repeated-Measures 2x2 chuẩn mực cho:
      - Factor M (Encoder Memory: 0=Clean, 1=Adv)
      - Factor Q (Query Prior: 0=Clean, 1=Adv)
    với Y_00=Arm A, Y_01=Arm D_full, Y_10=Arm C_full, Y_11=Arm B.
    Tách bạch rõ:
      1. Hiệu ứng biên ME(M), ME(Q), Inter(M x Q) kèm 95% Bootstrap CI.
      2. Tỷ lệ L1 trên giá trị tuyệt đối của hiệu ứng biên vs Tổng suy giảm Y_11 - Y_00.
      3. Bảng ANOVA chuẩn (SS, MS, F, p-value, eta^2_conditions, partial eta^2_p).
    """
    valid = (~np.isnan(y_00)) & (~np.isnan(y_11)) & (~np.isnan(y_10)) & (~np.isnan(y_01))
    ya = np.asarray(y_00[valid], dtype=np.float64)
    yb = np.asarray(y_11[valid], dtype=np.float64)
    yc = np.asarray(y_10[valid], dtype=np.float64)
    yd = np.asarray(y_01[valid], dtype=np.float64)
    n = len(ya)
    if n < 2:
        raise ValueError(f"Số mẫu hợp lệ không đủ cho ANOVA (N={n})")

    # Hiệu ứng biên theo từng ảnh
    vec_me_M = 0.5 * ((yc - ya) + (yb - yd))
    vec_me_Q = 0.5 * ((yd - ya) + (yb - yc))
    vec_inter = 0.5 * ((yb - yc) - (yd - ya))
    vec_total = yb - ya  # Đúng bằng vec_me_M + vec_me_Q

    mean_me_M = float(np.mean(vec_me_M))
    mean_me_Q = float(np.mean(vec_me_Q))
    mean_inter = float(np.mean(vec_inter))
    mean_total = float(np.mean(vec_total))

    rng = np.random.RandomState(seed)
    idx_mat = rng.randint(0, n, size=(n_boot, n))
    ci_M = np.percentile(vec_me_M[idx_mat].mean(axis=1), [2.5, 97.5]).tolist()
    ci_Q = np.percentile(vec_me_Q[idx_mat].mean(axis=1), [2.5, 97.5]).tolist()
    ci_Inter = np.percentile(vec_inter[idx_mat].mean(axis=1), [2.5, 97.5]).tolist()
    ci_Total = np.percentile(vec_total[idx_mat].mean(axis=1), [2.5, 97.5]).tolist()

    # Ma trận Y [N, 2, 2] với axis 1 = M (0,1), axis 2 = Q (0,1)
    y_mat = np.zeros((n, 2, 2), dtype=np.float64)
    y_mat[:, 0, 0] = ya
    y_mat[:, 0, 1] = yd
    y_mat[:, 1, 0] = yc
    y_mat[:, 1, 1] = yb

    grand_mean = np.mean(y_mat)
    cell_means = np.mean(y_mat, axis=0)
    m_means = np.mean(y_mat, axis=(0, 2))
    q_means = np.mean(y_mat, axis=(0, 1))

    ss_M = float(2 * n * np.sum((m_means - grand_mean) ** 2))
    ss_Q = float(2 * n * np.sum((q_means - grand_mean) ** 2))
    ss_inter = float(n * np.sum((cell_means - m_means[:, None] - q_means[None, :] + grand_mean) ** 2))

    # Sai số nội chủ thể cho thiết kế repeated-measures 2x2
    subj_mean = np.mean(y_mat, axis=(1, 2), keepdims=True)
    ss_subjects = float(4 * np.sum((subj_mean - grand_mean) ** 2))

    m_subj = np.mean(y_mat, axis=2)  # [N, 2]
    ss_err_M = float(2 * np.sum((m_subj - m_means[None, :] - subj_mean[:, :, 0] + grand_mean) ** 2))

    q_subj = np.mean(y_mat, axis=1)  # [N, 2]
    ss_err_Q = float(2 * np.sum((q_subj - q_means[None, :] - subj_mean[:, 0, :] + grand_mean) ** 2))

    resid_inter = (
        y_mat
        - m_subj[:, :, None]
        - q_subj[:, None, :]
        - cell_means[None, :, :]
        + m_means[None, :, None]
        + q_means[None, None, :]
        + subj_mean
        - grand_mean
    )
    ss_err_inter = float(np.sum(resid_inter ** 2))
    ss_total_all = float(np.sum((y_mat - grand_mean) ** 2))

    df_eff = 1
    df_err = n - 1
    f_M = (ss_M / df_eff) / max(ss_err_M / df_err, 1e-12)
    f_Q = (ss_Q / df_eff) / max(ss_err_Q / df_err, 1e-12)
    f_inter = (ss_inter / df_eff) / max(ss_err_inter / df_err, 1e-12)

    p_M = float(1.0 - stats.f.cdf(f_M, df_eff, df_err))
    p_Q = float(1.0 - stats.f.cdf(f_Q, df_eff, df_err))
    p_inter = float(1.0 - stats.f.cdf(f_inter, df_eff, df_err))

    ss_cond = max(ss_M + ss_Q + ss_inter, 1e-12)
    l1_sum = max(abs(mean_me_M) + abs(mean_me_Q) + abs(mean_inter), 1e-12)
    signed_denom = mean_total if abs(mean_total) > 1e-12 else 1.0

    return {
        "N": int(n),
        "df": [df_eff, df_err],
        "marginal_effects": {
            "ME_M": mean_me_M,
            "ME_M_ci95": [float(ci_M[0]), float(ci_M[1])],
            "ME_Q": mean_me_Q,
            "ME_Q_ci95": [float(ci_Q[0]), float(ci_Q[1])],
            "Interaction_MxQ": mean_inter,
            "Interaction_ci95": [float(ci_Inter[0]), float(ci_Inter[1])],
            "Total_Y11_minus_Y00": mean_total,
            "Total_ci95": [float(ci_Total[0]), float(ci_Total[1])],
        },
        "attribution_shares": {
            "signed_fraction_of_total_drop_pct": {
                "Memory_M": float(mean_me_M / signed_denom * 100.0),
                "Query_Q": float(mean_me_Q / signed_denom * 100.0),
                "note": "Y_11 - Y_00 = ME_M + ME_Q (tổng đúng 100%)",
            },
            "l1_marginal_share_pct": {
                "Memory_M": float(abs(mean_me_M) / l1_sum * 100.0),
                "Query_Q": float(abs(mean_me_Q) / l1_sum * 100.0),
                "Interaction_MxQ": float(abs(mean_inter) / l1_sum * 100.0),
                "note": "|ME| / (|ME_M| + |ME_Q| + |Inter|)",
            },
            "anova_condition_ss_share_pct": {
                "Memory_M": float(ss_M / ss_cond * 100.0),
                "Query_Q": float(ss_Q / ss_cond * 100.0),
                "Interaction_MxQ": float(ss_inter / ss_cond * 100.0),
                "note": "SS_factor / (SS_M + SS_Q + SS_inter)",
            },
            "partial_eta_squared": {
                "Memory_M": float(ss_M / max(ss_M + ss_err_M, 1e-12)),
                "Query_Q": float(ss_Q / max(ss_Q + ss_err_Q, 1e-12)),
                "Interaction_MxQ": float(ss_inter / max(ss_inter + ss_err_inter, 1e-12)),
            },
        },
        "anova_table": {
            "SS_subjects": ss_subjects,
            "SS_total": ss_total_all,
            "Memory_M": {"SS": ss_M, "SS_err": ss_err_M, "F": float(f_M), "p": p_M},
            "Query_Q": {"SS": ss_Q, "SS_err": ss_err_Q, "F": float(f_Q), "p": p_Q},
            "Interaction_MxQ": {"SS": ss_inter, "SS_err": ss_err_inter, "F": float(f_inter), "p": p_inter},
        },
    }
