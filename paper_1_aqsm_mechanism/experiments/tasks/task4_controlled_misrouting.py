# -*- coding: utf-8 -*-
"""
Module: paper_1_aqsm_mechanism.experiments.tasks.task4_controlled_misrouting
Mục đích: Thực nghiệm kiểm chứng tính đặc hiệu không gian (Controlled Spatial Misrouting)
trên toàn bộ tập ảnh COCO val2017 -> Phục vụ Table 8 và Paired TOST / Per-Query L2 Verification.

Bản chất khoa học:
  Giữ nguyên 100% Memory sạch M_clean và Content Queries Q_clean ở trạng thái hoàn hảo;
  chỉ can thiệp vào tọa độ bounding box tham chiếu ban đầu B_ref nạp vào Deformable Decoder:
    1. Clean Reference Baseline: B_clean chuẩn (L2 = 0.0000).
    2. Random Reassignment (mặc định 5 seeds): Hoán vị ngẫu nhiên các boxes sạch.
    3. Random Shift (Matched Center & Size L2 ở cấp từng query): Dịch chuyển đẳng hướng
       khớp đồng thời Center L2 và Size L2 ở cấp từng query trong [0, 1]^4 kèm log sai số max/p99.
    4. Decoupled Random Shift: Kết hợp Random Shift và Random Slot Permutation.
    5. AQSM Spatial Misrouting: Sử dụng chính xác tọa độ reference boxes bị lái lệch bởi AQSM.
"""

import gc
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from tqdm import tqdm

WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from paper_1_aqsm_mechanism.core.adv_cache import load_cached_adv_images
from paper_1_aqsm_mechanism.core.coco_gt import (
    bootstrap_paired_diff,
    compute_image_gt_union_area,
    compute_image_uniform_grid_fg_hit,
    compute_per_image_coco_aps,
    evaluate_coco_predictions,
    postprocess_rtdetr_predictions,
)
from paper_1_aqsm_mechanism.core.dataset_loader import FlexibleCOCODataset
from paper_1_aqsm_mechanism.core.detr_wrapper import RTDETRModelWrapper
from paper_1_aqsm_mechanism.core.mechanism_analyzer import (
    _box_iou_np,
    _cxcywh_to_xyxy_np,
    compute_directional_misrouting_metrics,
    compute_spatial_clustering_metrics,
    extract_encoder_components,
    run_decoder_from_components,
)
from paper_1_aqsm_mechanism.experiments.audit_logger import ExperimentAuditLogger
from paper_1_aqsm_mechanism.experiments.config import (
    DEFAULT_MULTI_SEEDS,
    MODEL_NAME,
    N_BOOTSTRAP,
    RANDOM_SEED,
    configure_cpu_threads,
    detect_hardware_environment,
    get_device,
    get_raw_csv_dir,
    get_summary_json_dir,
    resolve_model_name,
    update_experiment_paths,
)
from paper_1_aqsm_mechanism.utils.cli import create_base_parser
from paper_1_aqsm_mechanism.utils.io_helpers import save_json, write_csv_rows
from paper_1_aqsm_mechanism.utils.stats import evaluate_multiseed_statistical_evidence


def compute_ref_fg_hit(ref_pts: np.ndarray, gt_boxes_xyxy: np.ndarray) -> float:
    """Đo tỷ lệ reference points [300, 2] (hoặc cx, cy) rơi vào vùng GT box."""
    if len(gt_boxes_xyxy) == 0:
        return 0.0
    cx = ref_pts[:, 0]
    cy = ref_pts[:, 1]
    in_x = (cx[:, None] >= gt_boxes_xyxy[None, :, 0]) & (cx[:, None] <= gt_boxes_xyxy[None, :, 2])
    in_y = (cy[:, None] >= gt_boxes_xyxy[None, :, 1]) & (cy[:, None] <= gt_boxes_xyxy[None, :, 3])
    return float(np.mean(np.any(in_x & in_y, axis=1)))


def compute_object_recall(
    preds: Dict[str, Any],
    gt_boxes_xyxy: np.ndarray,
    iou_thresh: float = 0.5,
    score_thresh: float = 0.35,
) -> float:
    """Tỷ lệ GT boxes được phát hiện thành công."""
    if len(gt_boxes_xyxy) == 0:
        return 0.0
    scores = preds["scores"]
    boxes = preds["boxes"]
    active = scores >= score_thresh
    act_boxes = boxes[active]
    if len(act_boxes) == 0:
        return 0.0

    ious = _box_iou_np(gt_boxes_xyxy, act_boxes)
    max_ious = np.max(ious, axis=1)
    return float(np.mean(max_ious >= iou_thresh))


def _sample_exact_2d_with_diagnostics(
    p: np.ndarray,
    target_d: float,
    rng: np.random.RandomState,
    max_tries: int = 50,
    box_lo: float = 1e-5,
    box_hi: float = 1.0 - 1e-5,
) -> Tuple[np.ndarray, float, bool]:
    """
    Lấy mẫu điểm ngẫu nhiên 2D q trong [box_lo, box_hi]^2 sao cho ||q - p||_2 khớp chính xác target_d.
    Vì [box_lo, box_hi]^2 lồi và tồn tại ít nhất một góc đạt khoảng cách cực đại >= target_d
    khi điểm gốc trong [0, 1]^2, tìm cung góc hợp lệ trước khi fallback theo dây cung về góc xa nhất.
    Trả về (q, |realized_dist - target_d|, used_fallback).
    """
    p_safe = np.clip(np.asarray(p, dtype=np.float64), box_lo, box_hi)
    if target_d <= 1e-7:
        return p_safe.copy(), 0.0, False

    # Bước 1: Thử lấy mẫu ngẫu nhiên thuần trong toàn vòng tròn [0, 2*pi)
    thetas = rng.uniform(0.0, 2.0 * np.pi, size=max_tries)
    cands = p_safe[None, :] + target_d * np.stack([np.cos(thetas), np.sin(thetas)], axis=1)
    valid_mask = np.all((cands >= box_lo) & (cands <= box_hi), axis=1)
    if np.any(valid_mask):
        q = cands[int(np.flatnonzero(valid_mask)[0])]
        err = abs(float(np.linalg.norm(q - p_safe)) - target_d)
        return q, err, False

    # Bước 2: Quét lưới góc phân tầng 360 độ với pha ngẫu nhiên trên cung hợp lệ
    grid_thetas = rng.uniform(0.0, 2.0 * np.pi / 360.0) + np.linspace(0.0, 2.0 * np.pi, 360, endpoint=False)
    grid_cands = p_safe[None, :] + target_d * np.stack([np.cos(grid_thetas), np.sin(grid_thetas)], axis=1)
    grid_valid = np.all((grid_cands >= box_lo) & (grid_cands <= box_hi), axis=1)
    if np.any(grid_valid):
        valid_idx = np.flatnonzero(grid_valid)
        chosen = int(rng.choice(valid_idx))
        q = grid_cands[chosen]
        err = abs(float(np.linalg.norm(q - p_safe)) - target_d)
        return q, err, False

    # Bước 3: Fallback theo dây cung hướng tới góc hợp lệ có bán kính chứa được target_d
    corners = np.array(
        [[box_lo, box_lo], [box_lo, box_hi], [box_hi, box_lo], [box_hi, box_hi]],
        dtype=np.float64,
    )
    corner_vecs = corners - p_safe[None, :]
    corner_dists = np.linalg.norm(corner_vecs, axis=1)
    viable = np.flatnonzero(corner_dists >= target_d - 1e-6)
    c_idx = int(rng.choice(viable)) if len(viable) > 0 else int(np.argmax(corner_dists))
    c_dist = float(corner_dists[c_idx])
    if c_dist <= 1e-9:
        return p_safe.copy(), abs(target_d), True
    step = min(target_d, c_dist)
    q = np.clip(p_safe + step * (corner_vecs[c_idx] / c_dist), box_lo, box_hi)
    err = abs(float(np.linalg.norm(q - p_safe)) - target_d)
    return q, err, True


def sample_exact_matched_shift_4d_with_diagnostics(
    clean_box: np.ndarray,
    aqsm_box: np.ndarray,
    rng: np.random.RandomState,
    scale_factor: float = 1.0,
    max_tries: int = 50,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Sinh tọa độ box dịch chuyển ngẫu nhiên [cx, cy, w, h] khớp đồng thời Center L2
    và Size L2 của AQSM ở cấp từng query, kèm thông tin sai số thực tế và cờ fallback.
    """
    d_center = float(np.linalg.norm(aqsm_box[:2] - clean_box[:2])) * scale_factor
    d_size = float(np.linalg.norm(aqsm_box[2:] - clean_box[2:])) * scale_factor

    center_shifted, err_c, fb_c = _sample_exact_2d_with_diagnostics(clean_box[:2], d_center, rng, max_tries=max_tries)
    size_shifted, err_s, fb_s = _sample_exact_2d_with_diagnostics(clean_box[2:], d_size, rng, max_tries=max_tries)
    diag = {
        "center_l2_err": err_c,
        "size_l2_err": err_s,
        "fallback_center": fb_c,
        "fallback_size": fb_s,
    }
    return np.concatenate([center_shifted, size_shifted]), diag


def sample_exact_matched_shift_4d(
    clean_box: np.ndarray,
    aqsm_box: np.ndarray,
    rng: np.random.RandomState,
    scale_factor: float = 1.0,
    max_tries: int = 50,
) -> np.ndarray:
    """
    Wrapper giữ tương thích giao diện cho các script gọi ngoài (trả về tensor 4D thuần).
    """
    shifted, _ = sample_exact_matched_shift_4d_with_diagnostics(
        clean_box=clean_box,
        aqsm_box=aqsm_box,
        rng=rng,
        scale_factor=scale_factor,
        max_tries=max_tries,
    )
    return shifted


def run_controlled_misrouting_evaluation(
    wrapper: RTDETRModelWrapper,
    dataset: Any,
    adv_imgs: Optional[torch.Tensor] = None,
    batch_size: int = 10,
    n_boot: int = N_BOOTSTRAP,
    cache_dir: Optional[Path] = None,
    audit_logger: Any = None,
    n_dose_samples: Optional[int] = None,
    seeds: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Thực hiện đánh giá toàn diện kiểm chứng Spatial Specificity và phản bác Tautology.
    """
    model = wrapper.model
    device = wrapper.device
    clean_imgs = dataset.tensors
    gt_list = dataset.gt_list
    image_ids = dataset.image_ids
    n_total = len(dataset)
    gt_boxes_batch = [gt.get("boxes_xyxy", np.zeros((0, 4), dtype=np.float32)) for gt in gt_list]

    if adv_imgs is None:
        adv_imgs, _ = load_cached_adv_images(
            attack="aqsm",
            eps_num=16,
            n_eval=n_total,
            clean_imgs=clean_imgs,
            expected_image_ids=image_ids,
            cache_dir=cache_dir,
            wrapper=wrapper,
        )

    seeds = list(seeds) if seeds else list(DEFAULT_MULTI_SEEDS)

    conditions = ["Clean_Baseline", "AQSM_Misrouting"]
    for s in seeds:
        conditions.append(f"Random_Reassign_seed{s}")
    for s in seeds:
        conditions.append(f"Random_Shift_seed{s}")
    for s in seeds:
        conditions.append(f"Decoupled_Random_Shift_seed{s}")

    preds_by_cond: Dict[str, List[Dict[str, Any]]] = {c: [] for c in conditions}
    disp_metrics_by_cond: Dict[str, List[Dict[str, float]]] = {c: [] for c in conditions}
    # Lưu sai số khớp cấp từng query (per-query center/size L2 residual) cho Random Shift & Decoupled Random Shift
    per_query_match_diags: Dict[str, Dict[str, List[float]]] = {
        c: {
            "center_err": [],
            "size_err": [],
            "fb_center": [],
            "fb_size": [],
        }
        for c in conditions
        if "Shift" in c
    }

    n_batches = int(np.ceil(n_total / batch_size))
    n_queries_model: Optional[int] = None

    print("\n=======================================================")
    print(f"--> [Controlled Misrouting] Khởi chạy can thiệp B_ref trên N={n_total} ảnh...")
    print("=======================================================")

    for b_idx in tqdm(range(n_batches), desc="Controlled Misrouting"):
        start_i = b_idx * batch_size
        end_i = min(start_i + batch_size, n_total)
        cur_bsz = end_i - start_i

        b_clean = clean_imgs[start_i:end_i].to(device)
        b_adv = adv_imgs[start_i:end_i].to(device)

        with torch.no_grad():
            c_comp = extract_encoder_components(model, b_clean)
            a_comp = extract_encoder_components(model, b_adv)

        frozen_target = c_comp["target"]
        frozen_source = c_comp["source_flatten"]
        c_ref_sig = torch.sigmoid(c_comp["init_reference_points"])
        a_ref_sig = torch.sigmoid(a_comp["init_reference_points"])

        c_ref_np = c_ref_sig.detach().cpu().numpy()
        a_ref_np = a_ref_sig.detach().cpu().numpy()
        n_queries_batch = int(c_ref_np.shape[1])
        if n_queries_model is None:
            n_queries_model = n_queries_batch

        b_ref_tensors: Dict[str, torch.Tensor] = {}
        b_ref_tensors["Clean_Baseline"] = c_comp["init_reference_points"]
        b_ref_tensors["AQSM_Misrouting"] = a_comp["init_reference_points"]

        perms_by_seed: Dict[int, np.ndarray] = {}
        for s in seeds:
            rng = np.random.RandomState(s)
            perm = rng.permutation(n_queries_batch)
            perms_by_seed[s] = perm
            perm_boxes_np = c_ref_np[:, perm, :]
            perm_boxes_safe = np.clip(perm_boxes_np, 1e-5, 1.0 - 1e-5)
            perm_logits = torch.from_numpy(np.log(perm_boxes_safe / (1.0 - perm_boxes_safe))).to(device).float()
            b_ref_tensors[f"Random_Reassign_seed{s}"] = perm_logits

        # Lưu thêm post-sigmoid residual cho từng ảnh trong lô của các nhánh Shift
        batch_shift_q_diags: Dict[str, Dict[str, List[float]]] = {}
        for s in seeds:
            rng = np.random.RandomState(s)
            shifted_boxes_np = np.zeros_like(c_ref_np)
            c_errs: List[float] = []
            s_errs: List[float] = []
            fb_cs: List[float] = []
            fb_ss: List[float] = []
            for b in range(cur_bsz):
                img_c_errs: List[float] = []
                img_s_errs: List[float] = []
                for i in range(n_queries_batch):
                    box_s, q_diag = sample_exact_matched_shift_4d_with_diagnostics(
                        c_ref_np[b, i], a_ref_np[b, i], rng, scale_factor=1.0
                    )
                    shifted_boxes_np[b, i] = box_s
                    c_errs.append(q_diag["center_l2_err"])
                    s_errs.append(q_diag["size_l2_err"])
                    fb_cs.append(1.0 if q_diag["fallback_center"] else 0.0)
                    fb_ss.append(1.0 if q_diag["fallback_size"] else 0.0)
                    img_c_errs.append(q_diag["center_l2_err"])
                    img_s_errs.append(q_diag["size_l2_err"])
                batch_shift_q_diags.setdefault(f"Random_Shift_seed{s}_max_c", []).append(float(np.max(img_c_errs)))
                batch_shift_q_diags.setdefault(f"Random_Shift_seed{s}_max_s", []).append(float(np.max(img_s_errs)))

            shifted_boxes_safe = np.clip(shifted_boxes_np, 1e-5, 1.0 - 1e-5)
            shift_logits = torch.from_numpy(np.log(shifted_boxes_safe / (1.0 - shifted_boxes_safe))).to(device).float()
            shift_cond = f"Random_Shift_seed{s}"
            decoupled_cond = f"Decoupled_Random_Shift_seed{s}"
            b_ref_tensors[shift_cond] = shift_logits

            # Decoupled Random Shift: kết hợp đồng thời Random Shift và Permutation across slots
            decoupled_boxes_safe = shifted_boxes_safe[:, perms_by_seed[s], :]
            decoupled_logits = torch.from_numpy(
                np.log(decoupled_boxes_safe / (1.0 - decoupled_boxes_safe))
            ).to(device).float()
            b_ref_tensors[decoupled_cond] = decoupled_logits

            per_query_match_diags[shift_cond]["center_err"].extend(c_errs)
            per_query_match_diags[shift_cond]["size_err"].extend(s_errs)
            per_query_match_diags[shift_cond]["fb_center"].extend(fb_cs)
            per_query_match_diags[shift_cond]["fb_size"].extend(fb_ss)

            per_query_match_diags[decoupled_cond]["center_err"].extend(c_errs)
            per_query_match_diags[decoupled_cond]["size_err"].extend(s_errs)
            per_query_match_diags[decoupled_cond]["fb_center"].extend(fb_cs)
            per_query_match_diags[decoupled_cond]["fb_size"].extend(fb_ss)

        for cond, ref_tensor in b_ref_tensors.items():
            dec_out = run_decoder_from_components(
                model=model,
                target=frozen_target,
                source_flatten=frozen_source,
                init_reference_points=ref_tensor,
                spatial_shapes=c_comp["spatial_shapes"],
                spatial_shapes_list=c_comp["spatial_shapes_list"],
                level_start_index=c_comp["level_start_index"],
            )
            batch_preds = postprocess_rtdetr_predictions(
                dec_out["logits"], dec_out["pred_boxes"], mode="hf_multilabel", max_dets=100
            )
            preds_by_cond[cond].extend(batch_preds)

            ref_sig_np = torch.sigmoid(ref_tensor).detach().cpu().numpy()
            diff_full = ref_sig_np - c_ref_np
            l2_full_img = np.mean(np.sqrt(np.sum(diff_full ** 2, axis=-1)), axis=1)
            diff_center = diff_full[:, :, :2]
            l2_center_img = np.mean(np.sqrt(np.sum(diff_center ** 2, axis=-1)), axis=1)
            dcx_img = np.mean(np.abs(diff_center[:, :, 0]), axis=1)
            dcy_img = np.mean(np.abs(diff_center[:, :, 1]), axis=1)
            diff_size = diff_full[:, :, 2:]
            l2_size_img = np.mean(np.sqrt(np.sum(diff_size ** 2, axis=-1)), axis=1)
            dw_img = np.mean(np.abs(diff_size[:, :, 0]), axis=1)
            dh_img = np.mean(np.abs(diff_size[:, :, 1]), axis=1)

            for b in range(cur_bsz):
                img_idx = start_i + b
                gt_boxes = gt_boxes_batch[img_idx]
                fg_hit = compute_ref_fg_hit(ref_sig_np[b, :, :2], gt_boxes)
                recall = compute_object_recall(batch_preds[b], gt_boxes)

                # Đo khoảng cách tập hợp bất biến hoán vị (Hungarian, Chamfer) & GT Object Coverage
                cost_mat = cdist(c_ref_np[b, :, :2], ref_sig_np[b, :, :2])
                r_ind, c_ind = linear_sum_assignment(cost_mat)
                hungarian_d = float(np.mean(cost_mat[r_ind, c_ind]))
                chamfer_d = float(0.5 * (np.mean(np.min(cost_mat, axis=1)) + np.mean(np.min(cost_mat, axis=0))))

                if len(gt_boxes) > 0:
                    cx_b, cy_b = ref_sig_np[b, :, 0], ref_sig_np[b, :, 1]
                    in_x = (cx_b[:, None] >= gt_boxes[None, :, 0]) & (cx_b[:, None] <= gt_boxes[None, :, 2])
                    in_y = (cy_b[:, None] >= gt_boxes[None, :, 1]) & (cy_b[:, None] <= gt_boxes[None, :, 3])
                    gt_cov_center = float(np.mean(np.any(in_x & in_y, axis=0)))
                    ref_xyxy = _cxcywh_to_xyxy_np(ref_sig_np[b])
                    ious = _box_iou_np(ref_xyxy, gt_boxes)
                    gt_cov_iou30 = float(np.mean(ious.max(axis=0) >= 0.30))
                else:
                    gt_cov_center = float("nan")
                    gt_cov_iou30 = float("nan")

                # Đo đạc gom cụm không gian ANND, độ phân tán spatial variance và hướng dịch chuyển
                clust_metrics = compute_spatial_clustering_metrics(ref_sig_np[b, :, :2])
                dir_metrics = compute_directional_misrouting_metrics(c_ref_np[b, :, :2], ref_sig_np[b, :, :2])

                rec_dict: Dict[str, float] = {
                    "l2_full": float(l2_full_img[b]),
                    "l2_center": float(l2_center_img[b]),
                    "delta_cx": float(dcx_img[b]),
                    "delta_cy": float(dcy_img[b]),
                    "l2_size": float(l2_size_img[b]),
                    "delta_w": float(dw_img[b]),
                    "delta_h": float(dh_img[b]),
                    "fg_hit": fg_hit,
                    "recall": recall,
                    "hungarian_dist": hungarian_d,
                    "chamfer_dist": chamfer_d,
                    "gt_cov_center": gt_cov_center,
                    "gt_cov_iou30": gt_cov_iou30,
                    "annd": clust_metrics["annd"],
                    "spatial_variance": clust_metrics["spatial_variance"],
                    "spatial_std_x": clust_metrics["spatial_std_x"],
                    "spatial_std_y": clust_metrics["spatial_std_y"],
                    "mean_shift_vector_norm": dir_metrics["mean_shift_vector_norm"],
                    "directional_resultant_length": dir_metrics["directional_resultant_length"],
                    "radial_outward_bias": dir_metrics["radial_outward_bias"],
                }
                if cond.startswith("Random_Shift_seed"):
                    rec_dict["max_center_l2_err"] = batch_shift_q_diags[f"{cond}_max_c"][b]
                    rec_dict["max_size_l2_err"] = batch_shift_q_diags[f"{cond}_max_s"][b]
                disp_metrics_by_cond[cond].append(rec_dict)

    print(">>> Đang đánh giá COCO mAP toàn cục cho tất cả các nhánh...")
    map_results: Dict[str, Dict[str, Any]] = {}
    per_image_aps: Dict[str, np.ndarray] = {}

    for cond in conditions:
        eval_m = evaluate_coco_predictions(preds_by_cond[cond], gt_list)
        ap50_vec, _ = compute_per_image_coco_aps(preds_by_cond[cond], gt_list)
        recs = disp_metrics_by_cond[cond]
        cov_c_arr = [m["gt_cov_center"] for m in recs if not np.isnan(m["gt_cov_center"])]
        cov_i_arr = [m["gt_cov_iou30"] for m in recs if not np.isnan(m["gt_cov_iou30"])]
        cond_entry: Dict[str, Any] = {
            "mAP@50": float(eval_m["mAP@50"]),
            "mAP@75": float(eval_m["mAP@75"]),
            "mAP@50:95": float(eval_m["mAP@50:95"]),
            "recall": float(np.mean([m["recall"] for m in recs]) * 100.0),
            "fg_hit": float(np.mean([m["fg_hit"] for m in recs]) * 100.0),
            "fg_hit_std": float(np.std([m["fg_hit"] * 100.0 for m in recs], ddof=1)) if len(recs) > 1 else 0.0,
            "delta_l2": float(np.mean([m["l2_full"] for m in recs])),
            "delta_center_l2": float(np.mean([m["l2_center"] for m in recs])),
            "delta_size_l2": float(np.mean([m["l2_size"] for m in recs])),
            "hungarian_dist": float(np.mean([m["hungarian_dist"] for m in recs])),
            "chamfer_dist": float(np.mean([m["chamfer_dist"] for m in recs])),
            "gt_object_cov_pct": float(np.mean(cov_c_arr) * 100.0) if cov_c_arr else 0.0,
            "gt_object_cov_iou30_pct": float(np.mean(cov_i_arr) * 100.0) if cov_i_arr else 0.0,
            "annd": float(np.mean([m["annd"] for m in recs])),
            "spatial_variance": float(np.mean([m["spatial_variance"] for m in recs])),
            "mean_shift_vector_norm": float(np.mean([m["mean_shift_vector_norm"] for m in recs])),
            "directional_resultant_length": float(np.mean([m["directional_resultant_length"] for m in recs])),
            "radial_outward_bias": float(np.mean([m["radial_outward_bias"] for m in recs])),
        }
        if cond in per_query_match_diags and per_query_match_diags[cond]["center_err"]:
            c_arr = np.asarray(per_query_match_diags[cond]["center_err"], dtype=np.float64)
            s_arr = np.asarray(per_query_match_diags[cond]["size_err"], dtype=np.float64)
            cond_entry["per_query_matching_diagnostics"] = {
                "n_queries_checked": int(len(c_arr)),
                "center_l2_err_mean": float(np.mean(c_arr)),
                "center_l2_err_p99": float(np.percentile(c_arr, 99.0)),
                "center_l2_err_max": float(np.max(c_arr)),
                "size_l2_err_mean": float(np.mean(s_arr)),
                "size_l2_err_p99": float(np.percentile(s_arr, 99.0)),
                "size_l2_err_max": float(np.max(s_arr)),
                "clip_fallback_rate_center": float(np.mean(per_query_match_diags[cond]["fb_center"])),
                "clip_fallback_rate_size": float(np.mean(per_query_match_diags[cond]["fb_size"])),
            }
        map_results[cond] = cond_entry
        per_image_aps[cond] = ap50_vec

    def _agg_seed_list(prefix: str, key: str) -> List[float]:
        return [float(map_results[f"{prefix}_seed{s}"][key]) for s in seeds]

    reassign_map50 = _agg_seed_list("Random_Reassign", "mAP@50")
    reassign_map95 = _agg_seed_list("Random_Reassign", "mAP@50:95")
    reassign_recall = _agg_seed_list("Random_Reassign", "recall")
    reassign_fghit = _agg_seed_list("Random_Reassign", "fg_hit")

    shift_map50 = _agg_seed_list("Random_Shift", "mAP@50")
    shift_map95 = _agg_seed_list("Random_Shift", "mAP@50:95")
    shift_recall = _agg_seed_list("Random_Shift", "recall")
    shift_fghit = _agg_seed_list("Random_Shift", "fg_hit")
    shift_l2 = _agg_seed_list("Random_Shift", "delta_l2")
    shift_center_l2 = _agg_seed_list("Random_Shift", "delta_center_l2")
    shift_size_l2 = _agg_seed_list("Random_Shift", "delta_size_l2")

    decoupled_map50 = _agg_seed_list("Decoupled_Random_Shift", "mAP@50")
    decoupled_map95 = _agg_seed_list("Decoupled_Random_Shift", "mAP@50:95")
    decoupled_recall = _agg_seed_list("Decoupled_Random_Shift", "recall")
    decoupled_fghit = _agg_seed_list("Decoupled_Random_Shift", "fg_hit")

    clean_ap50 = per_image_aps["Clean_Baseline"]
    boot_ci_reassign = bootstrap_paired_diff(
        np.mean([per_image_aps[f"Random_Reassign_seed{s}"] for s in seeds], axis=0),
        clean_ap50,
        n_boot=n_boot,
    )
    boot_ci_shift = bootstrap_paired_diff(
        np.mean([per_image_aps[f"Random_Shift_seed{s}"] for s in seeds], axis=0),
        clean_ap50,
        n_boot=n_boot,
    )
    boot_ci_decoupled = bootstrap_paired_diff(
        np.mean([per_image_aps[f"Decoupled_Random_Shift_seed{s}"] for s in seeds], axis=0),
        clean_ap50,
        n_boot=n_boot,
    )
    boot_ci_aqsm = bootstrap_paired_diff(
        per_image_aps["AQSM_Misrouting"],
        clean_ap50,
        n_boot=n_boot,
    )

    rs_m50_mean = float(np.mean(shift_map50))
    drs_m50_mean = float(np.mean(decoupled_map50))
    aq_m50_val = float(map_results["AQSM_Misrouting"]["mAP@50"])
    clean_m50_val = float(map_results["Clean_Baseline"]["mAP@50"])
    clean_fg_vec = np.asarray(
        [m["fg_hit"] * 100.0 for m in disp_metrics_by_cond["Clean_Baseline"]], dtype=np.float64
    )
    clean_fg_val = float(map_results["Clean_Baseline"]["fg_hit"])

    def _build_cond_multiseed_evidence(prefix: str) -> Dict[str, Any]:
        seed_ap_map = {s: per_image_aps[f"{prefix}_seed{s}"] for s in seeds}
        seed_ap_scalar = {s: float(map_results[f"{prefix}_seed{s}"]["mAP@50"]) for s in seeds}
        seed_fg_map = {
            s: np.asarray(
                [m["fg_hit"] * 100.0 for m in disp_metrics_by_cond[f"{prefix}_seed{s}"]],
                dtype=np.float64,
            )
            for s in seeds
        }
        seed_fg_scalar = {s: float(map_results[f"{prefix}_seed{s}"]["fg_hit"]) for s in seeds}
        return {
            "ap50_vs_clean": evaluate_multiseed_statistical_evidence(
                baseline_per_image=clean_ap50,
                seed_per_image_map=seed_ap_map,
                seed_scalar_summary=seed_ap_scalar,
                clean_scalar_baseline=clean_m50_val,
                n_boot=n_boot,
                seed=42,
            ),
            "fg_hit_vs_clean": evaluate_multiseed_statistical_evidence(
                baseline_per_image=clean_fg_vec,
                seed_per_image_map=seed_fg_map,
                seed_scalar_summary=seed_fg_scalar,
                clean_scalar_baseline=clean_fg_val,
                n_boot=n_boot,
                seed=42,
            ),
        }

    ev_reassign = _build_cond_multiseed_evidence("Random_Reassign")
    ev_shift = _build_cond_multiseed_evidence("Random_Shift")
    ev_decoupled = _build_cond_multiseed_evidence("Decoupled_Random_Shift")

    def _tier1_fields(ev_dict: Dict[str, Any]) -> Dict[str, Any]:
        t1 = ev_dict["ap50_vs_clean"]["tier1_scalar_stability"] or {}
        return {
            "mAP@50_min": round(float(t1["min"]), 2) if t1.get("min") is not None else None,
            "mAP@50_max": round(float(t1["max"]), 2) if t1.get("max") is not None else None,
            "mAP@50_snr_seed": round(float(t1["snr_seed"]), 2) if t1.get("snr_seed") is not None else None,
        }

    summary_table8 = {
        "Clean_Baseline": {
            "mAP@50": round(map_results["Clean_Baseline"]["mAP@50"], 2),
            "mAP@50:95": round(map_results["Clean_Baseline"]["mAP@50:95"], 2),
            "recall_pct": round(map_results["Clean_Baseline"]["recall"], 2),
            "fg_hit_pct": round(map_results["Clean_Baseline"]["fg_hit"], 2),
            "delta_l2": 0.0000,
            "hungarian_dist": round(map_results["Clean_Baseline"]["hungarian_dist"], 4),
            "chamfer_dist": round(map_results["Clean_Baseline"]["chamfer_dist"], 4),
            "gt_object_cov_pct": round(map_results["Clean_Baseline"]["gt_object_cov_pct"], 2),
            "gt_object_cov_iou30_pct": round(map_results["Clean_Baseline"]["gt_object_cov_iou30_pct"], 4),
            "annd": round(map_results["Clean_Baseline"]["annd"], 4),
            "spatial_variance": round(map_results["Clean_Baseline"]["spatial_variance"], 4),
            "delta_map50_ci95": [0.0, 0.0],
        },
        "Random_Reassignment": {
            "mAP@50_mean": round(float(np.mean(reassign_map50)), 2),
            "mAP@50_std": round(float(np.std(reassign_map50, ddof=1)) if len(reassign_map50) > 1 else 0.0, 2),
            **_tier1_fields(ev_reassign),
            "mAP@50:95_mean": round(float(np.mean(reassign_map95)), 2),
            "mAP@50:95_std": round(float(np.std(reassign_map95, ddof=1)) if len(reassign_map95) > 1 else 0.0, 2),
            "recall_pct_mean": round(float(np.mean(reassign_recall)), 2),
            "recall_pct_std": round(float(np.std(reassign_recall, ddof=1)) if len(reassign_recall) > 1 else 0.0, 2),
            "fg_hit_pct_mean": round(float(np.mean(reassign_fghit)), 2),
            "delta_l2": round(float(np.mean(_agg_seed_list("Random_Reassign", "delta_l2"))), 4),
            "hungarian_dist": round(float(np.mean(_agg_seed_list("Random_Reassign", "hungarian_dist"))), 4),
            "chamfer_dist": round(float(np.mean(_agg_seed_list("Random_Reassign", "chamfer_dist"))), 4),
            "gt_object_cov_pct": round(float(np.mean(_agg_seed_list("Random_Reassign", "gt_object_cov_pct"))), 2),
            "gt_object_cov_iou30_pct": round(float(np.mean(_agg_seed_list("Random_Reassign", "gt_object_cov_iou30_pct"))), 4),
            "annd_mean": round(float(np.mean(_agg_seed_list("Random_Reassign", "annd"))), 4),
            "spatial_variance_mean": round(float(np.mean(_agg_seed_list("Random_Reassign", "spatial_variance"))), 4),
            "directional_resultant_length_mean": round(float(np.mean(_agg_seed_list("Random_Reassign", "directional_resultant_length"))), 4),
            "radial_outward_bias_mean": round(float(np.mean(_agg_seed_list("Random_Reassign", "radial_outward_bias"))), 4),
            "delta_map50_ci95": [round(boot_ci_reassign["ci_95"][0], 2), round(boot_ci_reassign["ci_95"][1], 2)],
            "wilcoxon_p": boot_ci_reassign["wilcoxon_p"],
            "multiseed_statistical_evidence": ev_reassign,
        },
        "Matched_L2_Random_Shift": {
            "mAP@50_mean": round(rs_m50_mean, 2),
            "mAP@50_std": round(float(np.std(shift_map50, ddof=1)) if len(shift_map50) > 1 else 0.0, 2),
            **_tier1_fields(ev_shift),
            "mAP@50:95_mean": round(float(np.mean(shift_map95)), 2),
            "mAP@50:95_std": round(float(np.std(shift_map95, ddof=1)) if len(shift_map95) > 1 else 0.0, 2),
            "recall_pct_mean": round(float(np.mean(shift_recall)), 2),
            "recall_pct_std": round(float(np.std(shift_recall, ddof=1)) if len(shift_recall) > 1 else 0.0, 2),
            "fg_hit_pct_mean": round(float(np.mean(shift_fghit)), 2),
            "delta_l2": round(float(np.mean(shift_l2)), 4),
            "delta_l2_std": round(float(np.std(shift_l2, ddof=1)) if len(shift_l2) > 1 else 0.0, 4),
            "delta_center_l2": round(float(np.mean(shift_center_l2)), 4),
            "delta_size_l2": round(float(np.mean(shift_size_l2)), 4),
            "hungarian_dist": round(float(np.mean(_agg_seed_list("Random_Shift", "hungarian_dist"))), 4),
            "chamfer_dist": round(float(np.mean(_agg_seed_list("Random_Shift", "chamfer_dist"))), 4),
            "gt_object_cov_pct": round(float(np.mean(_agg_seed_list("Random_Shift", "gt_object_cov_pct"))), 2),
            "gt_object_cov_iou30_pct": round(float(np.mean(_agg_seed_list("Random_Shift", "gt_object_cov_iou30_pct"))), 4),
            "annd_mean": round(float(np.mean(_agg_seed_list("Random_Shift", "annd"))), 4),
            "spatial_variance_mean": round(float(np.mean(_agg_seed_list("Random_Shift", "spatial_variance"))), 4),
            "directional_resultant_length_mean": round(float(np.mean(_agg_seed_list("Random_Shift", "directional_resultant_length"))), 4),
            "radial_outward_bias_mean": round(float(np.mean(_agg_seed_list("Random_Shift", "radial_outward_bias"))), 4),
            "per_query_center_l2_err_max": round(float(max(map_results[f"Random_Shift_seed{s}"]["per_query_matching_diagnostics"]["center_l2_err_max"] for s in seeds)), 6),
            "per_query_size_l2_err_max": round(float(max(map_results[f"Random_Shift_seed{s}"]["per_query_matching_diagnostics"]["size_l2_err_max"] for s in seeds)), 6),
            "clip_fallback_rate_center": round(float(np.mean([map_results[f"Random_Shift_seed{s}"]["per_query_matching_diagnostics"]["clip_fallback_rate_center"] for s in seeds])), 6),
            "clip_fallback_rate_size": round(float(np.mean([map_results[f"Random_Shift_seed{s}"]["per_query_matching_diagnostics"]["clip_fallback_rate_size"] for s in seeds])), 6),
            "delta_map50_ci95": [round(boot_ci_shift["ci_95"][0], 2), round(boot_ci_shift["ci_95"][1], 2)],
            "wilcoxon_p": boot_ci_shift["wilcoxon_p"],
            "multiseed_statistical_evidence": ev_shift,
        },
        "Decoupled_Random_Shift": {
            "description": "Matched Center/Size Random Shift combined with Random Slot Permutation across queries",
            "mAP@50_mean": round(drs_m50_mean, 2),
            "mAP@50_std": round(float(np.std(decoupled_map50, ddof=1)) if len(decoupled_map50) > 1 else 0.0, 2),
            **_tier1_fields(ev_decoupled),
            "mAP@50:95_mean": round(float(np.mean(decoupled_map95)), 2),
            "mAP@50:95_std": round(float(np.std(decoupled_map95, ddof=1)) if len(decoupled_map95) > 1 else 0.0, 2),
            "recall_pct_mean": round(float(np.mean(decoupled_recall)), 2),
            "recall_pct_std": round(float(np.std(decoupled_recall, ddof=1)) if len(decoupled_recall) > 1 else 0.0, 2),
            "fg_hit_pct_mean": round(float(np.mean(decoupled_fghit)), 2),
            "delta_l2": round(float(np.mean(_agg_seed_list("Decoupled_Random_Shift", "delta_l2"))), 4),
            "hungarian_dist": round(float(np.mean(_agg_seed_list("Decoupled_Random_Shift", "hungarian_dist"))), 4),
            "chamfer_dist": round(float(np.mean(_agg_seed_list("Decoupled_Random_Shift", "chamfer_dist"))), 4),
            "gt_object_cov_pct": round(float(np.mean(_agg_seed_list("Decoupled_Random_Shift", "gt_object_cov_pct"))), 2),
            "gt_object_cov_iou30_pct": round(float(np.mean(_agg_seed_list("Decoupled_Random_Shift", "gt_object_cov_iou30_pct"))), 4),
            "annd_mean": round(float(np.mean(_agg_seed_list("Decoupled_Random_Shift", "annd"))), 4),
            "spatial_variance_mean": round(float(np.mean(_agg_seed_list("Decoupled_Random_Shift", "spatial_variance"))), 4),
            "directional_resultant_length_mean": round(float(np.mean(_agg_seed_list("Decoupled_Random_Shift", "directional_resultant_length"))), 4),
            "radial_outward_bias_mean": round(float(np.mean(_agg_seed_list("Decoupled_Random_Shift", "radial_outward_bias"))), 4),
            "observed_gap_from_shift_pp": round(rs_m50_mean - drs_m50_mean, 2),
            "observed_gap_to_aqsm_pp": round(drs_m50_mean - aq_m50_val, 2),
            "delta_map50_ci95": [round(boot_ci_decoupled["ci_95"][0], 2), round(boot_ci_decoupled["ci_95"][1], 2)],
            "wilcoxon_p": boot_ci_decoupled["wilcoxon_p"],
            "multiseed_statistical_evidence": ev_decoupled,
        },
        "AQSM_Spatial_Misrouting": {
            "mAP@50": round(aq_m50_val, 2),
            "mAP@50:95": round(map_results["AQSM_Misrouting"]["mAP@50:95"], 2),
            "recall_pct": round(map_results["AQSM_Misrouting"]["recall"], 2),
            "fg_hit_pct": round(map_results["AQSM_Misrouting"]["fg_hit"], 2),
            "fg_hit_std": round(map_results["AQSM_Misrouting"]["fg_hit_std"], 2),
            "delta_l2": round(map_results["AQSM_Misrouting"]["delta_l2"], 4),
            "delta_center_l2": round(map_results["AQSM_Misrouting"]["delta_center_l2"], 4),
            "delta_size_l2": round(map_results["AQSM_Misrouting"]["delta_size_l2"], 4),
            "hungarian_dist": round(map_results["AQSM_Misrouting"]["hungarian_dist"], 4),
            "chamfer_dist": round(map_results["AQSM_Misrouting"]["chamfer_dist"], 4),
            "gt_object_cov_pct": round(map_results["AQSM_Misrouting"]["gt_object_cov_pct"], 2),
            "gt_object_cov_iou30_pct": round(map_results["AQSM_Misrouting"]["gt_object_cov_iou30_pct"], 4),
            "annd": round(map_results["AQSM_Misrouting"]["annd"], 4),
            "spatial_variance": round(map_results["AQSM_Misrouting"]["spatial_variance"], 4),
            "mean_shift_vector_norm": round(map_results["AQSM_Misrouting"]["mean_shift_vector_norm"], 4),
            "directional_resultant_length": round(map_results["AQSM_Misrouting"]["directional_resultant_length"], 4),
            "radial_outward_bias": round(map_results["AQSM_Misrouting"]["radial_outward_bias"], 4),
            "delta_map50_ci95": [round(boot_ci_aqsm["ci_95"][0], 2), round(boot_ci_aqsm["ci_95"][1], 2)],
            "wilcoxon_p": boot_ci_aqsm["wilcoxon_p"],
        },
    }

    # Đánh giá Dose-Response Curve theo các mức delta in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    print("\n>>> Đang tính toán đường Dose-Response Curve (delta in [0.1 .. 1.0])...")
    deltas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    dose_curve = {}
    rng_dose = np.random.RandomState(42)
    n_dose_eval = n_total if n_dose_samples is None else min(n_dose_samples, n_total)
    sub_gt = gt_list[:n_dose_eval]

    for d_val in deltas:
        d_preds = []
        d_l2_list = []
        for b_idx in range(int(np.ceil(n_dose_eval / batch_size))):
            s_i = b_idx * batch_size
            e_i = min(s_i + batch_size, n_dose_eval)
            b_clean = clean_imgs[s_i:e_i].to(device)
            b_adv = adv_imgs[s_i:e_i].to(device)
            with torch.no_grad():
                cc = extract_encoder_components(model, b_clean)
                ca = extract_encoder_components(model, b_adv)
                c_np = torch.sigmoid(cc["init_reference_points"]).cpu().numpy()
                a_np = torch.sigmoid(ca["init_reference_points"]).cpu().numpy()
                bsz_d = e_i - s_i
                n_dose_queries = int(c_np.shape[1])
                shifted_np = np.zeros_like(c_np)
                for b_sub in range(bsz_d):
                    for q_i in range(n_dose_queries):
                        shifted_np[b_sub, q_i] = sample_exact_matched_shift_4d(
                            c_np[b_sub, q_i], a_np[b_sub, q_i], rng_dose, scale_factor=d_val
                        )
                d_l2_list.append(float(np.mean(np.sqrt(np.sum((shifted_np - c_np) ** 2, axis=-1)))))
                sh_safe = np.clip(shifted_np, 1e-6, 1.0 - 1e-6)
                sh_logits = torch.from_numpy(np.log(sh_safe / (1.0 - sh_safe))).to(device).float()
                dec_out = run_decoder_from_components(
                    model=model,
                    target=cc["target"],
                    source_flatten=cc["source_flatten"],
                    init_reference_points=sh_logits,
                    spatial_shapes=cc["spatial_shapes"],
                    spatial_shapes_list=cc["spatial_shapes_list"],
                    level_start_index=cc["level_start_index"],
                )
                d_preds.extend(postprocess_rtdetr_predictions(dec_out["logits"], dec_out["pred_boxes"], mode="hf_multilabel"))

        eval_d = evaluate_coco_predictions(d_preds, sub_gt)
        dose_curve[f"delta_{d_val}"] = {
            "delta": d_val,
            "mean_l2_displacement": round(float(np.mean(d_l2_list)), 4),
            "mAP@50": round(float(eval_d["mAP@50"]), 2),
            "mAP@50:95": round(float(eval_d["mAP@50:95"]), 2),
        }
        print(f"   * Delta = {d_val:.1f} | Realized L2 = {np.mean(d_l2_list):.4f} | mAP@50 = {eval_d['mAP@50']:.2f}%")

    summary_json_dir = get_summary_json_dir()
    summary_json_dir.mkdir(parents=True, exist_ok=True)
    raw_csv_dir = get_raw_csv_dir()
    raw_csv_dir.mkdir(parents=True, exist_ok=True)

    json_path = summary_json_dir / "tautology_control_results.json"
    t8_payload = {
        "experiment": "Table 8: Spatial Specificity Verification & Tautology Refutation",
        "n_images": n_total,
        "seeds": seeds,
        "model": getattr(wrapper, "pretrained_model_name", None) or resolve_model_name(),
        "device": str(device),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "table8_summary": summary_table8,
        "detailed_by_condition": map_results,
        "dose_response_curve": dose_curve,
    }
    save_json(t8_payload, json_path)
    save_json(t8_payload, summary_json_dir / "controlled_misrouting_results.json")
    print(f"\n>>> Đã xuất JSON Table 8: {json_path}")

    raw_csv_path = raw_csv_dir / "tautology_refutation_raw.csv"
    csv_header = [
        "image_index", "image_id", "clean_ap50", "aqsm_ap50",
        "reassign_ap50_mean", "shift_ap50_mean", "decoupled_shift_ap50_mean",
        "aqsm_l2", "shift_l2", "aqsm_center_l2", "aqsm_size_l2",
        "shift_center_l2", "shift_size_l2", "shift_center_l2_max_err", "shift_size_l2_max_err",
        "clean_fg_hit_pct", "aqsm_fg_hit_pct", "reassign_fg_hit_mean_pct",
        "shift_fg_hit_mean_pct", "decoupled_shift_fg_hit_mean_pct",
        "gt_union_null_pct", "uniform_grid_300_hit_pct",
        "aqsm_dir_alignment", "shift_dir_alignment",
        "aqsm_radial_bias", "shift_radial_bias",
    ]
    csv_rows: List[List[Any]] = []
    for i in range(n_total):
        reassign_mean_i = float(np.mean([per_image_aps[f"Random_Reassign_seed{s}"][i] for s in seeds]))
        shift_mean_i = float(np.mean([per_image_aps[f"Random_Shift_seed{s}"][i] for s in seeds]))
        decoupled_mean_i = float(np.mean([per_image_aps[f"Decoupled_Random_Shift_seed{s}"][i] for s in seeds]))

        gt_boxes_i = gt_boxes_batch[i]
        crowd_i = gt_list[i].get("iscrowd", np.zeros(len(gt_boxes_i), dtype=bool))
        non_crowd_i = gt_boxes_i[~crowd_i] if len(gt_boxes_i) > 0 and len(crowd_i) == len(gt_boxes_i) else gt_boxes_i
        gt_union_i = compute_image_gt_union_area(non_crowd_i, resolution=640) * 100.0
        n_q_eff = n_queries_model if n_queries_model is not None else 300
        grid_hit_i = compute_image_uniform_grid_fg_hit(non_crowd_i, num_queries=n_q_eff) * 100.0

        csv_rows.append([
            i,
            image_ids[i],
            f"{clean_ap50[i]:.6f}",
            f"{per_image_aps['AQSM_Misrouting'][i]:.6f}",
            f"{reassign_mean_i:.6f}",
            f"{shift_mean_i:.6f}",
            f"{decoupled_mean_i:.6f}",
            f"{disp_metrics_by_cond['AQSM_Misrouting'][i]['l2_full']:.5f}",
            f"{float(np.mean([disp_metrics_by_cond[f'Random_Shift_seed{s}'][i]['l2_full'] for s in seeds])):.5f}",
            f"{disp_metrics_by_cond['AQSM_Misrouting'][i]['l2_center']:.5f}",
            f"{disp_metrics_by_cond['AQSM_Misrouting'][i]['l2_size']:.5f}",
            f"{float(np.mean([disp_metrics_by_cond[f'Random_Shift_seed{s}'][i]['l2_center'] for s in seeds])):.5f}",
            f"{float(np.mean([disp_metrics_by_cond[f'Random_Shift_seed{s}'][i]['l2_size'] for s in seeds])):.5f}",
            f"{float(max(disp_metrics_by_cond[f'Random_Shift_seed{s}'][i]['max_center_l2_err'] for s in seeds)):.7f}",
            f"{float(max(disp_metrics_by_cond[f'Random_Shift_seed{s}'][i]['max_size_l2_err'] for s in seeds)):.7f}",
            f"{disp_metrics_by_cond['Clean_Baseline'][i]['fg_hit'] * 100.0:.4f}",
            f"{disp_metrics_by_cond['AQSM_Misrouting'][i]['fg_hit'] * 100.0:.4f}",
            f"{float(np.mean([disp_metrics_by_cond[f'Random_Reassign_seed{s}'][i]['fg_hit'] for s in seeds])) * 100.0:.4f}",
            f"{float(np.mean([disp_metrics_by_cond[f'Random_Shift_seed{s}'][i]['fg_hit'] for s in seeds])) * 100.0:.4f}",
            f"{float(np.mean([disp_metrics_by_cond[f'Decoupled_Random_Shift_seed{s}'][i]['fg_hit'] for s in seeds])) * 100.0:.4f}",
            f"{gt_union_i:.4f}",
            f"{grid_hit_i:.4f}",
            f"{disp_metrics_by_cond['AQSM_Misrouting'][i]['directional_resultant_length']:.6f}",
            f"{float(np.mean([disp_metrics_by_cond[f'Random_Shift_seed{s}'][i]['directional_resultant_length'] for s in seeds])):.6f}",
            f"{disp_metrics_by_cond['AQSM_Misrouting'][i]['radial_outward_bias']:.6f}",
            f"{float(np.mean([disp_metrics_by_cond[f'Random_Shift_seed{s}'][i]['radial_outward_bias'] for s in seeds])):.6f}",
        ])
    write_csv_rows(raw_csv_path, csv_header, csv_rows)
    write_csv_rows(raw_csv_dir / "controlled_misrouting_per_image.csv", csv_header, csv_rows)
    print(f">>> Đã ghi nhận số liệu thô per-image (paired + per-query L2): {raw_csv_path}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return t8_payload


def main():
    parser = create_base_parser("Controlled Misrouting & Tautology Refutation (Table 8)", task_name="Task 4")
    parser.add_argument("--n_boot", type=int, default=N_BOOTSTRAP, help="Số vòng lặp Bootstrap")
    parser.add_argument("--n_dose_samples", type=int, default=None, help="Số ảnh tính toán Dose-Response Curve (mặc định lấy toàn bộ cohort)")
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(map(str, DEFAULT_MULTI_SEEDS)),
        help="Danh sách random seeds cho Random Reassignment & Matched-L2 Random Shift (phân tách bởi dấu phẩy)",
    )
    args = parser.parse_args()

    parsed_seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not parsed_seeds:
        parsed_seeds = list(DEFAULT_MULTI_SEEDS)

    chosen_model = resolve_model_name(getattr(args, "model", None))
    update_experiment_paths(
        folder_image=args.folder_image,
        annotations=args.annotations,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        model=chosen_model,
    )

    configure_cpu_threads(args.threads)
    hw_info = detect_hardware_environment(preferred_device=args.device)
    device = get_device(preferred=args.device)
    batch_size = args.batch_size if args.batch_size is not None else hw_info["recommended_batch_size"]

    audit_logger = ExperimentAuditLogger(task_name="task4_controlled_misrouting")

    wrapper = RTDETRModelWrapper(pretrained_model_name=chosen_model, device=device)
    wrapper.eval()

    dataset = FlexibleCOCODataset(
        folder_image=args.folder_image,
        annotations_file=args.annotations,
        img_size=640,
        id2label=wrapper.model.config.id2label,
        lazy_tensors=True,
        device=device,
        max_samples=args.max_samples,
    )

    audit_logger.log_metadata({
        "folder_image": str(args.folder_image),
        "total_images": len(dataset),
        "batch_size": batch_size,
        "threads": args.threads,
        "device": str(device),
        "hardware_info": hw_info,
        "random_seed": RANDOM_SEED,
        "seeds": parsed_seeds,
        "model_name": chosen_model,
        "max_samples": args.max_samples,
    })

    run_controlled_misrouting_evaluation(
        wrapper=wrapper,
        dataset=dataset,
        batch_size=batch_size,
        n_boot=args.n_boot,
        cache_dir=Path(args.cache_dir),
        audit_logger=audit_logger,
        n_dose_samples=args.n_dose_samples,
        seeds=parsed_seeds,
    )

    audit_logger.close()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print("\n[HOÀN THÀNH TOÀN BỘ] Task 4: Controlled Misrouting & Tautology Refutation thành công!")


if __name__ == "__main__":
    main()
