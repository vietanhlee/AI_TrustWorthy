# -*- coding: utf-8 -*-
"""
Module: paper_1_aqsm_mechanism.experiments.tasks.task4_controlled_misrouting
Mục đích: Thực nghiệm kiểm chứng tính đặc hiệu không gian và bác bỏ giả thuyết Tautology
(Controlled Spatial Misrouting and Spatial Specificity Verification) trên toàn bộ tập ảnh COCO val2017 -> Phục vụ Table 8.

Bản chất khoa học:
  Giữ nguyên 100% Memory sạch M_clean và Content Queries Q_clean ở trạng thái hoàn hảo;
  chỉ can thiệp vào tọa độ bounding box tham chiếu ban đầu B_ref nạp vào Deformable Decoder:
    1. Clean Reference Baseline: B_clean chuẩn (L2 = 0.0000).
    2. Random Reassignment (5 seeds: 42, 100, 2024, 777, 9999): Hoán vị ngẫu nhiên 300 boxes sạch.
    3. Matched-L2 Random Shift (5 seeds: 42, 100, 2024, 777, 9999): Dịch chuyển đẳng hướng ngẫu nhiên
       được chuẩn hóa sao cho độ lệch L2 sau khi clip [0, 1] khớp chính xác 100% với L2 của AQSM.
    4. AQSM Spatial Misrouting: Sử dụng chính xác tọa độ reference boxes bị lái lệch bởi đòn tấn công AQSM.

Đặc tính chuẩn công nghiệp (Production Standards):
  - Nhận tham số linh hoạt qua create_base_parser (--folder_image, --annotations, --output_dir, --cache_dir, --max_samples).
  - Nạp ảnh theo lô lười (Lazy Batch Loading) chống tràn RAM khi chạy quy mô 5,000 ảnh.
  - Ghi nhận raw CSV từng ảnh và summary JSON đầy đủ ngữ cảnh phục vụ điền bài báo.
"""

import argparse
import gc
import json
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from paper_1_aqsm_mechanism.core.adv_cache import load_cached_adv_images
from paper_1_aqsm_mechanism.core.coco_gt import (
    bootstrap_paired_diff,
    compute_per_image_coco_aps,
    evaluate_coco_predictions,
    postprocess_rtdetr_predictions,
)
from paper_1_aqsm_mechanism.core.dataset_loader import FlexibleCOCODataset
from paper_1_aqsm_mechanism.core.detr_wrapper import RTDETRModelWrapper
from paper_1_aqsm_mechanism.core.mechanism_analyzer import (
    _box_iou_np,
    compute_spatial_clustering_metrics,
    extract_encoder_components,
    run_decoder_from_components,
)
from paper_1_aqsm_mechanism.experiments.audit_logger import ExperimentAuditLogger
from paper_1_aqsm_mechanism.experiments.config import (
    MODEL_NAME,
    N_BOOTSTRAP,
    RANDOM_SEED,
    configure_cpu_threads,
    detect_hardware_environment,
    get_device,
    get_model_tag,
    get_raw_csv_dir,
    get_summary_json_dir,
    resolve_model_name,
    update_experiment_paths,
)
from paper_1_aqsm_mechanism.utils.cli import create_base_parser
from paper_1_aqsm_mechanism.utils.io_helpers import format_duration, save_json, write_csv_rows


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


def sample_exact_matched_shift_4d(
    clean_box: np.ndarray,
    aqsm_box: np.ndarray,
    rng: np.random.RandomState,
    scale_factor: float = 1.0,
    max_tries: int = 50,
) -> np.ndarray:
    """
    Sinh tọa độ box dịch chuyển ngẫu nhiên [cx, cy, w, h] sao cho sau phép chiếu/clipping
    vào [0, 1]^4, độ dịch chuyển thực tế sau clipping (realized post-clipping displacement)
    khớp CHÍNH XÁC 100% với độ dịch chuyển AQSM cả về Center L2 và Size L2.
    """
    d_center = float(np.linalg.norm(aqsm_box[:2] - clean_box[:2])) * scale_factor
    d_size = float(np.linalg.norm(aqsm_box[2:] - clean_box[2:])) * scale_factor

    def _sample_exact_2d(p: np.ndarray, target_d: float) -> np.ndarray:
        if target_d <= 1e-7:
            return p.copy()
        for _ in range(max_tries):
            theta = rng.uniform(0, 2 * np.pi)
            u = np.array([np.cos(theta), np.sin(theta)])
            p_cand = p + target_d * u
            if np.all(p_cand >= 0.0) and np.all(p_cand <= 1.0):
                return p_cand
        # Fallback bisection search nếu góc ngẫu nhiên chạm biên canvas
        u = np.array([0.5, 0.5]) - p
        u_norm = np.linalg.norm(u)
        if u_norm > 1e-6:
            u = u / u_norm
            lo, hi = 0.0, 2.0
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                clipped = np.clip(p + mid * u, 0.0, 1.0)
                dist = np.linalg.norm(clipped - p)
                if dist < target_d:
                    lo = mid
                else:
                    hi = mid
            return np.clip(p + 0.5 * (lo + hi) * u, 0.0, 1.0)
        return np.clip(p, 0.0, 1.0)

    center_shifted = _sample_exact_2d(clean_box[:2], d_center)
    size_shifted = _sample_exact_2d(clean_box[2:], d_size)
    return np.concatenate([center_shifted, size_shifted])


def run_controlled_misrouting_evaluation(
    wrapper: RTDETRModelWrapper,
    dataset: Any,
    adv_imgs: Optional[torch.Tensor] = None,
    batch_size: int = 10,
    n_boot: int = N_BOOTSTRAP,
    cache_dir: Optional[Path] = None,
    audit_logger: Any = None,
    n_dose_samples: Optional[int] = None,
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

    seeds = [42, 100, 2024, 777, 9999]

    conditions = ["Clean_Baseline", "AQSM_Misrouting"]
    for s in seeds:
        conditions.append(f"Random_Reassign_seed{s}")
    for s in seeds:
        conditions.append(f"Random_Shift_seed{s}")

    preds_by_cond: Dict[str, List[Dict[str, Any]]] = {c: [] for c in conditions}
    disp_metrics_by_cond: Dict[str, List[Dict[str, float]]] = {c: [] for c in conditions}

    n_batches = int(np.ceil(n_total / batch_size))

    print(f"\n=======================================================")
    print(f"--> [Controlled Misrouting] Khởi chạy can thiệp B_ref trên N={n_total} ảnh...")
    print(f"=======================================================")

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

        b_ref_tensors: Dict[str, torch.Tensor] = {}
        b_ref_tensors["Clean_Baseline"] = c_comp["init_reference_points"]
        b_ref_tensors["AQSM_Misrouting"] = a_comp["init_reference_points"]

        l2_diff = np.sqrt(np.sum((a_ref_np - c_ref_np) ** 2, axis=-1))
        mean_l2_per_img = np.mean(l2_diff, axis=1, keepdims=True)

        for s in seeds:
            rng = np.random.RandomState(s)
            perm = rng.permutation(300)
            perm_boxes_np = c_ref_np[:, perm, :]
            perm_boxes_safe = np.clip(perm_boxes_np, 1e-6, 1.0 - 1e-6)
            perm_logits = torch.from_numpy(np.log(perm_boxes_safe / (1.0 - perm_boxes_safe))).to(device).float()
            b_ref_tensors[f"Random_Reassign_seed{s}"] = perm_logits

        for s in seeds:
            rng = np.random.RandomState(s)
            shifted_boxes_np = np.zeros_like(c_ref_np)
            for b in range(cur_bsz):
                for i in range(300):
                    shifted_boxes_np[b, i] = sample_exact_matched_shift_4d(
                        c_ref_np[b, i], a_ref_np[b, i], rng, scale_factor=1.0
                    )
            shifted_boxes_safe = np.clip(shifted_boxes_np, 1e-6, 1.0 - 1e-6)
            shift_logits = torch.from_numpy(np.log(shifted_boxes_safe / (1.0 - shifted_boxes_safe))).to(device).float()
            b_ref_tensors[f"Random_Shift_seed{s}"] = shift_logits

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
            l2_full = float(np.mean(np.sqrt(np.sum(diff_full ** 2, axis=-1))))
            diff_center = diff_full[:, :, :2]
            l2_center = float(np.mean(np.sqrt(np.sum(diff_center ** 2, axis=-1))))
            dcx = float(np.mean(np.abs(diff_center[:, :, 0])))
            dcy = float(np.mean(np.abs(diff_center[:, :, 1])))
            diff_size = diff_full[:, :, 2:]
            l2_size = float(np.mean(np.sqrt(np.sum(diff_size ** 2, axis=-1))))
            dw = float(np.mean(np.abs(diff_size[:, :, 0])))
            dh = float(np.mean(np.abs(diff_size[:, :, 1])))

            for b in range(cur_bsz):
                img_idx = start_i + b
                gt_boxes = gt_boxes_batch[img_idx]
                fg_hit = compute_ref_fg_hit(ref_sig_np[b, :, :2], gt_boxes)
                recall = compute_object_recall(batch_preds[b], gt_boxes)

                # Đo đạc gom cụm không gian ANND và độ phân tán spatial variance
                clust_metrics = compute_spatial_clustering_metrics(ref_sig_np[b, :, :2])

                disp_metrics_by_cond[cond].append({
                    "l2_full": l2_full,
                    "l2_center": l2_center,
                    "delta_cx": dcx,
                    "delta_cy": dcy,
                    "l2_size": l2_size,
                    "delta_w": dw,
                    "delta_h": dh,
                    "fg_hit": fg_hit,
                    "recall": recall,
                    "annd": clust_metrics["annd"],
                    "spatial_variance": clust_metrics["spatial_variance"],
                    "spatial_std_x": clust_metrics["spatial_std_x"],
                    "spatial_std_y": clust_metrics["spatial_std_y"],
                })

    print(">>> Đang đánh giá COCO mAP toàn cục cho tất cả các nhánh...")
    map_results: Dict[str, Dict[str, float]] = {}
    per_image_aps: Dict[str, np.ndarray] = {}

    for cond in conditions:
        eval_m = evaluate_coco_predictions(preds_by_cond[cond], gt_list)
        ap50_vec, _ = compute_per_image_coco_aps(preds_by_cond[cond], gt_list)
        map_results[cond] = {
            "mAP@50": float(eval_m["mAP@50"]),
            "mAP@75": float(eval_m["mAP@75"]),
            "mAP@50:95": float(eval_m["mAP@50:95"]),
            "recall": float(np.mean([m["recall"] for m in disp_metrics_by_cond[cond]]) * 100.0),
            "fg_hit": float(np.mean([m["fg_hit"] for m in disp_metrics_by_cond[cond]]) * 100.0),
            "delta_l2": float(np.mean([m["l2_full"] for m in disp_metrics_by_cond[cond]])),
            "delta_center_l2": float(np.mean([m["l2_center"] for m in disp_metrics_by_cond[cond]])),
            "delta_size_l2": float(np.mean([m["l2_size"] for m in disp_metrics_by_cond[cond]])),
            "annd": float(np.mean([m["annd"] for m in disp_metrics_by_cond[cond]])),
            "spatial_variance": float(np.mean([m["spatial_variance"] for m in disp_metrics_by_cond[cond]])),
        }
        per_image_aps[cond] = ap50_vec

    reassign_map50 = [map_results[f"Random_Reassign_seed{s}"]["mAP@50"] for s in seeds]
    reassign_map95 = [map_results[f"Random_Reassign_seed{s}"]["mAP@50:95"] for s in seeds]
    reassign_recall = [map_results[f"Random_Reassign_seed{s}"]["recall"] for s in seeds]
    reassign_fghit = [map_results[f"Random_Reassign_seed{s}"]["fg_hit"] for s in seeds]

    shift_map50 = [map_results[f"Random_Shift_seed{s}"]["mAP@50"] for s in seeds]
    shift_map95 = [map_results[f"Random_Shift_seed{s}"]["mAP@50:95"] for s in seeds]
    shift_recall = [map_results[f"Random_Shift_seed{s}"]["recall"] for s in seeds]
    shift_fghit = [map_results[f"Random_Shift_seed{s}"]["fg_hit"] for s in seeds]
    shift_l2 = [map_results[f"Random_Shift_seed{s}"]["delta_l2"] for s in seeds]

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
    boot_ci_aqsm = bootstrap_paired_diff(
        per_image_aps["AQSM_Misrouting"],
        clean_ap50,
        n_boot=n_boot,
    )

    summary_table8 = {
        "Clean_Baseline": {
            "mAP@50": round(map_results["Clean_Baseline"]["mAP@50"], 2),
            "mAP@50:95": round(map_results["Clean_Baseline"]["mAP@50:95"], 2),
            "recall_pct": round(map_results["Clean_Baseline"]["recall"], 2),
            "fg_hit_pct": round(map_results["Clean_Baseline"]["fg_hit"], 2),
            "delta_l2": 0.0000,
            "annd": round(map_results["Clean_Baseline"]["annd"], 4),
            "spatial_variance": round(map_results["Clean_Baseline"]["spatial_variance"], 4),
            "delta_map50_ci95": [0.0, 0.0],
        },
        "Random_Reassignment": {
            "mAP@50_mean": round(float(np.mean(reassign_map50)), 2),
            "mAP@50_std": round(float(np.std(reassign_map50)), 2),
            "mAP@50:95_mean": round(float(np.mean(reassign_map95)), 2),
            "mAP@50:95_std": round(float(np.std(reassign_map95)), 2),
            "recall_pct_mean": round(float(np.mean(reassign_recall)), 2),
            "fg_hit_pct_mean": round(float(np.mean(reassign_fghit)), 2),
            "delta_l2": round(map_results["Random_Reassign_seed42"]["delta_l2"], 4),
            "annd_mean": round(float(np.mean([map_results[f"Random_Reassign_seed{s}"]["annd"] for s in seeds])), 4),
            "spatial_variance_mean": round(float(np.mean([map_results[f"Random_Reassign_seed{s}"]["spatial_variance"] for s in seeds])), 4),
            "delta_map50_ci95": [round(boot_ci_reassign["ci_95"][0], 2), round(boot_ci_reassign["ci_95"][1], 2)],
            "wilcoxon_p": boot_ci_reassign["wilcoxon_p"],
        },
        "Matched_L2_Random_Shift": {
            "mAP@50_mean": round(float(np.mean(shift_map50)), 2),
            "mAP@50_std": round(float(np.std(shift_map50)), 2),
            "mAP@50:95_mean": round(float(np.mean(shift_map95)), 2),
            "mAP@50:95_std": round(float(np.std(shift_map95)), 2),
            "recall_pct_mean": round(float(np.mean(shift_recall)), 2),
            "fg_hit_pct_mean": round(float(np.mean(shift_fghit)), 2),
            "delta_l2": round(float(np.mean(shift_l2)), 4),
            "annd_mean": round(float(np.mean([map_results[f"Random_Shift_seed{s}"]["annd"] for s in seeds])), 4),
            "spatial_variance_mean": round(float(np.mean([map_results[f"Random_Shift_seed{s}"]["spatial_variance"] for s in seeds])), 4),
            "delta_map50_ci95": [round(boot_ci_shift["ci_95"][0], 2), round(boot_ci_shift["ci_95"][1], 2)],
            "wilcoxon_p": boot_ci_shift["wilcoxon_p"],
        },
        "AQSM_Spatial_Misrouting": {
            "mAP@50": round(map_results["AQSM_Misrouting"]["mAP@50"], 2),
            "mAP@50:95": round(map_results["AQSM_Misrouting"]["mAP@50:95"], 2),
            "recall_pct": round(map_results["AQSM_Misrouting"]["recall"], 2),
            "fg_hit_pct": round(map_results["AQSM_Misrouting"]["fg_hit"], 2),
            "delta_l2": round(map_results["AQSM_Misrouting"]["delta_l2"], 4),
            "delta_center_l2": round(map_results["AQSM_Misrouting"]["delta_center_l2"], 4),
            "delta_size_l2": round(map_results["AQSM_Misrouting"]["delta_size_l2"], 4),
            "annd": round(map_results["AQSM_Misrouting"]["annd"], 4),
            "spatial_variance": round(map_results["AQSM_Misrouting"]["spatial_variance"], 4),
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
                shifted_np = np.zeros_like(c_np)
                for b_sub in range(bsz_d):
                    for q_i in range(300):
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
        "model": MODEL_NAME,
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
        "reassign_ap50_mean", "shift_ap50_mean", "aqsm_l2", "shift_l2",
    ]
    csv_rows: List[List[Any]] = []
    for i in range(n_total):
        reassign_mean_i = float(np.mean([per_image_aps[f"Random_Reassign_seed{s}"][i] for s in seeds]))
        shift_mean_i = float(np.mean([per_image_aps[f"Random_Shift_seed{s}"][i] for s in seeds]))
        csv_rows.append([
            i,
            image_ids[i],
            f"{clean_ap50[i]:.6f}",
            f"{per_image_aps['AQSM_Misrouting'][i]:.6f}",
            f"{reassign_mean_i:.6f}",
            f"{shift_mean_i:.6f}",
            f"{disp_metrics_by_cond['AQSM_Misrouting'][i]['l2_full']:.5f}",
            f"{disp_metrics_by_cond['Random_Shift_seed42'][i]['l2_full']:.5f}",
        ])
    write_csv_rows(raw_csv_path, csv_header, csv_rows)
    print(f">>> Đã ghi nhận số liệu thô per-image: {raw_csv_path}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return t8_payload


def main():
    parser = create_base_parser("Controlled Misrouting & Tautology Refutation (Table 8)", task_name="Task 4")
    parser.add_argument("--n_boot", type=int, default=N_BOOTSTRAP, help="Số vòng lặp Bootstrap")
    parser.add_argument("--n_dose_samples", type=int, default=None, help="Số ảnh tính toán Dose-Response Curve (mặc định lấy toàn bộ cohort)")
    args = parser.parse_args()

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
        "model_name": MODEL_NAME,
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
    )

    audit_logger.close()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print("\n[HOÀN THÀNH TOÀN BỘ] Task 4: Controlled Misrouting & Tautology Refutation thành công!")


if __name__ == "__main__":
    main()
