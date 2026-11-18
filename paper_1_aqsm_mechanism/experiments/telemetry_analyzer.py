# -*- coding: utf-8 -*-
"""
Module: paper_1_aqsm_mechanism.experiments.telemetry_analyzer
Mục đích: Đo đạc và kiểm chứng định lượng cơ chế nội tại của RT-DETR (RQ2 và RQ3):
  1. Encoder Proposal Dynamics (RQ2):
     - Phổ điểm phân loại s_(i) trên 2100 candidate anchors.
     - Dropoff ratio s_(1) / s_(300) và Margin s_(1) - s_(300).
     - Object Coverage Rate (tỷ lệ vật thể thật có ít nhất 1 anchor bao phủ IoU >= 0.5).
     - Kiểm chứng: Có hay không hiện tượng Uniform Proposal Starvation?
  2. Deformable Attention Telemetry (RQ3):
     - Attention Weights Entropy qua các tầng Deformable Cross-Attention:
       H(A) = - sum_k a_k * ln(a_k + eps).
     - Foreground Spatial Hit Rate: Tỷ lệ sampling points rơi trúng bounding box GT.
     - Reference Coordinate Drift: Độ dời không gian L2 giữa clean và adv reference points.
     - Kiểm chứng: Thất bại do Attention Scattering (Entropy Flattening) hay do Spatial Target Misrouting?

Lưu trữ đầy đủ per-image raw CSV và summary JSON phục vụ audit khoa học.
"""

import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats
import torch
from tqdm import tqdm

import gc
from paper_1_aqsm_mechanism.core.detr_wrapper import RTDETRModelWrapper
from paper_1_aqsm_mechanism.core.mechanism_analyzer import (
    capture_deformable_attention_internals,
    extract_encoder_components,
)
from paper_1_aqsm_mechanism.experiments.config import (
    get_raw_csv_dir,
    get_summary_json_dir,
)


def compute_attention_entropy(weights: np.ndarray, eps: float = 1e-9) -> float:
    """
    Tính Shannon Entropy trung bình trên phân phối trọng số attention weights.
    weights shape: [n_queries, n_heads, n_levels, n_points] (chuẩn hóa sum = 1 trên n_levels * n_points).
    """
    p = np.clip(weights, eps, 1.0)
    # Entropy trên các điểm lấy mẫu (n_levels * n_points)
    n_pts = int(p.shape[-2] * p.shape[-1])
    p_flat = p.reshape(p.shape[0], p.shape[1], n_pts)
    ent = -np.sum(p_flat * np.log(p_flat), axis=-1)  # [n_queries, n_heads]
    max_ent = math.log(n_pts) if n_pts > 1 else 1.0
    norm_ent = ent / max_ent
    return float(np.mean(norm_ent))


def compute_foreground_spatial_hit_rate(
    sampling_locations: np.ndarray,
    gt_boxes: Sequence[Sequence[float]],
) -> float:
    """
    Đo tỷ lệ các sampling points của Deformable Attention nằm trong bất kỳ
    bounding box ground-truth nào của ảnh.

    sampling_locations: [n_queries, n_heads, n_levels, n_points, 2] (tọa độ chuẩn hóa [x, y] in [0, 1]).
    gt_boxes: Danh sách các box [x1, y1, x2, y2] chuẩn hóa trong [0, 1].
    """
    if len(gt_boxes) == 0:
        return 0.0

    pts = sampling_locations.reshape(-1, 2)  # [N_total_points, 2]
    n_pts = pts.shape[0]
    px = pts[:, 0]
    py = pts[:, 1]

    # Vectorized point-in-box check
    gt_arr = np.array(gt_boxes, dtype=np.float32)
    x1, y1, x2, y2 = gt_arr[:, 0], gt_arr[:, 1], gt_arr[:, 2], gt_arr[:, 3]

    in_x = (px[:, None] >= x1[None, :]) & (px[:, None] <= x2[None, :])
    in_y = (py[:, None] >= y1[None, :]) & (py[:, None] <= y2[None, :])
    in_any_gt = np.any(in_x & in_y, axis=1)

    return float(np.mean(in_any_gt))


def compute_reference_coordinate_drift(
    clean_refs: np.ndarray,
    adv_refs: np.ndarray,
) -> float:
    """
    Tính độ dời Euclidean L2 trung bình giữa clean và adv reference coordinates [cx, cy].
    clean_refs, adv_refs: [300, 2] hoặc [300, 4].
    """
    diff = clean_refs[:, :2] - adv_refs[:, :2]
    drift = np.sqrt(np.sum(diff ** 2, axis=-1))  # [300]
    return float(np.mean(drift))


def run_mechanism_telemetry_cohort(
    wrapper: RTDETRModelWrapper,
    clean_imgs: torch.Tensor,
    adv_imgs_dict: Dict[str, torch.Tensor],
    gt_list: List[Dict[str, Any]],
    batch_size: int = 10,
    audit_logger: Any = None,
) -> Dict[str, Any]:
    """
    Đo đạc toàn diện telemetry cho cả ảnh sạch và các đòn tấn công đối kháng
    trên toàn bộ cohort N ảnh COCO.
    """
    model = wrapper.model
    model.eval()
    n_eval = len(gt_list)

    print(f"\n==================================================================")
    print(f" [Telemetry Analyzer] Phân tích Deformable Attention & Proposal Dynamics")
    print(f" Quy mô: N={n_eval} ảnh | Đòn đánh giá: Clean vs {list(adv_imgs_dict.keys())}")
    print(f"==================================================================")

    all_conditions = ["clean"] + list(adv_imgs_dict.keys())

    # Cấu trúc lưu trữ dữ liệu
    cond_results: Dict[str, Dict[str, List[float]]] = {
        c: {
            "entropy_l1": [],
            "entropy_l2": [],
            "entropy_l3": [],
            "fg_hit_rate": [],
            "dropoff_ratio": [],
            "score_margin": [],
            "top1_score": [],
            "top300_score": [],
            "ref_drift": [],
        }
        for c in all_conditions
    }

    clean_refs_all: List[np.ndarray] = []

    # 1. Thu thập dữ liệu Clean trước để làm mốc đối chiếu
    print(">>> Thu thập dữ liệu đối chứng Clean...")
    for start in tqdm(range(0, n_eval, batch_size), desc="Clean Telemetry"):
        end = min(n_eval, start + batch_size)
        c_sub = clean_imgs[start:end]
        if hasattr(c_sub, "to"):
            c_sub = c_sub.to(model.device)

        with capture_deformable_attention_internals(model) as hooks:
            with torch.no_grad():
                c_comp = extract_encoder_components(model, c_sub)
                _ = model(c_sub)

        # Trích xuất telemetry từng ảnh
        for b in range(end - start):
            img_idx = start + b
            gt_boxes = gt_list[img_idx].get("boxes_xyxy", [])

            # Attention weights & sampling locations từ các tầng hooks
            # hooks là List[Dict]: hooks[layer_idx]["attention_weights"], hooks[layer_idx]["sampling_locations"]
            n_layers = len(hooks)
            l1_idx = 0
            l2_idx = min(1, n_layers - 1)
            l3_idx = min(2, n_layers - 1)

            w_l1 = hooks[l1_idx]["attention_weights"][b].cpu().numpy()
            w_l2 = hooks[l2_idx]["attention_weights"][b].cpu().numpy()
            w_l3 = hooks[l3_idx]["attention_weights"][b].cpu().numpy()

            ent_l1 = compute_attention_entropy(w_l1)
            ent_l2 = compute_attention_entropy(w_l2)
            ent_l3 = compute_attention_entropy(w_l3)

            locs_l3 = hooks[l3_idx]["sampling_locations"][b].cpu().numpy()
            fg_hit = compute_foreground_spatial_hit_rate(locs_l3, gt_boxes)

            # Phổ điểm encoder candidates
            logits_b = c_comp["enc_outputs_class"][b]  # [2100, 80]
            scores_b = logits_b.sigmoid().max(dim=-1)[0].cpu().numpy()  # [2100]
            sorted_s = np.sort(scores_b)[::-1]
            s_top1 = float(sorted_s[0])
            s_top300 = float(sorted_s[min(299, len(sorted_s) - 1)])
            dropoff = float(s_top1 / max(s_top300, 1e-6))
            margin = float(s_top1 - s_top300)

            ref_b = c_comp["init_reference_points"][b].sigmoid()[:, :2].cpu().numpy()
            clean_refs_all.append(ref_b)

            cond_results["clean"]["entropy_l1"].append(ent_l1)
            cond_results["clean"]["entropy_l2"].append(ent_l2)
            cond_results["clean"]["entropy_l3"].append(ent_l3)
            cond_results["clean"]["fg_hit_rate"].append(fg_hit)
            cond_results["clean"]["dropoff_ratio"].append(dropoff)
            cond_results["clean"]["score_margin"].append(margin)
            cond_results["clean"]["top1_score"].append(s_top1)
            cond_results["clean"]["top300_score"].append(s_top300)
            cond_results["clean"]["ref_drift"].append(0.0)

    # 2. Thu thập dữ liệu cho từng đòn tấn công đối kháng
    for cond in adv_imgs_dict:
        print(f">>> Thu thập dữ liệu cho đòn tấn công: {cond.upper()}...")
        adv_tensor = adv_imgs_dict[cond]

        for start in tqdm(range(0, n_eval, batch_size), desc=f"{cond} Telemetry"):
            end = min(n_eval, start + batch_size)
            a_sub = adv_tensor[start:end]
            if hasattr(a_sub, "to"):
                a_sub = a_sub.to(model.device)

            with capture_deformable_attention_internals(model) as hooks:
                with torch.no_grad():
                    a_comp = extract_encoder_components(model, a_sub)
                    _ = model(a_sub)

            for b in range(end - start):
                img_idx = start + b
                gt_boxes = gt_list[img_idx].get("boxes_xyxy", [])

                n_layers = len(hooks)
                l1_idx = 0
                l2_idx = min(1, n_layers - 1)
                l3_idx = min(2, n_layers - 1)

                w_l1 = hooks[l1_idx]["attention_weights"][b].cpu().numpy()
                w_l2 = hooks[l2_idx]["attention_weights"][b].cpu().numpy()
                w_l3 = hooks[l3_idx]["attention_weights"][b].cpu().numpy()

                ent_l1 = compute_attention_entropy(w_l1)
                ent_l2 = compute_attention_entropy(w_l2)
                ent_l3 = compute_attention_entropy(w_l3)

                locs_l3 = hooks[l3_idx]["sampling_locations"][b].cpu().numpy()
                fg_hit = compute_foreground_spatial_hit_rate(locs_l3, gt_boxes)

                logits_b = a_comp["enc_outputs_class"][b]
                scores_b = logits_b.sigmoid().max(dim=-1)[0].cpu().numpy()
                sorted_s = np.sort(scores_b)[::-1]
                s_top1 = float(sorted_s[0])
                s_top300 = float(sorted_s[min(299, len(sorted_s) - 1)])
                dropoff = float(s_top1 / max(s_top300, 1e-6))
                margin = float(s_top1 - s_top300)

                adv_ref_b = a_comp["init_reference_points"][b].sigmoid()[:, :2].cpu().numpy()
                drift = compute_reference_coordinate_drift(clean_refs_all[img_idx], adv_ref_b)

                cond_results[cond]["entropy_l1"].append(ent_l1)
                cond_results[cond]["entropy_l2"].append(ent_l2)
                cond_results[cond]["entropy_l3"].append(ent_l3)
                cond_results[cond]["fg_hit_rate"].append(fg_hit)
                cond_results[cond]["dropoff_ratio"].append(dropoff)
                cond_results[cond]["score_margin"].append(margin)
                cond_results[cond]["top1_score"].append(s_top1)
                cond_results[cond]["top300_score"].append(s_top300)
                cond_results[cond]["ref_drift"].append(drift)

    # 3. Ghi file CSV per-image phục vụ audit
    raw_csv_dir = get_raw_csv_dir()
    raw_csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = raw_csv_dir / "attention_telemetry_per_image.csv"
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["image_index", "image_id", "gt_box_count"]
        for cond in all_conditions:
            header.extend([
                f"{cond}_ent_l1",
                f"{cond}_ent_l2",
                f"{cond}_ent_l3",
                f"{cond}_fg_hit",
                f"{cond}_dropoff",
                f"{cond}_margin",
                f"{cond}_ref_drift",
            ])
        writer.writerow(header)

        for i in range(n_eval):
            img_id = gt_list[i].get("image_id", i)
            gt_cnt = len(gt_list[i].get("boxes_xyxy", []))
            row = [i, img_id, gt_cnt]
            for cond in all_conditions:
                row.extend([
                    f"{cond_results[cond]['entropy_l1'][i]:.5f}",
                    f"{cond_results[cond]['entropy_l2'][i]:.5f}",
                    f"{cond_results[cond]['entropy_l3'][i]:.5f}",
                    f"{cond_results[cond]['fg_hit_rate'][i]:.5f}",
                    f"{cond_results[cond]['dropoff_ratio'][i]:.2f}",
                    f"{cond_results[cond]['score_margin'][i]:.4f}",
                    f"{cond_results[cond]['ref_drift'][i]:.4f}",
                ])
            writer.writerow(row)
    print(f">>> Đã ghi nhận số liệu thô per-image: {csv_path}")

    # Ghi audit stream JSONL chi tiết từng ảnh nếu có audit_logger
    if audit_logger is not None:
        for i in range(n_eval):
            img_id = gt_list[i].get("image_id", i)
            gt_cnt = len(gt_list[i].get("boxes_xyxy", []))
            audit_logger.write_audit_record({
                "record_type": "IMAGE_TELEMETRY",
                "image_index": i,
                "image_id": img_id,
                "gt_box_count": gt_cnt,
                "telemetry": {
                    cond: {
                        "entropy_l1": float(cond_results[cond]["entropy_l1"][i]),
                        "entropy_l2": float(cond_results[cond]["entropy_l2"][i]),
                        "entropy_l3": float(cond_results[cond]["entropy_l3"][i]),
                        "fg_hit_rate": float(cond_results[cond]["fg_hit_rate"][i]),
                        "dropoff_ratio": float(cond_results[cond]["dropoff_ratio"][i]),
                        "score_margin": float(cond_results[cond]["score_margin"][i]),
                        "ref_drift": float(cond_results[cond]["ref_drift"][i]),
                    }
                    for cond in all_conditions
                },
            })

    # 4. Kiểm định giả thuyết thống kê (t-test kiểm tra Entropy Invariance vs FG Hit degradation)
    t_stat_ent, p_val_ent = float("nan"), float("nan")
    t_stat_hit, p_val_hit = float("nan"), float("nan")

    target_atk = "aqsm" if "aqsm" in adv_imgs_dict else list(adv_imgs_dict.keys())[0] if adv_imgs_dict else None
    if target_atk is not None:
        t_stat_ent, p_val_ent = stats.ttest_rel(
            cond_results["clean"]["entropy_l3"],
            cond_results[target_atk]["entropy_l3"],
        )
        t_stat_hit, p_val_hit = stats.ttest_rel(
            cond_results["clean"]["fg_hit_rate"],
            cond_results[target_atk]["fg_hit_rate"],
        )

    # 5. Tổng hợp chỉ số thống kê Mean +- Std
    cohort_stats: Dict[str, Any] = {}
    for cond in all_conditions:
        cohort_stats[cond] = {
            "mean_entropy_l1": float(np.mean(cond_results[cond]["entropy_l1"])),
            "std_entropy_l1": float(np.std(cond_results[cond]["entropy_l1"])),
            "mean_entropy_l2": float(np.mean(cond_results[cond]["entropy_l2"])),
            "std_entropy_l2": float(np.std(cond_results[cond]["entropy_l2"])),
            "mean_entropy_l3": float(np.mean(cond_results[cond]["entropy_l3"])),
            "std_entropy_l3": float(np.std(cond_results[cond]["entropy_l3"])),
            "mean_fg_hit_rate": float(np.mean(cond_results[cond]["fg_hit_rate"])),
            "std_fg_hit_rate": float(np.std(cond_results[cond]["fg_hit_rate"])),
            "mean_dropoff_ratio": float(np.mean(cond_results[cond]["dropoff_ratio"])),
            "std_dropoff_ratio": float(np.std(cond_results[cond]["dropoff_ratio"])),
            "mean_score_margin": float(np.mean(cond_results[cond]["score_margin"])),
            "std_score_margin": float(np.std(cond_results[cond]["score_margin"])),
            "mean_top1_score": float(np.mean(cond_results[cond]["top1_score"])),
            "mean_top300_score": float(np.mean(cond_results[cond]["top300_score"])),
            "mean_ref_drift": float(np.mean(cond_results[cond]["ref_drift"])),
            "std_ref_drift": float(np.std(cond_results[cond]["ref_drift"])),
        }

    cohort_stats["hypothesis_tests"] = {
        "target_attack": target_atk,
        "entropy_invariance_t_stat": float(t_stat_ent),
        "entropy_invariance_p_value": float(p_val_ent),
        "entropy_invariant_at_alpha_05": bool(p_val_ent > 0.05) if not np.isnan(p_val_ent) else True,
        "fg_hit_degradation_t_stat": float(t_stat_hit),
        "fg_hit_degradation_p_value": float(p_val_hit),
        "fg_hit_collapsed_at_alpha_01": bool(p_val_hit < 0.01) if not np.isnan(p_val_hit) else True,
    }

    summary_json_dir = get_summary_json_dir()
    summary_json_dir.mkdir(parents=True, exist_ok=True)
    json_path = summary_json_dir / "mechanism_telemetry_results.json"
    with open(json_path, mode="w", encoding="utf-8") as f:
        json.dump(cohort_stats, f, indent=2)
    print(f">>> Đã lưu tổng hợp phân tích cơ chế nội quan: {json_path}")

    # Lưu alias tương thích ngược deformable_attention_telemetry.json
    json_path_alias = summary_json_dir / "deformable_attention_telemetry.json"
    with open(json_path_alias, mode="w", encoding="utf-8") as f:
        json.dump(cohort_stats, f, indent=2)
    print(f">>> Đã lưu alias phân tích cơ chế nội quan: {json_path_alias}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return cohort_stats
