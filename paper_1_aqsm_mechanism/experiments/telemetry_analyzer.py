# -*- coding: utf-8 -*-
"""
Module: paper_1_aqsm_mechanism.experiments.telemetry_analyzer
Mục đích: Đo đạc và kiểm chứng định lượng cơ chế nội tại của RT-DETR (RQ2 và RQ3):
  1. Encoder Proposal Dynamics (RQ2):
     - Phổ điểm phân loại s_(i) trên S candidate anchors (S = 8,400 ở native 640x640; 2,100 ở 320x320).
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

import gc
import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from scipy import stats
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from tqdm import tqdm

from paper_1_aqsm_mechanism.tools.compute_nonparametric_statistics import compute_paired_tost_equivalence

from paper_1_aqsm_mechanism.core.detr_wrapper import RTDETRModelWrapper
from paper_1_aqsm_mechanism.core.mechanism_analyzer import (
    capture_deformable_attention_internals,
    extract_encoder_components,
)
from paper_1_aqsm_mechanism.experiments.config import (
    get_raw_csv_dir,
    get_summary_json_dir,
)
from paper_1_aqsm_mechanism.utils.io_helpers import save_json, write_csv_rows


def compute_attention_distribution_details(weights: np.ndarray, eps: float = 1e-9) -> Dict[str, Any]:
    """
    Trích xuất thống kê chi tiết cấp Head và Multi-Scale từ tensor trọng số Deformable Attention
    `weights` shape `[n_queries, n_heads, n_levels, n_points]`:
      - head_entropies: [n_heads] entropy chuẩn hóa trên từng attention head riêng biệt.
      - mean_marginal_weights: [n_heads, n_levels * n_points] trọng số trung bình trên từng head.
      - top1_weight: trọng số tập trung cực đại trung bình.
      - scale_mass: [n_levels] phân bổ trọng số trung bình lên S3/S4/S5.
    """
    p = np.clip(weights, eps, 1.0)
    n_q, n_h, n_lvl, n_pts_lvl = p.shape
    n_pts = int(n_lvl * n_pts_lvl)
    p_flat = p.reshape(n_q, n_h, n_pts)
    max_ent = math.log(n_pts) if n_pts > 1 else 1.0
    ent_qh = -np.sum(p_flat * np.log(p_flat), axis=-1) / max_ent  # [n_q, n_h]
    head_ent = np.mean(ent_qh, axis=0)  # [n_h]
    marginal_w = np.mean(p_flat, axis=0)  # [n_h, n_pts]
    top1_w = float(np.mean(np.max(p_flat, axis=-1)))
    scale_mass = np.sum(p, axis=(0, 1, 3)) / float(max(1, n_q * n_h))
    return {
        "head_entropies": head_ent,
        "marginal_weights": marginal_w,
        "top1_weight": top1_w,
        "scale_mass": scale_mass,
    }


def compute_attention_entropy(weights: np.ndarray, eps: float = 1e-9) -> float:
    """
    Tính Shannon Entropy trung bình trên phân phối trọng số attention weights.
    weights shape: [n_queries, n_heads, n_levels, n_points] (chuẩn hóa sum = 1 trên n_levels * n_points).
    """
    p = np.clip(weights, eps, 1.0)
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
    """
    if len(gt_boxes) == 0:
        return 0.0

    pts = sampling_locations.reshape(-1, 2)  # [N_total_points, 2]
    px = pts[:, 0]
    py = pts[:, 1]

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
    Tính độ dời Euclidean L2 trung bình theo cặp thứ hạng (rank-paired)
    giữa clean và adv reference coordinates [cx, cy] (tức Δ_p, nhạy với hoán vị slot).
    """
    diff = clean_refs[:, :2] - adv_refs[:, :2]
    drift = np.sqrt(np.sum(diff ** 2, axis=-1))  # [N_q]
    return float(np.mean(drift))


def compute_hungarian_reference_drift(
    clean_refs: np.ndarray,
    adv_refs: np.ndarray,
) -> float:
    """
    Tính khoảng cách tập hợp bất biến hoán vị (Hungarian Optimal Matching Distance, Δ_Hungarian)
    giữa hai tập tọa độ reference [cx, cy] trên [0, 1]^2, tách biệt hoàn toàn dịch chuyển không gian
    khỏi hiệu ứng hoán đổi chỉ số slot thuần túy.
    """
    c_xy = np.asarray(clean_refs[:, :2], dtype=np.float64)
    a_xy = np.asarray(adv_refs[:, :2], dtype=np.float64)
    if len(c_xy) == 0 or len(a_xy) == 0:
        return 0.0
    cost_mat = cdist(c_xy, a_xy, metric="euclidean")
    r_ind, c_ind = linear_sum_assignment(cost_mat)
    return float(np.mean(cost_mat[r_ind, c_ind]))


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
    n_queries_model = int(getattr(wrapper, "num_queries", getattr(getattr(model, "config", None), "num_queries", 300)) or 300)

    print("\n==================================================================")
    print(" [Telemetry Analyzer] Phân tích Deformable Attention & Proposal Dynamics")
    print(f" Quy mô: N={n_eval} ảnh | Đòn đánh giá: Clean vs {list(adv_imgs_dict.keys())}")
    print("==================================================================")

    all_conditions = ["clean"] + list(adv_imgs_dict.keys())

    cond_results: Dict[str, Dict[str, List[float]]] = {
        c: {
            "entropy_l1": [],
            "entropy_l2": [],
            "entropy_l3": [],
            "entropy_all_layers": [],
            "fg_hit_rate": [],
            "dropoff_ratio": [],
            "score_margin": [],
            "top1_score": [],
            "top300_score": [],
            "ref_drift": [],
            "hungarian_ref_drift": [],
            "top1_attn_weight": [],
            "head_entropy_max_abs_delta": [],
            "head_attn_tv_shift": [],
            "scale_mass_s3": [],
            "scale_mass_s4": [],
            "scale_mass_s5": [],
        }
        for c in all_conditions
    }

    clean_refs_all: List[np.ndarray] = []
    clean_head_ent_all: List[Optional[np.ndarray]] = []
    clean_head_marg_all: List[Optional[np.ndarray]] = []

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

        for b in range(end - start):
            img_idx = start + b
            gt_boxes = gt_list[img_idx].get("boxes_xyxy", [])

            n_layers = len(hooks)
            if n_layers > 0:
                l1_idx = 0
                l2_idx = min(1, n_layers - 1)
                l3_idx = min(2, n_layers - 1)

                w_l1 = hooks[l1_idx]["attention_weights"][b].cpu().numpy()
                w_l2 = hooks[l2_idx]["attention_weights"][b].cpu().numpy()
                w_l3 = hooks[l3_idx]["attention_weights"][b].cpu().numpy()

                ent_l1 = compute_attention_entropy(w_l1)
                ent_l2 = compute_attention_entropy(w_l2)
                ent_l3 = compute_attention_entropy(w_l3)
                ent_all = float(np.mean([compute_attention_entropy(hooks[li]["attention_weights"][b].cpu().numpy()) for li in range(n_layers)]))
                locs_last = hooks[-1]["sampling_locations"][b].cpu().numpy()
                fg_hit = compute_foreground_spatial_hit_rate(locs_last, gt_boxes)

                head_details = [compute_attention_distribution_details(hooks[li]["attention_weights"][b].cpu().numpy()) for li in range(n_layers)]
                c_head_ent_l = np.stack([d["head_entropies"] for d in head_details], axis=0)  # [L, H]
                c_head_marg_l = np.stack([d["marginal_weights"] for d in head_details], axis=0)  # [L, H, L*K]
                c_top1_w = float(np.mean([d["top1_weight"] for d in head_details]))
                c_sm = np.mean(np.stack([d["scale_mass"] for d in head_details], axis=0), axis=0)
            else:
                ent_l1 = ent_l2 = ent_l3 = ent_all = 0.0
                fg_hit = 0.0
                c_head_ent_l = None
                c_head_marg_l = None
                c_top1_w = 0.0
                c_sm = np.zeros(3, dtype=np.float64)

            clean_head_ent_all.append(c_head_ent_l)
            clean_head_marg_all.append(c_head_marg_l)

            logits_b = c_comp["enc_outputs_class"][b]
            scores_b = logits_b.sigmoid().max(dim=-1)[0].cpu().numpy()
            sorted_s = np.sort(scores_b)[::-1]
            s_top1 = float(sorted_s[0])
            s_top300 = float(sorted_s[min(n_queries_model - 1, len(sorted_s) - 1)])
            dropoff = float(s_top1 / max(s_top300, 1e-6))
            margin = float(s_top1 - s_top300)

            ref_b = c_comp["init_reference_points"][b].sigmoid()[:, :2].cpu().numpy()
            clean_refs_all.append(ref_b)

            cond_results["clean"]["entropy_l1"].append(ent_l1)
            cond_results["clean"]["entropy_l2"].append(ent_l2)
            cond_results["clean"]["entropy_l3"].append(ent_l3)
            cond_results["clean"]["entropy_all_layers"].append(ent_all)
            cond_results["clean"]["fg_hit_rate"].append(fg_hit)
            cond_results["clean"]["dropoff_ratio"].append(dropoff)
            cond_results["clean"]["score_margin"].append(margin)
            cond_results["clean"]["top1_score"].append(s_top1)
            cond_results["clean"]["top300_score"].append(s_top300)
            cond_results["clean"]["ref_drift"].append(0.0)
            cond_results["clean"]["hungarian_ref_drift"].append(0.0)
            cond_results["clean"]["top1_attn_weight"].append(c_top1_w)
            cond_results["clean"]["head_entropy_max_abs_delta"].append(0.0)
            cond_results["clean"]["head_attn_tv_shift"].append(0.0)
            cond_results["clean"]["scale_mass_s3"].append(float(c_sm[0]) if len(c_sm) > 0 else 0.0)
            cond_results["clean"]["scale_mass_s4"].append(float(c_sm[1]) if len(c_sm) > 1 else 0.0)
            cond_results["clean"]["scale_mass_s5"].append(float(c_sm[2]) if len(c_sm) > 2 else 0.0)

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
                if n_layers > 0:
                    l1_idx = 0
                    l2_idx = min(1, n_layers - 1)
                    l3_idx = min(2, n_layers - 1)

                    w_l1 = hooks[l1_idx]["attention_weights"][b].cpu().numpy()
                    w_l2 = hooks[l2_idx]["attention_weights"][b].cpu().numpy()
                    w_l3 = hooks[l3_idx]["attention_weights"][b].cpu().numpy()

                    ent_l1 = compute_attention_entropy(w_l1)
                    ent_l2 = compute_attention_entropy(w_l2)
                    ent_l3 = compute_attention_entropy(w_l3)
                    ent_all = float(np.mean([compute_attention_entropy(hooks[li]["attention_weights"][b].cpu().numpy()) for li in range(n_layers)]))
                    locs_last = hooks[-1]["sampling_locations"][b].cpu().numpy()
                    fg_hit = compute_foreground_spatial_hit_rate(locs_last, gt_boxes)

                    a_details = [compute_attention_distribution_details(hooks[li]["attention_weights"][b].cpu().numpy()) for li in range(n_layers)]
                    a_head_ent_l = np.stack([d["head_entropies"] for d in a_details], axis=0)
                    a_head_marg_l = np.stack([d["marginal_weights"] for d in a_details], axis=0)
                    a_top1_w = float(np.mean([d["top1_weight"] for d in a_details]))
                    a_sm = np.mean(np.stack([d["scale_mass"] for d in a_details], axis=0), axis=0)
                    c_he = clean_head_ent_all[img_idx]
                    c_hm = clean_head_marg_all[img_idx]
                    if c_he is not None and c_he.shape == a_head_ent_l.shape:
                        head_ent_max_delta = float(np.max(np.abs(np.mean(a_head_ent_l - c_he, axis=0))))
                    else:
                        head_ent_max_delta = 0.0
                    if c_hm is not None and c_hm.shape == a_head_marg_l.shape:
                        head_tv = float(0.5 * np.mean(np.sum(np.abs(a_head_marg_l - c_hm), axis=-1)))
                    else:
                        head_tv = 0.0
                else:
                    ent_l1 = ent_l2 = ent_l3 = ent_all = 0.0
                    fg_hit = 0.0
                    a_top1_w = 0.0
                    head_ent_max_delta = 0.0
                    head_tv = 0.0
                    a_sm = np.zeros(3, dtype=np.float64)

                logits_b = a_comp["enc_outputs_class"][b]
                scores_b = logits_b.sigmoid().max(dim=-1)[0].cpu().numpy()
                sorted_s = np.sort(scores_b)[::-1]
                s_top1 = float(sorted_s[0])
                s_top300 = float(sorted_s[min(n_queries_model - 1, len(sorted_s) - 1)])
                dropoff = float(s_top1 / max(s_top300, 1e-6))
                margin = float(s_top1 - s_top300)

                adv_ref_b = a_comp["init_reference_points"][b].sigmoid()[:, :2].cpu().numpy()
                drift = compute_reference_coordinate_drift(clean_refs_all[img_idx], adv_ref_b)
                h_drift = compute_hungarian_reference_drift(clean_refs_all[img_idx], adv_ref_b)

                cond_results[cond]["entropy_l1"].append(ent_l1)
                cond_results[cond]["entropy_l2"].append(ent_l2)
                cond_results[cond]["entropy_l3"].append(ent_l3)
                cond_results[cond]["entropy_all_layers"].append(ent_all)
                cond_results[cond]["fg_hit_rate"].append(fg_hit)
                cond_results[cond]["dropoff_ratio"].append(dropoff)
                cond_results[cond]["score_margin"].append(margin)
                cond_results[cond]["top1_score"].append(s_top1)
                cond_results[cond]["top300_score"].append(s_top300)
                cond_results[cond]["ref_drift"].append(drift)
                cond_results[cond]["hungarian_ref_drift"].append(h_drift)
                cond_results[cond]["top1_attn_weight"].append(a_top1_w)
                cond_results[cond]["head_entropy_max_abs_delta"].append(head_ent_max_delta)
                cond_results[cond]["head_attn_tv_shift"].append(head_tv)
                cond_results[cond]["scale_mass_s3"].append(float(a_sm[0]) if len(a_sm) > 0 else 0.0)
                cond_results[cond]["scale_mass_s4"].append(float(a_sm[1]) if len(a_sm) > 1 else 0.0)
                cond_results[cond]["scale_mass_s5"].append(float(a_sm[2]) if len(a_sm) > 2 else 0.0)

    raw_csv_dir = get_raw_csv_dir()
    raw_csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = raw_csv_dir / "attention_telemetry_per_image.csv"

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
            f"{cond}_hungarian_ref_drift",
        ])

    csv_rows: List[List[Any]] = []
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
                f"{cond_results[cond]['hungarian_ref_drift'][i]:.4f}",
            ])
        csv_rows.append(row)

    write_csv_rows(csv_path, header, csv_rows)
    print(f">>> Đã ghi nhận số liệu thô per-image: {csv_path}")

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

    t_stat_ent, p_val_ent = float("nan"), float("nan")
    t_stat_hit, p_val_hit = float("nan"), float("nan")
    w_stat_ent, p_val_w_ent = float("nan"), float("nan")
    w_stat_hit, p_val_w_hit = float("nan"), float("nan")
    d_ent, d_hit = float("nan"), float("nan")

    target_atk = "aqsm" if "aqsm" in adv_imgs_dict else list(adv_imgs_dict.keys())[0] if adv_imgs_dict else None
    if target_atk is not None and n_eval >= 2:
        diff_ent = np.array(cond_results[target_atk]["entropy_l3"]) - np.array(cond_results["clean"]["entropy_l3"])
        diff_hit = np.array(cond_results[target_atk]["fg_hit_rate"]) - np.array(cond_results["clean"]["fg_hit_rate"])

        if np.std(diff_ent, ddof=1) > 0:
            t_stat_ent, p_val_ent = stats.ttest_rel(
                cond_results["clean"]["entropy_l3"],
                cond_results[target_atk]["entropy_l3"],
            )
        else:
            t_stat_ent, p_val_ent = 0.0, 1.0
        if np.std(diff_hit, ddof=1) > 0:
            t_stat_hit, p_val_hit = stats.ttest_rel(
                cond_results["clean"]["fg_hit_rate"],
                cond_results[target_atk]["fg_hit_rate"],
            )
        else:
            t_stat_hit, p_val_hit = 0.0, 1.0
        if np.any(diff_ent != 0):
            w_stat_ent, p_val_w_ent = stats.wilcoxon(diff_ent)
        if np.any(diff_hit != 0):
            w_stat_hit, p_val_w_hit = stats.wilcoxon(diff_hit)
        d_ent = float(np.mean(diff_ent) / (np.std(diff_ent, ddof=1) + 1e-12))
        d_hit = float(np.mean(diff_hit) / (np.std(diff_hit, ddof=1) + 1e-12))

    cohort_stats: Dict[str, Any] = {}
    for cond in all_conditions:
        n_c = len(cond_results[cond]["entropy_l1"])
        sqrt_n = np.sqrt(n_c) if n_c > 0 else 1.0
        ddof_c = 1 if n_c > 1 else 0
        std_el1 = float(np.std(cond_results[cond]["entropy_l1"], ddof=ddof_c)) if n_c > 0 else 0.0
        std_el2 = float(np.std(cond_results[cond]["entropy_l2"], ddof=ddof_c)) if n_c > 0 else 0.0
        std_el3 = float(np.std(cond_results[cond]["entropy_l3"], ddof=ddof_c)) if n_c > 0 else 0.0
        std_eall = float(np.std(cond_results[cond]["entropy_all_layers"], ddof=ddof_c)) if n_c > 0 else 0.0
        std_fg = float(np.std(cond_results[cond]["fg_hit_rate"], ddof=ddof_c)) if n_c > 0 else 0.0
        std_drop = float(np.std(cond_results[cond]["dropoff_ratio"], ddof=ddof_c)) if n_c > 0 else 0.0
        std_margin = float(np.std(cond_results[cond]["score_margin"], ddof=ddof_c)) if n_c > 0 else 0.0
        std_drift = float(np.std(cond_results[cond]["ref_drift"], ddof=ddof_c)) if n_c > 0 else 0.0
        cohort_stats[cond] = {
            "mean_entropy_l1": float(np.mean(cond_results[cond]["entropy_l1"])),
            "std_entropy_l1": std_el1,
            "sem_entropy_l1": float(std_el1 / sqrt_n),
            "mean_entropy_l2": float(np.mean(cond_results[cond]["entropy_l2"])),
            "std_entropy_l2": std_el2,
            "sem_entropy_l2": float(std_el2 / sqrt_n),
            "mean_entropy_l3": float(np.mean(cond_results[cond]["entropy_l3"])),
            "std_entropy_l3": std_el3,
            "sem_entropy_l3": float(std_el3 / sqrt_n),
            "mean_entropy_all_layers": float(np.mean(cond_results[cond]["entropy_all_layers"])),
            "std_entropy_all_layers": std_eall,
            "sem_entropy_all_layers": float(std_eall / sqrt_n),
            "mean_fg_hit_rate": float(np.mean(cond_results[cond]["fg_hit_rate"])),
            "std_fg_hit_rate": std_fg,
            "sem_fg_hit_rate": float(std_fg / sqrt_n),
            "mean_dropoff_ratio": float(np.mean(cond_results[cond]["dropoff_ratio"])),
            "std_dropoff_ratio": std_drop,
            "mean_score_margin": float(np.mean(cond_results[cond]["score_margin"])),
            "std_score_margin": std_margin,
            "mean_top1_score": float(np.mean(cond_results[cond]["top1_score"])),
            "mean_top300_score": float(np.mean(cond_results[cond]["top300_score"])),
            "mean_ref_drift": float(np.mean(cond_results[cond]["ref_drift"])),
            "std_ref_drift": std_drift,
            "sem_ref_drift": float(std_drift / sqrt_n),
            "mean_hungarian_ref_drift": float(np.mean(cond_results[cond]["hungarian_ref_drift"])),
            "std_hungarian_ref_drift": float(np.std(cond_results[cond]["hungarian_ref_drift"], ddof=ddof_c)) if n_c > 0 else 0.0,
            "mean_top1_attn_weight": float(np.mean(cond_results[cond]["top1_attn_weight"])),
            "mean_head_entropy_max_abs_delta": float(np.mean(cond_results[cond]["head_entropy_max_abs_delta"])),
            "mean_head_attn_tv_shift": float(np.mean(cond_results[cond]["head_attn_tv_shift"])),
            "mean_scale_mass_s3": float(np.mean(cond_results[cond]["scale_mass_s3"])),
            "mean_scale_mass_s4": float(np.mean(cond_results[cond]["scale_mass_s4"])),
            "mean_scale_mass_s5": float(np.mean(cond_results[cond]["scale_mass_s5"])),
        }

    ent_tost_02 = None
    ent_tost_05 = None
    if target_atk is not None and n_eval >= 2:
        ent_tost_02 = compute_paired_tost_equivalence(
            cond_results[target_atk]["entropy_all_layers"],
            cond_results["clean"]["entropy_all_layers"],
            bound=0.02,
            alpha=0.05,
        )
        ent_tost_05 = compute_paired_tost_equivalence(
            cond_results[target_atk]["entropy_all_layers"],
            cond_results["clean"]["entropy_all_layers"],
            bound=0.05,
            alpha=0.05,
        )

    cohort_stats["hypothesis_tests"] = {
        "target_attack": target_atk,
        "entropy_invariance_t_stat": float(t_stat_ent),
        "entropy_invariance_p_value": float(p_val_ent),
        "entropy_wilcoxon_w": float(w_stat_ent),
        "entropy_wilcoxon_p": float(p_val_w_ent),
        "entropy_cohen_d": float(d_ent),
        "entropy_invariant_at_alpha_05": bool(p_val_ent > 0.05) if not np.isnan(p_val_ent) else True,
        "entropy_equivalence_tost_bound_0.02": ent_tost_02,
        "entropy_equivalence_tost_bound_0.05": ent_tost_05,
        "fg_hit_degradation_t_stat": float(t_stat_hit),
        "fg_hit_degradation_p_value": float(p_val_hit),
        "fg_hit_wilcoxon_w": float(w_stat_hit),
        "fg_hit_wilcoxon_p": float(p_val_w_hit),
        "fg_hit_cohen_d": float(d_hit),
        "fg_hit_collapsed_at_alpha_01": bool(p_val_hit < 0.01) if not np.isnan(p_val_hit) else True,
    }

    summary_json_dir = get_summary_json_dir()
    summary_json_dir.mkdir(parents=True, exist_ok=True)
    json_path = summary_json_dir / "mechanism_telemetry_results.json"
    save_json(cohort_stats, json_path)
    print(f">>> Đã lưu tổng hợp phân tích cơ chế nội quan: {json_path}")

    json_path_alias = summary_json_dir / "deformable_attention_telemetry.json"
    save_json(cohort_stats, json_path_alias)
    print(f">>> Đã lưu alias phân tích cơ chế nội quan: {json_path_alias}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return cohort_stats
