# -*- coding: utf-8 -*-
"""
Module: edge_sec_pro.experiments.step_mechanism_analysis
Mục đích: Thực thi bộ phân tích cơ chế nội tại toàn diện (Mục 3, 4, 5 và 6) trên N=300 ảnh COCO val2017:
  - Mục 3: Phân tích Encoder Memory (Cosine Similarity, Relative L2 theo scale S3/S4/S5 & Foreground vs Background).
  - Mục 4: Phân tích Query Selection (Top-K Overlap, Spearman/Kendall, Object Coverage Rate, kiểm chứng Proposal Starvation).
  - Mục 5: Phân tích Deformable Attention (Sampling Locations, Reference Points, Attention Entropy, FG Hit Rate,
           kiểm chứng Attention Scattering định lượng).
  - Mục 6: Đối chiếu đa tầng dưới cùng threat model & ngân sách L_inf giữa AQSM, PGD Vanishing, Standard Task PGD
           (và đối chứng tự nhiên Gaussian Noise sigma=16) tại eps=16/255 (N=300) và eps in {4/255, 8/255, 16/255} (N=100).
"""

import argparse
import functools
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
from scipy import stats
import torch
from tqdm import tqdm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

print = functools.partial(print, flush=True)

WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from edge_sec_pro.core.coco_gt import (
    bootstrap_paired_diff,
    compute_per_image_coco_aps,
    evaluate_coco_predictions,
    load_coco_val2017_gt_cohort,
    postprocess_rtdetr_predictions,
)
from edge_sec_pro.core.config import (
    CHECKPOINTS_DIR,
    PGD_NUM_STEPS,
    PGD_STEP_SIZE,
    RESULTS_DIR,
)
from edge_sec_pro.core.dataset import apply_natural_corruption
from edge_sec_pro.core.metrics import (
    calculate_task_level_asr,
    clopper_pearson_interval,
    compute_auroc_and_bootstrap_ci,
    make_serializable,
)
from edge_sec_pro.experiments.step_generate_adv_cache import (
    cache_path,
    generate_single_cache,
    load_cached_adv_images,
)
from edge_sec_pro.transformer_monitors.detr_wrapper import RTDETRModelWrapper
from edge_sec_pro.transformer_monitors.rtdetr_advanced_features import RTDETRDeepFeatureExtractor
from edge_sec_pro.transformer_monitors.rtdetr_mechanism_analyzer import (
    SCALE_NAMES,
    analyze_deformable_attention_batch,
    analyze_encoder_memory_batch,
    analyze_query_selection_batch,
    extract_encoder_components,
)


def _summarize_scalar(arr: Sequence[float], n_boot: int = 2000, seed: int = 42) -> Dict[str, Any]:
    a = np.asarray(arr, dtype=np.float64)
    a = a[~np.isnan(a)]
    if len(a) == 0:
        return {"n": 0, "mean": 0.0, "std": 0.0, "median": 0.0, "q25": 0.0, "q75": 0.0, "ci_95": [0.0, 0.0]}
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(a), size=(n_boot, len(a)))
    ci = np.percentile(a[idx].mean(axis=1), [2.5, 97.5]).tolist()
    return {
        "n": int(len(a)),
        "mean": float(np.mean(a)),
        "std": float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
        "median": float(np.median(a)),
        "q25": float(np.percentile(a, 25)),
        "q75": float(np.percentile(a, 75)),
        "ci_95": [float(ci[0]), float(ci[1])],
    }


def _aggregate_encoder_memory(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "overall": {
            k: _summarize_scalar([r[k] for r in records])
            for k in [
                "all_cosine", "all_rel_l2", "all_rel_frob",
                "fg_cosine", "fg_rel_l2", "fg_rel_frob",
                "bg_cosine", "bg_rel_l2", "bg_rel_frob",
            ]
        },
        "fg_vs_bg_paired": {
            "cosine_fg_minus_bg": bootstrap_paired_diff(
                np.array([r["fg_cosine"] for r in records]),
                np.array([r["bg_cosine"] for r in records]),
            ),
            "rel_l2_fg_minus_bg": bootstrap_paired_diff(
                np.array([r["fg_rel_l2"] for r in records]),
                np.array([r["bg_rel_l2"] for r in records]),
            ),
        },
        "per_scale": {},
    }
    fg_mean = max(out["overall"]["fg_rel_l2"]["mean"], 0.0)
    bg_mean = max(out["overall"]["bg_rel_l2"]["mean"], 1e-8)
    out["fg_vs_bg_paired"]["rel_l2_ratio_fg_over_bg"] = float(fg_mean / bg_mean)

    for s_name in SCALE_NAMES:
        out["per_scale"][s_name] = {
            m_k: _summarize_scalar([r["per_scale"][s_name][m_k] for r in records if s_name in r["per_scale"]])
            for m_k in [
                "cosine", "rel_l2", "rel_frob",
                "fg_cosine", "fg_rel_l2", "bg_cosine", "bg_rel_l2",
                "pre_proj_cosine", "pre_proj_rel_l2",
            ]
        }
    return out


def _aggregate_query_selection(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = [
        "top300_overlap", "top300_jaccard", "top100_overlap", "top50_overlap",
        "spearman_2100", "kendall_2100", "spearman_fg_tokens", "median_rank_shift_fg_tokens",
        "evicted_top15_ratio", "median_rank_shift_top15",
        "fg_query_ratio_clean_top300", "fg_query_ratio_adv_top300",
        "cov_center_clean", "cov_center_adv",
        "cov_iou50_clean", "cov_iou50_adv",
        "cov_iou30_clean", "cov_iou30_adv",
        "clean_s1", "clean_s50", "clean_s300", "clean_dropoff", "clean_enc_margin",
        "adv_s1", "adv_s50", "adv_s300", "adv_dropoff", "adv_enc_margin",
    ]
    summary = {k: _summarize_scalar([r[k] for r in records]) for k in keys}
    summary["paired_deltas"] = {
        "cov_center_adv_minus_clean": bootstrap_paired_diff(
            np.array([r["cov_center_adv"] for r in records]),
            np.array([r["cov_center_clean"] for r in records]),
        ),
        "cov_iou50_adv_minus_clean": bootstrap_paired_diff(
            np.array([r["cov_iou50_adv"] for r in records]),
            np.array([r["cov_iou50_clean"] for r in records]),
        ),
        "cov_iou30_adv_minus_clean": bootstrap_paired_diff(
            np.array([r["cov_iou30_adv"] for r in records]),
            np.array([r["cov_iou30_clean"] for r in records]),
        ),
        "fg_query_ratio_adv_minus_clean": bootstrap_paired_diff(
            np.array([r["fg_query_ratio_adv_top300"] for r in records]),
            np.array([r["fg_query_ratio_clean_top300"] for r in records]),
        ),
    }
    return summary


def _aggregate_deformable_attention(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    clean_keys = [
        "n_active_queries", "mean_entropy_norm", "mean_top1_weight",
        "mean_dispersion_radius", "mean_fg_spatial_hit",
        "mean_fg_spatial_hit_active_q", "mean_oob_rate",
        "mean_scale_mass_s3", "mean_scale_mass_s4", "mean_scale_mass_s5",
    ]
    delta_keys = [
        "delta_entropy_norm", "delta_top1_weight", "delta_dispersion_radius",
        "delta_fg_spatial_hit", "init_ref_xy_shift_all", "init_ref_xy_shift_fg_slots",
        "mean_ref_xy_shift", "mean_sampling_loc_shift", "mean_sampling_loc_shift_fg_slots",
        "mean_sampling_offset_shift", "mean_attn_weight_tv_shift",
    ]
    out: Dict[str, Any] = {
        "clean_distribution": {k: _summarize_scalar([r["clean"][k] for r in records]) for k in clean_keys},
        "adv_distribution": {k: _summarize_scalar([r["adv"][k] for r in records]) for k in clean_keys},
        "deltas_and_shifts": {k: _summarize_scalar([r[k] for r in records]) for k in delta_keys},
        "paired_tests": {
            "entropy_adv_minus_clean": bootstrap_paired_diff(
                np.array([r["adv"]["mean_entropy_norm"] for r in records]),
                np.array([r["clean"]["mean_entropy_norm"] for r in records]),
            ),
            "dispersion_adv_minus_clean": bootstrap_paired_diff(
                np.array([r["adv"]["mean_dispersion_radius"] for r in records]),
                np.array([r["clean"]["mean_dispersion_radius"] for r in records]),
            ),
            "fg_hit_adv_minus_clean": bootstrap_paired_diff(
                np.array([r["adv"]["mean_fg_spatial_hit"] for r in records]),
                np.array([r["clean"]["mean_fg_spatial_hit"] for r in records]),
            ),
            "top1_weight_adv_minus_clean": bootstrap_paired_diff(
                np.array([r["adv"]["mean_top1_weight"] for r in records]),
                np.array([r["clean"]["mean_top1_weight"] for r in records]),
            ),
        },
        "per_layer": {},
    }
    for l_idx, l_name in enumerate(["L1", "L2", "L3"]):
        out["per_layer"][l_name] = {
            "clean_entropy_norm": _summarize_scalar([r["clean"]["per_layer"][l_name]["entropy_norm"] for r in records if l_name in r["clean"]["per_layer"]]),
            "adv_entropy_norm": _summarize_scalar([r["adv"]["per_layer"][l_name]["entropy_norm"] for r in records if l_name in r["adv"]["per_layer"]]),
            "clean_fg_hit": _summarize_scalar([r["clean"]["per_layer"][l_name]["fg_spatial_hit_all"] for r in records if l_name in r["clean"]["per_layer"]]),
            "adv_fg_hit": _summarize_scalar([r["adv"]["per_layer"][l_name]["fg_spatial_hit_all"] for r in records if l_name in r["adv"]["per_layer"]]),
            "clean_dispersion": _summarize_scalar([r["clean"]["per_layer"][l_name]["dispersion_radius"] for r in records if l_name in r["clean"]["per_layer"]]),
            "adv_dispersion": _summarize_scalar([r["adv"]["per_layer"][l_name]["dispersion_radius"] for r in records if l_name in r["adv"]["per_layer"]]),
            "ref_xy_shift": _summarize_scalar([r["layer_ref_xy_shift"][l_idx] for r in records]),
            "sampling_loc_shift": _summarize_scalar([r["layer_sampling_loc_shift"][l_idx] for r in records]),
            "sampling_offset_shift": _summarize_scalar([r["layer_sampling_offset_shift"][l_idx] for r in records]),
        }
    return out


def _evaluate_monitor_and_coco_metrics(
    wrapper: RTDETRModelWrapper,
    extractor: RTDETRDeepFeatureExtractor,
    detectors: Dict[str, Any],
    cal_thresholds: Dict[str, float],
    clean_test_scores: Dict[str, np.ndarray],
    clean_imgs: torch.Tensor,
    adv_imgs: torch.Tensor,
    gt_list: List[Dict[str, Any]],
    clean_logits_all: torch.Tensor,
    condition_key: str,
    batch_size: int = 10,
) -> Dict[str, Any]:
    """Đánh giá song song COCO mAP thật, Task ASR và TPR @ 5% FPR trên bộ giám sát đã calibrate."""
    feats_adv = []
    adv_logits_l: List[torch.Tensor] = []
    preds_adv: List[Dict[str, np.ndarray]] = []

    with torch.no_grad():
        for start in range(0, len(adv_imgs), batch_size):
            sub_x = adv_imgs[start : start + batch_size].to(wrapper.device)
            st = wrapper(sub_x)
            feats_adv.extend(extractor.extract_from_state(st))
            adv_logits_l.append(st.logits.cpu())
            preds_adv.extend(
                postprocess_rtdetr_predictions(st.logits, st.pred_boxes, mode="hf_multilabel", max_dets=100)
            )
            del st

    adv_logits = torch.cat(adv_logits_l, dim=0)
    coco_m = evaluate_coco_predictions(preds_adv, gt_list)
    ap50_img, _ = compute_per_image_coco_aps(preds_adv, gt_list)

    # ASR theo loại tấn công (nếu là attack)
    asr_stats: Dict[str, Any] = {}
    if condition_key in ["aqsm", "vanishing", "taskpgd"]:
        if condition_key == "aqsm":
            asr_val, succ, tot, _ = calculate_task_level_asr("AQSM", clean_logits_all, adv_logits)
            asr_stats["slot_asr"] = {"asr_pct": asr_val * 100.0, "succ": succ, "total": tot, "ci_95": [v * 100 for v in clopper_pearson_interval(succ, tot)]}
        asr_van, succ_v, tot_v, _ = calculate_task_level_asr("Vanishing", clean_logits_all, adv_logits)
        asr_stats["vanishing_maxlt020_asr"] = {"asr_pct": asr_van * 100.0, "succ": succ_v, "total": tot_v, "ci_95": [v * 100 for v in clopper_pearson_interval(succ_v, tot_v)]}

    # Monitor TPR & AUROC
    mon_metrics: Dict[str, Any] = {}
    raw_scores_dict: Dict[str, np.ndarray] = {}
    for ch_name, det in detectors.items():
        sc_a = det.score(feats_adv)
        raw_scores_dict[ch_name] = np.asarray(sc_a, dtype=np.float64)
        sc_c = clean_test_scores[ch_name]
        y_bin = np.concatenate([np.zeros(len(sc_c)), np.ones(len(sc_a))])
        s_all = np.concatenate([sc_c, sc_a])
        auroc, auroc_ci = compute_auroc_and_bootstrap_ci(y_bin, s_all, n_boot=300)
        th = float(cal_thresholds[ch_name])
        tp_cnt = int(np.sum(np.asarray(sc_a) >= th))
        n_s = len(sc_a)
        tpr_ci = clopper_pearson_interval(tp_cnt, n_s)
        mon_metrics[ch_name] = {
            "auroc": float(auroc),
            "auroc_ci": [float(auroc_ci[0]), float(auroc_ci[1])],
            "tpr_pct": float(tp_cnt / max(n_s, 1) * 100.0),
            "tp_count": tp_cnt,
            "n_samples": n_s,
            "tpr_ci_95": [float(tpr_ci[0] * 100.0), float(tpr_ci[1] * 100.0)],
        }

    return {
        "coco_map": coco_m,
        "per_image_ap50_valid_mean": float(np.nanmean(ap50_img)),
        "asr": asr_stats,
        "monitor_detection": mon_metrics,
        "adv_logits_cpu": adv_logits,
        "raw_scores": raw_scores_dict,
    }


def run_mechanism_analysis(
    n_eval: int = 300,
    eps_main: int = 16,
    sweep_eps: Sequence[int] = (4, 8, 16),
    n_sweep: int = 100,
    batch_size: int = 10,
    threads: int = 4,
) -> Dict[str, Any]:
    print("=" * 92)
    print(f"PHÂN TÍCH CƠ CHẾ NỘI TẠI RT-DETR (MỤC 3, 4, 5 & 6) | N_main={n_eval}, N_sweep={n_sweep}")
    print("=" * 92)

    torch.set_num_threads(threads)
    device = torch.device("cpu")
    wrapper = RTDETRModelWrapper(pretrained_model_name="PekingU/rtdetr_r18vd", device=device)
    wrapper.model.eval()
    extractor = RTDETRDeepFeatureExtractor(conf_threshold=0.35, top_k_queries=10, projection_dim=64)

    ckpt_file = CHECKPOINTS_DIR / "calibration_and_detectors.pkl"
    with open(ckpt_file, "rb") as f:
        ckpt = pickle.load(f)
    detectors = ckpt["detectors"]
    cal_thresholds = ckpt["cal_thresholds"]
    clean_test_scores = {k: np.asarray(v) for k, v in ckpt["clean_test_scores"].items()}

    cohort = load_coco_val2017_gt_cohort(
        n_cal=1200,
        n_test=600,
        eval_start=0,
        n_eval=n_eval,
        seed=42,
        id2label=wrapper.model.config.id2label,
        load_tensors=True,
    )
    clean_imgs: torch.Tensor = cohort["tensors"]
    gt_list: List[Dict[str, Any]] = cohort["gt_list"]
    image_ids: List[int] = cohort["image_ids"]
    gt_boxes_noncrowd = [g["boxes_xyxy"][~g["iscrowd"]] for g in gt_list]

    # 1. Clean baseline pass
    print(f"\n[1/4] Clean Baseline Pass trên N={n_eval} ảnh COCO GT thật...")
    clean_logits_l: List[torch.Tensor] = []
    clean_preds: List[Dict[str, np.ndarray]] = []
    with torch.no_grad():
        for start in range(0, n_eval, batch_size):
            st = wrapper(clean_imgs[start : start + batch_size].to(device))
            clean_logits_l.append(st.logits.cpu())
            clean_preds.extend(postprocess_rtdetr_predictions(st.logits, st.pred_boxes, mode="hf_multilabel", max_dets=100))
    clean_logits_all = torch.cat(clean_logits_l, dim=0)
    clean_coco_map = evaluate_coco_predictions(clean_preds, gt_list)
    clean_ap50_img, _ = compute_per_image_coco_aps(clean_preds, gt_list)
    print(
        f"  -> Clean Baseline (COCO val2017 GT, 320x320, N={n_eval}): "
        f"mAP@50 = {clean_coco_map['mAP@50']:.2f}%, mAP@50:95 = {clean_coco_map['mAP@50:95']:.2f}%"
    )

    # 2. Tạo ảnh Gaussian Noise control (cố định seed=42) + tải các nhánh attack từ cache N=300
    torch.manual_seed(42)
    noise_imgs = apply_natural_corruption(clean_imgs, "gaussian_noise", 16.0 / 255.0)

    main_conditions: Dict[str, Dict[str, Any]] = {
        "gaussian_noise_ctrl": {"imgs": noise_imgs, "meta": {"attack": "gaussian_noise", "sigma": 16.0 / 255.0}},
    }
    for atk in ["aqsm", "vanishing", "taskpgd"]:
        if not cache_path(atk, eps_main, n_eval).exists():
            generate_single_cache(
                attack=atk,
                eps_num=eps_main,
                n_eval=n_eval,
                batch_size=batch_size,
                steps=PGD_NUM_STEPS,
                alpha=PGD_STEP_SIZE,
                threads=threads,
            )
        adv_x, meta = load_cached_adv_images(atk, eps_main, n_eval, clean_imgs, image_ids)
        main_conditions[atk] = {"imgs": adv_x, "meta": meta}

    # 3. Phân tích chi tiết Mục 3, 4, 5, 6 trên N=300
    print(f"\n[2/4] Chạy phân tích Mục 3 (Encoder), Mục 4 (Query Selection), Mục 5 (Attention) trên N={n_eval}...")
    main_analysis: Dict[str, Any] = {}
    npz_export: Dict[str, np.ndarray] = {
        "clean__logits_max": clean_logits_all.max(dim=-1).values.numpy().astype(np.float32)
    }

    for cond_name, cond_obj in main_conditions.items():
        t_c = time.time()
        adv_x = cond_obj["imgs"]
        enc_recs: List[Dict[str, Any]] = []
        qs_recs: List[Dict[str, Any]] = []
        attn_recs: List[Dict[str, Any]] = []

        for start in tqdm(range(0, n_eval, batch_size), desc=f"Analyze {cond_name}"):
            end = min(n_eval, start + batch_size)
            c_sub = clean_imgs[start:end].to(device)
            a_sub = adv_x[start:end].to(device)
            gt_sub = gt_boxes_noncrowd[start:end]

            c_comp = extract_encoder_components(wrapper.model, c_sub)
            a_comp = extract_encoder_components(wrapper.model, a_sub)
            enc_recs.extend(analyze_encoder_memory_batch(c_comp, a_comp, gt_sub))
            qs_recs.extend(analyze_query_selection_batch(c_comp, a_comp, gt_sub))
            del c_comp, a_comp

            attn_recs.extend(analyze_deformable_attention_batch(wrapper.model, c_sub, a_sub, gt_sub))

        eval_res = _evaluate_monitor_and_coco_metrics(
            wrapper=wrapper,
            extractor=extractor,
            detectors=detectors,
            cal_thresholds=cal_thresholds,
            clean_test_scores=clean_test_scores,
            clean_imgs=clean_imgs,
            adv_imgs=adv_x,
            gt_list=gt_list,
            clean_logits_all=clean_logits_all,
            condition_key=cond_name,
            batch_size=batch_size,
        )

        # Thống kê thêm tương quan theo cặp ảnh giữa Query Selection / Encoder Memory và hiệu năng
        rel_drop_map = ((eval_res["coco_map"]["mAP@50"] - clean_coco_map["mAP@50"]) / max(clean_coco_map["mAP@50"], 1e-6)) * 100.0
        enc_agg = _aggregate_encoder_memory(enc_recs)
        qs_agg = _aggregate_query_selection(qs_recs)
        attn_agg = _aggregate_deformable_attention(attn_recs)

        main_analysis[cond_name] = {
            "meta": cond_obj["meta"],
            "elapsed_sec": float(time.time() - t_c),
            "coco_map": eval_res["coco_map"],
            "relative_drop_map50_pct": float(rel_drop_map),
            "asr": eval_res["asr"],
            "monitor_detection": eval_res["monitor_detection"],
            "encoder_memory": enc_agg,
            "query_selection": qs_agg,
            "deformable_attention": attn_agg,
        }

        # Lưu vector per-image thô vào NPZ phục vụ biểu đồ thật 100%
        npz_export[f"{cond_name}__logits_max"] = eval_res["adv_logits_cpu"].max(dim=-1).values.numpy().astype(np.float32)
        npz_export[f"{cond_name}__all_cosine"] = np.array([r["all_cosine"] for r in enc_recs], dtype=np.float32)
        npz_export[f"{cond_name}__all_rel_l2"] = np.array([r["all_rel_l2"] for r in enc_recs], dtype=np.float32)
        npz_export[f"{cond_name}__fg_rel_l2"] = np.array([r["fg_rel_l2"] for r in enc_recs], dtype=np.float32)
        npz_export[f"{cond_name}__bg_rel_l2"] = np.array([r["bg_rel_l2"] for r in enc_recs], dtype=np.float32)
        npz_export[f"{cond_name}__top300_overlap_qs"] = np.array([r["top300_overlap"] for r in qs_recs], dtype=np.float32)
        npz_export[f"{cond_name}__cov_iou50_clean"] = np.array([r["cov_iou50_clean"] for r in qs_recs], dtype=np.float32)
        npz_export[f"{cond_name}__cov_iou50_adv"] = np.array([r["cov_iou50_adv"] for r in qs_recs], dtype=np.float32)
        npz_export[f"{cond_name}__cov_center_adv"] = np.array([r["cov_center_adv"] for r in qs_recs], dtype=np.float32)
        npz_export[f"{cond_name}__attn_entropy_clean"] = np.array([r["clean"]["mean_entropy_norm"] for r in attn_recs], dtype=np.float32)
        npz_export[f"{cond_name}__attn_entropy_adv"] = np.array([r["adv"]["mean_entropy_norm"] for r in attn_recs], dtype=np.float32)
        npz_export[f"{cond_name}__attn_fg_hit_clean"] = np.array([r["clean"]["mean_fg_spatial_hit"] for r in attn_recs], dtype=np.float32)
        npz_export[f"{cond_name}__attn_fg_hit_adv"] = np.array([r["adv"]["mean_fg_spatial_hit"] for r in attn_recs], dtype=np.float32)
        npz_export[f"{cond_name}__attn_loc_shift"] = np.array([r["mean_sampling_loc_shift"] for r in attn_recs], dtype=np.float32)

        print(
            f"  [{cond_name}] mAP@50={eval_res['coco_map']['mAP@50']:.2f}% ({rel_drop_map:+.1f}%) | "
            f"Mem cos={enc_agg['overall']['all_cosine']['mean']:.3f}, relL2 FG/BG={enc_agg['overall']['fg_rel_l2']['mean']:.3f}/{enc_agg['overall']['bg_rel_l2']['mean']:.3f} | "
            f"Top300 ovl={qs_agg['top300_overlap']['mean']*100:.1f}%, Cov@50={qs_agg['cov_iou50_clean']['mean']*100:.1f}%->{qs_agg['cov_iou50_adv']['mean']*100:.1f}% | "
            f"Attn Ent={attn_agg['clean_distribution']['mean_entropy_norm']['mean']:.3f}->{attn_agg['adv_distribution']['mean_entropy_norm']['mean']:.3f}, "
            f"FGHit={attn_agg['clean_distribution']['mean_fg_spatial_hit']['mean']*100:.1f}%->{attn_agg['adv_distribution']['mean_fg_spatial_hit']['mean']*100:.1f}%"
        )

    # 4. Đối chiếu ngân sách nhiễu matched (eps in {4/255, 8/255, 16/255}) trên N=n_sweep ảnh đầu
    print(f"\n[3/4] Đối chiếu đa ngân sách eps={list(sweep_eps)}/255 trên N={n_sweep} ảnh (Mục 6: AQSM vs Vanishing vs Task PGD)...")
    sweep_results: Dict[str, Dict[str, Any]] = {}
    sub_clean = clean_imgs[:n_sweep]
    sub_gt = gt_list[:n_sweep]
    sub_gt_nc = gt_boxes_noncrowd[:n_sweep]
    sub_ids = image_ids[:n_sweep]
    sub_clean_logits = clean_logits_all[:n_sweep]
    sub_clean_map = evaluate_coco_predictions(clean_preds[:n_sweep], sub_gt)

    for eps_n in sweep_eps:
        eps_key = f"{eps_n}/255"
        sweep_results[eps_key] = {}
        for atk in ["aqsm", "vanishing", "taskpgd"]:
            if eps_n == eps_main and cache_path(atk, eps_main, n_eval).exists():
                full_adv, meta = load_cached_adv_images(atk, eps_main, n_eval, clean_imgs, image_ids)
                adv_sub = full_adv[:n_sweep]
            else:
                if not cache_path(atk, eps_n, n_sweep).exists():
                    generate_single_cache(
                        attack=atk,
                        eps_num=eps_n,
                        n_eval=n_sweep,
                        batch_size=batch_size,
                        steps=PGD_NUM_STEPS,
                        alpha=PGD_STEP_SIZE,
                        threads=threads,
                    )
                adv_sub, meta = load_cached_adv_images(atk, eps_n, n_sweep, sub_clean, sub_ids)

            # Đo nhanh các chỉ số cốt lõi trên N=n_sweep
            qs_s: List[Dict[str, Any]] = []
            enc_s: List[Dict[str, Any]] = []
            for st_i in range(0, n_sweep, batch_size):
                ed_i = min(n_sweep, st_i + batch_size)
                c_c = extract_encoder_components(wrapper.model, sub_clean[st_i:ed_i].to(device))
                a_c = extract_encoder_components(wrapper.model, adv_sub[st_i:ed_i].to(device))
                enc_s.extend(analyze_encoder_memory_batch(c_c, a_c, sub_gt_nc[st_i:ed_i]))
                qs_s.extend(analyze_query_selection_batch(c_c, a_c, sub_gt_nc[st_i:ed_i]))
                del c_c, a_c

            ev = _evaluate_monitor_and_coco_metrics(
                wrapper=wrapper,
                extractor=extractor,
                detectors=detectors,
                cal_thresholds=cal_thresholds,
                clean_test_scores=clean_test_scores,
                clean_imgs=sub_clean,
                adv_imgs=adv_sub,
                gt_list=sub_gt,
                clean_logits_all=sub_clean_logits,
                condition_key=atk,
                batch_size=batch_size,
            )
            sweep_results[eps_key][atk] = {
                "meta": meta,
                "coco_map": ev["coco_map"],
                "relative_retention_50_pct": float(ev["coco_map"]["mAP@50"] / max(sub_clean_map["mAP@50"], 1e-6) * 100.0),
                "top300_overlap_mean": float(np.mean([r["top300_overlap"] for r in qs_s])),
                "cov_iou50_adv_mean": float(np.nanmean([r["cov_iou50_adv"] for r in qs_s])),
                "enc_all_cosine_mean": float(np.mean([r["all_cosine"] for r in enc_s])),
                "enc_fg_rel_l2_mean": float(np.nanmean([r["fg_rel_l2"] for r in enc_s])),
                "monitor_tpr": {
                    ch: ev["monitor_detection"][ch]["tpr_pct"]
                    for ch in ev["monitor_detection"]
                },
            }
            print(
                f"  [Sweep {eps_key} | {atk:<9}] mAP@50={ev['coco_map']['mAP@50']:.2f}% "
                f"(Ret={sweep_results[eps_key][atk]['relative_retention_50_pct']:.1f}%) | "
                f"Cos={sweep_results[eps_key][atk]['enc_all_cosine_mean']:.3f} | "
                f"Top300Ovl={sweep_results[eps_key][atk]['top300_overlap_mean']*100:.1f}% | "
                f"MaxOR={ev['monitor_detection']['Calibrated_Max_OR']['tpr_pct']:.1f}% "
                f"ESM={ev['monitor_detection']['ESM_Dropoff']['tpr_pct']:.1f}%"
            )

    # 5. So sánh trực tiếp AQSM vs Task PGD & AQSM vs Vanishing trên N=300 (paired Wilcoxon)
    print(f"\n[4/4] Tổng hợp kiểm định so sánh trực tiếp cặp đôi AQSM vs TaskPGD vs Vanishing (Mục 6)...")
    pairwise_m6: Dict[str, Any] = {}
    compare_metrics = [
        ("all_cosine", "Encoder Cosine Similarity"),
        ("fg_rel_l2", "Encoder FG Relative L2"),
        ("bg_rel_l2", "Encoder BG Relative L2"),
        ("top300_overlap_qs", "Top-300 Selection Overlap"),
        ("cov_iou50_adv", "GT Object Coverage (IoU>=0.5)"),
        ("attn_entropy_adv", "Deformable Attention Normalized Entropy"),
        ("attn_fg_hit_adv", "Deformable Attention FG Spatial Hit"),
        ("attn_loc_shift", "Deformable Attention Sampling Loc Shift"),
    ]
    for target_other in ["taskpgd", "vanishing"]:
        comp_key = f"aqsm_vs_{target_other}"
        pairwise_m6[comp_key] = {}
        for npz_k, desc in compare_metrics:
            v_aqsm = npz_export[f"aqsm__{npz_k}"]
            v_oth = npz_export[f"{target_other}__{npz_k}"]
            pairwise_m6[comp_key][npz_k] = {
                "description": desc,
                **bootstrap_paired_diff(v_aqsm, v_oth),
            }

    # Export JSON + NPZ
    payload = {
        "protocol": {
            "model": "PekingU/rtdetr_r18vd",
            "resolution": 320,
            "threat_model": f"L_inf PGD (T={PGD_NUM_STEPS}, alpha=2/255, eps={eps_main}/255)",
            "n_main": int(n_eval),
            "n_sweep": int(n_sweep),
            "ground_truth": "Official COCO val2017 instances_val2017.json",
        },
        "clean_baseline": {
            "n_eval": int(n_eval),
            "coco_map": clean_coco_map,
            "per_image_ap50_valid_mean": float(np.nanmean(clean_ap50_img)),
            "n_valid_gt_images": int(np.sum(~np.isnan(clean_ap50_img))),
        },
        "sweep_clean_baseline": sub_clean_map,
        "main_n300_analysis": main_analysis,
        "epsilon_budget_comparison": sweep_results,
        "pairwise_mechanism_comparison_m6": pairwise_m6,
    }

    out_json = RESULTS_DIR / "rtdetr_mechanism_analysis_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(make_serializable(payload), f, indent=2)

    out_npz = CHECKPOINTS_DIR / "raw_mechanism_analysis_n300.npz"
    np.savez_compressed(out_npz, **npz_export)

    print(f"\n[OK] Hoàn tất phân tích cơ chế! Đã lưu JSON: {out_json} & NPZ: {out_npz}")
    return payload


def main():
    parser = argparse.ArgumentParser(description="Mechanism Analysis: Encoder Memory, Query Selection, Deformable Attention & Task PGD Comparison")
    parser.add_argument("--n-eval", type=int, default=300)
    parser.add_argument("--eps-main", type=int, default=16)
    parser.add_argument("--sweep-eps", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--n-sweep", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    run_mechanism_analysis(
        n_eval=args.n_eval,
        eps_main=args.eps_main,
        sweep_eps=args.sweep_eps,
        n_sweep=args.n_sweep,
        batch_size=args.batch_size,
        threads=args.threads,
    )


if __name__ == "__main__":
    main()
