# -*- coding: utf-8 -*-
"""
Module: paper_1_aqsm_mechanism.experiments.tasks.task10_multiseed_stability
Mục đích: Đo đạc tự động tính ổn định đa hạt giống (Multi-Seed AQSM Stability)
trên các biến thể RT-DETR (r18 / r50) và cập nhật động tệp
outputs/summary_json/multimodel_and_seed_stability_summary.json.

Đặc tính chuẩn công nghiệp:
  - Chạy AQSM với nhiều random seeds (mặc định: 42, 100, 2024, 777, 9999)
    điều khiển bước chọn lọc anchor nền xa C_bg (hạng > 500).
  - Đo thực tế Clean mAP@50, AQSM mAP@50, Hungarian Set Drift, và Anchor FG Hit %.
  - Cập nhật động nhánh dữ liệu đúng cho kiến trúc đang đánh giá trong
    multimodel_and_seed_stability_summary.json (không ghi đè cứng các kiến trúc khác).
"""

import csv
import json
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import torch
from tqdm import tqdm

WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from paper_1_aqsm_mechanism.core.attack_generator import DETRAttackGenerator
from paper_1_aqsm_mechanism.core.coco_gt import (
    compute_per_image_coco_aps,
    evaluate_coco_predictions,
    postprocess_rtdetr_predictions,
)
from paper_1_aqsm_mechanism.core.dataset_loader import FlexibleCOCODataset
from paper_1_aqsm_mechanism.core.detr_wrapper import RTDETRModelWrapper
from paper_1_aqsm_mechanism.core.mechanism_analyzer import extract_encoder_components
from paper_1_aqsm_mechanism.experiments.config import (
    DEFAULT_MULTI_SEEDS,
    MODEL_NAME,
    configure_cpu_threads,
    detect_hardware_environment,
    get_device,
    get_epsilon_l_inf,
    get_img_size,
    get_model_tag,
    get_pgd_num_steps,
    get_pgd_step_size,
    get_raw_csv_dir,
    get_summary_json_dir,
    resolve_model_name,
    update_experiment_paths,
)
from paper_1_aqsm_mechanism.utils.cli import create_base_parser
from paper_1_aqsm_mechanism.utils.io_helpers import save_json
from paper_1_aqsm_mechanism.utils.stats import (
    compute_inter_seed_consistency,
    evaluate_multiseed_statistical_evidence,
    summarize_across_seed_stability,
)


def _compute_anchor_fg_hit(ref_xywh: np.ndarray, gt_xyxy: np.ndarray) -> float:
    """Tính tỷ lệ % tâm 300 anchor rơi vào trong bất kỳ GT bounding box nào."""
    if len(ref_xywh) == 0 or len(gt_xyxy) == 0:
        return 0.0
    cx = ref_xywh[:, 0:1]
    cy = ref_xywh[:, 1:2]
    inside_x = (cx >= gt_xyxy[:, 0][None, :]) & (cx <= gt_xyxy[:, 2][None, :])
    inside_y = (cy >= gt_xyxy[:, 1][None, :]) & (cy <= gt_xyxy[:, 3][None, :])
    hits = (inside_x & inside_y).any(axis=1)
    return float(np.mean(hits) * 100.0)


def run_multiseed_stability(args) -> Dict[str, Any]:
    """Thực thi đánh giá đa seed AQSM cho kiến trúc được chọn và cập nhật JSON động."""
    configure_cpu_threads(args.threads)
    device = get_device(args.device)

    chosen_model = resolve_model_name(getattr(args, "model", None) or MODEL_NAME)
    eps_255_val = int(args.eps_255) if getattr(args, "eps_255", None) is not None else int(round(get_epsilon_l_inf() * 255.0))
    pgd_steps_val = int(args.pgd_steps) if getattr(args, "pgd_steps", None) is not None else int(get_pgd_num_steps())
    update_experiment_paths(
        folder_image=args.folder_image,
        annotations=args.annotations,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        model=chosen_model,
        pgd_steps=pgd_steps_val,
        eps_l_inf=float(eps_255_val) / 255.0,
    )

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        seeds = list(DEFAULT_MULTI_SEEDS)

    print("=" * 80)
    print(f" [TASK 10] ĐÁNH GIÁ ỔN ĐỊNH ĐA HẠT GIỐNG AQSM ({chosen_model})")
    print(f" Seeds: {seeds} | Max Samples: {args.max_samples} | Device: {device}")
    print("=" * 80)

    hw_info = detect_hardware_environment(preferred_device=device.type)
    wrapper = RTDETRModelWrapper(pretrained_model_name=chosen_model, device=device)
    wrapper.eval()
    model = wrapper.model
    batch_size = args.batch_size if args.batch_size is not None else hw_info["recommended_batch_size"]
    eff_img_size = int(get_img_size())

    dataset = FlexibleCOCODataset(
        folder_image=args.folder_image,
        annotations_file=args.annotations,
        img_size=eff_img_size,
        id2label=model.config.id2label,
        lazy_tensors=True,
        device=device,
        max_samples=args.max_samples,
    )
    clean_tensors = dataset.tensors
    gt_list = dataset.gt_list
    n_eval = len(dataset)

    # 1. Thu thập Clean Baseline Predictions & Clean Reference Points
    clean_preds: List[Dict[str, Any]] = []
    clean_refs_all: List[np.ndarray] = []
    for start in tqdm(range(0, n_eval, batch_size), desc="Clean Reference"):
        end = min(start + batch_size, n_eval)
        batch_imgs = clean_tensors[start:end].to(device)
        with torch.no_grad():
            outputs = model(pixel_values=batch_imgs, return_dict=True)
            comp = extract_encoder_components(model, batch_imgs)
            refs = torch.sigmoid(comp["init_reference_points"]).cpu().numpy()
        preds = postprocess_rtdetr_predictions(
            outputs.logits,
            outputs.pred_boxes,
            score_thresh=0.01,
            mode="hf_multilabel",
        )
        clean_preds.extend(preds)
        clean_refs_all.extend([refs[b] for b in range(end - start)])

    clean_coco = evaluate_coco_predictions(clean_preds, gt_list)
    clean_map50 = round(float(clean_coco.get("mAP@50", 0.0)), 2)
    clean_ap50_vec, _ = compute_per_image_coco_aps(clean_preds, gt_list)
    clean_anchor_hits = [
        _compute_anchor_fg_hit(
            clean_refs_all[idx],
            gt_list[idx].get("boxes_xyxy", np.zeros((0, 4), dtype=np.float32)),
        )
        for idx in range(n_eval)
    ]
    clean_fg_hit_mean = float(np.mean(clean_anchor_hits)) if clean_anchor_hits else 0.0

    # 2. Đánh giá AQSM theo từng seed
    base_alpha = float(get_pgd_step_size())
    step_alpha = base_alpha if eps_255_val >= 8 else min(base_alpha, 1.0 / 255.0)
    attacker = DETRAttackGenerator(
        wrapper,
        epsilon=float(eps_255_val) / 255.0,
        alpha=step_alpha,
        num_steps=pgd_steps_val,
    )

    per_seed_metrics: Dict[int, Dict[str, float]] = {}
    seed_ap50_vecs: Dict[int, np.ndarray] = {}
    seed_drift_vecs: Dict[int, np.ndarray] = {}
    seed_hit_vecs: Dict[int, np.ndarray] = {}

    for seed in seeds:
        adv_preds: List[Dict[str, Any]] = []
        hungarian_dists: List[float] = []
        anchor_hits: List[float] = []

        for start in tqdm(range(0, n_eval, batch_size), desc=f"AQSM Seed {seed}"):
            end = min(start + batch_size, n_eval)
            batch_imgs = clean_tensors[start:end].to(device)
            adv_batch = attacker.generate_query_selection_attack(
                batch_imgs,
                steps=pgd_steps_val,
                bg_seed=seed,
            )
            with torch.no_grad():
                outputs = model(pixel_values=adv_batch, return_dict=True)
                comp = extract_encoder_components(model, adv_batch)
                adv_refs = torch.sigmoid(comp["init_reference_points"]).cpu().numpy()
            preds = postprocess_rtdetr_predictions(
                outputs.logits,
                outputs.pred_boxes,
                score_thresh=0.01,
                mode="hf_multilabel",
            )
            adv_preds.extend(preds)

            for b in range(end - start):
                idx = start + b
                c_ref = clean_refs_all[idx]
                a_ref = adv_refs[b]
                cost = cdist(c_ref[:, :2], a_ref[:, :2])
                r_idx, c_idx = linear_sum_assignment(cost)
                hungarian_dists.append(float(np.mean(cost[r_idx, c_idx])))
                gt_boxes = gt_list[idx].get("boxes_xyxy", np.zeros((0, 4), dtype=np.float32))
                anchor_hits.append(_compute_anchor_fg_hit(a_ref, gt_boxes))

        adv_coco = evaluate_coco_predictions(adv_preds, gt_list)
        adv_map50 = round(float(adv_coco.get("mAP@50", 0.0)), 2)
        adv_ap50_vec, _ = compute_per_image_coco_aps(adv_preds, gt_list)

        seed_ap50_vecs[seed] = np.asarray(adv_ap50_vec, dtype=np.float64)
        seed_drift_vecs[seed] = np.asarray(hungarian_dists, dtype=np.float64)
        seed_hit_vecs[seed] = np.asarray(anchor_hits, dtype=np.float64)

        per_seed_metrics[seed] = {
            "aqsm_map50": adv_map50,
            "hungarian_set_drift": round(float(np.mean(hungarian_dists)), 4),
            "anchor_fg_hit_pct": round(float(np.mean(anchor_hits)), 2),
        }
        print(
            f"  [Seed {seed}] AQSM mAP@50={adv_map50:.2f}% | "
            f"Hungarian={per_seed_metrics[seed]['hungarian_set_drift']:.4f} | "
            f"Anchor FG Hit={per_seed_metrics[seed]['anchor_fg_hit_pct']:.2f}%"
        )

    # 3. Tổng hợp thống kê 2 tầng (Tier 1: Mean/Std/Min-Max/SNR_seed; Tier 2: Worst-seed paired & Inter-seed ICC/Spearman)
    maps = [per_seed_metrics[s]["aqsm_map50"] for s in seeds]
    drifts = [per_seed_metrics[s]["hungarian_set_drift"] for s in seeds]
    hits = [per_seed_metrics[s]["anchor_fg_hit_pct"] for s in seeds]

    aqsm_mean = round(float(np.mean(maps)), 2)
    aqsm_std = round(float(np.std(maps, ddof=1)) if len(maps) > 1 else 0.0, 2)
    drift_mean = round(float(np.mean(drifts)), 4)
    drift_std = round(float(np.std(drifts, ddof=1)) if len(drifts) > 1 else 0.0, 4)
    hit_mean = round(float(np.mean(hits)), 2)
    hit_std = round(float(np.std(hits, ddof=1)) if len(hits) > 1 else 0.0, 2)
    rel_drop_mean = round(((aqsm_mean - clean_map50) / max(clean_map50, 1e-6)) * 100.0, 2)

    ap50_evidence = evaluate_multiseed_statistical_evidence(
        baseline_per_image=clean_ap50_vec,
        seed_per_image_map=seed_ap50_vecs,
        seed_scalar_summary={s: per_seed_metrics[s]["aqsm_map50"] for s in seeds},
        clean_scalar_baseline=clean_map50,
        n_boot=2000,
        seed=42,
    )
    fg_hit_evidence = evaluate_multiseed_statistical_evidence(
        baseline_per_image=clean_anchor_hits,
        seed_per_image_map=seed_hit_vecs,
        seed_scalar_summary={s: per_seed_metrics[s]["anchor_fg_hit_pct"] for s in seeds},
        clean_scalar_baseline=clean_fg_hit_mean,
        n_boot=2000,
        seed=42,
    )
    drift_tier1 = summarize_across_seed_stability(drifts, baseline_value=0.0)
    drift_consistency = compute_inter_seed_consistency(seed_drift_vecs)
    ap50_tier1 = ap50_evidence["tier1_scalar_stability"] or summarize_across_seed_stability(
        maps, baseline_value=clean_map50
    )
    hit_tier1 = fg_hit_evidence["tier1_scalar_stability"] or summarize_across_seed_stability(
        hits, baseline_value=clean_fg_hit_mean
    )

    arch_prefix = get_model_tag(chosen_model)
    lower_m = chosen_model.lower()
    if "r50" in lower_m:
        arch_label = "RT-DETR-R50VD (Deep)"
        backbone_desc = f"ResNet-50 ({wrapper.decoder_layers} dec.)"
    elif "r101" in lower_m:
        arch_label = "RT-DETR-R101VD"
        backbone_desc = f"ResNet-101 ({wrapper.decoder_layers} dec.)"
    elif "r18" in lower_m:
        arch_label = "RT-DETR-R18VD"
        backbone_desc = f"ResNet-18 ({wrapper.decoder_layers} dec.)"
    else:
        arch_label = chosen_model.split("/")[-1]
        backbone_desc = f"{arch_label} ({wrapper.decoder_layers} dec.)"

    # Lưu vết per-image CSV đa seed phục vụ kiểm toán độc lập
    raw_csv_dir = get_raw_csv_dir()
    out_csv = raw_csv_dir / f"multiseed_{arch_prefix}_per_image.csv"
    fieldnames = ["image_idx", "clean_ap50", "clean_anchor_fg_hit_pct"]
    for s in seeds:
        fieldnames.extend([f"seed{s}_ap50", f"seed{s}_hungarian_drift", f"seed{s}_anchor_fg_hit_pct"])
    with open(out_csv, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(n_eval):
            row: Dict[str, Any] = {
                "image_idx": idx,
                "clean_ap50": (
                    round(float(clean_ap50_vec[idx]), 4)
                    if np.isfinite(clean_ap50_vec[idx])
                    else ""
                ),
                "clean_anchor_fg_hit_pct": round(float(clean_anchor_hits[idx]), 4),
            }
            for s in seeds:
                ap_v = seed_ap50_vecs[s][idx]
                row[f"seed{s}_ap50"] = round(float(ap_v), 4) if np.isfinite(ap_v) else ""
                row[f"seed{s}_hungarian_drift"] = round(float(seed_drift_vecs[s][idx]), 6)
                row[f"seed{s}_anchor_fg_hit_pct"] = round(float(seed_hit_vecs[s][idx]), 4)
            writer.writerow(row)

    summary_dir = get_summary_json_dir()
    out_json = summary_dir / "multimodel_and_seed_stability_summary.json"
    alpha_255_str = f"{step_alpha * 255.0:.2g}/255"
    if out_json.exists():
        with open(out_json, "r", encoding="utf-8") as fp:
            payload = json.load(fp)
    else:
        payload = {
            "title": "Table 12: Multi-Seed Optimization Stability and Cross-Variant Generalization",
            "dataset": f"COCO val2017 (N={n_eval} images)",
            "resolution": f"{eff_img_size}x{eff_img_size}",
            "threat_budget": f"eps={eps_255_val}/255, T={pgd_steps_val}, alpha={alpha_255_str}",
            "records": {},
        }
    payload["dataset"] = f"COCO val2017 (N={n_eval} images)"
    payload["resolution"] = f"{eff_img_size}x{eff_img_size}"
    payload["threat_budget"] = f"eps={eps_255_val}/255, T={pgd_steps_val}, alpha={alpha_255_str}"

    records = payload.setdefault("records", {})
    can_seed = 42 if 42 in per_seed_metrics else seeds[0]
    can_m = per_seed_metrics[can_seed]
    can_rel = round(((can_m["aqsm_map50"] - clean_map50) / max(clean_map50, 1e-6)) * 100.0, 2)

    records[f"{arch_prefix}_canonical"] = {
        "model": arch_label,
        "backbone": backbone_desc,
        "seed": can_seed,
        "protocol": f"Seed {can_seed} (N={n_eval})",
        "clean_map50": clean_map50,
        "aqsm_map50": can_m["aqsm_map50"],
        "relative_drop_pct": can_rel,
        "hungarian_set_drift": can_m["hungarian_set_drift"],
        "anchor_fg_hit_pct": can_m["anchor_fg_hit_pct"],
        "n_evaluated": n_eval,
    }
    records[f"{arch_prefix}_multiseed"] = {
        "model": arch_label,
        "backbone": backbone_desc,
        "seeds": seeds,
        "protocol": f"{len(seeds)} seeds (N={n_eval}, mean +/- std, worst-case per-image paired & ICC)",
        "clean_map50": clean_map50,
        "clean_anchor_fg_hit_pct": round(clean_fg_hit_mean, 2),
        "aqsm_map50_mean": aqsm_mean,
        "aqsm_map50_std": aqsm_std,
        "aqsm_map50_min": round(float(ap50_tier1["min"]), 2) if ap50_tier1["min"] is not None else None,
        "aqsm_map50_max": round(float(ap50_tier1["max"]), 2) if ap50_tier1["max"] is not None else None,
        "aqsm_map50_snr_seed": round(float(ap50_tier1["snr_seed"]), 2) if ap50_tier1["snr_seed"] is not None else None,
        "relative_drop_pct": rel_drop_mean,
        "hungarian_set_drift_mean": drift_mean,
        "hungarian_set_drift_std": drift_std,
        "hungarian_set_drift_min": round(float(drift_tier1["min"]), 4) if drift_tier1["min"] is not None else None,
        "hungarian_set_drift_max": round(float(drift_tier1["max"]), 4) if drift_tier1["max"] is not None else None,
        "hungarian_set_drift_snr_seed": round(float(drift_tier1["snr_seed"]), 2) if drift_tier1["snr_seed"] is not None else None,
        "anchor_fg_hit_pct_mean": hit_mean,
        "anchor_fg_hit_pct_std": hit_std,
        "anchor_fg_hit_pct_min": round(float(hit_tier1["min"]), 2) if hit_tier1["min"] is not None else None,
        "anchor_fg_hit_pct_max": round(float(hit_tier1["max"]), 2) if hit_tier1["max"] is not None else None,
        "anchor_fg_hit_pct_snr_seed": round(float(hit_tier1["snr_seed"]), 2) if hit_tier1["snr_seed"] is not None else None,
        "tier1_across_seed_stability": {
            "aqsm_map50": ap50_tier1,
            "hungarian_set_drift": drift_tier1,
            "anchor_fg_hit_pct": hit_tier1,
        },
        "tier2_within_seed_and_consistency": {
            "ap50_vs_clean": ap50_evidence,
            "anchor_fg_hit_vs_clean": fg_hit_evidence,
            "hungarian_drift_consistency": drift_consistency,
        },
        "per_image_csv": str(out_csv),
        "per_seed": {str(k): v for k, v in per_seed_metrics.items()},
        "n_evaluated": n_eval,
    }

    payload["last_updated_model"] = chosen_model
    payload["last_updated_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_json(payload, out_json)
    print(f" [DONE] Đã cập nhật {arch_prefix}_canonical & {arch_prefix}_multiseed tại: {out_json}")
    return payload


def main() -> None:
    parser = create_base_parser(description="Task 10: Multi-Seed AQSM Stability Benchmark")
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(map(str, DEFAULT_MULTI_SEEDS)),
        help="Danh sách random seeds cho lấy mẫu C_bg (phân tách bởi dấu phẩy)",
    )
    parser.add_argument("--eps_255", type=int, default=None, help="Biên độ nhiễu trên thang 255 (mặc định đọc từ cấu hình)")
    parser.add_argument("--pgd_steps", type=int, default=None, help="Số bước lặp PGD (mặc định đọc từ cấu hình)")
    args = parser.parse_args()
    run_multiseed_stability(args)


if __name__ == "__main__":
    main()
