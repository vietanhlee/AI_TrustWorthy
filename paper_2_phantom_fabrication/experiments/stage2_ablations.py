# -*- coding: utf-8 -*-
"""
Module: paper_2_phantom_fabrication.experiments.stage2_ablations
Mục đích: Thực thi Giai đoạn 2 (Stage 2) - Phân tích độ nhạy (Sensitivity Sweeps)
và Bộ mô phỏng tác động không gian mạng - vật lý (Tables 2 & 3, Figures 2 & 3 trong bài báo Elsevier).
"""

import gc
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from transformers import RTDetrForObjectDetection

from paper_2_phantom_fabrication.core.coco_gt import (
    box_cxcywh_to_xyxy,
    compute_iou_matrix,
    evaluate_coco_predictions,
    load_flexible_coco_cohort,
    postprocess_rtdetr_predictions,
)
from paper_2_phantom_fabrication.core.matcher import match_detections_against_gt
from paper_2_phantom_fabrication.downstream.aeb_simulation import simulate_aeb_impact
from paper_2_phantom_fabrication.downstream.bytetrack_sim import simulate_bytetrack_complexity
from paper_2_phantom_fabrication.defenses.sqd_filter import apply_sqd_filter
from paper_2_phantom_fabrication.experiments.streaming_evaluator import (
    get_cohort_dataloader,
    run_streaming_condition_eval,
)

logger = logging.getLogger("Stage2Ablations")


def run_stage_2_ablations(
    image_folder: Path,
    annotations_path: Optional[Path],
    n_samples: int,
    device: torch.device,
    hw_info: Dict[str, Any],
    batch_size: int,
    cache_dir: Optional[str] = None,
    skip_cache_gen: bool = False,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    """
    Thực thi Stage 2: Sweeps tham số và Downstream Suite (Tables 2 & 3, Figures 2 & 3).
    Bao gồm:
      - Budget Sweep: epsilon in {4, 8, 12, 16}/255
      - Iteration Sweep: T in {5, 10, 15, 20} steps
      - Cross-Class Generalization: 5 COCO classes
      - Size-Stratified Breakdown: Small, Medium, Large object eviction
      - Pairwise Coordinate Extraction for Figure 2a
      - ByteTrack Hungarian Assignment Simulation
      - AEB 3D Pinhole Hazard Projection
      - SQD Radius Sweep
      - Untargeted Baseline for Figure 2b ECDF
    """
    logger.info("=" * 80)
    logger.info(f"KHỞI ĐỘNG STAGE 2: ABLATIONS & DOWNSTREAM SUITE (N={n_samples} images)")
    logger.info("=" * 80)

    raw_model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r18vd")
    raw_model.eval()
    raw_model.to(device)

    # Tự động kích hoạt Multi-GPU DataParallel nếu phát hiện nhiều hơn 1 GPU
    if hw_info.get("num_gpus", 0) > 1 and device.type == "cuda":
        model = torch.nn.DataParallel(raw_model)
        logger.info(f"Đã tự động kích hoạt Data Parallelism trên {hw_info['num_gpus']} GPUs cho Stage 2: {hw_info['gpu_names']}")
    else:
        model = raw_model

    cohort = load_flexible_coco_cohort(
        image_folder, annotations_path, n_samples=n_samples, seed=42, img_size=640, preload_images=False
    )
    gt_list = cohort["gt_list"]
    dataloader = get_cohort_dataloader(cohort, batch_size=batch_size, device=device)

    # Lấy clean top-k queries làm tham chiếu
    res_clean_ref = run_streaming_condition_eval(
        model=model, dataloader=dataloader, condition="Clean", gt_list=gt_list,
        device=device, desc="Stage 2 [Clean Ref]"
    )
    clean_topk = res_clean_ref["clean_topk"]

    # 1. Budget Sweep (Table 2)
    logger.info("  [1/4] Chạy Budget Sweep (eps in {4, 8, 12, 16}/255)...")
    budget_sweep = {}
    epsilons = [4.0/255.0, 8.0/255.0, 12.0/255.0, 16.0/255.0]
    eps_names = ["eps_4", "eps_8", "eps_12", "eps_16"]

    for eps, name in zip(epsilons, eps_names):
        sweep_entry = {"eps_float": float(eps)}
        for a_type in ["EQF", "DQF"]:
            res_sweep = run_streaming_condition_eval(
                model=model, dataloader=dataloader, condition=a_type, gt_list=gt_list,
                target_class_idx=1, epsilon=eps, alpha=2.0/255.0, steps=8, device=device,
                clean_topk_reference=clean_topk, desc=f"Budget {name} [{a_type}]",
                image_ids=cohort["image_ids"], cache_dir=cache_dir, skip_cache_gen=skip_cache_gen,
                force_recompute=force_recompute,
            )
            m_stats = res_sweep["matched_stats"]
            displs = res_sweep["displacement_rates_raw"]
            n_succ = sum(m["is_asr"] for m in m_stats)
            n_tot = len(m_stats)
            asr_pct = (n_succ / n_tot) * 100.0
            z = 1.96
            p_hat = n_succ / n_tot
            denom = 1 + z**2 / n_tot
            center = (p_hat + z**2 / (2 * n_tot)) / denom
            margin = (z * math.sqrt(p_hat * (1 - p_hat) / n_tot + z**2 / (4 * n_tot**2))) / denom
            ci_low = max(0.0, (center - margin) * 100.0)
            ci_high = min(100.0, (center + margin) * 100.0)

            sweep_entry[a_type] = {
                "mAP50_operational": float(res_sweep["mAP50_operational"]),
                "active_mean": float(res_sweep["active_mean"]),
                "phantoms_mean": float(res_sweep["phantoms_mean"]),
                "precision_mean": float(res_sweep["precision_mean"]),
                "recall_mean": float(res_sweep["recall_mean"]),
                "asr_fab_pct": float(asr_pct),
                "asr_ci_95": [float(round(ci_low, 1)), float(round(ci_high, 1))],
                "displacement_rate_mean": float(res_sweep["displacement_rate_mean"]),
                "displacement_rates_raw": displs,
            }
        budget_sweep[name] = sweep_entry

    # 2. Iteration Sweep (Table 2)
    logger.info("  [2/4] Chạy Iteration Sweep (T in {5, 10, 15, 20} steps)...")
    iteration_sweep = {}
    steps_list = [5, 10, 15, 20]
    step_names = ["T_5", "T_10", "T_15", "T_20"]

    for steps, name in zip(steps_list, step_names):
        it_entry = {"steps": steps}
        for a_type in ["EQF", "DQF"]:
            res_it = run_streaming_condition_eval(
                model=model, dataloader=dataloader, condition=a_type, gt_list=gt_list,
                target_class_idx=1, epsilon=16.0/255.0, alpha=2.0/255.0, steps=steps, device=device,
                clean_topk_reference=clean_topk, desc=f"Iteration {name} [{a_type}]",
                image_ids=cohort["image_ids"], cache_dir=cache_dir, skip_cache_gen=skip_cache_gen,
                force_recompute=force_recompute,
            )
            m_stats = res_it["matched_stats"]
            displs = res_it["displacement_rates_raw"]
            n_succ = sum(m["is_asr"] for m in m_stats)
            n_tot = len(m_stats)
            asr_pct = (n_succ / n_tot) * 100.0
            z = 1.96
            p_hat = n_succ / n_tot
            denom = 1 + z**2 / n_tot
            center = (p_hat + z**2 / (2 * n_tot)) / denom
            margin = (z * math.sqrt(p_hat * (1 - p_hat) / n_tot + z**2 / (4 * n_tot**2))) / denom
            ci_low = max(0.0, (center - margin) * 100.0)
            ci_high = min(100.0, (center + margin) * 100.0)

            it_entry[a_type] = {
                "mAP50_operational": float(res_it["mAP50_operational"]),
                "active_mean": float(res_it["active_mean"]),
                "phantoms_mean": float(res_it["phantoms_mean"]),
                "precision_mean": float(res_it["precision_mean"]),
                "recall_mean": float(res_it["recall_mean"]),
                "asr_fab_pct": float(asr_pct),
                "asr_ci_95": [float(round(ci_low, 1)), float(round(ci_high, 1))],
                "displacement_rate_mean": float(res_it["displacement_rate_mean"]),
                "displacement_rates_raw": displs,
            }
        iteration_sweep[name] = it_entry

    # 3. Cross-Class Generalization (Table 3)
    logger.info("  [3/4] Chạy Multi-Class Generalization (5 classes)...")
    cross_classes = [
        {"name": "person", "id": 0},
        {"name": "bicycle", "id": 1},
        {"name": "car", "id": 2},
        {"name": "traffic light", "id": 9},
        {"name": "stop sign", "id": 11},
    ]
    cross_class_results = {}

    for c_info in cross_classes:
        c_name = c_info["name"]
        c_id = c_info["id"]
        c_entry = {"class_id": c_id}

        for a_type in ["EQF", "DQF"]:
            res_cls = run_streaming_condition_eval(
                model=model, dataloader=dataloader, condition=a_type, gt_list=gt_list,
                target_class_idx=c_id, epsilon=16.0/255.0, alpha=2.0/255.0, steps=8, device=device,
                clean_topk_reference=clean_topk, desc=f"Class {c_name} [{a_type}]",
                image_ids=cohort["image_ids"], cache_dir=cache_dir, skip_cache_gen=skip_cache_gen,
                force_recompute=force_recompute,
            )
            c_entry[a_type] = {
                "mAP50_operational": float(res_cls["mAP50_operational"]),
                "active_mean": float(res_cls["active_mean"]),
                "phantoms_mean": float(res_cls["phantoms_mean"]),
                "precision_mean": float(res_cls["precision_mean"]),
                "recall_mean": float(res_cls["recall_mean"]),
                "asr_fab_pct": float(res_cls["asr_fab_pct"]),
                "displacement_rate_mean": float(res_cls["displacement_rate_mean"]),
            }
        cross_class_results[c_name] = c_entry

    # 4. Size-Stratified Breakdown, ByteTrack, AEB, Pairwise Points & SQD Sweep
    logger.info("  [4/4] Đánh giá phân rã kích thước (Size Breakdown) và Downstream Suite...")
    res_clean_down = run_streaming_condition_eval(
        model=model, dataloader=dataloader, condition="Clean", gt_list=gt_list,
        device=device, save_logits_and_boxes=True, desc="Downstream [Clean]"
    )
    clean_logits = res_clean_down["logits"]
    clean_boxes = res_clean_down["pred_boxes"]

    res_eqf_down = run_streaming_condition_eval(
        model=model, dataloader=dataloader, condition="EQF", gt_list=gt_list,
        target_class_idx=1, epsilon=16.0/255.0, alpha=2.0/255.0, steps=8, device=device,
        save_logits_and_boxes=True, desc="Downstream [EQF]",
        image_ids=cohort["image_ids"], cache_dir=cache_dir, skip_cache_gen=skip_cache_gen,
        force_recompute=force_recompute,
    )
    eqf_logits = res_eqf_down["logits"]
    eqf_boxes = res_eqf_down["pred_boxes"]

    res_dqf_down = run_streaming_condition_eval(
        model=model, dataloader=dataloader, condition="DQF", gt_list=gt_list,
        target_class_idx=1, epsilon=16.0/255.0, alpha=2.0/255.0, steps=8, device=device,
        save_logits_and_boxes=True, desc="Downstream [DQF]",
        image_ids=cohort["image_ids"], cache_dir=cache_dir, skip_cache_gen=skip_cache_gen,
        force_recompute=force_recompute,
    )
    dqf_logits = res_dqf_down["logits"]
    dqf_boxes = res_dqf_down["pred_boxes"]

    clean_post = postprocess_rtdetr_predictions(clean_logits, clean_boxes, score_thresh=0.35)
    eqf_post = postprocess_rtdetr_predictions(eqf_logits, eqf_boxes, score_thresh=0.35)

    # Size-stratified evaluation
    size_bins = {
        "small": {"max_area": 32.0 * 32.0 / (640.0 * 640.0), "min_area": 0.0},
        "medium": {"min_area": 32.0 * 32.0 / (640.0 * 640.0), "max_area": 96.0 * 96.0 / (640.0 * 640.0)},
        "large": {"min_area": 96.0 * 96.0 / (640.0 * 640.0), "max_area": 1e6},
    }
    size_breakdown = {}

    for s_name, s_range in size_bins.items():
        sub_gt = []
        tot_objs = 0
        evicted_cnt = 0

        for i in range(len(gt_list)):
            gb = gt_list[i]["boxes_xyxy"]
            gl = gt_list[i]["labels"]
            areas = (gb[:, 2] - gb[:, 0]) * (gb[:, 3] - gb[:, 1])
            m = (areas >= s_range["min_area"]) & (areas < s_range["max_area"])
            sub_gt.append({
                "boxes_xyxy": gb[m], "labels": gl[m],
                "iscrowd": gt_list[i]["iscrowd"][m], "image_id": gt_list[i]["image_id"]
            })
            tot_objs += int(np.sum(m))

            c_p = clean_post[i]
            e_p = eqf_post[i]
            if len(gb[m]) > 0:
                iou_c = compute_iou_matrix(c_p["boxes"], gb[m]) if len(c_p["boxes"]) > 0 else np.zeros((0, len(gb[m])))
                iou_e = compute_iou_matrix(e_p["boxes"], gb[m]) if len(e_p["boxes"]) > 0 else np.zeros((0, len(gb[m])))
                for obj_j in range(len(gb[m])):
                    was_det_clean = bool(np.max(iou_c[:, obj_j]) >= 0.50) if len(iou_c) > 0 else False
                    is_det_eqf = bool(np.max(iou_e[:, obj_j]) >= 0.50) if len(iou_e) > 0 else False
                    if was_det_clean and not is_det_eqf:
                        evicted_cnt += 1

        c_eval_s = evaluate_coco_predictions(clean_post, sub_gt)["mAP@50"]
        e_eval_s = evaluate_coco_predictions(eqf_post, sub_gt)["mAP@50"]
        evic_rate = (evicted_cnt / max(tot_objs, 1)) * 100.0

        z = 1.96
        p_hat = evicted_cnt / max(tot_objs, 1)
        denom = 1 + z**2 / max(tot_objs, 1)
        center = (p_hat + z**2 / (2 * max(tot_objs, 1))) / denom
        margin = (z * math.sqrt(p_hat * (1 - p_hat) / max(tot_objs, 1) + z**2 / (4 * max(tot_objs, 1)**2))) / denom
        ci_low = max(0.0, (center - margin) * 100.0)
        ci_high = min(100.0, (center + margin) * 100.0)

        rel_drop = ((e_eval_s - c_eval_s) / max(c_eval_s, 1e-6)) * 100.0
        size_breakdown[s_name] = {
            "total_objects": tot_objs,
            "eviction_rate": float(round(evic_rate, 1)),
            "eviction_ci_95": [float(round(ci_low, 1)), float(round(ci_high, 1))],
            "clean_mAP50": float(round(c_eval_s, 2)),
            "eqf_mAP50": float(round(e_eval_s, 2)),
            "relative_ap_dropoff": float(round(rel_drop, 1)),
        }

    # Extract 2,000+ Pairwise Phantom Coordinates for Figure 2a
    center_dists_all, ious_all = [], []
    eqf_probs = torch.sigmoid(eqf_logits)
    eqf_p_boxes = box_cxcywh_to_xyxy(eqf_boxes)

    for i in range(len(gt_list)):
        act_m = eqf_probs[i].max(dim=-1)[0] >= 0.35
        p_boxes = eqf_p_boxes[i, act_m].numpy()
        if len(p_boxes) > 1:
            centers = np.stack([(p_boxes[:, 0] + p_boxes[:, 2])/2, (p_boxes[:, 1] + p_boxes[:, 3])/2], axis=-1)
            ious = compute_iou_matrix(p_boxes, p_boxes)
            for j in range(len(p_boxes)):
                for k in range(j + 1, len(p_boxes)):
                    c_dist = float(np.linalg.norm(centers[j] - centers[k]))
                    iou_val = float(ious[j, k])
                    center_dists_all.append(round(c_dist, 4))
                    ious_all.append(round(iou_val, 4))

    # Real ByteTrack Hungarian Assignment Complexity
    bytetrack_metrics = simulate_bytetrack_complexity(
        clean_post=clean_post,
        eqf_post=eqf_post,
        dqf_logits=dqf_logits,
        dqf_boxes=dqf_boxes,
        gt_list=gt_list,
    )

    # AEB 3D Pinhole Hazard Projection
    dists_grid = np.linspace(10, 80, 36)
    aeb_metrics = simulate_aeb_impact(
        clean_post=clean_post,
        eqf_post=eqf_post,
        dqf_logits=dqf_logits,
        dqf_boxes=dqf_boxes,
        dists_grid=dists_grid,
    )

    # SQD Radius Sweep
    sqd_radii = [0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30]
    sqd_radius_sweep = {}
    for r in sqd_radii:
        c_filt = apply_sqd_filter(clean_logits, clean_boxes, score_thresh=0.35, dist_thresh=r)
        e_filt = apply_sqd_filter(eqf_logits, eqf_boxes, score_thresh=0.35, dist_thresh=r)
        d_filt = apply_sqd_filter(dqf_logits, dqf_boxes, score_thresh=0.35, dist_thresh=r)

        c_ret = np.mean([len(p["boxes"]) for p in c_filt]) / max(np.mean([len(p["boxes"]) for p in clean_post]), 1e-6) * 100.0
        c_map = evaluate_coco_predictions(c_filt, gt_list)["mAP@50"]
        c_base = evaluate_coco_predictions(clean_post, gt_list)["mAP@50"]

        e_surv = np.mean([match_detections_against_gt(p["boxes"], p["scores"], p["labels"], gt_list[i]["boxes_xyxy"], gt_list[i]["labels"], target_cls=1)["phantoms"] for i, p in enumerate(e_filt)])
        d_surv = np.mean([match_detections_against_gt(p["boxes"], p["scores"], p["labels"], gt_list[i]["boxes_xyxy"], gt_list[i]["labels"], target_cls=1)["phantoms"] for i, p in enumerate(d_filt)])
        e_base_pht = np.mean([match_detections_against_gt(p["boxes"], p["scores"], p["labels"], gt_list[i]["boxes_xyxy"], gt_list[i]["labels"], target_cls=1)["phantoms"] for i, p in enumerate(eqf_post)])
        d_base_pht = np.mean([match_detections_against_gt(p["boxes"], p["scores"], p["labels"], gt_list[i]["boxes_xyxy"], gt_list[i]["labels"], target_cls=1)["phantoms"] for i, p in enumerate(postprocess_rtdetr_predictions(dqf_logits, dqf_boxes, score_thresh=0.35))])

        sqd_radius_sweep[f"dist_{r:.2f}"] = {
            "radius": r,
            "eqf_rejection_pct": float(round((1.0 - e_surv / max(e_base_pht, 1e-6)) * 100.0, 1)),
            "dqf_rejection_pct": float(round((1.0 - d_surv / max(d_base_pht, 1e-6)) * 100.0, 1)),
            "surviving_eqf": float(round(e_surv, 1)),
            "surviving_dqf": float(round(d_surv, 1)),
            "clean_retention_pct": float(round(c_ret, 1)),
            "clean_map50_shift": float(round(c_map - c_base, 2)),
        }

    # Untargeted Baseline for Figure 2b ECDF
    res_untarg = run_streaming_condition_eval(
        model=model, dataloader=dataloader, condition="Untargeted_PGD", gt_list=gt_list,
        target_class_idx=1, epsilon=16.0/255.0, alpha=2.0/255.0, steps=8, device=device,
        clean_topk_reference=clean_topk, desc="Downstream [Untargeted]",
        image_ids=cohort["image_ids"], cache_dir=cache_dir, skip_cache_gen=skip_cache_gen,
        force_recompute=force_recompute,
    )
    untarg_displs = res_untarg["displacement_rates_raw"]

    del clean_logits, clean_boxes, eqf_logits, eqf_boxes, dqf_logits, dqf_boxes
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    downstream_and_figures = {
        "size_breakdown": size_breakdown,
        "pairwise_phantom_points": {
            "center_distances": center_dists_all[:2500],
            "ious": ious_all[:2500],
        },
        "bytetrack": bytetrack_metrics,
        "aeb_hazard": aeb_metrics,
        "sqd_radius_sweep": sqd_radius_sweep,
        "untargeted_baseline": {
            "displacement_rates_raw": [float(round(d, 2)) for d in untarg_displs],
            "displacement_rate_mean": float(np.mean(untarg_displs)),
        }
    }

    return {
        "budget_sweep": budget_sweep,
        "iteration_sweep": iteration_sweep,
        "cross_class": cross_class_results,
        "downstream_and_figures": downstream_and_figures,
    }
