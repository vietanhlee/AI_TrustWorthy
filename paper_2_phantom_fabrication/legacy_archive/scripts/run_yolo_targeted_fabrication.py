# -*- coding: utf-8 -*-
"""
Authentic YOLOv8 Targeted Fabrication Benchmark for Paper 2
Direct Empirical Response to Peer Reviewer Critique:
Evaluating convolutional NMS suppression under identical targeted fabrication objectives.

Runs genuine PGD targeted fabrication (eps=16/255, T=15, alpha=2/255) on YOLOv8-Small
across the official COCO val2017 cohort (N=300).
Measures:
1. Pre-NMS candidate count (how many anchors are elevated >= 0.35)
2. Post-NMS surviving detections (after IoU=0.50 NMS)
3. Exact NMS rejection/suppression rate (%) on YOLOv8
4. Surviving phantom query count vs RT-DETR
5. CPU inference latency under attack
"""

import os
import sys
import time
import json
from pathlib import Path
import numpy as np
import torch
import torchvision.ops as ops
from ultralytics import YOLO

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Setup path
EXPERIMENTS_DIR = Path(__file__).resolve().parent
PAPER2_DIR = EXPERIMENTS_DIR.parent
WORKSPACE = PAPER2_DIR.parent

if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from paper_2_phantom_fabrication.core.coco_gt import (
    load_coco_val2017_gt_cohort,
    compute_iou_matrix,
    evaluate_coco_predictions,
)
from paper_2_phantom_fabrication.experiments.exp_logger_p2 import (
    setup_experiment_logger,
    SampleAuditTracker,
    compute_distribution_stats,
    RESULTS_DIR,
)


def run_yolo_targeted_fabrication():
    logger, log_file = setup_experiment_logger("yolo_targeted_fabrication")
    logger.info("=" * 80)
    logger.info("STARTING AUTHENTIC YOLOV8 TARGETED FABRICATION BENCHMARK (N=300)")
    logger.info("Evaluating Pre-NMS vs Post-NMS candidate survival on YOLOv8-Small")
    logger.info("=" * 80)

    sample_tracker = SampleAuditTracker("yolo_targeted_fabrication")

    # 1. Load dataset cohort (N=300, seed=42)
    cohort = load_coco_val2017_gt_cohort(n_eval=300, seed=42)
    clean_tensors = cohort["tensors"]  # [300, 3, 320, 320]
    gt_list = cohort["gt_list"]
    image_ids = cohort["image_ids"]
    n_images = len(image_ids)

    # 2. Load YOLOv8-Small
    yolo_weight = (
        EXPERIMENTS_DIR / "yolov8s.pt"
        if (EXPERIMENTS_DIR / "yolov8s.pt").exists()
        else (WORKSPACE / "yolov8s.pt" if (WORKSPACE / "yolov8s.pt").exists() else "yolov8s.pt")
    )
    logger.info(f"Loading YOLOv8-Small weights from: {yolo_weight}")
    yolo = YOLO(str(yolo_weight))
    model = yolo.model
    model.eval()

    target_cls = 1  # bicycle (matching primary RT-DETR benchmark)
    alpha = 2.0 / 255.0
    eps = 16.0 / 255.0
    T_steps = 15
    batch_size = 5
    n_batches = int(np.ceil(n_images / batch_size))

    # Containers for results
    pre_nms_active_clean = []
    post_nms_active_clean = []
    clean_latencies = []

    pre_nms_active_adv = []
    post_nms_active_adv = []
    surviving_phantoms_adv = []
    nms_rejection_rates = []
    adv_latencies = []

    clean_yolo_preds = []
    adv_yolo_preds = []

    logger.info(f"\nRunning Targeted Fabrication on {n_images} images (batch size {batch_size}, T={T_steps} steps)...")
    start_total_time = time.time()

    for b in range(n_batches):
        s_idx = b * batch_size
        e_idx = min(s_idx + batch_size, n_images)
        b_len = e_idx - s_idx
        batch_x = clean_tensors[s_idx:e_idx].clone()

        # --- A. Clean Evaluation on Batch ---
        t0_clean = time.perf_counter()
        with torch.no_grad():
            clean_out = model(batch_x)[0]  # [b_len, 84, 2100]
        t1_clean = time.perf_counter()
        clean_batch_lat = (t1_clean - t0_clean) * 1000.0 / b_len

        # --- B. Targeted Adversarial Attack (PGD-15) ---
        x_adv = batch_x.clone().detach().requires_grad_(True)
        for step in range(T_steps):
            out = model(x_adv)[0]  # [b_len, 84, 2100]
            # out[:, 4+target_cls, :] is the target class score for each of 2100 anchors
            target_scores = out[:, 4 + target_cls, :]  # [b_len, 2100]
            loss = -target_scores.topk(50, dim=-1).values.mean()
            loss.backward()
            with torch.no_grad():
                x_adv = x_adv - alpha * x_adv.grad.sign()
                delta = torch.clamp(x_adv - batch_x, -eps, eps)
                x_adv = torch.clamp(batch_x + delta, 0.0, 1.0).requires_grad_(True)

        # --- C. Adversarial Evaluation on Batch ---
        t0_adv = time.perf_counter()
        with torch.no_grad():
            adv_out = model(x_adv.detach())[0]  # [b_len, 84, 2100]
        t1_adv = time.perf_counter()
        adv_batch_lat = (t1_adv - t0_adv) * 1000.0 / b_len

        # --- D. Analyze Each Image in Batch ---
        for j in range(b_len):
            img_idx = s_idx + j
            clean_latencies.append(clean_batch_lat)
            adv_latencies.append(adv_batch_lat)

            # Ground truth
            gt_boxes = gt_list[img_idx]["boxes_xyxy"]
            gt_lbls = gt_list[img_idx]["labels"]
            gt_target_boxes = gt_boxes[gt_lbls == target_cls] if len(gt_lbls) > 0 else np.zeros((0, 4))

            # === CLEAN PREDICTIONS ===
            c_boxes_raw = clean_out[j, :4, :].t()  # [2100, 4] (cx, cy, w, h in px)
            c_scores_raw = clean_out[j, 4:, :].t()  # [2100, 80]
            c_target_scores = c_scores_raw[:, target_cls]

            # Pre-NMS
            c_pre_mask = c_target_scores >= 0.35
            n_c_pre = int(c_pre_mask.sum().item())
            pre_nms_active_clean.append(n_c_pre)

            # Post-NMS (Standard torchvision NMS on active target boxes)
            c_cx, c_cy, c_w, c_h = c_boxes_raw[:, 0], c_boxes_raw[:, 1], c_boxes_raw[:, 2], c_boxes_raw[:, 3]
            c_xyxy = torch.stack([
                (c_cx - c_w / 2.0) / 320.0,
                (c_cy - c_h / 2.0) / 320.0,
                (c_cx + c_w / 2.0) / 320.0,
                (c_cy + c_h / 2.0) / 320.0,
            ], dim=1).clamp(0.0, 1.0)

            if n_c_pre > 0:
                c_keep = ops.nms(c_xyxy[c_pre_mask], c_target_scores[c_pre_mask], iou_threshold=0.50)
                n_c_post = len(c_keep)
            else:
                n_c_post = 0
            post_nms_active_clean.append(n_c_post)

            # === ADVERSARIAL PREDICTIONS ===
            a_boxes_raw = adv_out[j, :4, :].t()  # [2100, 4]
            a_scores_raw = adv_out[j, 4:, :].t()  # [2100, 80]
            a_target_scores = a_scores_raw[:, target_cls]

            # Pre-NMS
            a_pre_mask = a_target_scores >= 0.35
            n_a_pre = int(a_pre_mask.sum().item())
            pre_nms_active_adv.append(n_a_pre)

            # Post-NMS
            a_cx, a_cy, a_w, a_h = a_boxes_raw[:, 0], a_boxes_raw[:, 1], a_boxes_raw[:, 2], a_boxes_raw[:, 3]
            a_xyxy = torch.stack([
                (a_cx - a_w / 2.0) / 320.0,
                (a_cy - a_h / 2.0) / 320.0,
                (a_cx + a_w / 2.0) / 320.0,
                (a_cy + a_h / 2.0) / 320.0,
            ], dim=1).clamp(0.0, 1.0)

            if n_a_pre > 0:
                a_keep = ops.nms(a_xyxy[a_pre_mask], a_target_scores[a_pre_mask], iou_threshold=0.50)
                surv_boxes = a_xyxy[a_pre_mask][a_keep].cpu().numpy()
                n_a_post = len(a_keep)
            else:
                surv_boxes = np.zeros((0, 4))
                n_a_post = 0
            post_nms_active_adv.append(n_a_post)

            # Rejection rate of NMS on YOLOv8
            if n_a_pre > 0:
                rej_rate = (n_a_pre - n_a_post) / float(n_a_pre) * 100.0
            else:
                rej_rate = 0.0
            nms_rejection_rates.append(rej_rate)

            # Surviving phantoms (IoU < 0.30 against GT of target class)
            n_phantoms = 0
            for sb in surv_boxes:
                if len(gt_target_boxes) == 0:
                    n_phantoms += 1
                else:
                    ious_gt = compute_iou_matrix(sb[np.newaxis, :], gt_target_boxes)[0]
                    if float(np.max(ious_gt)) < 0.30:
                        n_phantoms += 1
            surviving_phantoms_adv.append(n_phantoms)

            # Record sample
            sample_tracker.record_sample(
                sample_idx=img_idx,
                image_id=image_ids[img_idx],
                condition="YOLOv8_Targeted_Fabrication",
                metrics={
                    "clean_pre_nms": n_c_pre,
                    "clean_post_nms": n_c_post,
                    "adv_pre_nms": n_a_pre,
                    "adv_post_nms": n_a_post,
                    "nms_rejection_rate": round(rej_rate, 2),
                    "surviving_phantoms": n_phantoms,
                    "is_asr_fab": 1 if n_phantoms > 0 else 0,
                    "clean_latency_ms": round(clean_batch_lat, 3),
                    "adv_latency_ms": round(adv_batch_lat, 3),
                }
            )

        if (b + 1) % 10 == 0 or (b + 1) == n_batches:
            logger.info(
                f"  Batch {b+1:2d}/{n_batches:2d} -> Processed {e_idx:3d}/{n_images} images | "
                f"Current Mean Pre-NMS: {np.mean(pre_nms_active_adv):.1f} | "
                f"Post-NMS: {np.mean(post_nms_active_adv):.2f} | "
                f"NMS Rejection: {np.mean(nms_rejection_rates):.1f}%"
            )

    elapsed_total = time.time() - start_total_time
    logger.info(f"\nCompleted YOLOv8 Targeted Fabrication Benchmark in {elapsed_total:.1f}s")

    # Compute Summary Statistics
    summary = {
        "benchmark": "YOLOv8-Small Targeted Fabrication Benchmark",
        "dataset": "COCO val2017 (N=300)",
        "target_class": target_cls,
        "perturbation": f"PGD-15 (eps=16/255, alpha=2/255, L_inf)",
        "pre_nms_adv": compute_distribution_stats(pre_nms_active_adv),
        "post_nms_adv": compute_distribution_stats(post_nms_active_adv),
        "nms_rejection_rate_pct": compute_distribution_stats(nms_rejection_rates),
        "surviving_phantoms": compute_distribution_stats(surviving_phantoms_adv),
        "asr_fab_pct": float(np.mean([1 if p > 0 else 0 for p in surviving_phantoms_adv]) * 100.0),
        "clean_latencies_ms": compute_distribution_stats(clean_latencies),
        "adv_latencies_ms": compute_distribution_stats(adv_latencies),
    }

    logger.info("\n" + "=" * 80)
    logger.info("FINAL YOLOV8 TARGETED FABRICATION BENCHMARK RESULTS:")
    logger.info(f"  Pre-NMS Candidate Proposals (>= 0.35): {summary['pre_nms_adv']['mean']:.2f} ± {summary['pre_nms_adv']['std']:.2f} (Median: {summary['pre_nms_adv']['median']:.1f})")
    logger.info(f"  Post-NMS Surviving Boxes (IoU=0.50):   {summary['post_nms_adv']['mean']:.2f} ± {summary['post_nms_adv']['std']:.2f} (Median: {summary['post_nms_adv']['median']:.1f})")
    logger.info(f"  NMS Suppression / Rejection Rate:      {summary['nms_rejection_rate_pct']['mean']:.2f}% ± {summary['nms_rejection_rate_pct']['std']:.2f}% (Median: {summary['nms_rejection_rate_pct']['median']:.1f}%)")
    logger.info(f"  Surviving Phantom Boxes:               {summary['surviving_phantoms']['mean']:.2f} ± {summary['surviving_phantoms']['std']:.2f}")
    logger.info(f"  Fabrication ASR:                       {summary['asr_fab_pct']:.1f}%")
    logger.info(f"  Mean Clean CPU Latency:                {summary['clean_latencies_ms']['mean']:.2f} ms")
    logger.info(f"  Mean Adv CPU Latency:                  {summary['adv_latencies_ms']['mean']:.2f} ms")
    logger.info("=" * 80)

    # Save artifacts
    csv_path = sample_tracker.save_csv()
    json_path = RESULTS_DIR / "yolo_targeted_fabrication_audit.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved sample audit to: {csv_path}")
    logger.info(f"Saved summary JSON to: {json_path}")

    return summary


if __name__ == "__main__":
    run_yolo_targeted_fabrication()
