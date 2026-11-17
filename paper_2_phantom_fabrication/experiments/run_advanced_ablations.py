# -*- coding: utf-8 -*-
"""
Advanced Empirical Ablations Script for Paper 2
Directly addresses peer review critique points:
1. Stratified ASR (target class present in image vs absent in image)
2. Mechanistic Isolation of DQF Displacement (Encoder logit perturbation Delta L_enc)
3. SQD-Filter Distance Parameter Sweep (Pareto Knee Point Analysis)
"""

import os
import sys
import time
import json
from pathlib import Path
import numpy as np
import torch
import torchvision.ops as ops
from transformers import RTDetrForObjectDetection

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Setup path
EXPERIMENTS_DIR = Path(__file__).resolve().parent
PAPER2_DIR = EXPERIMENTS_DIR.parent
WORKSPACE = PAPER2_DIR.parent

if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from paper_2_phantom_fabrication.core.coco_gt import (
    load_coco_val2017_gt_cohort,
    evaluate_coco_predictions,
    postprocess_rtdetr_predictions,
    compute_iou_matrix,
    box_cxcywh_to_xyxy,
)
from paper_2_phantom_fabrication.experiments.exp_logger_p2 import (
    setup_experiment_logger,
    SampleAuditTracker,
    compute_distribution_stats,
    RESULTS_DIR,
)
from paper_2_phantom_fabrication.experiments.sqd_filter import SpatialQueryDeduplicationFilter


def run_advanced_ablations():
    logger, log_file = setup_experiment_logger("advanced_ablations")
    logger.info("=" * 80)
    logger.info("STARTING ADVANCED EMPIRICAL ABLATIONS FOR PAPER 2")
    logger.info("1. Stratified ASR | 2. DQF Eviction Mechanism | 3. SQD-Filter Pareto Sweep")
    logger.info("=" * 80)

    sample_tracker = SampleAuditTracker("advanced_ablations")

    # 1. Load COCO test cohort (N=300, seed=42)
    cohort = load_coco_val2017_gt_cohort(n_eval=300, seed=42)
    clean_tensors = cohort["tensors"]  # [300, 3, 320, 320]
    gt_list = cohort["gt_list"]
    image_ids = cohort["image_ids"]
    n_images = len(image_ids)

    # 2. Load authentic adversarial caches
    local_adv_dir = EXPERIMENTS_DIR / "adv_cache"
    eqf_path = local_adv_dir / "adv_cache_eqf_eps16_n300.npz"
    dqf_path = local_adv_dir / "adv_cache_dqf_eps16_n300.npz"

    eqf_npz = np.load(eqf_path)
    dqf_npz = np.load(dqf_path)
    eqf_deltas = torch.from_numpy(eqf_npz["delta_f16"]).float()
    dqf_deltas = torch.from_numpy(dqf_npz["delta_f16"]).float()

    # Gaussian noise (sigma=16/255)
    torch.manual_seed(42)
    noise_deltas = (torch.randn_like(clean_tensors) * (16.0 / 255.0)).clamp(-16.0 / 255.0, 16.0 / 255.0)

    # 3. Load model
    logger.info("Loading RT-DETR-R18VD model...")
    model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r18vd")
    model.eval()

    conditions = {
        "Clean": clean_tensors,
        "Noise": (clean_tensors + noise_deltas).clamp(0.0, 1.0),
        "EQF": (clean_tensors + eqf_deltas).clamp(0.0, 1.0),
        "DQF": (clean_tensors + dqf_deltas).clamp(0.0, 1.0),
    }

    # Run inference for all conditions and collect logits, pred_boxes, enc_outputs
    batch_size = 10
    n_batches = int(np.ceil(n_images / batch_size))
    model_outputs = {}

    for c_name, c_tensors in conditions.items():
        logger.info(f"Running inference for condition: {c_name} (N={n_images})...")
        logits_list = []
        boxes_list = []
        enc_class_list = []
        enc_topk_list = []

        with torch.no_grad():
            for b in range(n_batches):
                s = b * batch_size
                e = min(s + batch_size, n_images)
                bx = c_tensors[s:e]
                out = model(pixel_values=bx)

                logits_list.append(out.logits.cpu())
                boxes_list.append(out.pred_boxes.cpu())

                if hasattr(out, "enc_outputs_class") and out.enc_outputs_class is not None:
                    enc_class_list.append(out.enc_outputs_class.cpu())  # [B, 2100, 80]
                    scores, _ = out.enc_outputs_class.max(dim=-1)
                    topk = torch.topk(scores, k=300, dim=-1).indices.cpu()
                    enc_topk_list.append(topk)

        model_outputs[c_name] = {
            "logits": torch.cat(logits_list, dim=0),
            "pred_boxes": torch.cat(boxes_list, dim=0),
            "enc_class": torch.cat(enc_class_list, dim=0) if enc_class_list else None,
            "enc_topk": torch.cat(enc_topk_list, dim=0) if enc_topk_list else None,
        }

    # =========================================================================
    # ABLATION 1: Stratified ASR (Target Class Present vs Absent)
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("ABLATION 1: STRATIFIED ASR (TARGET CLASS PRESENT VS ABSENT)")
    logger.info("=" * 80)

    stratified_results = {}
    target_classes_to_test = [
        (1, "bicycle"),
        (0, "person"),
        (2, "car"),
    ]

    for cid, cname in target_classes_to_test:
        # Check GT presence in cohort
        has_gt = [1 if cid in gt["labels"] else 0 for gt in gt_list]
        n_has_gt = sum(has_gt)
        n_no_gt = n_images - n_has_gt
        logger.info(f"\nAnalyzing Class ID {cid} ({cname}): {n_has_gt}/{n_images} images contain GT ({n_has_gt/n_images*100:.1f}%)")

        class_strat = {}
        for cond in ["EQF", "DQF"]:
            logits = model_outputs[cond]["logits"]
            pred_boxes = model_outputs[cond]["pred_boxes"]
            probs = torch.sigmoid(logits)
            max_p, pred_cls = probs.max(dim=-1)
            boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes).clamp(0.0, 1.0)

            asr_with_gt = []
            asr_without_gt = []
            phantoms_with_gt = []
            phantoms_without_gt = []

            for i in range(n_images):
                act_mask = (max_p[i] >= 0.35) & (pred_cls[i] == cid)
                act_boxes = boxes_xyxy[i, act_mask].numpy()

                gt_boxes = gt_list[i]["boxes_xyxy"]
                gt_lbls = gt_list[i]["labels"]
                gt_tgt = gt_boxes[gt_lbls == cid] if len(gt_lbls) > 0 else np.zeros((0, 4))

                n_pht = 0
                for b in act_boxes:
                    if len(gt_tgt) == 0:
                        n_pht += 1
                    else:
                        ious = compute_iou_matrix(b[np.newaxis, :], gt_tgt)[0]
                        if float(np.max(ious)) < 0.30:
                            n_pht += 1

                is_asr = 1 if n_pht > 0 else 0
                if has_gt[i]:
                    asr_with_gt.append(is_asr)
                    phantoms_with_gt.append(n_pht)
                else:
                    asr_without_gt.append(is_asr)
                    phantoms_without_gt.append(n_pht)

            asr_w_val = float(np.mean(asr_with_gt) * 100.0) if len(asr_with_gt) > 0 else 0.0
            pht_w_val = float(np.mean(phantoms_with_gt)) if len(phantoms_with_gt) > 0 else 0.0
            asr_wo_val = float(np.mean(asr_without_gt) * 100.0) if len(asr_without_gt) > 0 else 0.0
            pht_wo_val = float(np.mean(phantoms_without_gt)) if len(phantoms_without_gt) > 0 else 0.0

            class_strat[cond] = {
                "with_gt": {"n": n_has_gt, "asr_pct": asr_w_val, "phantoms_mean": pht_w_val},
                "without_gt": {"n": n_no_gt, "asr_pct": asr_wo_val, "phantoms_mean": pht_wo_val},
                "overall": {
                    "asr_pct": float(np.mean(asr_with_gt + asr_without_gt) * 100.0),
                    "phantoms_mean": float(np.mean(phantoms_with_gt + phantoms_without_gt)),
                }
            }
            logger.info(
                f"  [{cond}] With GT ({n_has_gt:3d} img): ASR={asr_w_val:.1f}%, Phantoms={pht_w_val:.2f} | "
                f"Without GT ({n_no_gt:3d} img): ASR={asr_wo_val:.1f}%, Phantoms={pht_wo_val:.2f}"
            )

        stratified_results[cname] = {
            "class_id": cid,
            "n_with_gt": n_has_gt,
            "n_without_gt": n_no_gt,
            "metrics": class_strat,
        }

    # =========================================================================
    # ABLATION 2: Mechanistic Isolation of DQF Eviction
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("ABLATION 2: MECHANISTIC ISOLATION OF DQF EVICTION")
    logger.info("Evaluating Encoder Candidate Logit Shift Delta L_enc and Rank Eviction")
    logger.info("=" * 80)

    clean_enc = model_outputs["Clean"]["enc_class"]  # [300, 2100, 80]
    clean_topk = model_outputs["Clean"]["enc_topk"]  # [300, 300]

    dqf_isolation_results = {}

    for cond in ["Noise", "EQF", "DQF"]:
        adv_enc = model_outputs[cond]["enc_class"]
        adv_topk = model_outputs[cond]["enc_topk"]

        # 1. Absolute logit shift across all candidate anchors
        abs_logit_diff = (adv_enc - clean_enc).abs()
        mean_logit_shift = float(abs_logit_diff.mean().item())
        max_logit_shift = float(abs_logit_diff.max().item())

        # 2. Logit shift on target class (cid=1)
        target_logit_diff = (adv_enc[:, :, 1] - clean_enc[:, :, 1]).mean().item()

        # 3. Top-K eviction rate
        eviction_rates = []
        for i in range(n_images):
            c_set = set(clean_topk[i].numpy())
            a_set = set(adv_topk[i].numpy())
            evicted = len(c_set - a_set) / 300.0
            eviction_rates.append(evicted * 100.0)

        mean_eviction = float(np.mean(eviction_rates))
        std_eviction = float(np.std(eviction_rates))

        dqf_isolation_results[cond] = {
            "mean_logit_shift": round(mean_logit_shift, 4),
            "max_logit_shift": round(max_logit_shift, 4),
            "target_logit_shift": round(target_logit_diff, 4),
            "eviction_rate_mean": round(mean_eviction, 2),
            "eviction_rate_std": round(std_eviction, 2),
        }
        logger.info(
            f"  [{cond:<5}] Mean Logit Shift: {mean_logit_shift:.4f} | "
            f"Target Logit Shift: {target_logit_diff:+.4f} | "
            f"Candidate Eviction Rate: {mean_eviction:.2f}% ± {std_eviction:.2f}%"
        )

    # =========================================================================
    # ABLATION 3: SQD-Filter Distance Parameter Sweep & Pareto Knee Point
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("ABLATION 3: SQD-FILTER DISTANCE PARAMETER SWEEP (PARETO KNEE POINT)")
    logger.info("Evaluating d_thresh in [0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30]")
    logger.info("=" * 80)

    d_thresholds = [0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30]
    sqd_sweep_results = []

    # Clean predictions for recall baseline
    clean_logits = model_outputs["Clean"]["logits"]
    clean_boxes = model_outputs["Clean"]["pred_boxes"]
    clean_probs = torch.sigmoid(clean_logits)
    clean_max_p, clean_pred_cls = clean_probs.max(dim=-1)
    clean_boxes_xyxy = box_cxcywh_to_xyxy(clean_boxes).clamp(0.0, 1.0)

    # Baseline nominal active clean detection count
    n_clean_act_total = sum(int((clean_max_p[i] >= 0.35).sum().item()) for i in range(n_images))

    for d_th in d_thresholds:
        filt = SpatialQueryDeduplicationFilter(distance_threshold=d_th, conf_threshold=0.35)

        # 1. Clean retention
        clean_retained_count = 0
        clean_filtered_preds = []
        for i in range(n_images):
            b_i = clean_boxes[i:i+1]
            l_i = clean_logits[i:i+1]
            fb, fp, tel = filt(b_i, l_i)
            fp_max, fp_cls = fp[0].max(dim=-1)
            act_m = (fp_max >= 0.35)
            clean_retained_count += int(act_m.sum().item())

            # For mAP
            b_xyxy = box_cxcywh_to_xyxy(fb[0, act_m]).clamp(0.0, 1.0).numpy() if act_m.sum() > 0 else np.zeros((0, 4))
            sc = fp_max[act_m].numpy() if act_m.sum() > 0 else np.zeros((0,))
            lb = fp_cls[act_m].numpy() if act_m.sum() > 0 else np.zeros((0,), dtype=int)
            clean_filtered_preds.append({"boxes": b_xyxy, "scores": sc, "labels": lb})

        clean_ret_pct = float(clean_retained_count / max(1, n_clean_act_total) * 100.0)
        clean_eval = evaluate_coco_predictions(clean_filtered_preds, gt_list)
        clean_map = float(clean_eval["mAP@50"])
        clean_map_drop = float(clean_map - 41.02)  # relative to nominal thresholded 41.02%

        # 2. EQF rejection
        eqf_logits = model_outputs["EQF"]["logits"]
        eqf_boxes = model_outputs["EQF"]["pred_boxes"]
        eqf_surv = []
        for i in range(n_images):
            fb, fp, tel = filt(eqf_boxes[i:i+1], eqf_logits[i:i+1])
            fp_max, fp_cls = fp[0].max(dim=-1)
            act_m = (fp_max >= 0.35) & (fp_cls == 1)
            fb_xyxy = box_cxcywh_to_xyxy(fb[0, act_m]).clamp(0.0, 1.0).numpy() if act_m.sum() > 0 else np.zeros((0, 4))

            gt_boxes = gt_list[i]["boxes_xyxy"]
            gt_lbls = gt_list[i]["labels"]
            gt_tgt = gt_boxes[gt_lbls == 1] if len(gt_lbls) > 0 else np.zeros((0, 4))

            n_pht = 0
            for b in fb_xyxy:
                if len(gt_tgt) == 0:
                    n_pht += 1
                else:
                    ious = compute_iou_matrix(b[np.newaxis, :], gt_tgt)[0]
                    if float(np.max(ious)) < 0.30:
                        n_pht += 1
            eqf_surv.append(n_pht)

        eqf_surv_mean = float(np.mean(eqf_surv))
        eqf_rej_pct = float((11.2533 - eqf_surv_mean) / 11.2533 * 100.0)

        # 3. DQF rejection
        dqf_logits = model_outputs["DQF"]["logits"]
        dqf_boxes = model_outputs["DQF"]["pred_boxes"]
        dqf_surv = []
        for i in range(n_images):
            fb, fp, tel = filt(dqf_boxes[i:i+1], dqf_logits[i:i+1])
            fp_max, fp_cls = fp[0].max(dim=-1)
            act_m = (fp_max >= 0.35) & (fp_cls == 1)
            fb_xyxy = box_cxcywh_to_xyxy(fb[0, act_m]).clamp(0.0, 1.0).numpy() if act_m.sum() > 0 else np.zeros((0, 4))

            gt_boxes = gt_list[i]["boxes_xyxy"]
            gt_lbls = gt_list[i]["labels"]
            gt_tgt = gt_boxes[gt_lbls == 1] if len(gt_lbls) > 0 else np.zeros((0, 4))

            n_pht = 0
            for b in fb_xyxy:
                if len(gt_tgt) == 0:
                    n_pht += 1
                else:
                    ious = compute_iou_matrix(b[np.newaxis, :], gt_tgt)[0]
                    if float(np.max(ious)) < 0.30:
                        n_pht += 1
            dqf_surv.append(n_pht)

        dqf_surv_mean = float(np.mean(dqf_surv))
        dqf_rej_pct = float((7.11 - dqf_surv_mean) / 7.11 * 100.0)

        sweep_row = {
            "d_thresh": d_th,
            "clean_retention_pct": round(clean_ret_pct, 2),
            "clean_map50": round(clean_map, 2),
            "clean_map_drop_pp": round(clean_map_drop, 2),
            "eqf_rejection_pct": round(eqf_rej_pct, 2),
            "eqf_surviving_phantoms": round(eqf_surv_mean, 2),
            "dqf_rejection_pct": round(dqf_rej_pct, 2),
            "dqf_surviving_phantoms": round(dqf_surv_mean, 2),
        }
        sqd_sweep_results.append(sweep_row)
        logger.info(
            f"  d_thresh={d_th:.2f} -> "
            f"EQF Rej: {eqf_rej_pct:5.1f}% (Surv: {eqf_surv_mean:.2f}) | "
            f"DQF Rej: {dqf_rej_pct:5.1f}% (Surv: {dqf_surv_mean:.2f}) | "
            f"Clean Ret: {clean_ret_pct:5.1f}% | "
            f"mAP@50 Shift: {clean_map_drop:+.2f} pp"
        )

    # Save comprehensive advanced ablation results
    final_audit = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stratified_asr": stratified_results,
        "dqf_mechanism_isolation": dqf_isolation_results,
        "sqd_filter_sweep": sqd_sweep_results,
    }

    out_json = RESULTS_DIR / "advanced_ablations_audit.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(final_audit, f, indent=2, ensure_ascii=False)
    logger.info(f"\nSaved all advanced ablation results to: {out_json}")
    logger.info("=" * 80)
    return final_audit


if __name__ == "__main__":
    run_advanced_ablations()
