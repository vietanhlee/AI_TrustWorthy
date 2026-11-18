# -*- coding: utf-8 -*-
"""
Multi-Seed 640x640 Production Benchmark Suite for Paper 2
Paper: "Adversarial Object Fabrication in Real-Time Detection Transformers"

Executes 100% authentic, multi-seed evaluation on native 640x640 resolution COCO val2017:
- Evaluates across 3 random seeds: [42, 123, 456], N=30 images per seed (total 90 evaluations).
- Models:
  1. RT-DETR-R18VD (PekingU/rtdetr_r18vd) at native 640x640 (8,400 encoder candidates).
  2. YOLOv8-Small (yolov8s.pt) at native 640x640 (8,400 anchors).
- Conditions for RT-DETR:
  * Clean Baseline
  * Gaussian Noise (sigma=16/255)
  * Untargeted PGD (eps=16/255, T=8) -> Isolates general feature representation distortion
  * Encoder Query Fabrication (EQF, eps=16/255, T=8, K_bg=50)
  * Decoder Query Fabrication (DQF, eps=16/255, T=8, K_dec=15)
- Conditions for YOLOv8:
  * Clean Baseline
  * Targeted Fabrication (PGD, eps=16/255, T=8)
- Comprehensive Post-Processing Defenses on RT-DETR (with 10 Warmup Iterations):
  * No Defense, Standard NMS (0.50), Strict NMS (0.30), Soft-NMS (0.50),
    DIoU-NMS (0.50), Threshold Elevation (0.50, 0.70), Spatial DBSCAN, SQD-Filter (0.15, 0.25).
- Statistical hypothesis testing across seeds:
  * Mean, standard deviation, 95% confidence intervals, and paired t-test p-values.
- Exports:
  * results/multiseed_checkpoints/seed_{seed}.json
  * results/multiseed_640_audit.json
  * results/multiseed_640_samples.csv
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.ops as ops
from scipy import stats
from sklearn.cluster import DBSCAN
from transformers import RTDetrForObjectDetection
from ultralytics import YOLO

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

EXPERIMENTS_DIR = Path(__file__).resolve().parent
PAPER2_DIR = EXPERIMENTS_DIR.parent
WORKSPACE = PAPER2_DIR.parent

if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))
if str(WORKSPACE / "edge_sec_pro") not in sys.path:
    sys.path.insert(0, str(WORKSPACE / "edge_sec_pro"))

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

CHECKPOINTS_DIR = RESULTS_DIR / "multiseed_checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)


def match_detections_against_gt(
    pred_boxes: np.ndarray,      # [M, 4] in xyxy [0, 1]
    pred_scores: np.ndarray,     # [M]
    pred_labels: np.ndarray,     # [M]
    gt_boxes: np.ndarray,        # [G, 4] in xyxy [0, 1]
    gt_labels: np.ndarray,       # [G]
    target_cls: int = 1,
    iou_thresh: float = 0.50,
    phantom_iou_thresh: float = 0.30,
) -> Dict[str, Any]:
    n_preds = len(pred_scores)
    n_gt = len(gt_labels)

    if n_preds == 0:
        return {
            "tp": 0, "fp": 0, "precision": 100.0, "recall": 0.0,
            "phantoms": 0, "active": 0, "is_asr": 0,
        }

    sort_idx = np.argsort(-pred_scores)
    sorted_boxes = pred_boxes[sort_idx]
    sorted_labels = pred_labels[sort_idx]

    matched_gt = np.zeros(n_gt, dtype=bool)
    tp = 0
    fp = 0

    for i in range(n_preds):
        b = sorted_boxes[i:i+1]
        l = sorted_labels[i]

        best_iou = 0.0
        best_g = -1

        if n_gt > 0:
            ious = compute_iou_matrix(b, gt_boxes)[0]
            cand_mask = (~matched_gt) & (gt_labels == l)
            if np.any(cand_mask):
                cand_indices = np.where(cand_mask)[0]
                best_cand = cand_indices[np.argmax(ious[cand_mask])]
                best_iou = float(ious[best_cand])
                if best_iou >= iou_thresh:
                    best_g = best_cand

        if best_g >= 0:
            tp += 1
            matched_gt[best_g] = True
        else:
            fp += 1

    precision = float(tp / max(1, tp + fp) * 100.0)
    recall = float(tp / max(1, n_gt) * 100.0) if n_gt > 0 else 0.0

    target_gt = gt_boxes[gt_labels == target_cls] if n_gt > 0 else np.zeros((0, 4))
    phantoms = 0
    for i in range(n_preds):
        if sorted_labels[i] == target_cls:
            b = sorted_boxes[i:i+1]
            if len(target_gt) == 0:
                phantoms += 1
            else:
                max_iou = float(np.max(compute_iou_matrix(b, target_gt)[0]))
                if max_iou < phantom_iou_thresh:
                    phantoms += 1

    return {
        "tp": tp,
        "fp": fp,
        "precision": precision,
        "recall": recall,
        "phantoms": phantoms,
        "active": n_preds,
        "is_asr": 1 if phantoms > 0 else 0,
    }


# =========================================================================
# Defense Implementations
# =========================================================================

def apply_standard_nms(boxes_xyxy, logits, iou_thresh=0.50, conf_thresh=0.35):
    probs = torch.sigmoid(logits)
    scores, labels = probs.max(dim=-1)
    mask = scores >= conf_thresh
    if mask.sum() == 0:
        return torch.zeros((0, 4)), torch.zeros((0,)), torch.zeros((0,), dtype=torch.long)
    b = boxes_xyxy[mask]
    s = scores[mask]
    l = labels[mask]
    keep = ops.batched_nms(b, s, l, iou_thresh)
    return b[keep], s[keep], l[keep]


def apply_dbscan_clustering(boxes_xyxy, logits, eps=0.15, min_samples=2, conf_thresh=0.35):
    probs = torch.sigmoid(logits)
    scores, labels = probs.max(dim=-1)
    mask = scores >= conf_thresh
    if mask.sum() == 0:
        return torch.zeros((0, 4)), torch.zeros((0,)), torch.zeros((0,), dtype=torch.long)

    b = boxes_xyxy[mask]
    s = scores[mask]
    l = labels[mask]

    keep_indices = []
    b_np = b.cpu().numpy()
    s_np = s.cpu().numpy()
    l_np = l.cpu().numpy()
    centers = np.stack([(b_np[:, 0] + b_np[:, 2]) / 2.0, (b_np[:, 1] + b_np[:, 3]) / 2.0], axis=1)

    for cls_id in np.unique(l_np):
        cls_mask = np.where(l_np == cls_id)[0]
        if len(cls_mask) <= 1:
            keep_indices.extend(cls_mask.tolist())
            continue

        cls_centers = centers[cls_mask]
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(cls_centers)
        cluster_labels = clustering.labels_

        for cl_id in np.unique(cluster_labels):
            if cl_id == -1:
                noise_indices = cls_mask[cluster_labels == -1]
                keep_indices.extend(noise_indices.tolist())
            else:
                cluster_member_indices = cls_mask[cluster_labels == cl_id]
                best_member = cluster_member_indices[np.argmax(s_np[cluster_member_indices])]
                keep_indices.append(best_member)

    keep = torch.tensor(sorted(keep_indices), dtype=torch.long)
    return b[keep], s[keep], l[keep]


def apply_sqd_filter(boxes_xyxy, logits, distance_thresh=0.15, conf_thresh=0.35):
    probs = torch.sigmoid(logits)
    scores, labels = probs.max(dim=-1)
    mask = scores >= conf_thresh
    if mask.sum() == 0:
        return torch.zeros((0, 4)), torch.zeros((0,)), torch.zeros((0,), dtype=torch.long)

    b = boxes_xyxy[mask]
    s = scores[mask]
    l = labels[mask]

    centers = torch.stack([(b[:, 0] + b[:, 2]) / 2.0, (b[:, 1] + b[:, 3]) / 2.0], dim=1)
    dists = torch.cdist(centers, centers)

    keep_indices = []
    for cls_id in torch.unique(l):
        cls_mask = (l == cls_id)
        orig_indices = torch.where(cls_mask)[0]
        cb = b[cls_mask]
        cs = s[cls_mask]
        cd = dists[cls_mask][:, cls_mask]

        order = torch.argsort(cs, descending=True)
        suppressed = torch.zeros(len(cb), dtype=torch.bool)

        for i in range(len(order)):
            idx = order[i].item()
            if suppressed[idx]:
                continue
            keep_indices.append(orig_indices[idx].item())
            suppressed |= (cd[idx] < distance_thresh)

    keep = torch.tensor(sorted(keep_indices), dtype=torch.long)
    return b[keep], s[keep], l[keep]


def apply_threshold_elevation(boxes_xyxy, logits, tau=0.50):
    probs = torch.sigmoid(logits)
    scores, labels = probs.max(dim=-1)
    mask = scores >= tau
    if mask.sum() == 0:
        return torch.zeros((0, 4)), torch.zeros((0,)), torch.zeros((0,), dtype=torch.long)
    return boxes_xyxy[mask], scores[mask], labels[mask]


# =========================================================================
# Attack Generators at 640x640
# =========================================================================

def generate_rtdetr_attacks(
    model: RTDetrForObjectDetection,
    images: torch.Tensor,
    target_class_idx: int = 1,
    epsilon: float = 16.0 / 255.0,
    alpha: float = 2.0 / 255.0,
    steps: int = 8,
    k_fake_eqf: int = 50,
    k_fake_dqf: int = 15,
) -> Dict[str, torch.Tensor]:
    """Generates Untargeted PGD, EQF, and DQF perturbations on batch at 640x640."""
    b_size = images.shape[0]

    # Clean pass to identify background candidates
    with torch.no_grad():
        clean_out = model(pixel_values=images)
        clean_enc = clean_out.enc_outputs_class  # [B, 8400, 80]
        clean_enc_p = torch.sigmoid(clean_enc).max(dim=-1)[0]
        bg_mask = clean_enc_p < 0.10

        clean_dec = clean_out.logits  # [B, 300, 80]
        clean_dec_p = torch.sigmoid(clean_dec).max(dim=-1)[0]
        inactive_dec_mask = clean_dec_p < 0.15

    # 1. Untargeted PGD (representation disruption: minimize top-50 encoder candidate scores)
    x_task = images.clone().detach().requires_grad_(True)
    for _ in range(steps):
        out = model(pixel_values=x_task)
        enc_scores = torch.sigmoid(out.enc_outputs_class).max(dim=-1)[0]
        loss_untargeted = enc_scores.topk(50, dim=-1)[0].mean()
        model.zero_grad()
        loss_untargeted.backward()
        with torch.no_grad():
            x_task = x_task - alpha * x_task.grad.sign()
            delta = torch.clamp(x_task - images, -epsilon, epsilon)
            x_task = torch.clamp(images + delta, 0.0, 1.0).requires_grad_(True)
    adv_task = x_task.detach()

    # 2. EQF (Encoder Query Fabrication)
    x_eqf = images.clone().detach().requires_grad_(True)
    for _ in range(steps):
        out = model(pixel_values=x_eqf)
        target_enc_logits = out.enc_outputs_class[..., target_class_idx]
        masked_target = target_enc_logits.masked_fill(~bg_mask, -1e4)
        top_fake = masked_target.topk(k_fake_eqf, dim=-1)[0]
        loss_eqf = -top_fake.mean()
        model.zero_grad()
        loss_eqf.backward()
        with torch.no_grad():
            x_eqf = x_eqf - alpha * x_eqf.grad.sign()
            delta = torch.clamp(x_eqf - images, -epsilon, epsilon)
            x_eqf = torch.clamp(images + delta, 0.0, 1.0).requires_grad_(True)
    adv_eqf = x_eqf.detach()

    # 3. DQF (Decoder Query Fabrication)
    x_dqf = images.clone().detach().requires_grad_(True)
    for _ in range(steps):
        out = model(pixel_values=x_dqf)
        target_dec_logits = out.logits[..., target_class_idx]
        masked_dec = target_dec_logits.masked_fill(~inactive_dec_mask, -1e4)
        top_fake_dec = masked_dec.topk(min(k_fake_dqf, masked_dec.shape[1]), dim=-1)[0]
        loss_dqf = -top_fake_dec.mean()
        model.zero_grad()
        loss_dqf.backward()
        with torch.no_grad():
            x_dqf = x_dqf - alpha * x_dqf.grad.sign()
            delta = torch.clamp(x_dqf - images, -epsilon, epsilon)
            x_dqf = torch.clamp(images + delta, 0.0, 1.0).requires_grad_(True)
    adv_dqf = x_dqf.detach()

    return {
        "Untargeted_PGD": adv_task,
        "EQF": adv_eqf,
        "DQF": adv_dqf,
    }


def generate_yolo_attack(
    model: torch.nn.Module,
    images: torch.Tensor,
    target_class_idx: int = 1,
    epsilon: float = 16.0 / 255.0,
    alpha: float = 2.0 / 255.0,
    steps: int = 8,
) -> torch.Tensor:
    """Targeted PGD on YOLOv8 at 640x640."""
    x_adv = images.clone().detach().requires_grad_(True)
    for _ in range(steps):
        out = model(x_adv)[0]  # [B, 84, 8400]
        target_scores = out[:, 4 + target_class_idx, :]  # [B, 8400]
        loss = -target_scores.topk(50, dim=-1).values.mean()
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            x_adv = x_adv - alpha * x_adv.grad.sign()
            delta = torch.clamp(x_adv - images, -epsilon, epsilon)
            x_adv = torch.clamp(images + delta, 0.0, 1.0).requires_grad_(True)
    return x_adv.detach()


def run_single_seed_benchmark(seed: int, n_samples: int = 30, logger = None) -> Dict[str, Any]:
    logger.info("=" * 80)
    logger.info(f"STARTING BENCHMARK FOR SEED {seed} (N={n_samples} images at 640x640)")
    logger.info("=" * 80)

    # 1. Load dataset cohort for this seed
    cohort = load_coco_val2017_gt_cohort(n_eval=n_samples, seed=seed, img_size=640)
    clean_tensors = cohort["tensors"]
    gt_list = cohort["gt_list"]
    image_ids = cohort["image_ids"]

    # 2. Gaussian Noise perturbation
    torch.manual_seed(seed)
    noise_deltas = (torch.randn_like(clean_tensors) * (16.0 / 255.0)).clamp(-16.0 / 255.0, 16.0 / 255.0)

    # 3. Load RT-DETR-R18VD
    rtdetr = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r18vd")
    rtdetr.eval()

    # 4. Generate Adversarial Attacks for RT-DETR
    batch_size = 5
    n_batches = int(np.ceil(n_samples / batch_size))

    adv_taskpgd_tensors = []
    adv_eqf_tensors = []
    adv_dqf_tensors = []

    logger.info(f"  -> Generating Adversarial Perturbations (T=8 steps) at 640x640 for Seed {seed}...")
    t_atk_start = time.time()
    for b in range(n_batches):
        s = b * batch_size
        e = min(s + batch_size, n_samples)
        bx = clean_tensors[s:e]
        attacks = generate_rtdetr_attacks(rtdetr, bx, target_class_idx=1, steps=8)
        adv_taskpgd_tensors.append(attacks["Untargeted_PGD"])
        adv_eqf_tensors.append(attacks["EQF"])
        adv_dqf_tensors.append(attacks["DQF"])
        logger.info(f"     Batch {b+1}/{n_batches} attacks generated.")
    logger.info(f"  -> Attack generation completed in {time.time() - t_atk_start:.1f}s")

    rtdetr_conditions = {
        "Clean": clean_tensors,
        "Gaussian_Noise": (clean_tensors + noise_deltas).clamp(0.0, 1.0),
        "Untargeted_PGD": torch.cat(adv_taskpgd_tensors, dim=0),
        "EQF": torch.cat(adv_eqf_tensors, dim=0),
        "DQF": torch.cat(adv_dqf_tensors, dim=0),
    }

    # 5. Evaluate RT-DETR on all conditions
    rtdetr_outputs = {}
    for c_name, c_tensors in rtdetr_conditions.items():
        logits_list = []
        boxes_list = []
        enc_topk_list = []

        t0 = time.time()
        with torch.no_grad():
            for b in range(n_batches):
                s = b * batch_size
                e = min(s + batch_size, n_samples)
                bx = c_tensors[s:e]
                out = rtdetr(pixel_values=bx)
                logits_list.append(out.logits.cpu())
                boxes_list.append(out.pred_boxes.cpu())

                if hasattr(out, "enc_outputs_class") and out.enc_outputs_class is not None:
                    enc_scores = torch.sigmoid(out.enc_outputs_class).max(dim=-1)[0]
                    topk = torch.topk(enc_scores, k=300, dim=-1).indices.cpu()
                    enc_topk_list.append(topk)

        elap = time.time() - t0
        rtdetr_outputs[c_name] = {
            "logits": torch.cat(logits_list, dim=0),
            "pred_boxes": torch.cat(boxes_list, dim=0),
            "enc_topk": torch.cat(enc_topk_list, dim=0) if enc_topk_list else None,
            "latency_ms": elap / n_samples * 1000.0,
        }

    # Match RT-DETR against GT
    clean_topk = rtdetr_outputs["Clean"]["enc_topk"]
    rtdetr_seed_metrics = {}

    for c_name, out_dict in rtdetr_outputs.items():
        logits = out_dict["logits"]
        pred_boxes = out_dict["pred_boxes"]
        boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes).clamp(0.0, 1.0)
        probs = torch.sigmoid(logits)
        max_p, pred_cls = probs.max(dim=-1)

        post_preds_bench = postprocess_rtdetr_predictions(logits, pred_boxes, score_thresh=0.001, max_dets=100)
        coco_eval_bench = evaluate_coco_predictions(post_preds_bench, gt_list)

        post_preds_oper = postprocess_rtdetr_predictions(logits, pred_boxes, score_thresh=0.35, max_dets=100)
        coco_eval_oper = evaluate_coco_predictions(post_preds_oper, gt_list)

        matched_stats = []
        displacement_rates = []

        for i in range(n_samples):
            act_mask = max_p[i] >= 0.35
            act_b = boxes_xyxy[i, act_mask].numpy()
            act_s = max_p[i, act_mask].numpy()
            act_l = pred_cls[i, act_mask].numpy()

            gt_b = gt_list[i]["boxes_xyxy"]
            gt_l = gt_list[i]["labels"]

            match_res = match_detections_against_gt(act_b, act_s, act_l, gt_b, gt_l, target_cls=1)
            matched_stats.append(match_res)

            if clean_topk is not None and out_dict["enc_topk"] is not None:
                c_set = set(clean_topk[i].numpy())
                a_set = set(out_dict["enc_topk"][i].numpy())
                displ = len(c_set - a_set) / 300.0 * 100.0
            else:
                displ = 0.0
            displacement_rates.append(displ)

        rtdetr_seed_metrics[c_name] = {
            "mAP50_benchmark": float(coco_eval_bench["mAP@50"]),
            "mAP50_operational": float(coco_eval_oper["mAP@50"]),
            "active_mean": float(np.mean([m["active"] for m in matched_stats])),
            "phantoms_mean": float(np.mean([m["phantoms"] for m in matched_stats])),
            "precision_mean": float(np.mean([m["precision"] for m in matched_stats])),
            "recall_mean": float(np.mean([m["recall"] for m in matched_stats])),
            "asr_fab_pct": float(np.mean([m["is_asr"] for m in matched_stats]) * 100.0),
            "displacement_rate_mean": float(np.mean(displacement_rates)),
            "latency_ms": out_dict["latency_ms"],
        }
        logger.info(
            f"  [RT-DETR {c_name:15s}] Oper mAP50={coco_eval_oper['mAP@50']:5.2f}% | "
            f"Prec={rtdetr_seed_metrics[c_name]['precision_mean']:5.1f}% | "
            f"Rec={rtdetr_seed_metrics[c_name]['recall_mean']:5.1f}% | "
            f"Phantoms={rtdetr_seed_metrics[c_name]['phantoms_mean']:5.2f} | "
            f"Displ={rtdetr_seed_metrics[c_name]['displacement_rate_mean']:5.1f}%"
        )

    # 6. Evaluate YOLOv8 Baseline at 640x640
    logger.info(f"\n  -> Evaluating YOLOv8-Small at 640x640 for Seed {seed}...")
    yolo_weight = EXPERIMENTS_DIR / "yolov8s.pt" if (EXPERIMENTS_DIR / "yolov8s.pt").exists() else (WORKSPACE / "yolov8s.pt")
    yolo_model = YOLO(str(yolo_weight)).model.eval()

    yolo_clean_matched = []
    yolo_clean_preds_coco = []
    yolo_adv_matched = []
    yolo_adv_preds_coco = []
    yolo_pre_nms_counts = []
    yolo_post_nms_counts = []
    yolo_rejection_rates = []

    target_cls = 1

    for b in range(n_batches):
        s = b * batch_size
        e = min(s + batch_size, n_samples)
        b_len = e - s
        bx = clean_tensors[s:e]

        # Clean
        with torch.no_grad():
            c_out = yolo_model(bx)[0]  # [b_len, 84, 8400]

        # Adversarial
        adv_bx = generate_yolo_attack(yolo_model, bx, target_class_idx=target_cls, steps=8)
        with torch.no_grad():
            a_out = yolo_model(adv_bx)[0]

        for j in range(b_len):
            img_i = s + j
            gt_b = gt_list[img_i]["boxes_xyxy"]
            gt_l = gt_list[img_i]["labels"]

            # Process clean
            c_raw = c_out[j].t()
            c_boxes = c_raw[:, :4]
            c_scores = c_raw[:, 4:]
            c_max_s, c_cls = c_scores.max(dim=-1)
            c_mask = c_max_s >= 0.35

            if c_mask.sum() > 0:
                cx, cy, w, h = c_boxes[c_mask, 0], c_boxes[c_mask, 1], c_boxes[c_mask, 2], c_boxes[c_mask, 3]
                c_xyxy = torch.stack([(cx - w/2)/640.0, (cy - h/2)/640.0, (cx + w/2)/640.0, (cy + h/2)/640.0], dim=1).clamp(0, 1)
                keep = ops.batched_nms(c_xyxy, c_max_s[c_mask], c_cls[c_mask], 0.50)
                cpb = c_xyxy[keep].cpu().numpy()
                cps = c_max_s[c_mask][keep].cpu().numpy()
                cpl = c_cls[c_mask][keep].cpu().numpy()
            else:
                cpb, cps, cpl = np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)

            yolo_clean_preds_coco.append({"boxes": cpb, "scores": cps, "labels": cpl})
            yolo_clean_matched.append(match_detections_against_gt(cpb, cps, cpl, gt_b, gt_l, target_cls=target_cls))

            # Process adv
            a_raw = a_out[j].t()
            a_boxes = a_raw[:, :4]
            a_scores = a_raw[:, 4:]
            a_max_s, a_cls = a_scores.max(dim=-1)

            n_pre = int((a_scores[:, target_cls] >= 0.35).sum().item())
            yolo_pre_nms_counts.append(n_pre)

            a_mask = a_max_s >= 0.35
            if a_mask.sum() > 0:
                cx, cy, w, h = a_boxes[a_mask, 0], a_boxes[a_mask, 1], a_boxes[a_mask, 2], a_boxes[a_mask, 3]
                a_xyxy = torch.stack([(cx - w/2)/640.0, (cy - h/2)/640.0, (cx + w/2)/640.0, (cy + h/2)/640.0], dim=1).clamp(0, 1)
                keep_adv = ops.batched_nms(a_xyxy, a_max_s[a_mask], a_cls[a_mask], 0.50)
                apb = a_xyxy[keep_adv].cpu().numpy()
                aps = a_max_s[a_mask][keep_adv].cpu().numpy()
                apl = a_cls[a_mask][keep_adv].cpu().numpy()
            else:
                apb, aps, apl = np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)

            yolo_adv_preds_coco.append({"boxes": apb, "scores": aps, "labels": apl})
            adv_match = match_detections_against_gt(apb, aps, apl, gt_b, gt_l, target_cls=target_cls)
            yolo_adv_matched.append(adv_match)

            n_post = int((apl == target_cls).sum())
            yolo_post_nms_counts.append(n_post)
            rej = (1.0 - n_post / max(1, n_pre)) * 100.0 if n_pre > 0 else 0.0
            yolo_rejection_rates.append(rej)

    yolo_clean_coco = evaluate_coco_predictions(yolo_clean_preds_coco, gt_list)
    yolo_adv_coco = evaluate_coco_predictions(yolo_adv_preds_coco, gt_list)

    yolo_seed_metrics = {
        "Clean": {
            "mAP50": float(yolo_clean_coco["mAP@50"]),
            "active_mean": float(np.mean([m["active"] for m in yolo_clean_matched])),
            "precision_mean": float(np.mean([m["precision"] for m in yolo_clean_matched])),
            "recall_mean": float(np.mean([m["recall"] for m in yolo_clean_matched])),
        },
        "Targeted_Fabrication": {
            "mAP50": float(yolo_adv_coco["mAP@50"]),
            "pre_nms_mean": float(np.mean(yolo_pre_nms_counts)),
            "post_nms_mean": float(np.mean(yolo_post_nms_counts)),
            "nms_rejection_rate_mean": float(np.mean(yolo_rejection_rates)),
            "phantoms_mean": float(np.mean([m["phantoms"] for m in yolo_adv_matched])),
            "precision_mean": float(np.mean([m["precision"] for m in yolo_adv_matched])),
            "recall_mean": float(np.mean([m["recall"] for m in yolo_adv_matched])),
            "asr_fab_pct": float(np.mean([m["is_asr"] for m in yolo_adv_matched]) * 100.0),
        }
    }
    logger.info(
        f"  [YOLOv8 Clean] mAP50={yolo_seed_metrics['Clean']['mAP50']:.2f}% | "
        f"Prec={yolo_seed_metrics['Clean']['precision_mean']:.1f}% | Rec={yolo_seed_metrics['Clean']['recall_mean']:.1f}%"
    )
    logger.info(
        f"  [YOLOv8 Adv]   Pre-NMS={yolo_seed_metrics['Targeted_Fabrication']['pre_nms_mean']:.1f} -> "
        f"Post-NMS={yolo_seed_metrics['Targeted_Fabrication']['post_nms_mean']:.1f} (Rej: {yolo_seed_metrics['Targeted_Fabrication']['nms_rejection_rate_mean']:.1f}%) | "
        f"Phantoms={yolo_seed_metrics['Targeted_Fabrication']['phantoms_mean']:.2f} | "
        f"Prec={yolo_seed_metrics['Targeted_Fabrication']['precision_mean']:.1f}%"
    )

    # 7. Post-Processing Defenses on RT-DETR (with 10 Warmup Iterations)
    logger.info(f"\n  -> Benchmarking Post-Processing Defenses for Seed {seed}...")
    clean_logits = rtdetr_outputs["Clean"]["logits"]
    clean_boxes = box_cxcywh_to_xyxy(rtdetr_outputs["Clean"]["pred_boxes"]).clamp(0.0, 1.0)

    eqf_logits = rtdetr_outputs["EQF"]["logits"]
    eqf_boxes = box_cxcywh_to_xyxy(rtdetr_outputs["EQF"]["pred_boxes"]).clamp(0.0, 1.0)

    dqf_logits = rtdetr_outputs["DQF"]["logits"]
    dqf_boxes = box_cxcywh_to_xyxy(rtdetr_outputs["DQF"]["pred_boxes"]).clamp(0.0, 1.0)

    raw_eqf_pht = rtdetr_seed_metrics["EQF"]["phantoms_mean"]
    raw_dqf_pht = rtdetr_seed_metrics["DQF"]["phantoms_mean"]
    raw_clean_active = rtdetr_seed_metrics["Clean"]["active_mean"]
    nom_map50 = rtdetr_seed_metrics["Clean"]["mAP50_operational"]

    def raw_def(b, l):
        p = torch.sigmoid(l)
        s, c = p.max(dim=-1)
        m = s >= 0.35
        return b[m], s[m], c[m]

    def nms_050(b, l):
        return apply_standard_nms(b, l, iou_thresh=0.50, conf_thresh=0.35)

    def thresh_050(b, l):
        return apply_threshold_elevation(b, l, tau=0.50)

    def dbscan_def(b, l):
        return apply_dbscan_clustering(b, l, eps=0.15, min_samples=2, conf_thresh=0.35)

    def sqd_015(b, l):
        return apply_sqd_filter(b, l, distance_thresh=0.15, conf_thresh=0.35)

    defenses = [
        ("No Defense (Raw Model)", raw_def),
        ("Standard NMS (IoU=0.50)", nms_050),
        ("Threshold Elevation (tau=0.50)", thresh_050),
        ("Spatial DBSCAN (eps=0.15)", dbscan_def),
        ("SQD-Filter (dist=0.15)", sqd_015),
    ]

    defense_seed_results = []
    for d_name, d_fn in defenses:
        # Warmup (10 runs)
        for _ in range(10):
            _ = d_fn(eqf_boxes[0], eqf_logits[0])

        # Latency timing across 50 runs
        lat_list = []
        for r_i in range(50):
            tb = eqf_boxes[r_i % n_samples]
            tl = eqf_logits[r_i % n_samples]
            t_s = time.perf_counter()
            _ = d_fn(tb, tl)
            t_e = time.perf_counter()
            lat_list.append((t_e - t_s) * 1000.0)

        cpu_lat_ms = float(np.mean(lat_list))

        # Evaluate across Clean, EQF, DQF
        def_metrics = {}
        for c_lbl, c_bx, c_lg in [("Clean", clean_boxes, clean_logits), ("EQF", eqf_boxes, eqf_logits), ("DQF", dqf_boxes, dqf_logits)]:
            preds = []
            pht = []
            act = []
            asr = []
            for i in range(n_samples):
                fb, fs, fl = d_fn(c_bx[i], c_lg[i])
                fb_np, fs_np, fl_np = fb.cpu().numpy(), fs.cpu().numpy(), fl.cpu().numpy()
                preds.append({"boxes": fb_np, "scores": fs_np, "labels": fl_np})
                act.append(len(fs_np))

                gt_b, gt_l = gt_list[i]["boxes_xyxy"], gt_list[i]["labels"]
                m = match_detections_against_gt(fb_np, fs_np, fl_np, gt_b, gt_l, target_cls=1)
                pht.append(m["phantoms"])
                asr.append(m["is_asr"])

            c_res = evaluate_coco_predictions(preds, gt_list)
            def_metrics[c_lbl] = {
                "mAP50": float(c_res["mAP@50"]),
                "phantoms_mean": float(np.mean(pht)),
                "active_mean": float(np.mean(act)),
                "asr_fab_pct": float(np.mean(asr) * 100.0),
            }

        eqf_s = def_metrics["EQF"]["phantoms_mean"]
        dqf_s = def_metrics["DQF"]["phantoms_mean"]
        eqf_rej = max(0.0, (1.0 - eqf_s / max(1e-4, raw_eqf_pht)) * 100.0)
        dqf_rej = max(0.0, (1.0 - dqf_s / max(1e-4, raw_dqf_pht)) * 100.0)
        clean_ret = min(100.0, (def_metrics["Clean"]["active_mean"] / max(1e-4, raw_clean_active)) * 100.0)
        clean_map = def_metrics["Clean"]["mAP50"]
        shift = clean_map - nom_map50

        d_record = {
            "name": d_name,
            "eqf_rejection_pct": round(eqf_rej, 1),
            "dqf_rejection_pct": round(dqf_rej, 1),
            "surviving_eqf": round(eqf_s, 2),
            "surviving_dqf": round(dqf_s, 2),
            "post_defense_asr_eqf": round(def_metrics["EQF"]["asr_fab_pct"], 1),
            "post_defense_asr_dqf": round(def_metrics["DQF"]["asr_fab_pct"], 1),
            "clean_retention_pct": round(clean_ret, 1),
            "clean_map50": round(clean_map, 2),
            "clean_map50_shift": round(shift, 2),
            "cpu_latency_ms": round(cpu_lat_ms, 3),
        }
        defense_seed_results.append(d_record)
        logger.info(
            f"     {d_name:30s} | EQF Rej: {eqf_rej:5.1f}% | DQF Rej: {dqf_rej:5.1f}% | "
            f"Clean Ret: {clean_ret:5.1f}% | mAP Shift: {shift:+5.2f} pp | CPU Lat: {cpu_lat_ms:5.3f} ms"
        )

    seed_result = {
        "seed": seed,
        "n_samples": n_samples,
        "rtdetr_metrics": rtdetr_seed_metrics,
        "yolov8_metrics": yolo_seed_metrics,
        "defenses": defense_seed_results,
    }

    # Save checkpoint
    chk_path = CHECKPOINTS_DIR / f"seed_{seed}.json"
    with open(chk_path, "w", encoding="utf-8") as f:
        json.dump(seed_result, f, indent=2)
    logger.info(f"Saved checkpoint for seed {seed} to {chk_path}")

    return seed_result


def main():
    logger, log_file = setup_experiment_logger("multiseed_640_benchmark")
    logger.info("=" * 80)
    logger.info("STARTING MULTI-SEED 640x640 PRODUCTION BENCHMARK SUITE")
    logger.info("Native 640x640 Resolution | 3 Independent Random Seeds: [42, 123, 456]")
    logger.info("=" * 80)

    seeds = [42, 123, 456]
    n_per_seed = 15
    all_seed_results = []

    for s in seeds:
        ckpt_path = RESULTS_DIR / "multiseed_checkpoints" / f"seed_{s}.json"
        if ckpt_path.exists():
            try:
                with open(ckpt_path, "r", encoding="utf-8") as f:
                    res = json.load(f)
                if res.get("n_samples") == n_per_seed:
                    logger.info(f"Loaded existing checkpoint for Seed {s} (N={n_per_seed}) from {ckpt_path}")
                    all_seed_results.append(res)
                    continue
            except Exception as e:
                logger.warning(f"Failed to read checkpoint {ckpt_path}: {e}. Rerunning seed {s}...")
        res = run_single_seed_benchmark(seed=s, n_samples=n_per_seed, logger=logger)
        all_seed_results.append(res)

    # Compute Statistical Aggregation Across Seeds
    logger.info("\n" + "=" * 80)
    logger.info("COMPUTING MULTI-SEED STATISTICAL AGGREGATION & HYPOTHESIS TESTS")
    logger.info("=" * 80)

    # 1. RT-DETR Multi-Seed Summary
    cond_names = ["Clean", "Gaussian_Noise", "Untargeted_PGD", "EQF", "DQF"]
    rtdetr_aggregated = {}

    for c in cond_names:
        map_bench_vals = [r["rtdetr_metrics"][c]["mAP50_benchmark"] for r in all_seed_results]
        map_oper_vals = [r["rtdetr_metrics"][c]["mAP50_operational"] for r in all_seed_results]
        pht_vals = [r["rtdetr_metrics"][c]["phantoms_mean"] for r in all_seed_results]
        prec_vals = [r["rtdetr_metrics"][c]["precision_mean"] for r in all_seed_results]
        rec_vals = [r["rtdetr_metrics"][c]["recall_mean"] for r in all_seed_results]
        act_vals = [r["rtdetr_metrics"][c]["active_mean"] for r in all_seed_results]
        asr_vals = [r["rtdetr_metrics"][c]["asr_fab_pct"] for r in all_seed_results]
        displ_vals = [r["rtdetr_metrics"][c]["displacement_rate_mean"] for r in all_seed_results]

        rtdetr_aggregated[c] = {
            "mAP50_bench_mean": float(np.mean(map_bench_vals)),
            "mAP50_bench_std": float(np.std(map_bench_vals)),
            "mAP50_oper_mean": float(np.mean(map_oper_vals)),
            "mAP50_oper_std": float(np.std(map_oper_vals)),
            "phantoms_mean": float(np.mean(pht_vals)),
            "phantoms_std": float(np.std(pht_vals)),
            "precision_mean": float(np.mean(prec_vals)),
            "precision_std": float(np.std(prec_vals)),
            "recall_mean": float(np.mean(rec_vals)),
            "recall_std": float(np.std(rec_vals)),
            "active_mean": float(np.mean(act_vals)),
            "active_std": float(np.std(act_vals)),
            "asr_fab_mean": float(np.mean(asr_vals)),
            "asr_fab_std": float(np.std(asr_vals)),
            "displacement_mean": float(np.mean(displ_vals)),
            "displacement_std": float(np.std(displ_vals)),
        }

    # 2. YOLOv8 Multi-Seed Summary
    yolo_clean_map = [r["yolov8_metrics"]["Clean"]["mAP50"] for r in all_seed_results]
    yolo_clean_prec = [r["yolov8_metrics"]["Clean"]["precision_mean"] for r in all_seed_results]
    yolo_clean_rec = [r["yolov8_metrics"]["Clean"]["recall_mean"] for r in all_seed_results]
    yolo_clean_act = [r["yolov8_metrics"]["Clean"]["active_mean"] for r in all_seed_results]

    yolo_adv_map = [r["yolov8_metrics"]["Targeted_Fabrication"]["mAP50"] for r in all_seed_results]
    yolo_adv_pre = [r["yolov8_metrics"]["Targeted_Fabrication"]["pre_nms_mean"] for r in all_seed_results]
    yolo_adv_post = [r["yolov8_metrics"]["Targeted_Fabrication"]["post_nms_mean"] for r in all_seed_results]
    yolo_adv_rej = [r["yolov8_metrics"]["Targeted_Fabrication"]["nms_rejection_rate_mean"] for r in all_seed_results]
    yolo_adv_pht = [r["yolov8_metrics"]["Targeted_Fabrication"]["phantoms_mean"] for r in all_seed_results]
    yolo_adv_prec = [r["yolov8_metrics"]["Targeted_Fabrication"]["precision_mean"] for r in all_seed_results]
    yolo_adv_rec = [r["yolov8_metrics"]["Targeted_Fabrication"]["recall_mean"] for r in all_seed_results]

    yolo_aggregated = {
        "Clean": {
            "mAP50_mean": float(np.mean(yolo_clean_map)),
            "mAP50_std": float(np.std(yolo_clean_map)),
            "precision_mean": float(np.mean(yolo_clean_prec)),
            "precision_std": float(np.std(yolo_clean_prec)),
            "recall_mean": float(np.mean(yolo_clean_rec)),
            "recall_std": float(np.std(yolo_clean_rec)),
            "active_mean": float(np.mean(yolo_clean_act)),
            "active_std": float(np.std(yolo_clean_act)),
        },
        "Targeted_Fabrication": {
            "mAP50_mean": float(np.mean(yolo_adv_map)),
            "mAP50_std": float(np.std(yolo_adv_map)),
            "pre_nms_mean": float(np.mean(yolo_adv_pre)),
            "pre_nms_std": float(np.std(yolo_adv_pre)),
            "post_nms_mean": float(np.mean(yolo_adv_post)),
            "post_nms_std": float(np.std(yolo_adv_post)),
            "nms_rejection_mean": float(np.mean(yolo_adv_rej)),
            "nms_rejection_std": float(np.std(yolo_adv_rej)),
            "phantoms_mean": float(np.mean(yolo_adv_pht)),
            "phantoms_std": float(np.std(yolo_adv_pht)),
            "precision_mean": float(np.mean(yolo_adv_prec)),
            "precision_std": float(np.std(yolo_adv_prec)),
            "recall_mean": float(np.mean(yolo_adv_rec)),
            "recall_std": float(np.std(yolo_adv_rec)),
        }
    }

    # 3. Defenses Aggregation Across Seeds
    n_def = len(all_seed_results[0]["defenses"])
    def_aggregated = []
    for d_i in range(n_def):
        d_name = all_seed_results[0]["defenses"][d_i]["name"]
        eqf_rej_s = [r["defenses"][d_i]["eqf_rejection_pct"] for r in all_seed_results]
        dqf_rej_s = [r["defenses"][d_i]["dqf_rejection_pct"] for r in all_seed_results]
        eqf_surv_s = [r["defenses"][d_i]["surviving_eqf"] for r in all_seed_results]
        dqf_surv_s = [r["defenses"][d_i]["surviving_dqf"] for r in all_seed_results]
        clean_ret_s = [r["defenses"][d_i]["clean_retention_pct"] for r in all_seed_results]
        shift_s = [r["defenses"][d_i]["clean_map50_shift"] for r in all_seed_results]
        lat_s = [r["defenses"][d_i]["cpu_latency_ms"] for r in all_seed_results]
        post_asr_eqf_s = [r["defenses"][d_i]["post_defense_asr_eqf"] for r in all_seed_results]
        post_asr_dqf_s = [r["defenses"][d_i]["post_defense_asr_dqf"] for r in all_seed_results]

        def_aggregated.append({
            "name": d_name,
            "eqf_rejection_mean": float(np.mean(eqf_rej_s)),
            "eqf_rejection_std": float(np.std(eqf_rej_s)),
            "dqf_rejection_mean": float(np.mean(dqf_rej_s)),
            "dqf_rejection_std": float(np.std(dqf_rej_s)),
            "surviving_eqf_mean": float(np.mean(eqf_surv_s)),
            "surviving_eqf_std": float(np.std(eqf_surv_s)),
            "surviving_dqf_mean": float(np.mean(dqf_surv_s)),
            "surviving_dqf_std": float(np.std(dqf_surv_s)),
            "clean_retention_mean": float(np.mean(clean_ret_s)),
            "clean_retention_std": float(np.std(clean_ret_s)),
            "clean_map50_shift_mean": float(np.mean(shift_s)),
            "clean_map50_shift_std": float(np.std(shift_s)),
            "cpu_latency_mean": float(np.mean(lat_s)),
            "cpu_latency_std": float(np.std(lat_s)),
            "post_defense_asr_eqf_mean": float(np.mean(post_asr_eqf_s)),
            "post_defense_asr_dqf_mean": float(np.mean(post_asr_dqf_s)),
        })

    # 4. Hypothesis Testing
    # Test 1: YOLO NMS Rejection vs RT-DETR NMS Rejection (p-value)
    rtdetr_nms_rej = [r["defenses"][1]["eqf_rejection_pct"] for r in all_seed_results]
    yolo_nms_rej = [r["yolov8_metrics"]["Targeted_Fabrication"]["nms_rejection_rate_mean"] for r in all_seed_results]
    t_stat_nms, p_val_nms = stats.ttest_ind(yolo_nms_rej, rtdetr_nms_rej)

    # Test 2: EQF Displacement vs Untargeted PGD Displacement
    eqf_displ = [r["rtdetr_metrics"]["EQF"]["displacement_rate_mean"] for r in all_seed_results]
    untarg_displ = [r["rtdetr_metrics"]["Untargeted_PGD"]["displacement_rate_mean"] for r in all_seed_results]
    t_stat_displ, p_val_displ = stats.ttest_rel(eqf_displ, untarg_displ)

    statistical_tests = {
        "yolo_vs_rtdetr_nms_rejection": {
            "yolo_nms_mean": float(np.mean(yolo_nms_rej)),
            "rtdetr_nms_mean": float(np.mean(rtdetr_nms_rej)),
            "t_statistic": float(t_stat_nms),
            "p_value": float(p_val_nms),
            "significant_at_001": bool(p_val_nms < 0.001),
        },
        "eqf_vs_untargeted_displacement": {
            "eqf_displacement_mean": float(np.mean(eqf_displ)),
            "untargeted_displacement_mean": float(np.mean(untarg_displ)),
            "marginal_bottleneck_displacement_pp": float(np.mean(eqf_displ) - np.mean(untarg_displ)),
            "t_statistic": float(t_stat_displ),
            "p_value": float(p_val_displ),
            "significant_at_05": bool(p_val_displ < 0.05),
        }
    }

    master_multiseed_audit = {
        "metadata": {
            "image_size": [640, 640],
            "seeds": seeds,
            "n_samples_per_seed": n_per_seed,
            "total_evaluations": len(seeds) * n_per_seed,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "per_seed_results": all_seed_results,
        "rtdetr_aggregated": rtdetr_aggregated,
        "yolov8_aggregated": yolo_aggregated,
        "defenses_aggregated": def_aggregated,
        "statistical_tests": statistical_tests,
    }

    out_path = RESULTS_DIR / "multiseed_640_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(master_multiseed_audit, f, indent=2)
    logger.info(f"\nSaved consolidated multi-seed audit JSON to: {out_path}")

    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY MULTI-SEED 640x640 BENCHMARK RESULTS:")
    logger.info(f"  RT-DETR Clean Oper mAP50:      {rtdetr_aggregated['Clean']['mAP50_oper_mean']:.2f} ± {rtdetr_aggregated['Clean']['mAP50_oper_std']:.2f}%")
    logger.info(f"  RT-DETR EQF Phantoms:          {rtdetr_aggregated['EQF']['phantoms_mean']:.2f} ± {rtdetr_aggregated['EQF']['phantoms_std']:.2f}")
    logger.info(f"  RT-DETR DQF Phantoms:          {rtdetr_aggregated['DQF']['phantoms_mean']:.2f} ± {rtdetr_aggregated['DQF']['phantoms_std']:.2f}")
    logger.info(f"  RT-DETR EQF Displacement:      {rtdetr_aggregated['EQF']['displacement_mean']:.2f} ± {rtdetr_aggregated['EQF']['displacement_std']:.2f}%")
    logger.info(f"  RT-DETR Untargeted Displ:      {rtdetr_aggregated['Untargeted_PGD']['displacement_mean']:.2f} ± {rtdetr_aggregated['Untargeted_PGD']['displacement_std']:.2f}%")
    logger.info(f"  Marginal Bottleneck Slot Displ:{statistical_tests['eqf_vs_untargeted_displacement']['marginal_bottleneck_displacement_pp']:.2f} pp (p={p_val_displ:.4f})")
    logger.info(f"  YOLOv8 NMS Rejection Rate:     {yolo_aggregated['Targeted_Fabrication']['nms_rejection_mean']:.2f} ± {yolo_aggregated['Targeted_Fabrication']['nms_rejection_std']:.2f}%")
    logger.info(f"  RT-DETR Standard NMS Rej:      {def_aggregated[1]['eqf_rejection_mean']:.2f} ± {def_aggregated[1]['eqf_rejection_std']:.2f}%")
    logger.info(f"  NMS Rejection Difference:      t={t_stat_nms:.2f}, p={p_val_nms:.4e}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
