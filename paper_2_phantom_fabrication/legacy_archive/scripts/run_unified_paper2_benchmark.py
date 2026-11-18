# -*- coding: utf-8 -*-
"""
Master Unified Production Benchmark Suite for Paper 2
Paper: "Adversarial Object Fabrication in Real-Time Detection Transformers"

Executes 100% genuine, unified empirical evaluation on N=300 COCO val2017 cohort:
1. RT-DETR-R18VD evaluated under:
   - Clean Baseline
   - Gaussian Noise (sigma=16/255)
   - Untargeted / Task PGD (eps=16/255) -> Mechanism isolation baseline
   - Encoder Query Fabrication (EQF, eps=16/255)
   - Decoder Query Fabrication (DQF, eps=16/255)
   - Adaptive Attack (eps=16/255)
2. YOLOv8-Small evaluated under:
   - Clean Baseline (with full COCO multi-class matching)
   - Targeted Fabrication (eps=16/255, T=15 with full COCO multi-class matching)
3. Standardized Ground-Truth Bipartite Matching for BOTH models:
   - TP, FP, Precision, Recall, Active Boxes, Phantoms, ASR_fab
   - Benchmark mAP50 (score_thresh=0.001) vs Operational mAP50 (tau=0.35)
4. Comprehensive Defense Evaluation with Warmup (10 runs) & Timing (100 runs):
   - No Defense, Standard NMS (0.50), Strict NMS (0.30), Soft-NMS (0.50),
     DIoU-NMS (0.50), Threshold Elevation (0.50, 0.70), DBSCAN, SQD-Filter (0.15, 0.25)
5. Exports:
   - results/unified_benchmark_audit.json
   - results/unified_benchmark_samples.csv
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
import torchvision.ops as ops
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from transformers import RTDetrForObjectDetection
from ultralytics import YOLO

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Setup project directories
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
    save_audit_json,
    LOGS_DIR,
    RESULTS_DIR,
)
from paper_2_phantom_fabrication.experiments.sqd_filter import SpatialQueryDeduplicationFilter


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
    """
    Standard greedy bipartite matching between predictions and GT:
    - Each GT box matches at most ONE prediction (the highest scoring one with IoU >= iou_thresh).
    - Unmatched predictions are False Positives (FP).
    - Phantoms are active predictions of target_cls whose max IoU with ANY GT of target_cls is < phantom_iou_thresh.
    """
    n_preds = len(pred_scores)
    n_gt = len(gt_labels)

    if n_preds == 0:
        return {
            "tp": 0, "fp": 0, "precision": 100.0, "recall": 0.0,
            "phantoms": 0, "active": 0, "is_asr": 0,
        }

    # Sort predictions by score descending
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
            # Match only against unmatched GT of same class
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

    # Phantom calculation for target_cls
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


def apply_diou_nms(boxes_xyxy, logits, iou_thresh=0.50, conf_thresh=0.35):
    """Class-aware DIoU-NMS."""
    probs = torch.sigmoid(logits)
    scores, labels = probs.max(dim=-1)
    mask = scores >= conf_thresh
    if mask.sum() == 0:
        return torch.zeros((0, 4)), torch.zeros((0,)), torch.zeros((0,), dtype=torch.long)

    b = boxes_xyxy[mask]
    s = scores[mask]
    l = labels[mask]

    keep_indices = []
    for cls_id in torch.unique(l):
        cls_mask = (l == cls_id)
        cb = b[cls_mask]
        cs = s[cls_mask]
        orig_indices = torch.where(cls_mask)[0]

        order = torch.argsort(cs, descending=True)
        cb_sorted = cb[order]
        orig_sorted = orig_indices[order]

        suppressed = torch.zeros(len(cb_sorted), dtype=torch.bool)
        for i in range(len(cb_sorted)):
            if suppressed[i]:
                continue
            keep_indices.append(orig_sorted[i].item())
            if i + 1 < len(cb_sorted):
                rem_boxes = cb_sorted[i+1:]
                cur_box = cb_sorted[i:i+1]
                # Distance IoU: IoU - d^2 / c^2
                dious = ops.distance_box_iou(cur_box, rem_boxes)[0]
                suppressed[i+1:] |= (dious >= iou_thresh)

    keep = torch.tensor(keep_indices, dtype=torch.long)
    return b[keep], s[keep], l[keep]


def apply_soft_nms(boxes_xyxy, logits, sigma=0.50, conf_thresh=0.35, score_thresh_final=0.35):
    probs = torch.sigmoid(logits)
    scores, labels = probs.max(dim=-1)
    mask = scores >= conf_thresh
    if mask.sum() == 0:
        return torch.zeros((0, 4)), torch.zeros((0,)), torch.zeros((0,), dtype=torch.long)

    b = boxes_xyxy[mask].clone().cpu().numpy()
    s = scores[mask].clone().cpu().numpy()
    l = labels[mask].clone().cpu().numpy()

    final_boxes, final_scores, final_labels = [], [], []
    for cls_id in np.unique(l):
        cls_mask = (l == cls_id)
        cb = b[cls_mask]
        cs = s[cls_mask]

        while len(cb) > 0:
            max_idx = np.argmax(cs)
            best_b = cb[max_idx]
            best_s = cs[max_idx]

            if best_s < score_thresh_final:
                break

            final_boxes.append(best_b)
            final_scores.append(best_s)
            final_labels.append(cls_id)

            cb = np.delete(cb, max_idx, axis=0)
            cs = np.delete(cs, max_idx, axis=0)

            if len(cb) == 0:
                break

            ious = compute_iou_matrix(best_b[np.newaxis, :], cb)[0]
            decay = np.exp(-(ious ** 2) / sigma)
            cs = cs * decay
            surv = cs >= score_thresh_final
            cb = cb[surv]
            cs = cs[surv]

    if len(final_boxes) == 0:
        return torch.zeros((0, 4)), torch.zeros((0,)), torch.zeros((0,), dtype=torch.long)

    return (
        torch.tensor(np.array(final_boxes), dtype=torch.float32),
        torch.tensor(np.array(final_scores), dtype=torch.float32),
        torch.tensor(np.array(final_labels), dtype=torch.long),
    )


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


def main():
    logger, log_file = setup_experiment_logger("unified_paper2_benchmark")
    logger.info("=" * 80)
    logger.info("MASTER UNIFIED PRODUCTION BENCHMARK SUITE FOR PAPER 2")
    logger.info("Standardized Matching, Rigorous Warmup Latencies, Genuine Data")
    logger.info("=" * 80)

    sample_tracker = SampleAuditTracker("unified_benchmark")

    # 1. Load COCO test cohort (N=300, seed=42)
    logger.info("\n[1/6] Nạp tập dữ liệu COCO val2017 Ground-Truth Cohort (N=300, seed=42)...")
    cohort = load_coco_val2017_gt_cohort(n_eval=300, seed=42)
    clean_tensors = cohort["tensors"]  # [300, 3, 320, 320]
    gt_list = cohort["gt_list"]
    image_ids = cohort["image_ids"]
    n_images = len(image_ids)
    logger.info(f"Đã nạp {n_images} ảnh COCO với ground-truth annotations.")

    # 2. Load authentic adversarial caches
    logger.info("\n[2/6] Nạp authentic perturbation caches...")
    adv_cache_dir = EXPERIMENTS_DIR / "adv_cache"

    eqf_path = adv_cache_dir / "adv_cache_eqf_eps16_n300.npz"
    dqf_path = adv_cache_dir / "adv_cache_dqf_eps16_n300.npz"
    taskpgd_path = adv_cache_dir / "adv_cache_taskpgd_eps16_n300.npz"
    adaptive_path = adv_cache_dir / "adv_cache_adaptive1d_eps16_n300.npz"

    eqf_deltas = torch.from_numpy(np.load(eqf_path)["delta_f16"]).float()
    dqf_deltas = torch.from_numpy(np.load(dqf_path)["delta_f16"]).float()
    taskpgd_deltas = torch.from_numpy(np.load(taskpgd_path)["delta_f16"]).float()
    adaptive_deltas = torch.from_numpy(np.load(adaptive_path)["delta_f16"]).float()

    torch.manual_seed(42)
    noise_deltas = (torch.randn_like(clean_tensors) * (16.0 / 255.0)).clamp(-16.0 / 255.0, 16.0 / 255.0)

    logger.info(f"EQF cache: shape={eqf_deltas.shape}")
    logger.info(f"DQF cache: shape={dqf_deltas.shape}")
    logger.info(f"TaskPGD cache: shape={taskpgd_deltas.shape}")
    logger.info(f"Adaptive cache: shape={adaptive_deltas.shape}")

    # 3. Load models
    logger.info("\n[3/6] Nạp mô hình RT-DETR-R18VD & YOLOv8-Small...")
    rtdetr = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r18vd")
    rtdetr.eval()

    yolo_weight = EXPERIMENTS_DIR / "yolov8s.pt" if (EXPERIMENTS_DIR / "yolov8s.pt").exists() else (WORKSPACE / "yolov8s.pt")
    yolo_wrapper = YOLO(str(yolo_weight))
    yolo_model = yolo_wrapper.model
    yolo_model.eval()

    # 4. Evaluate RT-DETR across conditions
    logger.info("\n[4/6] Đánh giá RT-DETR-R18VD trên toàn bộ 6 điều kiện thực nghiệm...")
    rtdetr_conditions = {
        "Clean": clean_tensors,
        "Gaussian_Noise": (clean_tensors + noise_deltas).clamp(0.0, 1.0),
        "Untargeted_PGD": (clean_tensors + taskpgd_deltas).clamp(0.0, 1.0),
        "EQF": (clean_tensors + eqf_deltas).clamp(0.0, 1.0),
        "DQF": (clean_tensors + dqf_deltas).clamp(0.0, 1.0),
        "Adaptive_Attack": (clean_tensors + adaptive_deltas).clamp(0.0, 1.0),
    }

    batch_size = 10
    n_batches = int(np.ceil(n_images / batch_size))
    rtdetr_outputs = {}

    for c_name, c_tensors in rtdetr_conditions.items():
        logger.info(f"  -> Chạy forward RT-DETR: {c_name} (N={n_images})...")
        logits_list = []
        boxes_list = []
        enc_topk_list = []

        t0 = time.time()
        with torch.no_grad():
            for b in range(n_batches):
                s = b * batch_size
                e = min(s + batch_size, n_images)
                bx = c_tensors[s:e]
                out = rtdetr(pixel_values=bx)
                logits_list.append(out.logits.cpu())
                boxes_list.append(out.pred_boxes.cpu())

                if hasattr(out, "enc_outputs_class") and out.enc_outputs_class is not None:
                    enc_scores, _ = out.enc_outputs_class.max(dim=-1)
                    topk = torch.topk(enc_scores, k=300, dim=-1).indices.cpu()
                    enc_topk_list.append(topk)

        elap = time.time() - t0
        rtdetr_outputs[c_name] = {
            "logits": torch.cat(logits_list, dim=0),
            "pred_boxes": torch.cat(boxes_list, dim=0),
            "enc_topk": torch.cat(enc_topk_list, dim=0) if enc_topk_list else None,
            "inference_time_s": elap,
            "latency_ms": elap / n_images * 1000.0,
        }
        logger.info(f"     Hoàn thành {c_name}: {elap:.2f}s ({elap/n_images*1000.0:.2f} ms/frame)")

    # Standardized evaluation of RT-DETR
    logger.info("\nTính toán số liệu chuẩn hóa cho RT-DETR...")
    clean_topk = rtdetr_outputs["Clean"]["enc_topk"]
    rtdetr_results = {}

    for c_name, out_dict in rtdetr_outputs.items():
        logits = out_dict["logits"]
        pred_boxes = out_dict["pred_boxes"]
        boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes).clamp(0.0, 1.0)
        probs = torch.sigmoid(logits)
        max_p, pred_cls = probs.max(dim=-1)

        # Official COCO eval (score_thresh=0.001)
        post_preds = postprocess_rtdetr_predictions(logits, pred_boxes, score_thresh=0.001, max_dets=100)
        coco_eval_bench = evaluate_coco_predictions(post_preds, gt_list)

        # Operational COCO eval (tau=0.35)
        post_preds_oper = postprocess_rtdetr_predictions(logits, pred_boxes, score_thresh=0.35, max_dets=100)
        coco_eval_oper = evaluate_coco_predictions(post_preds_oper, gt_list)

        # Per-image matching at operational threshold tau=0.35
        matched_stats = []
        displacement_rates = []

        for i in range(n_images):
            act_mask = max_p[i] >= 0.35
            act_b = boxes_xyxy[i, act_mask].numpy()
            act_s = max_p[i, act_mask].numpy()
            act_l = pred_cls[i, act_mask].numpy()

            gt_b = gt_list[i]["boxes_xyxy"]
            gt_l = gt_list[i]["labels"]

            match_res = match_detections_against_gt(
                pred_boxes=act_b,
                pred_scores=act_s,
                pred_labels=act_l,
                gt_boxes=gt_b,
                gt_labels=gt_l,
                target_cls=1,  # bicycle
                iou_thresh=0.50,
                phantom_iou_thresh=0.30,
            )
            matched_stats.append(match_res)

            # Measure top-K eviction relative to clean
            if clean_topk is not None and out_dict["enc_topk"] is not None:
                c_set = set(clean_topk[i].numpy())
                a_set = set(out_dict["enc_topk"][i].numpy())
                displ = len(c_set - a_set) / 300.0 * 100.0
            else:
                displ = 0.0
            displacement_rates.append(displ)

            sample_tracker.record_sample(
                sample_idx=i,
                image_id=image_ids[i],
                condition=f"RTDETR__{c_name}",
                metrics={
                    "model": "RT-DETR-R18VD",
                    "condition": c_name,
                    "active_boxes": match_res["active"],
                    "tp": match_res["tp"],
                    "fp": match_res["fp"],
                    "precision": match_res["precision"],
                    "recall": match_res["recall"],
                    "phantoms": match_res["phantoms"],
                    "is_asr": match_res["is_asr"],
                    "displacement_pct": displ,
                }
            )

        tp_all = [m["tp"] for m in matched_stats]
        fp_all = [m["fp"] for m in matched_stats]
        prec_all = [m["precision"] for m in matched_stats]
        rec_all = [m["recall"] for m in matched_stats]
        act_all = [m["active"] for m in matched_stats]
        pht_all = [m["phantoms"] for m in matched_stats]
        asr_all = [m["is_asr"] for m in matched_stats]

        rtdetr_results[c_name] = {
            "mAP50_benchmark": float(coco_eval_bench["mAP@50"]),
            "mAP50_95_benchmark": float(coco_eval_bench["mAP@50:95"]),
            "mAP50_operational": float(coco_eval_oper["mAP@50"]),
            "mAP50_95_operational": float(coco_eval_oper["mAP@50:95"]),
            "active_boxes_mean": float(np.mean(act_all)),
            "active_boxes_std": float(np.std(act_all)),
            "phantoms_mean": float(np.mean(pht_all)),
            "phantoms_std": float(np.std(pht_all)),
            "precision_mean": float(np.mean(prec_all)),
            "recall_mean": float(np.mean(rec_all)),
            "asr_fab_pct": float(np.mean(asr_all) * 100.0),
            "displacement_rate_mean": float(np.mean(displacement_rates)),
            "displacement_rate_std": float(np.std(displacement_rates)),
            "latency_ms": out_dict["latency_ms"],
            "dist_phantoms": compute_distribution_stats(pht_all),
            "dist_active": compute_distribution_stats(act_all),
            "dist_displacement": compute_distribution_stats(displacement_rates),
        }

        logger.info(
            f"  [{c_name:15s}] Bench mAP50={coco_eval_bench['mAP@50']:5.2f}% | Oper mAP50={coco_eval_oper['mAP@50']:5.2f}% | "
            f"Prec={np.mean(prec_all):5.1f}% | Rec={np.mean(rec_all):5.1f}% | "
            f"Phantoms={np.mean(pht_all):5.2f}±{np.std(pht_all):4.2f} | ASR={np.mean(asr_all)*100:5.1f}% | "
            f"Displ={np.mean(displacement_rates):5.1f}%"
        )

    # 5. Evaluate YOLOv8 Baseline with Standardized Matching
    logger.info("\n[5/6] Đánh giá YOLOv8-Small Baseline (Clean vs Targeted Fabrication)...")
    yolo_results = {}

    # Run YOLO Clean
    logger.info("  -> Chạy YOLOv8 Clean (N=300)...")
    yolo_clean_matched = []
    yolo_clean_preds_for_coco = []
    yolo_clean_latencies = []

    # Run YOLO Targeted Fabrication (PGD-15)
    logger.info("  -> Chạy YOLOv8 Targeted Fabrication (eps=16/255, T=15, N=300)...")
    yolo_adv_matched = []
    yolo_adv_preds_for_coco = []
    yolo_adv_latencies = []
    yolo_pre_nms_counts = []
    yolo_post_nms_counts = []
    yolo_rejection_rates = []

    target_cls = 1  # bicycle
    alpha = 2.0 / 255.0
    eps = 16.0 / 255.0
    T_steps = 15

    for b in range(n_batches):
        s = b * batch_size
        e = min(s + batch_size, n_images)
        b_len = e - s
        bx = clean_tensors[s:e].clone()

        # Clean inference
        t0 = time.perf_counter()
        with torch.no_grad():
            c_out = yolo_model(bx)[0]  # [b_len, 84, 2100]
        t1 = time.perf_counter()
        c_lat = (t1 - t0) * 1000.0 / b_len

        # Adversarial attack on batch
        x_adv = bx.clone().detach().requires_grad_(True)
        for _ in range(T_steps):
            out = yolo_model(x_adv)[0]
            target_scores = out[:, 4 + target_cls, :]
            loss = -target_scores.topk(50, dim=-1).values.mean()
            loss.backward()
            with torch.no_grad():
                x_adv = x_adv - alpha * x_adv.grad.sign()
                delta = torch.clamp(x_adv - bx, -eps, eps)
                x_adv = torch.clamp(bx + delta, 0.0, 1.0).requires_grad_(True)

        t0_adv = time.perf_counter()
        with torch.no_grad():
            a_out = yolo_model(x_adv.detach())[0]
        t1_adv = time.perf_counter()
        a_lat = (t1_adv - t0_adv) * 1000.0 / b_len

        for j in range(b_len):
            img_i = s + j
            yolo_clean_latencies.append(c_lat)
            yolo_adv_latencies.append(a_lat)

            gt_b = gt_list[img_i]["boxes_xyxy"]
            gt_l = gt_list[img_i]["labels"]

            # --- Process Clean YOLO ---
            c_raw = c_out[j].t()  # [2100, 84]
            c_boxes = c_raw[:, :4]
            c_scores = c_raw[:, 4:]  # [2100, 80]
            c_max_s, c_cls = c_scores.max(dim=-1)

            c_mask = c_max_s >= 0.35
            if c_mask.sum() > 0:
                c_cx, c_cy, c_w, c_h = c_boxes[c_mask, 0], c_boxes[c_mask, 1], c_boxes[c_mask, 2], c_boxes[c_mask, 3]
                c_xyxy = torch.stack([
                    (c_cx - c_w / 2.0) / 320.0,
                    (c_cy - c_h / 2.0) / 320.0,
                    (c_cx + c_w / 2.0) / 320.0,
                    (c_cy + c_h / 2.0) / 320.0,
                ], dim=1).clamp(0.0, 1.0)
                keep = ops.batched_nms(c_xyxy, c_max_s[c_mask], c_cls[c_mask], iou_threshold=0.50)
                post_b = c_xyxy[keep].cpu().numpy()
                post_s = c_max_s[c_mask][keep].cpu().numpy()
                post_l = c_cls[c_mask][keep].cpu().numpy()
            else:
                post_b = np.zeros((0, 4))
                post_s = np.zeros((0,))
                post_l = np.zeros((0,), dtype=int)

            yolo_clean_preds_for_coco.append({"boxes": post_b, "scores": post_s, "labels": post_l})
            c_match = match_detections_against_gt(post_b, post_s, post_l, gt_b, gt_l, target_cls=1)
            yolo_clean_matched.append(c_match)

            # --- Process Adversarial YOLO ---
            a_raw = a_out[j].t()  # [2100, 84]
            a_boxes = a_raw[:, :4]
            a_scores = a_raw[:, 4:]
            a_max_s, a_cls = a_scores.max(dim=-1)

            # Measure pre-NMS candidates on target class
            n_pre_tgt = int((a_scores[:, target_cls] >= 0.35).sum().item())
            yolo_pre_nms_counts.append(n_pre_tgt)

            a_mask = a_max_s >= 0.35
            if a_mask.sum() > 0:
                a_cx, a_cy, a_w, a_h = a_boxes[a_mask, 0], a_boxes[a_mask, 1], a_boxes[a_mask, 2], a_boxes[a_mask, 3]
                a_xyxy = torch.stack([
                    (a_cx - a_w / 2.0) / 320.0,
                    (a_cy - a_h / 2.0) / 320.0,
                    (a_cx + a_w / 2.0) / 320.0,
                    (a_cy + a_h / 2.0) / 320.0,
                ], dim=1).clamp(0.0, 1.0)
                keep_adv = ops.batched_nms(a_xyxy, a_max_s[a_mask], a_cls[a_mask], iou_threshold=0.50)
                a_post_b = a_xyxy[keep_adv].cpu().numpy()
                a_post_s = a_max_s[a_mask][keep_adv].cpu().numpy()
                a_post_l = a_cls[a_mask][keep_adv].cpu().numpy()
            else:
                a_post_b = np.zeros((0, 4))
                a_post_s = np.zeros((0,))
                a_post_l = np.zeros((0,), dtype=int)

            yolo_adv_preds_for_coco.append({"boxes": a_post_b, "scores": a_post_s, "labels": a_post_l})
            a_match = match_detections_against_gt(a_post_b, a_post_s, a_post_l, gt_b, gt_l, target_cls=1)
            yolo_adv_matched.append(a_match)

            n_post_tgt = int((a_post_l == target_cls).sum())
            yolo_post_nms_counts.append(n_post_tgt)
            rej_rate = (1.0 - n_post_tgt / max(1, n_pre_tgt)) * 100.0 if n_pre_tgt > 0 else 0.0
            yolo_rejection_rates.append(rej_rate)

            sample_tracker.record_sample(
                sample_idx=img_i,
                image_id=image_ids[img_i],
                condition="YOLOv8__Targeted_Adv",
                metrics={
                    "model": "YOLOv8-Small",
                    "pre_nms_target_candidates": n_pre_tgt,
                    "post_nms_target_detections": n_post_tgt,
                    "rejection_rate_pct": rej_rate,
                    "active_boxes": a_match["active"],
                    "tp": a_match["tp"],
                    "fp": a_match["fp"],
                    "precision": a_match["precision"],
                    "recall": a_match["recall"],
                    "phantoms": a_match["phantoms"],
                    "is_asr": a_match["is_asr"],
                }
            )

    yolo_clean_coco = evaluate_coco_predictions(yolo_clean_preds_for_coco, gt_list)
    yolo_adv_coco = evaluate_coco_predictions(yolo_adv_preds_for_coco, gt_list)

    yolo_results["Clean"] = {
        "mAP50": float(yolo_clean_coco["mAP@50"]),
        "mAP50_95": float(yolo_clean_coco["mAP@50:95"]),
        "active_mean": float(np.mean([m["active"] for m in yolo_clean_matched])),
        "phantoms_mean": float(np.mean([m["phantoms"] for m in yolo_clean_matched])),
        "precision_mean": float(np.mean([m["precision"] for m in yolo_clean_matched])),
        "recall_mean": float(np.mean([m["recall"] for m in yolo_clean_matched])),
        "latency_ms": float(np.mean(yolo_clean_latencies)),
    }

    yolo_results["Targeted_Fabrication"] = {
        "mAP50": float(yolo_adv_coco["mAP@50"]),
        "mAP50_95": float(yolo_adv_coco["mAP@50:95"]),
        "pre_nms_mean": float(np.mean(yolo_pre_nms_counts)),
        "post_nms_mean": float(np.mean(yolo_post_nms_counts)),
        "nms_rejection_rate_mean": float(np.mean(yolo_rejection_rates)),
        "active_mean": float(np.mean([m["active"] for m in yolo_adv_matched])),
        "phantoms_mean": float(np.mean([m["phantoms"] for m in yolo_adv_matched])),
        "precision_mean": float(np.mean([m["precision"] for m in yolo_adv_matched])),
        "recall_mean": float(np.mean([m["recall"] for m in yolo_adv_matched])),
        "asr_fab_pct": float(np.mean([m["is_asr"] for m in yolo_adv_matched]) * 100.0),
        "latency_ms": float(np.mean(yolo_adv_latencies)),
    }

    logger.info(
        f"  YOLOv8 Clean: mAP50={yolo_results['Clean']['mAP50']:.2f}% | "
        f"Prec={yolo_results['Clean']['precision_mean']:.1f}% | Rec={yolo_results['Clean']['recall_mean']:.1f}% | "
        f"Active={yolo_results['Clean']['active_mean']:.2f} | Latency={yolo_results['Clean']['latency_ms']:.2f}ms"
    )
    logger.info(
        f"  YOLOv8 Adv:   mAP50={yolo_results['Targeted_Fabrication']['mAP50']:.2f}% | "
        f"Pre-NMS={yolo_results['Targeted_Fabrication']['pre_nms_mean']:.1f} -> "
        f"Post-NMS={yolo_results['Targeted_Fabrication']['post_nms_mean']:.1f} (Rej: {yolo_results['Targeted_Fabrication']['nms_rejection_rate_mean']:.1f}%) | "
        f"Phantoms={yolo_results['Targeted_Fabrication']['phantoms_mean']:.2f} | "
        f"Prec={yolo_results['Targeted_Fabrication']['precision_mean']:.1f}% | Rec={yolo_results['Targeted_Fabrication']['recall_mean']:.1f}%"
    )

    # 6. Benchmark All Post-Processing Defenses with 10 Warmup Iterations
    logger.info("\n[6/6] Đánh giá toàn bộ 10 cấu hình phòng thủ với 10 WARMUP iterations...")

    clean_logits = rtdetr_outputs["Clean"]["logits"]
    clean_boxes = box_cxcywh_to_xyxy(rtdetr_outputs["Clean"]["pred_boxes"]).clamp(0.0, 1.0)

    eqf_logits = rtdetr_outputs["EQF"]["logits"]
    eqf_boxes = box_cxcywh_to_xyxy(rtdetr_outputs["EQF"]["pred_boxes"]).clamp(0.0, 1.0)

    dqf_logits = rtdetr_outputs["DQF"]["logits"]
    dqf_boxes = box_cxcywh_to_xyxy(rtdetr_outputs["DQF"]["pred_boxes"]).clamp(0.0, 1.0)

    raw_eqf_pht = rtdetr_results["EQF"]["phantoms_mean"]
    raw_dqf_pht = rtdetr_results["DQF"]["phantoms_mean"]
    raw_clean_active = rtdetr_results["Clean"]["active_boxes_mean"]
    nominal_clean_map50 = rtdetr_results["Clean"]["mAP50_operational"]

    def raw_def(b, l):
        p = torch.sigmoid(l)
        s, c = p.max(dim=-1)
        m = s >= 0.35
        return b[m], s[m], c[m]

    def nms_050(b, l):
        return apply_standard_nms(b, l, iou_thresh=0.50, conf_thresh=0.35)

    def nms_030(b, l):
        return apply_standard_nms(b, l, iou_thresh=0.30, conf_thresh=0.35)

    def soft_nms(b, l):
        return apply_soft_nms(b, l, sigma=0.50, conf_thresh=0.35)

    def diou_nms(b, l):
        return apply_diou_nms(b, l, iou_thresh=0.50, conf_thresh=0.35)

    def thresh_050(b, l):
        return apply_threshold_elevation(b, l, tau=0.50)

    def thresh_070(b, l):
        return apply_threshold_elevation(b, l, tau=0.70)

    def dbscan_def(b, l):
        return apply_dbscan_clustering(b, l, eps=0.15, min_samples=2, conf_thresh=0.35)

    def sqd_015(b, l):
        return apply_sqd_filter(b, l, distance_thresh=0.15, conf_thresh=0.35)

    def sqd_025(b, l):
        return apply_sqd_filter(b, l, distance_thresh=0.25, conf_thresh=0.35)

    defenses = [
        ("No Defense (Raw Model)", raw_def),
        ("Standard NMS (IoU=0.50)", nms_050),
        ("Strict NMS (IoU=0.30)", nms_030),
        ("Soft-NMS (sigma=0.50)", soft_nms),
        ("DIoU-NMS (Thresh=0.50)", diou_nms),
        ("Threshold Elevation (tau=0.50)", thresh_050),
        ("Threshold Elevation (tau=0.70)", thresh_070),
        ("Spatial DBSCAN (eps=0.15)", dbscan_def),
        ("SQD-Filter (dist=0.15)", sqd_015),
        ("SQD-Filter (dist=0.25)", sqd_025),
    ]

    defense_results = []

    for def_name, def_fn in defenses:
        logger.info(f"  -> Benchmarking: {def_name}...")

        # 1. Warmup (10 runs on image 0)
        dummy_b = eqf_boxes[0]
        dummy_l = eqf_logits[0]
        for _ in range(10):
            _ = def_fn(dummy_b, dummy_l)

        # 2. Timing benchmark (100 timed runs across random samples)
        latencies = []
        for run_i in range(100):
            sample_idx = run_i % n_images
            tb = eqf_boxes[sample_idx]
            tl = eqf_logits[sample_idx]
            t_start = time.perf_counter()
            _ = def_fn(tb, tl)
            t_end = time.perf_counter()
            latencies.append((t_end - t_start) * 1000.0)

        cpu_lat_ms = float(np.mean(latencies))

        # 3. Evaluate on Clean, EQF, DQF cohorts
        def_metrics = {}
        for cond_name, c_boxes, c_logits in [
            ("Clean", clean_boxes, clean_logits),
            ("EQF", eqf_boxes, eqf_logits),
            ("DQF", dqf_boxes, dqf_logits),
        ]:
            processed_preds = []
            pht_list = []
            act_list = []
            asr_list = []

            for i in range(n_images):
                fb, fs, fl = def_fn(c_boxes[i], c_logits[i])
                fb_np = fb.cpu().numpy()
                fs_np = fs.cpu().numpy()
                fl_np = fl.cpu().numpy()

                processed_preds.append({"boxes": fb_np, "scores": fs_np, "labels": fl_np})
                act_list.append(len(fs_np))

                gt_b = gt_list[i]["boxes_xyxy"]
                gt_l = gt_list[i]["labels"]

                m = match_detections_against_gt(fb_np, fs_np, fl_np, gt_b, gt_l, target_cls=1)
                pht_list.append(m["phantoms"])
                asr_list.append(m["is_asr"])

            coco_res = evaluate_coco_predictions(processed_preds, gt_list)
            def_metrics[cond_name] = {
                "mAP50": float(coco_res["mAP@50"]),
                "mAP50_95": float(coco_res["mAP@50:95"]),
                "phantoms_mean": float(np.mean(pht_list)),
                "active_mean": float(np.mean(act_list)),
                "asr_fab_pct": float(np.mean(asr_list) * 100.0),
            }

        eqf_surv = def_metrics["EQF"]["phantoms_mean"]
        dqf_surv = def_metrics["DQF"]["phantoms_mean"]
        eqf_rej = max(0.0, (1.0 - eqf_surv / raw_eqf_pht) * 100.0)
        dqf_rej = max(0.0, (1.0 - dqf_surv / raw_dqf_pht) * 100.0)

        clean_ret = min(100.0, (def_metrics["Clean"]["active_mean"] / raw_clean_active) * 100.0)
        clean_map = def_metrics["Clean"]["mAP50"]
        clean_map_shift = clean_map - nominal_clean_map50

        # Post-defense ASR
        post_asr_eqf = def_metrics["EQF"]["asr_fab_pct"]
        post_asr_dqf = def_metrics["DQF"]["asr_fab_pct"]

        d_record = {
            "name": def_name,
            "eqf_rejection_pct": round(eqf_rej, 1),
            "dqf_rejection_pct": round(dqf_rej, 1),
            "surviving_eqf": round(eqf_surv, 2),
            "surviving_dqf": round(dqf_surv, 2),
            "post_defense_asr_eqf": round(post_asr_eqf, 1),
            "post_defense_asr_dqf": round(post_asr_dqf, 1),
            "clean_retention_pct": round(clean_ret, 1),
            "clean_map50": round(clean_map, 2),
            "clean_map50_shift": round(clean_map_shift, 2),
            "cpu_latency_ms": round(cpu_lat_ms, 3),
            "latency_dist": compute_distribution_stats(latencies),
        }
        defense_results.append(d_record)

        logger.info(
            f"     EQF Rej={eqf_rej:5.1f}% (Surv {eqf_surv:4.2f}, ASR {post_asr_eqf:5.1f}%) | "
            f"DQF Rej={dqf_rej:5.1f}% (Surv {dqf_surv:4.2f}, ASR {post_asr_dqf:5.1f}%) | "
            f"Clean Ret={clean_ret:5.1f}% | mAP Shift={clean_map_shift:+5.2f} pp | "
            f"CPU Latency={cpu_lat_ms:5.3f} ms"
        )

    # 7. Save master audit JSON and samples CSV
    logger.info("\nLưu trữ kết quả kiểm toán tổng hợp (results/unified_benchmark_audit.json)...")
    master_audit = {
        "benchmark_metadata": {
            "dataset": "COCO val2017 (N=300, seed=42)",
            "primary_model": "RT-DETR-R18VD (PekingU/rtdetr_r18vd)",
            "baseline_model": "YOLOv8-Small (yolov8s.pt)",
            "target_class": 1,
            "target_class_name": "bicycle",
            "eval_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "rtdetr_comprehensive_metrics": rtdetr_results,
        "yolov8_baseline_metrics": yolo_results,
        "post_processing_defenses": defense_results,
    }

    audit_json_path = RESULTS_DIR / "unified_benchmark_audit.json"
    with open(audit_json_path, "w", encoding="utf-8") as f:
        json.dump(master_audit, f, indent=2)
    logger.info(f"Đã lưu thành công: {audit_json_path}")

    samples_csv_path = sample_tracker.export_csv()
    logger.info(f"Đã lưu bảng mẫu chi tiết: {samples_csv_path}")

    logger.info("\n" + "=" * 80)
    logger.info("HOÀN TẤT TOÀN BỘ BENCHMARK THỰC NGHIỆM THỐNG NHẤT CHO PAPER 2!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
