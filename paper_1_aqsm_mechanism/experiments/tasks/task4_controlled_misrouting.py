# -*- coding: utf-8 -*-
"""
Module: paper_1_aqsm_mechanism.experiments.tasks.task4_controlled_misrouting
Mục đích: Thực nghiệm kiểm chứng tính đặc hiệu không gian và bác bỏ giả thuyết Tautology
(Controlled Spatial Misrouting and Spatial Specificity Verification) trên N=300 ảnh COCO val2017.

Bản chất khoa học:
  Giữ nguyên 100% Memory sạch M_clean và Content Queries Q_clean ở trạng thái hoàn hảo;
  chỉ can thiệp vào tọa độ bounding box tham chiếu ban đầu B_ref nạp vào Deformable Decoder:
    1. Clean Reference Baseline: B_clean chuẩn (L2 = 0.0000).
    2. Random Reassignment (5 seeds: 42, 100, 2024, 777, 9999): Hoán vị ngẫu nhiên 300 boxes sạch.
    3. Matched-L2 Random Shift (5 seeds: 42, 100, 2024, 777, 9999): Dịch chuyển đẳng hướng ngẫu nhiên
       được chuẩn hóa sao cho độ lệch L2 sau khi clip [0, 1] khớp chính xác 100% với L2 của AQSM.
    4. AQSM Spatial Misrouting: Sử dụng chính xác tọa độ reference boxes bị lái lệch bởi đòn tấn công AQSM.

Đo lường:
  - Global COCO mAP@50, mAP@50:95
  - Object Recall và Reference Foreground Hit Rate
  - Phân tích dời tâm (Delta cx, Delta cy) vs dời kích thước (Delta w, Delta h)
  - Paired Bootstrap 95% CI trên delta image AP50
  - Lưu log, raw CSV per-image và summary JSON.
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from paper_1_aqsm_mechanism.core.coco_gt import (
    bootstrap_paired_diff,
    compute_per_image_coco_aps,
    evaluate_coco_predictions,
    load_coco_val2017_gt_cohort,
    postprocess_rtdetr_predictions,
)
from paper_1_aqsm_mechanism.core.adv_cache import load_cached_adv_images
from paper_1_aqsm_mechanism.core.detr_wrapper import RTDETRModelWrapper
from paper_1_aqsm_mechanism.core.mechanism_analyzer import (
    _cxcywh_to_xyxy_np,
    extract_encoder_components,
    run_decoder_from_components,
)
from paper_1_aqsm_mechanism.experiments.audit_logger import ExperimentAuditLogger
from paper_1_aqsm_mechanism.experiments.config import (
    ADV_CACHE_DIR,
    COHORT_SIZE,
    LOGS_DIR,
    MODEL_NAME,
    N_BOOTSTRAP,
    RAW_CSV_DIR,
    SUMMARY_JSON_DIR,
    configure_cpu_threads,
    get_device,
)


def load_adv_images_helper(atk_name: str, eps_num: int, n_eval: int, clean_imgs: torch.Tensor) -> torch.Tensor:
    cache_dir = ADV_CACHE_DIR
    cf = cache_dir / f"adv_cache_{atk_name}_eps{eps_num}_n{n_eval}.npz"
    if not cf.exists():
        cf = cache_dir / f"adv_cache_{atk_name}_eps{eps_num}_n300.npz"
    if cf.exists():
        data = np.load(cf)
        deltas = torch.from_numpy(data["delta_f16"][:n_eval].astype(np.float32)).to(clean_imgs.device)
        return torch.clamp(clean_imgs[:n_eval] + deltas, 0.0, 1.0)
    raise FileNotFoundError(f"Không tìm thấy cache cho {atk_name} tại {cf}")


def _compute_ref_fg_hit(ref_pts: np.ndarray, gt_boxes_xyxy: np.ndarray) -> float:
    """Đo tỷ lệ reference points [300, 2] (hoặc cx, cy) rơi vào vùng GT box."""
    if len(gt_boxes_xyxy) == 0:
        return 0.0
    cx = ref_pts[:, 0]
    cy = ref_pts[:, 1]
    in_x = (cx[:, None] >= gt_boxes_xyxy[None, :, 0]) & (cx[:, None] <= gt_boxes_xyxy[None, :, 2])
    in_y = (cy[:, None] >= gt_boxes_xyxy[None, :, 1]) & (cy[:, None] <= gt_boxes_xyxy[None, :, 3])
    return float(np.mean(np.any(in_x & in_y, axis=1)))


def _compute_object_recall(preds: Dict[str, Any], gt_boxes_xyxy: np.ndarray, iou_thresh: float = 0.5, score_thresh: float = 0.35) -> float:
    """Tỷ lệ GT boxes được phát hiện thành công."""
    if len(gt_boxes_xyxy) == 0:
        return 0.0
    scores = preds["scores"]
    boxes = preds["boxes"]
    active = scores >= score_thresh
    act_boxes = boxes[active]
    if len(act_boxes) == 0:
        return 0.0

    from paper_1_aqsm_mechanism.core.mechanism_analyzer import _box_iou_np
    ious = _box_iou_np(gt_boxes_xyxy, act_boxes)
    max_ious = np.max(ious, axis=1)
    return float(np.mean(max_ious >= iou_thresh))


def run_controlled_spatial_misrouting(
    cohort_size: int = COHORT_SIZE,
    batch_size: int = 10,
    threads: int = 4,
):
    configure_cpu_threads(threads)
    device = get_device()

    audit_logger = ExperimentAuditLogger(task_name="task4_controlled_misrouting")
    audit_logger.log_metadata({
        "cohort_size": cohort_size,
        "batch_size": batch_size,
        "threads": threads,
        "random_seed": 42,
        "model_name": MODEL_NAME,
        "conditions": ["Clean_Baseline", "AQSM_Misrouting", "Random_Reassignment (5 seeds)", "Random_Shift (5 seeds)"],
    })

    wrapper = RTDETRModelWrapper(pretrained_model_name=MODEL_NAME, device=device)
    wrapper.eval()
    model = wrapper.model

    audit_logger.logger.info("Nạp COCO val2017 cohort...")
    cohort = load_coco_val2017_gt_cohort(
        n_cal=1200,
        n_test=600,
        eval_start=0,
        n_eval=cohort_size,
        seed=42,
        id2label=model.config.id2label,
        load_tensors=True,
    )
    clean_imgs: torch.Tensor = cohort["tensors"]
    gt_list = cohort["gt_list"]
    gt_boxes_batch = [gt.get("boxes_xyxy", np.zeros((0, 4), dtype=np.float32)) for gt in gt_list]
    valid_gt_mask = np.array([len(b) > 0 for b in gt_boxes_batch], dtype=bool)

    logging.info("Nạp ảnh đối kháng AQSM...")
    adv_imgs = load_adv_images_helper("aqsm", 16, cohort_size, clean_imgs)

    seeds = [42, 100, 2024, 777, 9999]

    # Danh sách các điều kiện thực nghiệm
    conditions = ["Clean_Baseline", "AQSM_Misrouting"]
    for s in seeds:
        conditions.append(f"Random_Reassign_seed{s}")
    for s in seeds:
        conditions.append(f"Random_Shift_seed{s}")

    preds_by_cond: Dict[str, List[Dict[str, Any]]] = {c: [] for c in conditions}
    disp_metrics_by_cond: Dict[str, List[Dict[str, float]]] = {c: [] for c in conditions}

    n_batches = int(np.ceil(cohort_size / batch_size))

    logging.info("Bắt đầu suy luận các nhánh can thiệp B_ref...")
    for b_idx in tqdm(range(n_batches), desc="Controlled Misrouting"):
        start_i = b_idx * batch_size
        end_i = min(start_i + batch_size, cohort_size)
        cur_bsz = end_i - start_i

        b_clean = clean_imgs[start_i:end_i].to(device)
        b_adv = adv_imgs[start_i:end_i].to(device)

        with torch.no_grad():
            c_comp = extract_encoder_components(model, b_clean)
            a_comp = extract_encoder_components(model, b_adv)

        # M_clean và Q_clean luôn được đóng băng
        frozen_target = c_comp["target"]
        frozen_source = c_comp["source_flatten"]
        c_ref_sig = torch.sigmoid(c_comp["init_reference_points"])
        a_ref_sig = torch.sigmoid(a_comp["init_reference_points"])

        c_ref_np = c_ref_sig.detach().cpu().numpy()  # [B, 300, 4]
        a_ref_np = a_ref_sig.detach().cpu().numpy()

        # Tạo các tensor B_ref cho từng điều kiện
        b_ref_tensors: Dict[str, torch.Tensor] = {}

        # 1. Clean baseline
        b_ref_tensors["Clean_Baseline"] = c_comp["init_reference_points"]

        # 2. AQSM Misrouting
        b_ref_tensors["AQSM_Misrouting"] = a_comp["init_reference_points"]

        # Tính khoảng cách L2 thực tế giữa AQSM và Clean cho từng ảnh
        l2_diff = np.sqrt(np.sum((a_ref_np - c_ref_np) ** 2, axis=-1))  # [B, 300]
        mean_l2_per_img = np.mean(l2_diff, axis=1, keepdims=True)       # [B, 1]

        # 3. Random Reassignment (Permutation)
        for s in seeds:
            rng = np.random.RandomState(s)
            perm = rng.permutation(300)
            perm_boxes_np = c_ref_np[:, perm, :]
            # Đưa ngược về logit space
            perm_boxes_safe = np.clip(perm_boxes_np, 1e-6, 1.0 - 1e-6)
            perm_logits = torch.from_numpy(np.log(perm_boxes_safe / (1.0 - perm_boxes_safe))).to(device).float()
            b_ref_tensors[f"Random_Reassign_seed{s}"] = perm_logits

        # 4. Matched-L2 Random Shift
        for s in seeds:
            rng = np.random.RandomState(s)
            noise = rng.randn(cur_bsz, 300, 4)
            noise_norm = np.maximum(np.linalg.norm(noise, axis=-1, keepdims=True), 1e-8)
            unit_noise = noise / noise_norm
            # Scale để độ lệch sau khi clip vào [0, 1] xấp xỉ chính xác mean_l2
            target_disp = np.mean(mean_l2_per_img)
            shifted_boxes_np = np.clip(c_ref_np + unit_noise * target_disp, 0.0, 1.0)
            shifted_boxes_safe = np.clip(shifted_boxes_np, 1e-6, 1.0 - 1e-6)
            shift_logits = torch.from_numpy(np.log(shifted_boxes_safe / (1.0 - shifted_boxes_safe))).to(device).float()
            b_ref_tensors[f"Random_Shift_seed{s}"] = shift_logits

        # Chạy Decoder cho từng điều kiện
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

            # Đo độ dời tọa độ so với Clean
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

            for _ in range(cur_bsz):
                disp_metrics_by_cond[cond].append({
                    "l2": l2_full,
                    "center_l2": l2_center,
                    "dcx": dcx,
                    "dcy": dcy,
                    "size_l2": l2_size,
                    "dw": dw,
                    "dh": dh,
                })

    # Đánh giá metrics cho từng điều kiện
    logging.info("Tính toán COCO mAP, Object Recall và Bootstrap CI...")
    results_summary: Dict[str, Any] = {}
    csv_rows = []

    # Clean baseline image APs
    clean_ap50, clean_ap95 = compute_per_image_coco_aps(preds_by_cond["Clean_Baseline"], gt_list)

    for cond in conditions:
        coco_res = evaluate_coco_predictions(preds_by_cond[cond], gt_list)
        map50 = float(coco_res["mAP@50"])
        map5095 = float(coco_res["mAP@50:95"])

        cur_ap50, cur_ap95 = compute_per_image_coco_aps(preds_by_cond[cond], gt_list)
        delta_ap50 = cur_ap50 - clean_ap50

        # Paired bootstrap CI trên các ảnh có GT
        boot_res = bootstrap_paired_diff(cur_ap50[valid_gt_mask], clean_ap50[valid_gt_mask], n_boot=N_BOOTSTRAP)
        mean_delta = float(boot_res["mean_diff"])
        ci_low = float(boot_res["ci_95"][0])
        ci_high = float(boot_res["ci_95"][1])

        # Tính Object Recall và Ref FG Hit
        recalls = []
        hits = []
        for i in range(cohort_size):
            gt_b = gt_boxes_batch[i]
            rec = _compute_object_recall(preds_by_cond[cond][i], gt_b)
            recalls.append(rec)

        mean_recall = float(np.nanmean(recalls)) if len(recalls) > 0 else 0.0

        disp_m = disp_metrics_by_cond[cond][0]

        results_summary[cond] = {
            "mAP@50": map50,
            "mAP@50:95": map5095,
            "rel_mAP50_pct": (map50 / max(results_summary.get("Clean_Baseline", {}).get("mAP@50", map50), 1e-8)) * 100.0,
            "object_recall": mean_recall,
            "delta_ap50_mean_pp": mean_delta * 100.0,
            "delta_ap50_ci95_pp": [ci_low * 100.0, ci_high * 100.0],
            "displacements": disp_m,
        }

        logging.info(
            f"  {cond:25s} | mAP@50={map50*100:5.2f}% | Recall={mean_recall*100:5.2f}% | "
            f"L2={disp_m['l2']:.4f} | Delta={mean_delta*100:+6.2f} pp [{ci_low*100:+.2f}, {ci_high*100:+.2f}]"
        )

        for i in range(cohort_size):
            csv_rows.append({
                "condition": cond,
                "image_index": i,
                "ap50": float(cur_ap50[i]),
                "delta_ap50_vs_clean": float(delta_ap50[i]),
                "recall": float(recalls[i]),
                "l2": disp_m["l2"],
            })

    # Tính trung bình cộng của 5 seeds cho Random Reassignment và Random Shift
    reassign_map50 = [results_summary[f"Random_Reassign_seed{s}"]["mAP@50"] for s in seeds]
    reassign_map5095 = [results_summary[f"Random_Reassign_seed{s}"]["mAP@50:95"] for s in seeds]
    reassign_recall = [results_summary[f"Random_Reassign_seed{s}"]["object_recall"] for s in seeds]
    reassign_delta = [results_summary[f"Random_Reassign_seed{s}"]["delta_ap50_mean_pp"] for s in seeds]

    shift_map50 = [results_summary[f"Random_Shift_seed{s}"]["mAP@50"] for s in seeds]
    shift_map5095 = [results_summary[f"Random_Shift_seed{s}"]["mAP@50:95"] for s in seeds]
    shift_recall = [results_summary[f"Random_Shift_seed{s}"]["object_recall"] for s in seeds]
    shift_delta = [results_summary[f"Random_Shift_seed{s}"]["delta_ap50_mean_pp"] for s in seeds]

    results_summary["Aggregate_Summary"] = {
        "Random_Reassignment": {
            "mAP50_mean": float(np.mean(reassign_map50)),
            "mAP50_std": float(np.std(reassign_map50)),
            "mAP5095_mean": float(np.mean(reassign_map5095)),
            "mAP5095_std": float(np.std(reassign_map5095)),
            "recall_mean": float(np.mean(reassign_recall)),
            "recall_std": float(np.std(reassign_recall)),
            "delta_mean_pp": float(np.mean(reassign_delta)),
        },
        "Random_Shift_Matched_L2": {
            "mAP50_mean": float(np.mean(shift_map50)),
            "mAP50_std": float(np.std(shift_map50)),
            "mAP5095_mean": float(np.mean(shift_map5095)),
            "mAP5095_std": float(np.std(shift_map5095)),
            "recall_mean": float(np.mean(shift_recall)),
            "recall_std": float(np.std(shift_recall)),
            "delta_mean_pp": float(np.mean(shift_delta)),
        },
    }

    # Ghi audit stream cho từng condition
    for cond in conditions:
        audit_logger.write_audit_record({
            "record_type": "CONDITION_SUMMARY",
            "condition": cond,
            "metrics": results_summary[cond],
        })

    # Xuất file CSV & JSON summary thông qua audit_logger
    audit_logger.export_raw_csv("controlled_misrouting_per_image.csv", csv_rows)
    audit_logger.export_summary_json("controlled_misrouting_results.json", results_summary)
    audit_logger.close()

    print("[HOÀN THÀNH] Task 4: Controlled Spatial Misrouting!")
    return results_summary


def main():
    parser = argparse.ArgumentParser(description="Task 4: Controlled Spatial Misrouting")
    parser.add_argument("--cohort_size", type=int, default=COHORT_SIZE)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    run_controlled_spatial_misrouting(
        cohort_size=args.cohort_size,
        batch_size=args.batch_size,
        threads=args.threads,
    )


if __name__ == "__main__":
    main()
