# -*- coding: utf-8 -*-
"""
Module: paper_1_aqsm_mechanism.experiments.tasks.task2_encoder_query_dynamics
Mục đích: Đánh giá thực nghiệm động lực học Encoder Memory & Query Selection trên N=300 ảnh COCO val2017:
  1. Phân rã độ biến đổi Multi-Scale Cosine Similarity qua các tầng S3 (40x40), S4 (20x20), S5 (10x10)
     và phân tầng không gian Foreground (vật thể) vs Background (nền).
  2. Phân tích độ xáo trộn thứ hạng Top-K Query Selection: Overlap Top-300, Top-100, Top-50, Top-15,
     tỷ lệ trục xuất (evicted ratio), độ dịch chuyển trung vị (median rank shift), tương quan Spearman rho và Kendall tau.
  3. Kiểm chứng thực nghiệm giả thuyết Proposal Starvation (Object Coverage IoU >= 0.30, IoU >= 0.50, Center-in-GT).
  4. Đo phổ điểm tuyển chọn Proposal Score Spectrum (s^(1), s^(50), s^(300)), Dropoff margin và Query Logit distribution.

Được thực thi độc lập và lưu log đầy đủ, xuất file CSV per-image raw metrics và file JSON tổng hợp thống kê.
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from tqdm import tqdm

WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from edge_sec_pro.core.coco_gt import load_coco_val2017_gt_cohort
from edge_sec_pro.core.config import PGD_NUM_STEPS, PGD_STEP_SIZE
from edge_sec_pro.core.metrics import clopper_pearson_interval, make_serializable
from edge_sec_pro.experiments.step_generate_adv_cache import load_cached_adv_images
from edge_sec_pro.transformer_monitors.detr_wrapper import RTDETRModelWrapper
from edge_sec_pro.transformer_monitors.rtdetr_mechanism_analyzer import (
    SCALE_NAMES,
    analyze_encoder_memory_batch,
    analyze_query_selection_batch,
    extract_encoder_components,
)
from paper_1_aqsm_mechanism.experiments.audit_logger import ExperimentAuditLogger
from paper_1_aqsm_mechanism.experiments.config import (
    COHORT_SIZE,
    LOGS_DIR,
    MODEL_NAME,
    RAW_CSV_DIR,
    SUMMARY_JSON_DIR,
    configure_cpu_threads,
    get_device,
)


def load_adv_images_helper(atk_name: str, eps_num: int, n_eval: int, clean_imgs: torch.Tensor) -> torch.Tensor:
    cache_dir = WORKSPACE_DIR / "edge_sec_pro" / "results" / "checkpoints" / "adv_cache"
    cf = cache_dir / f"adv_cache_{atk_name}_eps{eps_num}_n{n_eval}.npz"
    if not cf.exists():
        cf = cache_dir / f"adv_cache_{atk_name}_eps{eps_num}_n300.npz"
    if cf.exists():
        data = np.load(cf)
        deltas = torch.from_numpy(data["delta_f16"][:n_eval].astype(np.float32)).to(clean_imgs.device)
        return torch.clamp(clean_imgs[:n_eval] + deltas, 0.0, 1.0)
    raise FileNotFoundError(f"Không tìm thấy cache cho {atk_name} tại {cf}")


def _summarize_list(arr: Sequence[float]) -> Dict[str, float]:

    a = np.asarray(arr, dtype=np.float64)
    a = a[~np.isnan(a)]
    if len(a) == 0:
        return {"mean": 0.0, "std": 0.0, "median": 0.0, "q25": 0.0, "q75": 0.0}
    return {
        "mean": float(np.mean(a)),
        "std": float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
        "median": float(np.median(a)),
        "q25": float(np.percentile(a, 25)),
        "q75": float(np.percentile(a, 75)),
    }


def run_encoder_query_dynamics(
    cohort_size: int = COHORT_SIZE,
    batch_size: int = 10,
    threads: int = 4,
):
    configure_cpu_threads(threads)
    device = get_device()

    audit_logger = ExperimentAuditLogger(task_name="task2_encoder_query_dynamics")
    audit_logger.log_metadata({
        "cohort_size": cohort_size,
        "batch_size": batch_size,
        "threads": threads,
        "random_seed": 42,
        "model_name": MODEL_NAME,
        "attacks": ["aqsm", "vanishing", "taskpgd"],
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

    # Các đòn tấn công cần so sánh
    attacks_to_eval = ["aqsm", "vanishing", "taskpgd"]
    results_by_attack: Dict[str, Any] = {}

    # CSV Raw writer
    raw_csv_path = RAW_CSV_DIR / "encoder_query_dynamics_raw.csv"
    csv_rows = []

    for atk_name in attacks_to_eval:
        logging.info(f"--> Đang phân tích cơ chế nội tại cho đòn tấn công: {atk_name.upper()}...")
        adv_imgs = load_adv_images_helper(atk_name, 16, cohort_size, clean_imgs)

        all_mem_records: List[Dict[str, Any]] = []
        all_query_records: List[Dict[str, Any]] = []

        n_batches = int(np.ceil(cohort_size / batch_size))
        for b_idx in tqdm(range(n_batches), desc=f"Dynamics {atk_name.upper()}"):
            start_i = b_idx * batch_size
            end_i = min(start_i + batch_size, cohort_size)

            b_clean = clean_imgs[start_i:end_i].to(device)
            b_adv = adv_imgs[start_i:end_i].to(device)
            b_gt = gt_boxes_batch[start_i:end_i]

            with torch.no_grad():
                c_comp = extract_encoder_components(model, b_clean)
                a_comp = extract_encoder_components(model, b_adv)

            mem_batch = analyze_encoder_memory_batch(c_comp, a_comp, b_gt)
            query_batch = analyze_query_selection_batch(c_comp, a_comp, b_gt)

            all_mem_records.extend(mem_batch)
            all_query_records.extend(query_batch)

        # Tổng hợp thống kê
        # 1. Cosine similarity
        s3_cos = [r["per_scale"]["S3"]["cosine"] for r in all_mem_records]
        s4_cos = [r["per_scale"]["S4"]["cosine"] for r in all_mem_records]
        s5_cos = [r["per_scale"]["S5"]["cosine"] for r in all_mem_records]
        fg_cos = [r["fg_cosine"] for r in all_mem_records]
        bg_cos = [r["bg_cosine"] for r in all_mem_records]
        all_cos = [r["all_cosine"] for r in all_mem_records]

        # 2. Query selection overlap & eviction
        top300_ov = [r["top300_overlap"] for r in all_query_records]
        top50_ov = [r["top50_overlap"] for r in all_query_records]
        top15_evict = [r["evicted_top15_ratio"] for r in all_query_records]
        sp_2100 = [r["spearman_2100"] for r in all_query_records]
        cov_center_adv = [r["cov_center_adv"] for r in all_query_records]
        cov_iou30_adv = [r["cov_iou30_adv"] for r in all_query_records]
        cov_iou50_adv = [r["cov_iou50_adv"] for r in all_query_records]

        cov_center_clean = [r["cov_center_clean"] for r in all_query_records]
        cov_iou30_clean = [r["cov_iou30_clean"] for r in all_query_records]
        cov_iou50_clean = [r["cov_iou50_clean"] for r in all_query_records]

        # 3. Score spectrum
        clean_s1 = [r["clean_s1"] for r in all_query_records]
        clean_s50 = [r["clean_s50"] for r in all_query_records]
        clean_s300 = [r["clean_s300"] for r in all_query_records]
        clean_drop = [r["clean_dropoff"] for r in all_query_records]

        adv_s1 = [r["adv_s1"] for r in all_query_records]
        adv_s50 = [r["adv_s50"] for r in all_query_records]
        adv_s300 = [r["adv_s300"] for r in all_query_records]
        adv_drop = [r["adv_dropoff"] for r in all_query_records]

        summary = {
            "attack": atk_name,
            "n_eval": cohort_size,
            "cosine_similarity": {
                "S3": _summarize_list(s3_cos),
                "S4": _summarize_list(s4_cos),
                "S5": _summarize_list(s5_cos),
                "foreground": _summarize_list(fg_cos),
                "background": _summarize_list(bg_cos),
                "overall": _summarize_list(all_cos),
            },
            "query_selection_overlap": {
                "top300_overlap_count": _summarize_list(top300_ov),
                "top300_overlap_pct": {
                    "mean": float(np.mean(top300_ov)) / 300.0 * 100.0,
                    "std": float(np.std(top300_ov)) / 300.0 * 100.0,
                },
                "top50_overlap_count": _summarize_list(top50_ov),
                "top15_evicted_ratio": _summarize_list(top15_evict),
                "spearman_2100": _summarize_list(sp_2100),
            },
            "starvation_hypothesis": {
                "center_in_gt_clean": _summarize_list(cov_center_clean),
                "center_in_gt_adv": _summarize_list(cov_center_adv),
                "iou30_clean": _summarize_list(cov_iou30_clean),
                "iou30_adv": _summarize_list(cov_iou30_adv),
                "iou50_clean": _summarize_list(cov_iou50_clean),
                "iou50_adv": _summarize_list(cov_iou50_adv),
            },
            "score_spectrum": {
                "clean_s1": _summarize_list(clean_s1),
                "clean_s50": _summarize_list(clean_s50),
                "clean_s300": _summarize_list(clean_s300),
                "clean_dropoff_ratio": _summarize_list(clean_drop),
                "adv_s1": _summarize_list(adv_s1),
                "adv_s50": _summarize_list(adv_s50),
                "adv_s300": _summarize_list(adv_s300),
                "adv_dropoff_ratio": _summarize_list(adv_drop),
            },
        }

        results_by_attack[atk_name] = summary

        # Thu thập rows cho raw CSV
        for i in range(cohort_size):
            csv_rows.append({
            audit_record = {
                "record_type": "IMAGE_DYNAMICS",
                "attack": atk_name,
                "image_index": i,
                "s3_cosine": float(s3_cos[i]),
                "s4_cosine": float(s4_cos[i]),
                "s5_cosine": float(s5_cos[i]),
                "fg_cosine": float(fg_cos[i]),
                "bg_cosine": float(bg_cos[i]),
                "top300_overlap": int(top300_ov[i]),
                "top15_evicted": float(top15_evict[i]),
                "cov_center_adv": float(cov_center_adv[i]),
                "cov_iou30_adv": float(cov_iou30_adv[i]),
                "adv_s1": float(adv_s1[i]),
                "adv_s50": float(adv_s50[i]),
                "adv_s300": float(adv_s300[i]),
                "adv_dropoff": float(adv_drop[i]),
            }
            csv_rows.append(audit_record)
            audit_logger.write_audit_record(audit_record)

    # Ghi file CSV raw & JSON summary thông qua audit_logger
    audit_logger.export_raw_csv("encoder_query_dynamics_raw.csv", csv_rows)
    audit_logger.export_summary_json("encoder_query_dynamics.json", results_by_attack)
    audit_logger.close()

    print("[HOÀN THÀNH] Task 2: Encoder Memory & Query Dynamics!")
    return results_by_attack


def main():
    parser = argparse.ArgumentParser(description="Task 2: Encoder Memory & Query Selection Dynamics")
    parser.add_argument("--cohort_size", type=int, default=COHORT_SIZE)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    run_encoder_query_dynamics(
        cohort_size=args.cohort_size,
        batch_size=args.batch_size,
        threads=args.threads,
    )


if __name__ == "__main__":
    main()
