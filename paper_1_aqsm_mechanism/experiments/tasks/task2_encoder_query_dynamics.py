# -*- coding: utf-8 -*-
"""
Module: paper_1_aqsm_mechanism.experiments.tasks.task2_encoder_query_dynamics
Mục đích: Đánh giá thực nghiệm động lực học Encoder Memory & Query Selection trên toàn bộ tập ảnh COCO val2017:
  1. Phân rã độ biến đổi Multi-Scale Cosine Similarity qua các tầng S3 (80x80), S4 (40x40), S5 (20x20)
     và phân tầng không gian Foreground (vật thể) vs Background (nền) -> Phục vụ Table 5.
  2. Phân tích độ xáo trộn thứ hạng Top-K Query Selection: Overlap Top-300, Top-100, Top-50, Top-15,
     tỷ lệ trục xuất (evicted ratio), độ dịch chuyển trung vị (median rank shift), tương quan Spearman rho và Kendall tau -> Phục vụ Table 6.
  3. Kiểm chứng thực nghiệm giả thuyết Proposal Starvation (Object Coverage IoU >= 0.30, IoU >= 0.50, Center-in-GT).
  4. Đo phổ điểm tuyển chọn Proposal Score Spectrum (s^(1), s^(50), s^(300)), Dropoff margin và Query Logit distribution.

Đặc tính chuẩn công nghiệp (Production Standards):
  - Nhận tham số linh hoạt `--folder_image`, `--annotations`, `--output_dir`, `--cache_dir`, `--max_samples`.
  - Nạp ảnh theo lô lười (Lazy Batch Loading) chống tràn RAM khi chạy quy mô 5,000 ảnh.
  - Ghi nhận raw CSV từng ảnh và summary JSON đầy đủ ngữ cảnh phục vụ điền bài báo.
"""

import argparse
import csv
import gc
import json
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from tqdm import tqdm

WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from paper_1_aqsm_mechanism.core.adv_cache import load_cached_adv_images
from paper_1_aqsm_mechanism.core.dataset_loader import FlexibleCOCODataset
from paper_1_aqsm_mechanism.core.detr_wrapper import RTDETRModelWrapper
from paper_1_aqsm_mechanism.core.mechanism_analyzer import (
    SCALE_NAMES,
    analyze_encoder_memory_batch,
    analyze_query_selection_batch,
    extract_encoder_components,
)
from paper_1_aqsm_mechanism.experiments.audit_logger import ExperimentAuditLogger
from paper_1_aqsm_mechanism.experiments.config import (
    MODEL_NAME,
    RANDOM_SEED,
    configure_cpu_threads,
    detect_hardware_environment,
    get_device,
    get_raw_csv_dir,
    get_summary_json_dir,
    update_experiment_paths,
)
from paper_1_aqsm_mechanism.utils.cli import create_base_parser
from paper_1_aqsm_mechanism.utils.io_helpers import format_duration, save_json, write_csv_rows


def summarize_metric_list(arr: Sequence[float]) -> Dict[str, float]:
    """Tính toán bộ chỉ số thống kê mô tả (Mean, Std, Median, Min, Max, 95% CI)."""
    a = np.asarray(arr, dtype=np.float64)
    valid = a[~np.isnan(a)]
    if len(valid) == 0:
        return {"mean": float("nan"), "std": float("nan"), "median": float("nan"), "ci_95": [float("nan"), float("nan")]}
    mean = float(np.mean(valid))
    std = float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0
    med = float(np.median(valid))
    ci_low = float(np.percentile(valid, 2.5))
    ci_high = float(np.percentile(valid, 97.5))
    return {
        "mean": round(mean, 4),
        "std": round(std, 4),
        "median": round(med, 4),
        "ci_95": [round(ci_low, 4), round(ci_high, 4)],
    }


def run_encoder_query_dynamics_evaluation(
    wrapper: RTDETRModelWrapper,
    clean_imgs: Any,
    gt_list: List[Dict[str, Any]],
    image_ids: Sequence[int],
    attacks_to_eval: Sequence[str] = ("aqsm", "vanishing", "taskpgd"),
    batch_size: int = 10,
    cache_dir: Optional[Path] = None,
    audit_logger: Any = None,
) -> Dict[str, Any]:
    """
    Thực hiện phân tích toàn diện Encoder Memory & Query Selection Dynamics trên tập ảnh.
    Lưu trữ trực tiếp file CSV raw metrics và JSON summary Table 5, Table 6.
    """
    model = wrapper.model
    device = wrapper.device
    model.eval()

    n_total = len(gt_list)
    gt_boxes_batch = [gt.get("boxes_xyxy", np.zeros((0, 4), dtype=np.float32)) for gt in gt_list]

    results_by_attack: Dict[str, Any] = {}
    memory_results_by_attack: Dict[str, Any] = {}
    csv_rows = []

    for atk_name in attacks_to_eval:
        print(f"\n=======================================================")
        print(f"--> [Task 2] Phân tích cơ chế cho đòn: {atk_name.upper()} (N={n_total})...")
        print(f"=======================================================")

        adv_imgs, meta = load_cached_adv_images(
            attack=atk_name,
            eps_num=16,
            n_eval=n_total,
            clean_imgs=clean_imgs,
            expected_image_ids=image_ids,
            cache_dir=cache_dir,
        )

        all_mem_records: List[Dict[str, Any]] = []
        all_query_records: List[Dict[str, Any]] = []

        n_batches = int(np.ceil(n_total / batch_size))
        for b_idx in tqdm(range(n_batches), desc=f"Dynamics {atk_name.upper()}"):
            start_i = b_idx * batch_size
            end_i = min(start_i + batch_size, n_total)

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

        # 1. Tổng hợp Table 5: Encoder Memory Cosine Similarity & Relative L2
        s3_cos = [r["per_scale"]["S3"]["cosine"] for r in all_mem_records]
        s4_cos = [r["per_scale"]["S4"]["cosine"] for r in all_mem_records]
        s5_cos = [r["per_scale"]["S5"]["cosine"] for r in all_mem_records]
        fg_cos = [r["fg_cosine"] for r in all_mem_records]
        bg_cos = [r["bg_cosine"] for r in all_mem_records]
        all_cos = [r["all_cosine"] for r in all_mem_records]

        s3_l2 = [r["per_scale"]["S3"]["rel_l2"] for r in all_mem_records]
        s4_l2 = [r["per_scale"]["S4"]["rel_l2"] for r in all_mem_records]
        s5_l2 = [r["per_scale"]["S5"]["rel_l2"] for r in all_mem_records]
        fg_l2 = [r["fg_rel_l2"] for r in all_mem_records]
        bg_l2 = [r["bg_rel_l2"] for r in all_mem_records]
        all_l2 = [r["all_rel_l2"] for r in all_mem_records]

        memory_results_by_attack[atk_name] = {
            "cosine_similarity": {
                "S3": summarize_metric_list(s3_cos),
                "S4": summarize_metric_list(s4_cos),
                "S5": summarize_metric_list(s5_cos),
                "Foreground": summarize_metric_list(fg_cos),
                "Background": summarize_metric_list(bg_cos),
                "Overall": summarize_metric_list(all_cos),
            },
            "relative_l2_perturbation": {
                "S3": summarize_metric_list(s3_l2),
                "S4": summarize_metric_list(s4_l2),
                "S5": summarize_metric_list(s5_l2),
                "Foreground": summarize_metric_list(fg_l2),
                "Background": summarize_metric_list(bg_l2),
                "Overall": summarize_metric_list(all_l2),
            },
        }

        # 2. Tổng hợp Table 6: Query Selection Overlap, Eviction & Coverage
        top300_ov = [r["top300_overlap"] * 100.0 for r in all_query_records]
        top100_ov = [r["top100_overlap"] * 100.0 for r in all_query_records]
        top50_ov = [r["top50_overlap"] * 100.0 for r in all_query_records]
        evicted = [r["evicted_top15_ratio"] * 100.0 for r in all_query_records]
        rank_shift_top15 = [r["median_rank_shift_top15"] for r in all_query_records]
        rank_shift_fg = [r["median_rank_shift_fg_tokens"] for r in all_query_records]
        sp_rho = [r["spearman_2100"] for r in all_query_records]
        kt_tau = [r["kendall_2100"] for r in all_query_records]

        cov_center_c = [r["cov_center_clean"] * 100.0 for r in all_query_records]
        cov_center_a = [r["cov_center_adv"] * 100.0 for r in all_query_records]
        cov_iou50_c = [r["cov_iou50_clean"] * 100.0 for r in all_query_records]
        cov_iou50_a = [r["cov_iou50_adv"] * 100.0 for r in all_query_records]
        cov_iou30_c = [r["cov_iou30_clean"] * 100.0 for r in all_query_records]
        cov_iou30_a = [r["cov_iou30_adv"] * 100.0 for r in all_query_records]

        results_by_attack[atk_name] = {
            "top300_overlap_pct": summarize_metric_list(top300_ov),
            "top100_overlap_pct": summarize_metric_list(top100_ov),
            "top50_overlap_pct": summarize_metric_list(top50_ov),
            "evicted_top15_fg_pct": summarize_metric_list(evicted),
            "median_rank_shift_top15": summarize_metric_list(rank_shift_top15),
            "median_rank_shift_fg_tokens": summarize_metric_list(rank_shift_fg),
            "spearman_rho_2100": summarize_metric_list(sp_rho),
            "kendall_tau_2100": summarize_metric_list(kt_tau),
            "object_coverage": {
                "center_in_gt_clean_pct": summarize_metric_list(cov_center_c),
                "center_in_gt_adv_pct": summarize_metric_list(cov_center_a),
                "iou50_clean_pct": summarize_metric_list(cov_iou50_c),
                "iou50_adv_pct": summarize_metric_list(cov_iou50_a),
                "iou30_clean_pct": summarize_metric_list(cov_iou30_c),
                "iou30_adv_pct": summarize_metric_list(cov_iou30_a),
            },
            "score_spectrum": {
                "clean_s1": summarize_metric_list([r["clean_s1"] for r in all_query_records]),
                "clean_s50": summarize_metric_list([r["clean_s50"] for r in all_query_records]),
                "clean_s300": summarize_metric_list([r["clean_s300"] for r in all_query_records]),
                "clean_dropoff": summarize_metric_list([r["clean_dropoff"] for r in all_query_records]),
                "clean_margin": summarize_metric_list([r["clean_enc_margin"] for r in all_query_records]),
                "adv_s1": summarize_metric_list([r["adv_s1"] for r in all_query_records]),
                "adv_s50": summarize_metric_list([r["adv_s50"] for r in all_query_records]),
                "adv_s300": summarize_metric_list([r["adv_s300"] for r in all_query_records]),
                "adv_dropoff": summarize_metric_list([r["adv_dropoff"] for r in all_query_records]),
                "adv_margin": summarize_metric_list([r["adv_enc_margin"] for r in all_query_records]),
            },
        }

        for i in range(n_total):
            m_r = all_mem_records[i]
            q_r = all_query_records[i]
            csv_rows.append([
                atk_name,
                i,
                image_ids[i],
                m_r["per_scale"]["S3"]["cosine"],
                m_r["per_scale"]["S4"]["cosine"],
                m_r["per_scale"]["S5"]["cosine"],
                m_r["fg_cosine"],
                m_r["bg_cosine"],
                m_r["all_cosine"],
                q_r["top300_overlap"],
                q_r["top100_overlap"],
                q_r["top50_overlap"],
                q_r["evicted_top15_ratio"],
                q_r["median_rank_shift_top15"],
                q_r["spearman_2100"],
                q_r["kendall_2100"],
                q_r["cov_center_clean"],
                q_r["cov_center_adv"],
                q_r["cov_iou50_clean"],
                q_r["cov_iou50_adv"],
                q_r["clean_dropoff"],
                q_r["adv_dropoff"],
            ])

        del adv_imgs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    raw_csv_path = get_raw_csv_dir() / "encoder_query_dynamics_raw.csv"
    headers = [
        "attack", "image_index", "image_id",
        "cos_s3", "cos_s4", "cos_s5", "cos_fg", "cos_bg", "cos_all",
        "top300_overlap", "top100_overlap", "top50_overlap", "evicted_top15",
        "rank_shift_top15", "spearman_2100", "kendall_2100",
        "cov_center_clean", "cov_center_adv", "cov_iou50_clean", "cov_iou50_adv",
        "clean_dropoff", "adv_dropoff",
    ]
    write_csv_rows(raw_csv_path, headers, csv_rows)
    print(f"\n>>> Đã ghi nhận số liệu thô per-image: {raw_csv_path}")

    summary_json_dir = get_summary_json_dir()
    summary_json_dir.mkdir(parents=True, exist_ok=True)

    json_path_mem = summary_json_dir / "encoder_memory_analysis.json"
    save_json(json_path_mem, {
        "experiment": "Table 5: Multi-Scale Encoder Memory Deformation",
        "n_images": n_total,
        "model": MODEL_NAME,
        "device": str(device),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": memory_results_by_attack,
    })
    print(f">>> Đã xuất JSON Table 5: {json_path_mem}")

    json_path_query = summary_json_dir / "query_selection_dynamics.json"
    query_payload = {
        "experiment": "Table 6: Query Selection Perturbation Dynamics",
        "n_images": n_total,
        "model": MODEL_NAME,
        "device": str(device),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": results_by_attack,
    }
    save_json(json_path_query, query_payload)
    save_json(summary_json_dir / "encoder_query_dynamics.json", query_payload)
    print(f">>> Đã xuất JSON Table 6: {json_path_query}")

    return {
        "table5_memory": memory_results_by_attack,
        "table6_query_selection": results_by_attack,
    }


def main():
    parser = create_base_parser("Encoder Memory & Query Selection Dynamics", task_name="Task 2")
    parser.add_argument("--attacks", type=str, default="aqsm,vanishing,taskpgd", help="Danh sách đòn tấn công")
    args = parser.parse_args()

    update_experiment_paths(
        folder_image=args.folder_image,
        annotations=args.annotations,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
    )

    configure_cpu_threads(args.threads)
    hw_info = detect_hardware_environment(preferred_device=args.device)
    device = get_device(preferred=args.device)
    batch_size = args.batch_size if args.batch_size is not None else hw_info["recommended_batch_size"]

    audit_logger = ExperimentAuditLogger(task_name="task2_encoder_query_dynamics")

    # Nạp mô hình RT-DETR
    wrapper = RTDETRModelWrapper(pretrained_model_name=MODEL_NAME, device=device)
    wrapper.eval()
    model = wrapper.model

    # Nạp dataset linh hoạt hỗ trợ max_samples
    dataset = FlexibleCOCODataset(
        folder_image=args.folder_image,
        annotations_file=args.annotations,
        img_size=640,
        id2label=model.config.id2label,
        lazy_tensors=True,
        device=device,
        max_samples=args.max_samples,
    )

    clean_imgs = dataset.tensors
    gt_list = dataset.gt_list
    image_ids = dataset.image_ids
    n_total = len(dataset)

    audit_logger.log_metadata({
        "folder_image": str(args.folder_image),
        "total_images": n_total,
        "batch_size": batch_size,
        "threads": args.threads,
        "device": str(device),
        "hardware_info": hw_info,
        "attacks": args.attacks,
        "random_seed": RANDOM_SEED,
        "model_name": MODEL_NAME,
        "max_samples": args.max_samples,
    })

    attacks_to_eval = [a.strip().lower() for a in args.attacks.split(",") if a.strip()]

    run_encoder_query_dynamics_evaluation(
        wrapper=wrapper,
        clean_imgs=clean_imgs,
        gt_list=gt_list,
        image_ids=image_ids,
        attacks_to_eval=attacks_to_eval,
        batch_size=batch_size,
        cache_dir=Path(args.cache_dir),
        audit_logger=audit_logger,
    )

    audit_logger.close()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print("\n[HOÀN THÀNH TOÀN BỘ] Task 2: Encoder Memory & Query Selection Dynamics thành công!")


if __name__ == "__main__":
    main()
