# -*- coding: utf-8 -*-
"""
Module: paper_1_aqsm_mechanism.experiments.tasks.task5_matched_sweep
Mục đích: Task độc lập quét ngân sách nhiễu Matched-Budget (eps in {4/255, 8/255, 16/255})
cho AQSM, Query Vanishing, và Task PGD -> Phục vụ Table 9 và Figure 8.

Đặc tính chuẩn công nghiệp (Production Standards):
  - Nhận tham số linh hoạt `--folder_image`, `--annotations`, `--output_dir`, `--cache_dir`.
  - Nạp ảnh theo lô lười (Lazy Batch Loading) chống tràn RAM khi chạy quy mô 5,000 ảnh.
  - Tự động fallback và logging audit streams chi tiết.
"""

import argparse
import gc
import logging
import os
import sys
import time
from pathlib import Path

import torch

WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from paper_1_aqsm_mechanism.core.dataset_loader import FlexibleCOCODataset
from paper_1_aqsm_mechanism.core.detr_wrapper import RTDETRModelWrapper
from paper_1_aqsm_mechanism.experiments.audit_logger import ExperimentAuditLogger
from paper_1_aqsm_mechanism.experiments.config import (
    ADV_CACHE_DIR,
    COCO_ANN_FILE,
    COCO_VAL_DIR,
    LOGS_DIR,
    MODEL_NAME,
    OUTPUTS_DIR,
    RANDOM_SEED,
    SWEEP_EPSILONS,
    configure_cpu_threads,
    detect_hardware_environment,
    get_device,
    update_experiment_paths,
)
from paper_1_aqsm_mechanism.experiments.matched_budget_sweep import run_matched_budget_sweep


def main():
    parser = argparse.ArgumentParser(description="Task 5: Matched-Budget Sweep (Table 9 & Figure 8)")
    parser.add_argument("--folder_image", type=str, default=str(COCO_VAL_DIR), help="Đường dẫn thư mục ảnh COCO")
    parser.add_argument("--annotations", type=str, default=str(COCO_ANN_FILE), help="Đường dẫn file instances_val2017.json")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUTS_DIR), help="Thư mục xuất kết quả")
    parser.add_argument("--cache_dir", type=str, default=str(ADV_CACHE_DIR), help="Thư mục cache đối kháng")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size suy luận")
    parser.add_argument("--threads", type=int, default=8, help="Số luồng CPU cho task này")
    parser.add_argument("--device", type=str, default="auto", help="Thiết bị tính toán (cuda / cpu / auto)")
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

    audit_logger = ExperimentAuditLogger(task_name="task5_matched_sweep")

    # Nạp mô hình RT-DETR
    wrapper = RTDETRModelWrapper(pretrained_model_name=MODEL_NAME, device=device)
    wrapper.eval()

    # Nạp dataset linh hoạt
    dataset = FlexibleCOCODataset(
        folder_image=args.folder_image,
        annotations_file=args.annotations,
        img_size=640,
        id2label=wrapper.model.config.id2label,
        lazy_tensors=True,
        device=device,
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
        "random_seed": RANDOM_SEED,
        "model_name": MODEL_NAME,
        "epsilons": SWEEP_EPSILONS,
    })

    res = run_matched_budget_sweep(
        wrapper=wrapper,
        clean_imgs=clean_imgs,
        gt_list=gt_list,
        image_ids=image_ids,
        epsilons=SWEEP_EPSILONS,
        batch_size=batch_size,
        cache_dir=Path(args.cache_dir),
        audit_logger=audit_logger,
    )

    audit_logger.log_step(
        step_name="Sweep_Complete",
        step_idx=1,
        total_steps=1,
        metrics={"clean_map50": res["clean_baseline"]["mAP@50"]},
    )
    audit_logger.close()
    del clean_imgs, dataset, wrapper
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print(f"\n[HOÀN THÀNH TOÀN BỘ] Task 5: Matched-Budget Sweep thành công! Clean Baseline: {res['clean_baseline']['mAP@50']:.2f}%")


if __name__ == "__main__":
    main()
