# -*- coding: utf-8 -*-
"""
Module: paper_1_aqsm_mechanism.experiments.tasks.task4_matched_sweep
Mục đích: Task độc lập quét ngân sách nhiễu Matched-Budget (eps in {4/255, 8/255, 16/255}).
Có thể chạy độc lập hoặc chạy song song trong hồ bơi đa tiến trình.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import torch

WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from edge_sec_pro.core.coco_gt import load_coco_val2017_gt_cohort
from edge_sec_pro.transformer_monitors.detr_wrapper import RTDETRModelWrapper
from paper_1_aqsm_mechanism.experiments.audit_logger import ExperimentAuditLogger
from paper_1_aqsm_mechanism.experiments.config import (
    LOGS_DIR,
    MODEL_NAME,
    RANDOM_SEED,
    SWEEP_EPSILONS,
    configure_cpu_threads,
    get_device,
)
from paper_1_aqsm_mechanism.experiments.matched_budget_sweep import run_matched_budget_sweep


def main():
    parser = argparse.ArgumentParser(description="Task 4: Matched-Budget Sweep")
    parser.add_argument("--cohort_size", type=int, default=100, help="Số lượng ảnh COCO quét ngân sách")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size suy luận")
    parser.add_argument("--threads", type=int, default=4, help="Số luồng CPU cho task này")
    args = parser.parse_args()

    configure_cpu_threads(args.threads)
    device = get_device()

    audit_logger = ExperimentAuditLogger(task_name="task5_matched_sweep")
    audit_logger.log_metadata({
        "cohort_size": args.cohort_size,
        "batch_size": args.batch_size,
        "threads": args.threads,
        "random_seed": RANDOM_SEED,
        "model_name": MODEL_NAME,
        "epsilons": SWEEP_EPSILONS,
    })

    wrapper = RTDETRModelWrapper(pretrained_model_name=MODEL_NAME, device=device)
    wrapper.eval()

    cohort_data = load_coco_val2017_gt_cohort(n_eval=args.cohort_size, seed=RANDOM_SEED)
    clean_imgs = cohort_data["tensors"].to(device)
    gt_list = cohort_data["gt_list"]

    res = run_matched_budget_sweep(
        wrapper=wrapper,
        clean_imgs=clean_imgs,
        gt_list=gt_list,
        epsilons=SWEEP_EPSILONS,
        batch_size=args.batch_size,
        audit_logger=audit_logger,
    )
    audit_logger.log_step(
        step_name="Sweep_Complete",
        step_idx=1,
        total_steps=1,
        metrics={"clean_map50": res['clean_baseline']['mAP@50']},
    )
    audit_logger.close()
    print(f"Hoàn thành Task 5 thành công! Clean Baseline: {res['clean_baseline']['mAP@50']:.2f}%")


if __name__ == "__main__":
    main()
