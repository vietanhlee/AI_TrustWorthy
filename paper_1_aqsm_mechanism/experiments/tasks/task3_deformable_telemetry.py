# -*- coding: utf-8 -*-
"""
Module: paper_1_aqsm_mechanism.experiments.tasks.task2_telemetry_analyzer
Mục đích: Task độc lập đo đạc Deformable Attention Telemetry (Entropy, Foreground Spatial Hit, Reference Drift).
Có thể chạy độc lập hoặc chạy song song trong hồ bơi đa tiến trình.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from paper_1_aqsm_mechanism.core.coco_gt import load_coco_val2017_gt_cohort
from paper_1_aqsm_mechanism.core.detr_wrapper import RTDETRModelWrapper
from paper_1_aqsm_mechanism.experiments.audit_logger import ExperimentAuditLogger
from paper_1_aqsm_mechanism.experiments.config import (
    ADV_CACHE_DIR,
    COHORT_SIZE,
    LOGS_DIR,
    MODEL_NAME,
    RANDOM_SEED,
    configure_cpu_threads,
    get_device,
)
from paper_1_aqsm_mechanism.experiments.telemetry_analyzer import run_mechanism_telemetry_cohort


def main():
    parser = argparse.ArgumentParser(description="Task 2: Attention & Proposal Telemetry")
    parser.add_argument("--cohort_size", type=int, default=COHORT_SIZE, help="Số lượng ảnh COCO val2017")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size suy luận")
    parser.add_argument("--threads", type=int, default=4, help="Số luồng CPU cho task này")
    args = parser.parse_args()

    configure_cpu_threads(args.threads)
    device = get_device()

    audit_logger = ExperimentAuditLogger(task_name="task3_deformable_telemetry")
    audit_logger.log_metadata({
        "cohort_size": args.cohort_size,
        "batch_size": args.batch_size,
        "threads": args.threads,
        "random_seed": RANDOM_SEED,
        "model_name": MODEL_NAME,
        "attacks": ["aqsm", "vanishing", "task_pgd"],
    })

    wrapper = RTDETRModelWrapper(pretrained_model_name=MODEL_NAME, device=device)
    wrapper.eval()

    cohort_data = load_coco_val2017_gt_cohort(n_eval=args.cohort_size, seed=RANDOM_SEED)
    clean_imgs = cohort_data["tensors"].to(device)
    gt_list = cohort_data["gt_list"]
    n_actual = len(gt_list)

    cache_dir = ADV_CACHE_DIR

    def load_cached(atk_name: str) -> torch.Tensor:
        cf = cache_dir / f"adv_cache_{atk_name}_eps16_n300.npz"
        data = np.load(cf)
        deltas = torch.from_numpy(data["delta_f16"][:n_actual].astype(np.float32)).to(device)
        return torch.clamp(clean_imgs[:n_actual] + deltas, 0.0, 1.0)

    adv_dict = {
        "aqsm": load_cached("aqsm"),
        "vanishing": load_cached("vanishing"),
        "task_pgd": load_cached("taskpgd"),
    }

    res = run_mechanism_telemetry_cohort(
        wrapper=wrapper,
        clean_imgs=clean_imgs,
        adv_imgs_dict=adv_dict,
        gt_list=gt_list,
        batch_size=args.batch_size,
        audit_logger=audit_logger,
    )
    p_val = res['hypothesis_tests']['entropy_invariance_p_value']
    audit_logger.log_step(
        step_name="Telemetry_Complete",
        step_idx=1,
        total_steps=1,
        metrics={"entropy_invariance_p": p_val},
    )
    audit_logger.close()
    print(f"Hoàn thành Task 3 thành công! Entropy Invariance p={p_val:.4f}")


if __name__ == "__main__":
    main()
