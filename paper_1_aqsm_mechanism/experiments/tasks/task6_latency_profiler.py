# -*- coding: utf-8 -*-
"""
Module: paper_1_aqsm_mechanism.experiments.tasks.task3_latency_profiler
Mục đích: Task độc lập đo đạc thời gian thực thi phần cứng (Hardware Latency Telemetry qua 300 runs).
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

from paper_1_aqsm_mechanism.core.coco_gt import load_coco_val2017_gt_cohort
from paper_1_aqsm_mechanism.core.detr_wrapper import RTDETRModelWrapper
from paper_1_aqsm_mechanism.experiments.audit_logger import ExperimentAuditLogger
from paper_1_aqsm_mechanism.experiments.config import (
    LOGS_DIR,
    MODEL_NAME,
    RANDOM_SEED,
    configure_cpu_threads,
    get_device,
)
from paper_1_aqsm_mechanism.experiments.latency_profiler import profile_pipeline_latency


def main():
    parser = argparse.ArgumentParser(description="Task 6: Hardware Latency Profiler")
    parser.add_argument("--n_runs", type=int, default=300, help="Số lần chạy đo đạc")
    parser.add_argument("--cohort_size", type=int, default=300, help="Số lượng ảnh hoặc số lần chạy tương đương")
    parser.add_argument("--warmup", type=int, default=20, help="Số lần warm-up")
    parser.add_argument("--threads", type=int, default=4, help="Số luồng CPU cho task này")
    args = parser.parse_args()

    n_runs = args.n_runs if args.n_runs != 300 else args.cohort_size

    configure_cpu_threads(args.threads)
    device = get_device()

    audit_logger = ExperimentAuditLogger(task_name="task6_latency_profiler")
    audit_logger.log_metadata({
        "n_runs": n_runs,
        "n_warmup": args.warmup,
        "threads": args.threads,
        "model_name": MODEL_NAME,
    })

    wrapper = RTDETRModelWrapper(pretrained_model_name=MODEL_NAME, device=device)
    wrapper.eval()

    cohort_data = load_coco_val2017_gt_cohort(n_eval=1, seed=RANDOM_SEED)
    sample_img = cohort_data["tensors"].to(device)

    res = profile_pipeline_latency(
        wrapper=wrapper,
        sample_img=sample_img,
        n_runs=n_runs,
        n_warmup=args.warmup,
        audit_logger=audit_logger,
    )
    mean_lat = res['stages']['end_to_end_pipeline']['mean']
    audit_logger.log_step(
        step_name="Latency_Complete",
        step_idx=1,
        total_steps=1,
        metrics={"end_to_end_ms": mean_lat},
    )
    audit_logger.close()
    print(f"Hoàn thành Task 6 thành công! Tổng độ trễ: {mean_lat:.2f} ms")


if __name__ == "__main__":
    main()
