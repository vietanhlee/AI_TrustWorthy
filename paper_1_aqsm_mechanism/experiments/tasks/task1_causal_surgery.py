# -*- coding: utf-8 -*-
"""
Module: paper_1_aqsm_mechanism.experiments.tasks.task1_causal_surgery
Mục đích: Task độc lập chạy phẫu thuật can thiệp nhân quả 2x2 Factorial Tensor-Swap (6 arms)
trên N=300 ảnh COCO val2017 cho các đòn tấn công: AQSM, Query Vanishing, Standard Task PGD.
Có thể chạy độc lập hoặc chạy song song trong hồ bơi đa tiến trình (Multi-process Worker Pool).
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

from edge_sec_pro.core.coco_gt import load_coco_val2017_gt_cohort
from edge_sec_pro.experiments.step_generate_adv_cache import load_cached_adv_images
from edge_sec_pro.transformer_monitors.detr_wrapper import RTDETRModelWrapper
from paper_1_aqsm_mechanism.experiments.audit_logger import ExperimentAuditLogger
from paper_1_aqsm_mechanism.experiments.attack_engine import DiagnosticAttackEngine
from paper_1_aqsm_mechanism.experiments.causal_surgery_engine import run_causal_surgery
from paper_1_aqsm_mechanism.experiments.config import (
    COHORT_SIZE,
    EPSILON_L_INF,
    LOGS_DIR,
    MODEL_NAME,
    PGD_NUM_STEPS,
    PGD_STEP_SIZE,
    RANDOM_SEED,
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


def main():
    parser = argparse.ArgumentParser(description="Task 1: Causal Surgery 2x2 Factorial Swap")
    parser.add_argument("--cohort_size", type=int, default=COHORT_SIZE, help="Số lượng ảnh COCO val2017")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size suy luận")
    parser.add_argument("--threads", type=int, default=4, help="Số luồng CPU cho task này")
    parser.add_argument("--attacks", type=str, default="aqsm,vanishing,taskpgd", help="Danh sách đòn tấn công cách nhau bởi dấu phẩy")
    args = parser.parse_args()

    configure_cpu_threads(args.threads)
    device = get_device()

    audit_logger = ExperimentAuditLogger(task_name="task1_causal_surgery")
    audit_logger.log_metadata({
        "cohort_size": args.cohort_size,
        "batch_size": args.batch_size,
        "threads": args.threads,
        "attacks": args.attacks,
        "random_seed": RANDOM_SEED,
        "model_name": MODEL_NAME,
    })

    # Nạp mô hình & dữ liệu
    wrapper = RTDETRModelWrapper(pretrained_model_name=MODEL_NAME, device=device)
    wrapper.eval()

    cohort_data = load_coco_val2017_gt_cohort(n_eval=args.cohort_size, seed=RANDOM_SEED)
    clean_imgs = cohort_data["tensors"].to(device)
    gt_list = cohort_data["gt_list"]
    n_actual = len(gt_list)

    attack_list = [a.strip().lower() for a in args.attacks.split(",") if a.strip()]

    all_results = {}
    for atk in attack_list:
        audit_logger.logger.info(f"\n=======================================================")
        audit_logger.logger.info(f"=== Đang thực hiện Causal Surgery cho: {atk.upper()} ===")
        audit_logger.logger.info(f"=======================================================")

        adv_imgs = load_adv_images_helper(atk, 16, n_actual, clean_imgs)

        res = run_causal_surgery(
            wrapper=wrapper,
            clean_imgs=clean_imgs,
            adv_imgs=adv_imgs,
            gt_list=gt_list,
            attack_name=atk.upper(),
            batch_size=args.batch_size,
            audit_logger=audit_logger,
        )
        all_results[atk] = res

        eta_q = res["anova_repeated_measures_full"]["attribution_shares"]["partial_eta_squared"]["Query_Q"]
        eta_m = res["anova_repeated_measures_full"]["attribution_shares"]["partial_eta_squared"]["Memory_M"]
        audit_logger.log_step(
            step_name="Attack_Completed",
            step_idx=attack_list.index(atk) + 1,
            total_steps=len(attack_list),
            metrics={"attack": atk, "eta_p2_Q": eta_q, "eta_p2_M": eta_m},
        )
        audit_logger.logger.info(f"Hoàn thành {atk.upper()}: ANOVA eta_p^2(Q)={eta_q:.4f}, eta_p^2(M)={eta_m:.4f}")

    audit_logger.close()
    print("[HOÀN THÀNH TOÀN BỘ] Task 1: Causal Surgery!")


if __name__ == "__main__":
    main()
