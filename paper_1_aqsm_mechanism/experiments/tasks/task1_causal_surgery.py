# -*- coding: utf-8 -*-
"""
Module: paper_1_aqsm_mechanism.experiments.tasks.task1_causal_surgery
Mục đích: Task độc lập chạy phẫu thuật can thiệp nhân quả 2x2 Factorial Tensor-Swap (6 arms)
trên toàn bộ tập ảnh chỉ định (COCO val2017) cho các đòn tấn công: AQSM, Query Vanishing, Standard Task PGD.

Đặc tính chuẩn công nghiệp (Production Standards):
  - Nhận tham số linh hoạt `--folder_image`, `--annotations`, `--output_dir`, `--cache_dir`.
  - Nạp ảnh theo lô lười (Lazy Batch Loading) chống tràn RAM khi chạy quy mô 5,000 ảnh.
  - Phân tích ANOVA lặp lại 2x2 chuẩn mực (Repeated-Measures ANOVA) và Paired Bootstrap 95% CI.
  - Ghi nhận raw CSV từng ảnh và summary JSON đầy đủ ngữ cảnh phục vụ điền bài báo.
"""

import argparse
import gc
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from paper_1_aqsm_mechanism.core.dataset_loader import FlexibleCOCODataset
from paper_1_aqsm_mechanism.core.adv_cache import load_cached_adv_images
from paper_1_aqsm_mechanism.core.detr_wrapper import RTDETRModelWrapper
from paper_1_aqsm_mechanism.experiments.audit_logger import ExperimentAuditLogger
from paper_1_aqsm_mechanism.experiments.causal_surgery_engine import run_causal_surgery
from paper_1_aqsm_mechanism.experiments.config import (
    ADV_CACHE_DIR,
    COCO_ANN_FILE,
    COCO_VAL_DIR,
    LOGS_DIR,
    MODEL_NAME,
    OUTPUTS_DIR,
    RANDOM_SEED,
    configure_cpu_threads,
    detect_hardware_environment,
    get_device,
    update_experiment_paths,
)


def main():
    parser = argparse.ArgumentParser(description="Task 1: Causal Surgery 2x2 Factorial Swap")
    parser.add_argument("--folder_image", type=str, default=str(COCO_VAL_DIR), help="Đường dẫn thư mục ảnh COCO")
    parser.add_argument("--annotations", type=str, default=str(COCO_ANN_FILE), help="Đường dẫn file instances_val2017.json")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUTS_DIR), help="Thư mục xuất kết quả")
    parser.add_argument("--cache_dir", type=str, default=str(ADV_CACHE_DIR), help="Thư mục cache đối kháng")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size suy luận (tự động nếu None)")
    parser.add_argument("--threads", type=int, default=8, help="Số luồng CPU cho task này")
    parser.add_argument("--device", type=str, default="auto", help="Thiết bị tính toán (cuda / cpu / auto)")
    parser.add_argument("--attacks", type=str, default="aqsm,vanishing,taskpgd", help="Danh sách đòn tấn công cách nhau bởi dấu phẩy")
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

    audit_logger = ExperimentAuditLogger(task_name="task1_causal_surgery")

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
    n_actual = len(dataset)

    audit_logger.log_metadata({
        "folder_image": str(args.folder_image),
        "total_images": n_actual,
        "batch_size": batch_size,
        "threads": args.threads,
        "device": str(device),
        "hardware_info": hw_info,
        "attacks": args.attacks,
        "random_seed": RANDOM_SEED,
        "model_name": MODEL_NAME,
    })

    attack_list = [a.strip().lower() for a in args.attacks.split(",") if a.strip()]
    all_results = {}

    for atk in attack_list:
        audit_logger.logger.info(f"\n=======================================================")
        audit_logger.logger.info(f"=== Đang thực hiện Causal Surgery cho: {atk.upper()} (N={n_actual}) ===")
        audit_logger.logger.info(f"=======================================================")

        adv_imgs, meta = load_cached_adv_images(
            attack=atk,
            eps_num=16,
            n_eval=n_actual,
            clean_imgs=clean_imgs,
            expected_image_ids=image_ids,
            cache_dir=args.cache_dir,
        )

        res = run_causal_surgery(
            wrapper=wrapper,
            clean_imgs=clean_imgs,
            adv_imgs=adv_imgs,
            gt_list=gt_list,
            attack_name=atk.upper(),
            batch_size=batch_size,
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

        # Giải phóng bộ nhớ và VRAM sau mỗi đòn tấn công
        del adv_imgs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    audit_logger.close()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print("\n[HOÀN THÀNH TOÀN BỘ] Task 1: Causal Surgery 2x2 Factorial Swap thành công!")


if __name__ == "__main__":
    main()
