# -*- coding: utf-8 -*-
"""
Module: paper_1_aqsm_mechanism.experiments.tasks.task1_causal_surgery
Mục đích: CLI điều phối Task 1 - Phẫu thuật can thiệp nhân quả 2x2 Factorial Tensor-Swap (6 arms)
trên tập COCO val2017 cho các đòn tấn công: AQSM, Query Vanishing, Standard Task PGD -> Phục vụ Table 6 & 7.
"""

import gc
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
from pathlib import Path

import torch

WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from paper_1_aqsm_mechanism.core.adv_cache import load_cached_adv_images
from paper_1_aqsm_mechanism.core.dataset_loader import FlexibleCOCODataset
from paper_1_aqsm_mechanism.core.detr_wrapper import RTDETRModelWrapper
from paper_1_aqsm_mechanism.experiments.audit_logger import ExperimentAuditLogger
from paper_1_aqsm_mechanism.experiments.engines.causal_surgery_engine import run_causal_surgery
from paper_1_aqsm_mechanism.experiments.config import (
    MODEL_NAME,
    RANDOM_SEED,
    configure_cpu_threads,
    detect_hardware_environment,
    get_device,
    update_experiment_paths,
)
from paper_1_aqsm_mechanism.utils.cli import create_base_parser


def main():
    parser = create_base_parser("Causal Surgery 2x2 Factorial Swap", task_name="Task 1")
    parser.add_argument("--attacks", type=str, default="aqsm,vanishing,taskpgd", help="Danh sách đòn tấn công (phân tách bởi dấu phẩy)")
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
    wrapper = RTDETRModelWrapper(pretrained_model_name=MODEL_NAME, device=device)
    wrapper.eval()

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
    for atk in attack_list:
        audit_logger.logger.info(f"=== Đang thực hiện Causal Surgery cho: {atk.upper()} (N={n_actual}) ===")
        adv_imgs, _ = load_cached_adv_images(
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

        eta_q = res["anova_repeated_measures_full"]["attribution_shares"]["partial_eta_squared"]["Query_Q"]
        eta_m = res["anova_repeated_measures_full"]["attribution_shares"]["partial_eta_squared"]["Memory_M"]
        audit_logger.log_step(
            step_name="Attack_Completed",
            step_idx=attack_list.index(atk) + 1,
            total_steps=len(attack_list),
            metrics={"attack": atk, "eta_p2_Q": eta_q, "eta_p2_M": eta_m},
        )
        del adv_imgs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    audit_logger.close()
    print("\n[HOÀN THÀNH TOÀN BỘ] Task 1: Causal Surgery 2x2 Factorial Swap thành công!")


if __name__ == "__main__":
    main()
