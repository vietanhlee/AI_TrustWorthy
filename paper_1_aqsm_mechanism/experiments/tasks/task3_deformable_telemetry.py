# -*- coding: utf-8 -*-
"""
Module: paper_1_aqsm_mechanism.experiments.tasks.task3_deformable_telemetry
Mục đích: CLI điều phối Task 3 - Đo đạc Deformable Attention Telemetry (Entropy, Spatial FG Hit, Reference Drift)
trên tập COCO val2017 -> Phục vụ Table 7.
"""

import gc
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
from pathlib import Path
from typing import Any, Dict

import torch

WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from paper_1_aqsm_mechanism.core.adv_cache import load_cached_adv_images
from paper_1_aqsm_mechanism.core.dataset_loader import FlexibleCOCODataset
from paper_1_aqsm_mechanism.core.detr_wrapper import RTDETRModelWrapper
from paper_1_aqsm_mechanism.experiments.audit_logger import ExperimentAuditLogger
from paper_1_aqsm_mechanism.experiments.config import (
    MODEL_NAME,
    RANDOM_SEED,
    configure_cpu_threads,
    detect_hardware_environment,
    get_device,
    update_experiment_paths,
)
from paper_1_aqsm_mechanism.experiments.engines.telemetry_analyzer import run_mechanism_telemetry_cohort
from paper_1_aqsm_mechanism.utils.cli import create_base_parser


def main():
    parser = create_base_parser("Attention & Proposal Telemetry (Table 7)", task_name="Task 3")
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

    audit_logger = ExperimentAuditLogger(task_name="task3_deformable_telemetry")
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
        "attacks": args.attacks,
    })

    attacks_to_eval = [a.strip().lower() for a in args.attacks.split(",") if a.strip()]
    adv_dict: Dict[str, Any] = {}
    for atk in attacks_to_eval:
        adv_imgs, _ = load_cached_adv_images(
            attack=atk,
            eps_num=16,
            n_eval=n_total,
            clean_imgs=clean_imgs,
            expected_image_ids=image_ids,
            cache_dir=args.cache_dir,
        )
        adv_dict[atk] = adv_imgs

    print(f"\n=======================================================")
    print(f"--> [Task 3] Khởi chạy Telemetry Deformable Attention trên N={n_total} ảnh...")
    print(f"=======================================================")

    run_mechanism_telemetry_cohort(
        wrapper=wrapper,
        clean_imgs=clean_imgs,
        adv_imgs_dict=adv_dict,
        gt_list=gt_list,
        batch_size=batch_size,
        audit_logger=audit_logger,
    )

    audit_logger.close()
    del adv_dict, clean_imgs, dataset, wrapper
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print("\n[HOÀN THÀNH TOÀN BỘ] Task 3: Deformable Attention Telemetry thành công!")


if __name__ == "__main__":
    main()
