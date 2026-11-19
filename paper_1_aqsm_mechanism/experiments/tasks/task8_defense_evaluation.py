# -*- coding: utf-8 -*-
"""
Module: paper_1_aqsm_mechanism.experiments.tasks.task8_defense_evaluation
Mục đích: CLI điều phối Task 8 - Đánh giá cơ chế phòng thủ Encoder Selection Monitor (ESM)
và Calibrated Guardrail trên tập COCO val2017 -> Phục vụ Table 11 & Figure 9.
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
from paper_1_aqsm_mechanism.experiments.engines.defense_evaluation_engine import (
    run_defense_evaluation,
)
from paper_1_aqsm_mechanism.utils.cli import create_base_parser


def main():
    parser = create_base_parser("Empirical Defense & Adaptive Evasion Evaluation (Table 11)", task_name="Task 8")
    parser.add_argument("--cal_ratio", type=float, default=0.3, help="Tỷ lệ tập calibration không chồng lấn (mặc định 30%%)")
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

    audit_logger = ExperimentAuditLogger(task_name="task8_defense_evaluation")
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

    audit_logger.log_metadata({
        "folder_image": str(args.folder_image),
        "total_images": len(dataset),
        "batch_size": batch_size,
        "threads": args.threads,
        "device": str(device),
        "hardware_info": hw_info,
        "random_seed": RANDOM_SEED,
        "model_name": MODEL_NAME,
        "defense_name": "EncoderSelectionMonitor (ESM)",
    })

    defense_results = run_defense_evaluation(
        wrapper=wrapper,
        dataset=dataset,
        cal_ratio=args.cal_ratio,
        target_fpr=0.05,
        cache_dir=args.cache_dir,
        audit_logger=audit_logger,
    )

    audit_logger.log_step(
        step_name="Defense_Evaluation_Complete",
        step_idx=1,
        total_steps=1,
        metrics={
            "clean_fpr": defense_results["empirical_clean_fpr_pct"],
            "aqsm_tpr": defense_results["attacks"]["aqsm"]["tpr_detection_rate_pct"],
            "vanishing_tpr": defense_results["attacks"]["vanishing"]["tpr_detection_rate_pct"],
            "task_pgd_tpr": defense_results["attacks"]["taskpgd"]["tpr_detection_rate_pct"],
        },
    )
    audit_logger.close()
    del dataset, wrapper
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print("[HOÀN THÀNH TOÀN BỘ] Task 8: Defense Evaluation thành công!")


if __name__ == "__main__":
    main()
