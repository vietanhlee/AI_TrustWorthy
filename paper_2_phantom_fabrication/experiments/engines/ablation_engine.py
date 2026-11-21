# -*- coding: utf-8 -*-
"""
Module: paper_2_phantom_fabrication.experiments.engines.ablation_engine
Mục đích: Động cơ đo đạc thực nghiệm Stage 2 Sensitivity Sweeps (Table 2):
  1. Perturbation Budget Sweep: eps in {4/255, 8/255, 12/255, 16/255}.
  2. PGD Optimization Steps Sweep: steps in {2, 4, 8, 15}.
  - Đo đạc chi tiết: False Positive Count, mAP@50, mAP@75, F1 score.
"""

import copy
import gc
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from transformers import RTDetrForObjectDetection

from paper_2_phantom_fabrication.core.adv_cache import find_cache_file
from paper_2_phantom_fabrication.core.coco_gt import (
    discover_coco_annotations,
    load_flexible_coco_cohort,
)
from paper_2_phantom_fabrication.core.config import get_adv_cache_dir
from paper_2_phantom_fabrication.core.hardware import detect_hardware
from paper_2_phantom_fabrication.core.matcher import match_detections_against_gt
from paper_2_phantom_fabrication.experiments.engines.cache_engine import generate_single_attack_cache
from paper_2_phantom_fabrication.experiments.streaming_evaluator import (
    get_cohort_dataloader,
    run_streaming_condition_eval,
)

logger = logging.getLogger("AblationEngine")


def run_stage_2_sensitivity_sweeps(
    image_folder: Union[str, Path],
    annotations_path: Optional[Union[str, Path]] = None,
    n_samples: int = 0,
    device: Optional[torch.device] = None,
    hw_info: Optional[Dict[str, Any]] = None,
    batch_size: int = 8,
    cache_dir: Optional[Union[str, Path]] = None,
    skip_cache_gen: bool = False,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    """
    Thực thi quét độ nhạy ngân sách nhiễu và số bước lặp PGD (Table 2).
    """
    cdir = Path(cache_dir) if cache_dir is not None else get_adv_cache_dir()
    cdir.mkdir(parents=True, exist_ok=True)

    # Tự động dọn dẹp các file tạm dở dang tmp_* nếu có
    try:
        for tmp_f in cdir.glob("tmp_*"):
            try:
                tmp_f.unlink()
            except Exception:
                pass
    except Exception:
        pass

    import shutil
    disk_free_gb = shutil.disk_usage(cdir).free / (1024 ** 3)
    if disk_free_gb < 3.5:
        logger.warning(
            f"[DISK SAFE] Dung lượng đĩa khả dụng ({disk_free_gb:.2f} GB) đang thấp (< 3.5 GB). "
            f"Tự động chuyển Stage 2 Sensitivity Sweeps sang chế độ On-The-Fly GPU để không tốn ổ đĩa."
        )
        skip_cache_gen = True

    if hw_info is None:
        hw_info = detect_hardware("auto" if device is None else str(device))
    if device is None:
        device = torch.device(hw_info["primary_device"])

    img_folder = Path(image_folder).resolve()
    ann_file = discover_coco_annotations(img_folder, annotations_path)

    logger.info("=" * 80)
    logger.info(f"KHỞI CHẠY STAGE 2 ABLATIONS: SENSITIVITY SWEEPS (N={n_samples})")
    logger.info("=" * 80)

    cohort = load_flexible_coco_cohort(
        img_folder, ann_file, n_samples=n_samples, seed=42, img_size=640, preload_images=False
    )
    dataloader = get_cohort_dataloader(cohort, batch_size=batch_size, device=device)
    n_eval = len(cohort["image_ids"])

    raw_model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r18vd")
    raw_model.eval()
    raw_model.to(device)

    if hw_info.get("num_gpus", 0) > 1 and device.type == "cuda":
        model = torch.nn.DataParallel(raw_model)
    else:
        model = raw_model

    budget_sweep_results: Dict[str, Any] = {}
    iteration_sweep_results: Dict[str, Any] = {}

    # 1. Perturbation Budget Sweep: eps in {4, 8, 12, 16}
    epsilons = [4, 8, 12, 16]
    for eps_val in epsilons:
        eps_f = float(eps_val) / 255.0
        alpha_f = 2.0 / 255.0 if eps_val >= 8 else 1.0 / 255.0
        budget_sweep_results[f"eps_{eps_val}"] = {}

        for atk in ["EQF", "DQF"]:
            logger.info(f"  [Budget Sweep] Đánh giá {atk} tại eps={eps_val}/255...")
            if not skip_cache_gen:
                cf = find_cache_file(
                    attack=atk, eps_num=eps_val, steps=8, target_class_idx=1, n_eval=n_eval, cache_dir=cdir
                )
                if cf is None or force_recompute:
                    generate_single_attack_cache(
                        model=model,
                        dataloader=dataloader,
                        cohort=cohort,
                        attack_name=atk,
                        eps_int=eps_val,
                        steps=8,
                        target_class_idx=1,
                        cache_dir=cdir,
                        device=device,
                        force_recompute=force_recompute,
                    )

            out = run_streaming_condition_eval(
                model=model,
                dataloader=dataloader,
                cohort=cohort,
                condition_type=atk,
                target_class_idx=1,
                attack_eps=eps_f,
                attack_alpha=alpha_f,
                attack_steps=8,
                cache_dir=cdir,
                device=device,
                detector_backend="rtdetr",
                skip_cache_gen=skip_cache_gen,
            )
            matched = match_detections_against_gt(out["raw_detections"], cohort["gt_list"], iou_threshold=0.5)
            budget_sweep_results[f"eps_{eps_val}"][atk] = {
                "mAP50": round(out["metrics"]["mAP@50"], 2),
                "mAP75": round(out["metrics"]["mAP@75"], 2),
                "fp_count": matched["total_fp"],
                "tp_count": matched["total_tp"],
                "f1": round(matched["f1"], 4),
            }

    # 2. Optimization Steps Sweep: steps in {2, 4, 8, 15}
    steps_list = [2, 4, 8, 15]
    for st in steps_list:
        iteration_sweep_results[f"steps_{st}"] = {}
        for atk in ["EQF", "DQF"]:
            logger.info(f"  [Steps Sweep] Đánh giá {atk} tại steps={st}...")
            if not skip_cache_gen:
                cf = find_cache_file(
                    attack=atk, eps_num=16, steps=st, target_class_idx=1, n_eval=n_eval, cache_dir=cdir
                )
                if cf is None or force_recompute:
                    generate_single_attack_cache(
                        model=model,
                        dataloader=dataloader,
                        cohort=cohort,
                        attack_name=atk,
                        eps_int=16,
                        steps=st,
                        target_class_idx=1,
                        cache_dir=cdir,
                        device=device,
                        force_recompute=force_recompute,
                    )

            out = run_streaming_condition_eval(
                model=model,
                dataloader=dataloader,
                cohort=cohort,
                condition_type=atk,
                target_class_idx=1,
                attack_eps=16.0 / 255.0,
                attack_alpha=2.0 / 255.0,
                attack_steps=st,
                cache_dir=cdir,
                device=device,
                detector_backend="rtdetr",
                skip_cache_gen=skip_cache_gen,
            )
            matched = match_detections_against_gt(out["raw_detections"], cohort["gt_list"], iou_threshold=0.5)
            iteration_sweep_results[f"steps_{st}"][atk] = {
                "mAP50": round(out["metrics"]["mAP@50"], 2),
                "mAP75": round(out["metrics"]["mAP@75"], 2),
                "fp_count": matched["total_fp"],
                "tp_count": matched["total_tp"],
                "f1": round(matched["f1"], 4),
            }

    return {
        "budget_sweep": budget_sweep_results,
        "iteration_sweep": iteration_sweep_results,
    }
