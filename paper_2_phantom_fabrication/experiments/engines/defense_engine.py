# -*- coding: utf-8 -*-
"""
Module: paper_2_phantom_fabrication.experiments.engines.defense_engine
Mục đích: Động cơ đánh giá chuyên sâu giải pháp phòng thủ Spatial-Query Divergence (SQD) Filter
và so sánh đối chiếu với các baseline input transformation defenses (Table 4).
"""

import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
from transformers import RTDetrForObjectDetection

from paper_2_phantom_fabrication.core.coco_gt import (
    discover_coco_annotations,
    load_flexible_coco_cohort,
)
from paper_2_phantom_fabrication.core.config import get_adv_cache_dir
from paper_2_phantom_fabrication.core.hardware import detect_hardware
from paper_2_phantom_fabrication.core.matcher import match_detections_against_gt
from paper_2_phantom_fabrication.defenses.baseline_defenses import (
    BitDepthReductionDefense,
    GaussianBlurDefense,
    JPEGCompressionDefense,
    MedianFilterDefense,
)
from paper_2_phantom_fabrication.defenses.sqd_filter import (
    SQDFilter,
    evaluate_sqd_filter_on_detections,
)
from paper_2_phantom_fabrication.experiments.streaming_evaluator import (
    get_cohort_dataloader,
    run_streaming_condition_eval,
)

logger = logging.getLogger("DefenseEngine")


def run_comprehensive_defense_evaluation(
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
    Đánh giá chi tiết toàn bộ các cơ chế phòng thủ trước đòn tấn công DQF.
    """
    cdir = Path(cache_dir) if cache_dir is not None else get_adv_cache_dir()
    cdir.mkdir(parents=True, exist_ok=True)

    if hw_info is None:
        hw_info = detect_hardware("auto" if device is None else str(device))
    if device is None:
        device = torch.device(hw_info["primary_device"])

    img_folder = Path(image_folder).resolve()
    ann_file = discover_coco_annotations(img_folder, annotations_path)

    logger.info("=" * 80)
    logger.info(f"KHỞI CHẠY ĐÁNH GIÁ CHUYÊN SÂU PHÒNG THỦ: SQD FILTER VS BASELINES (N={n_samples})")
    logger.info("=" * 80)

    cohort = load_flexible_coco_cohort(
        img_folder, ann_file, n_samples=n_samples, seed=42, img_size=640, preload_images=False
    )
    dataloader = get_cohort_dataloader(cohort, batch_size=batch_size, device=device)

    raw_model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r18vd")
    raw_model.eval()
    raw_model.to(device)

    if hw_info.get("num_gpus", 0) > 1 and device.type == "cuda":
        model = torch.nn.DataParallel(raw_model)
    else:
        model = raw_model

    # 1. Đánh giá DQF raw
    dqf_res = run_streaming_condition_eval(
        model=model,
        dataloader=dataloader,
        cohort=cohort,
        condition_type="DQF",
        target_class_idx=1,
        attack_eps=16.0 / 255.0,
        attack_alpha=2.0 / 255.0,
        attack_steps=8,
        cache_dir=cdir,
        device=device,
        detector_backend="rtdetr",
    )
    dqf_raw = dqf_res["raw_detections"]
    dqf_match = match_detections_against_gt(dqf_raw, cohort["gt_list"], iou_threshold=0.5)

    defense_results: Dict[str, Any] = {
        "Unprotected_DQF": {
            "mAP50": round(dqf_res["metrics"]["mAP@50"], 2),
            "mAP75": round(dqf_res["metrics"]["mAP@75"], 2),
            "fp_count": dqf_match["total_fp"],
            "tp_count": dqf_match["total_tp"],
            "precision": round(dqf_match["precision"] * 100.0, 2),
            "recall": round(dqf_match["recall"] * 100.0, 2),
            "f1": round(dqf_match["f1"], 4),
        }
    }

    # 2. Đánh giá SQD Filter
    sqd_filtered = evaluate_sqd_filter_on_detections(dqf_raw, divergence_threshold=0.35)
    sqd_match = match_detections_against_gt(sqd_filtered, cohort["gt_list"], iou_threshold=0.5)
    defense_results["SQD_Filter_Ours"] = {
        "fp_count": sqd_match["total_fp"],
        "tp_count": sqd_match["total_tp"],
        "precision": round(sqd_match["precision"] * 100.0, 2),
        "recall": round(sqd_match["recall"] * 100.0, 2),
        "f1": round(sqd_match["f1"], 4),
        "fp_reduction_pct": round((1.0 - sqd_match["total_fp"] / max(1, dqf_match["total_fp"])) * 100.0, 2),
    }

    # 3. Đánh giá Baselines
    baselines = {
        "JPEG_Quality75": JPEGCompressionDefense(quality=75),
        "Gaussian_Blur": GaussianBlurDefense(kernel_size=5, sigma=1.0),
        "Median_Filter": MedianFilterDefense(kernel_size=3),
        "BitDepth_4bit": BitDepthReductionDefense(bits=4),
    }
    for b_name, b_obj in baselines.items():
        b_res = run_streaming_condition_eval(
            model=model,
            dataloader=dataloader,
            cohort=cohort,
            condition_type="DQF",
            target_class_idx=1,
            attack_eps=16.0 / 255.0,
            attack_alpha=2.0 / 255.0,
            attack_steps=8,
            cache_dir=cdir,
            device=device,
            detector_backend="rtdetr",
            preprocessor=b_obj,
        )
        b_match = match_detections_against_gt(b_res["raw_detections"], cohort["gt_list"], iou_threshold=0.5)
        defense_results[b_name] = {
            "mAP50": round(b_res["metrics"]["mAP@50"], 2),
            "mAP75": round(b_res["metrics"]["mAP@75"], 2),
            "fp_count": b_match["total_fp"],
            "tp_count": b_match["total_tp"],
            "f1": round(b_match["f1"], 4),
            "fp_reduction_pct": round((1.0 - b_match["total_fp"] / max(1, dqf_match["total_fp"])) * 100.0, 2),
        }

    return defense_results
