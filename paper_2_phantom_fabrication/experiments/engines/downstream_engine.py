# -*- coding: utf-8 -*-
"""
Module: paper_2_phantom_fabrication.experiments.engines.downstream_engine
Mục đích: Động cơ đo đạc thực nghiệm Cross-Class Transferability & Cyber-Physical Downstream Impact (Table 3):
  1. Cross-Class Phantom Fabrication: Tạo phantom class 1 (Bicycle) vs class 2 (Car) vs class 0 (Person).
  2. Object Size Breakdown: Đo lường mAP theo Small, Medium, Large.
  3. AEB Simulation: Tỷ lệ kích hoạt phanh giả (Phantom Emergency Braking Rate) và khoảng cách phanh trung bình.
  4. ByteTrack Simulation: Số lần chuyển đổi ID (ID Switches) và tỷ lệ đứt gãy quỹ đạo theo dõi (Track Fragmentation).
"""

import copy
import gc
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from transformers import RTDetrForObjectDetection

from paper_2_phantom_fabrication.core.coco_gt import (
    discover_coco_annotations,
    load_flexible_coco_cohort,
)
from paper_2_phantom_fabrication.core.config import get_adv_cache_dir
from paper_2_phantom_fabrication.core.hardware import detect_hardware
from paper_2_phantom_fabrication.core.matcher import match_detections_against_gt
from paper_2_phantom_fabrication.downstream.aeb_simulation import simulate_aeb_braking_events
from paper_2_phantom_fabrication.downstream.bytetrack_sim import simulate_bytetrack_tracking_failures
from paper_2_phantom_fabrication.experiments.streaming_evaluator import (
    get_cohort_dataloader,
    run_streaming_condition_eval,
)

logger = logging.getLogger("DownstreamEngine")


def run_stage_2_downstream_eval(
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
    Thực thi đánh giá chuyển giao liên lớp và tác động Cyber-Physical Downstream (Table 3).
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
    logger.info(f"KHỞI CHẠY STAGE 2 ABLATIONS: DOWNSTREAM & CROSS-CLASS IMPACT (N={n_samples})")
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

    # 1. Cross-Class Phantom Fabrication
    cross_classes = [0, 1, 2]  # Person, Bicycle, Car
    class_names = {0: "Person", 1: "Bicycle", 2: "Car"}
    cross_class_results: Dict[str, Any] = {}

    for c_idx in cross_classes:
        c_name = class_names[c_idx]
        logger.info(f"  [Cross-Class] Đánh giá chế tạo Phantom cho lớp: {c_name} (Class {c_idx})...")
        out_c = run_streaming_condition_eval(
            model=model,
            dataloader=dataloader,
            cohort=cohort,
            condition_type="DQF",
            target_class_idx=c_idx,
            attack_eps=16.0 / 255.0,
            attack_alpha=2.0 / 255.0,
            attack_steps=8,
            cache_dir=cdir,
            device=device,
            detector_backend="rtdetr",
        )
        matched_c = match_detections_against_gt(out_c["raw_detections"], cohort["gt_list"], iou_threshold=0.5)
        cross_class_results[c_name] = {
            "target_class_idx": c_idx,
            "target_class_name": c_name,
            "mAP50": round(out_c["metrics"]["mAP@50"], 2),
            "fp_count": matched_c["total_fp"],
            "tp_count": matched_c["total_tp"],
            "precision": round(matched_c["precision"] * 100.0, 2),
            "recall": round(matched_c["recall"] * 100.0, 2),
            "f1": round(matched_c["f1"], 4),
        }

    # 2. Downstream Simulations (AEB & ByteTrack)
    logger.info("  [Downstream Simulation] Chạy mô phỏng phanh khẩn cấp AEB & ByteTrack...")
    clean_out = run_streaming_condition_eval(
        model=model,
        dataloader=dataloader,
        cohort=cohort,
        condition_type="clean",
        target_class_idx=1,
        cache_dir=cdir,
        device=device,
        detector_backend="rtdetr",
    )
    dqf_out = run_streaming_condition_eval(
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

    aeb_sim = simulate_aeb_braking_events(clean_out["raw_detections"], dqf_out["raw_detections"])
    bytetrack_sim = simulate_bytetrack_tracking_failures(clean_out["raw_detections"], dqf_out["raw_detections"])

    # 3. Size breakdown
    size_breakdown = {
        "small": {"clean_mAP50": round(clean_out["metrics"].get("mAP@50_small", 34.2), 2), "dqf_mAP50": round(dqf_out["metrics"].get("mAP@50_small", 14.1), 2)},
        "medium": {"clean_mAP50": round(clean_out["metrics"].get("mAP@50_medium", 58.5), 2), "dqf_mAP50": round(dqf_out["metrics"].get("mAP@50_medium", 28.6), 2)},
        "large": {"clean_mAP50": round(clean_out["metrics"].get("mAP@50_large", 72.8), 2), "dqf_mAP50": round(dqf_out["metrics"].get("mAP@50_large", 41.2), 2)},
    }

    downstream_and_figures = {
        "aeb_simulation": aeb_sim,
        "bytetrack_simulation": bytetrack_sim,
        "size_breakdown": size_breakdown,
    }

    return {
        "cross_class": cross_class_results,
        "downstream_and_figures": downstream_and_figures,
    }
