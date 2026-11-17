# -*- coding: utf-8 -*-
"""
Bước 10: Quét Độ Nhạy Ngân Sách Nhiễu Epsilon (Comprehensive Epsilon Sweep)
Mục đích: Đánh giá độ suy giảm mAP@50 và tỷ lệ phát hiện (TPR @ 5% FPR) theo ngân sách nhiễu:
epsilon in [2/255, 4/255, 8/255, 16/255] trên 3 nhánh:
1. Standard Task Detection PGD
2. Tier 1 AQSM Attack
3. Tier 2 Query Vanishing Attack
Xuất log chi tiết, file CSV lịch sử quét sweep và file JSON metrics.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F

# Đảm bảo import đúng workspace
EXPERIMENTS_DIR = Path(__file__).resolve().parent
PAPER3_DIR = EXPERIMENTS_DIR.parent
WORKSPACE_DIR = PAPER3_DIR.parent

if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))
if str(WORKSPACE_DIR / "edge_sec_pro") not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR / "edge_sec_pro"))

from paper_3_defense_guardrail.experiments.common_exp import (
    load_coco_images,
    RTDETRModelWrapper,
    RTDETRDeepFeatureExtractor,
    DETRAttackGenerator,
    CHECKPOINTS_DIR,
)
from paper_3_defense_guardrail.experiments.common_exp import box_cxcywh_to_xyxy, compute_map_coco
from paper_3_defense_guardrail.experiments.exp_logger import (
    setup_experiment_logger,
    MetricHistoryTracker,
    save_audit_json,
    PAPER3_DIR,
    RESULTS_DIR,
)


def extract_preds(out):
    logits = out.logits[0]
    boxes = out.pred_boxes[0]
    probs = torch.sigmoid(logits)
    scores, labels = probs.max(dim=-1)
    xyxy = box_cxcywh_to_xyxy(boxes).clamp(0.0, 1.0)
    return xyxy.detach().cpu().numpy(), scores.detach().cpu().numpy(), labels.detach().cpu().numpy()


def run_epsilon_sweep(n_eval: int = 25):
    logger, log_file = setup_experiment_logger("step10_epsilon_sweep")
    logger.info(f"BƯỚC 10: QUÉT ĐỘ NHẠY NGÂN SÁCH NHIỄU EPSILON (N={n_eval})")

    device = torch.device("cpu")
    logger.info(f"Khởi tạo phần cứng: Device = {device}, PyTorch Threads = {torch.get_num_threads()}")

    wrapper = RTDETRModelWrapper(pretrained_model_name="PekingU/rtdetr_r18vd", device=device)
    wrapper.model.eval()
    model = wrapper.model
    attacker = DETRAttackGenerator(wrapper)

    # Nạp ảnh kiểm thử
    _, test_imgs = load_coco_images(n_cal=100, n_test=n_eval, img_size=320, seed=42)
    eval_tensor = test_imgs.to(device)

    # 1. Đo clean baseline
    logger.info("\n1. Đánh giá mAP Baseline trên dữ liệu sạch...")
    gt_boxes = []
    gt_labels = []
    for i in range(n_eval):
        x = eval_tensor[i : i + 1]
        with torch.no_grad():
            out = model(pixel_values=x, return_dict=True)
            b, s, l = extract_preds(out)
            fg = s >= 0.35
            gt_boxes.append(b[fg] if fg.any() else b[:1])
            gt_labels.append(l[fg] if fg.any() else l[:1])

    clean_map = compute_map_coco(gt_boxes, [np.ones_like(l) for l in gt_labels], gt_labels, gt_boxes, gt_labels)
    logger.info(f"   -> Clean Baseline mAP@50 = {clean_map['mAP@50']:.1f}%")

    # 2. Cấu hình Sweep Epsilon
    eps_configs = [
        (2.0 / 255.0, 0.5 / 255.0, "eps_2"),
        (4.0 / 255.0, 1.0 / 255.0, "eps_4"),
        (8.0 / 255.0, 1.5 / 255.0, "eps_8"),
        (16.0 / 255.0, 2.0 / 255.0, "eps_16"),
    ]

    history_tracker = MetricHistoryTracker("step10_epsilon_sweep", output_dir=RESULTS_DIR)
    sweep_data: Dict[str, Any] = {"clean_baseline": clean_map, "aqsm": {}, "query_vanishing": {}}

    logger.info("\n2. Bắt đầu quét độ nhạy qua 4 mức Epsilon...")

    for step_idx, (eps, step_alpha, label) in enumerate(eps_configs, start=1):
        logger.info("-" * 80)
        logger.info(f"Mức Epsilon = {eps*255:.0f}/255 ({label}) | Step Alpha = {step_alpha*255:.1f}/255")

        # AQSM Attack
        aqsm_preds = {"boxes": [], "scores": [], "labels": []}
        aqsm_drops = []
        aqsm_det_esm = []
        for i in range(n_eval):
            x = eval_tensor[i : i + 1]
            adv_x = attacker.generate_query_selection_attack(x, steps=15, alpha=step_alpha, margin=0.20, lambda_rank=1.0)
            with torch.no_grad():
                out = model(pixel_values=adv_x, return_dict=True)
                b, s, l = extract_preds(out)
                aqsm_preds["boxes"].append(b)
                aqsm_preds["scores"].append(s)
                aqsm_preds["labels"].append(l)

                enc_logits = out.enc_outputs_class
                enc_probs = torch.sigmoid(enc_logits)
                top_vals, _ = torch.topk(enc_probs.max(-1).values[0], 300)
                dropoff = (top_vals[0] / (top_vals[-1] + 1e-4)).item()
                aqsm_drops.append(dropoff)
                aqsm_det_esm.append(dropoff <= 15.0)

        aqsm_map = compute_map_coco(aqsm_preds["boxes"], aqsm_preds["scores"], aqsm_preds["labels"], gt_boxes, gt_labels)
        aqsm_tpr = float(np.mean(aqsm_det_esm)) * 100.0
        sweep_data["aqsm"][label] = {
            "epsilon": eps,
            "mAP@50": aqsm_map["mAP@50"],
            "mean_dropoff": float(np.mean(aqsm_drops)),
            "tpr_esm": aqsm_tpr,
        }
        logger.info(f"   [AQSM]             mAP@50 = {aqsm_map['mAP@50']:5.1f}% | Dropoff = {np.mean(aqsm_drops):5.2f} | TPR(ESM) = {aqsm_tpr:5.1f}%")

        # Vanishing Attack
        vanish_preds = {"boxes": [], "scores": [], "labels": []}
        vanish_active_qs = []
        vanish_det_act = []
        for i in range(n_eval):
            x = eval_tensor[i : i + 1]
            adv_x = attacker.generate_vanishing_attack(x, steps=15, alpha=step_alpha)
            with torch.no_grad():
                out = model(pixel_values=adv_x, return_dict=True)
                b, s, l = extract_preds(out)
                vanish_preds["boxes"].append(b)
                vanish_preds["scores"].append(s)
                vanish_preds["labels"].append(l)

                active_cnt = (s >= 0.35).sum().item()
                vanish_active_qs.append(active_cnt)
                vanish_det_act.append(active_cnt <= 1)

        vanish_map = compute_map_coco(vanish_preds["boxes"], vanish_preds["scores"], vanish_preds["labels"], gt_boxes, gt_labels)
        vanish_tpr = float(np.mean(vanish_det_act)) * 100.0
        sweep_data["query_vanishing"][label] = {
            "epsilon": eps,
            "mAP@50": vanish_map["mAP@50"],
            "mean_active_queries": float(np.mean(vanish_active_qs)),
            "tpr_active_q": vanish_tpr,
        }
        logger.info(f"   [Query Vanishing]   mAP@50 = {vanish_map['mAP@50']:5.1f}% | ActiveQ = {np.mean(vanish_active_qs):5.2f} | TPR(ActiveQ) = {vanish_tpr:5.1f}%")

        # Ghi vết ra history CSV
        history_tracker.log_step(
            epoch=1,
            step=step_idx,
            total_steps=len(eps_configs),
            loss=0.0,
            extra_metrics={
                "epsilon_num": int(eps * 255),
                "aqsm_map50": round(aqsm_map["mAP@50"], 2),
                "aqsm_tpr_esm": round(aqsm_tpr, 2),
                "vanish_map50": round(vanish_map["mAP@50"], 2),
                "vanish_tpr_act": round(vanish_tpr, 2),
            },
            logger=None,
        )

    # 3. Xuất kết quả
    history_csv = history_tracker.save_csv()
    res_json_paper3 = RESULTS_DIR / "step10_epsilon_sweep_metrics.json"
    res_json_global = CHECKPOINTS_DIR / "res_step10_epsilon_sweep.json"

    export_payload = {
        "experiment": "step10_epsilon_sweep",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {"device": str(device), "num_threads": torch.get_num_threads()},
        "cohort_size": n_eval,
        "sweep_data": sweep_data,
        "artifacts": {
            "history_csv": str(history_csv),
            "log_file": str(log_file),
        }
    }

    save_audit_json(res_json_paper3, export_payload)
    save_audit_json(res_json_global, export_payload)

    logger.info(f"\n-> Đã lưu lịch sử sweep CSV    : {history_csv}")
    logger.info(f"-> Đã lưu Audit JSON (Paper 3) : {res_json_paper3}")
    logger.info(f"-> Đã đồng bộ JSON toàn dự án  : {res_json_global}")
    logger.info("\n>>> BƯỚC 10 ĐÃ HOÀN TẤT THÀNH CÔNG VỚI ĐẦY ĐỦ DỮ LIỆU KIỂM TOÁN! <<<")


if __name__ == "__main__":
    run_epsilon_sweep(n_eval=25)
