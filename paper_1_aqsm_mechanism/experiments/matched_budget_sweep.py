# -*- coding: utf-8 -*-
"""
Module: paper_1_aqsm_mechanism.experiments.matched_budget_sweep
Mục đích: Thực hiện quét ngân sách nhiễu tương thích (Matched-Budget Perturbation Sweep)
theo chuẩn L_inf qua các mức eps in {4/255, 8/255, 16/255} (RQ4):
  - So sánh đối chiếu hiệu quả phá hoại mAP@50 giữa AQSM, Standard Task PGD, và Query Vanishing.
  - Phân tích động lực biến thiên của Query Eviction Ratio và Encoder Selection Margin theo ngân sách nhiễu.

Lưu trữ chi tiết kết quả thô ra CSV và tổng hợp thống kê ra JSON.
"""

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from edge_sec_pro.core.coco_gt import (
    evaluate_coco_predictions,
    postprocess_rtdetr_predictions,
)
from edge_sec_pro.transformer_monitors.detr_wrapper import RTDETRModelWrapper
from edge_sec_pro.transformer_monitors.rtdetr_mechanism_analyzer import extract_encoder_components
from paper_1_aqsm_mechanism.experiments.attack_engine import DiagnosticAttackEngine
from paper_1_aqsm_mechanism.experiments.config import (
    PGD_NUM_STEPS,
    PGD_STEP_SIZE,
    RAW_CSV_DIR,
    SUMMARY_JSON_DIR,
    SWEEP_EPSILONS,
)


def run_matched_budget_sweep(
    wrapper: RTDETRModelWrapper,
    clean_imgs: torch.Tensor,
    gt_list: List[Dict[str, Any]],
    epsilons: Optional[List[float]] = None,
    batch_size: int = 10,
    audit_logger: Any = None,
) -> Dict[str, Any]:
    """
    Quét ngân sách nhiễu trên tập ảnh kiểm thử và đo đạc mAP theo chuẩn COCO.
    """
    model = wrapper.model
    model.eval()
    n_eval = len(gt_list)
    eps_list = epsilons or SWEEP_EPSILONS

    print(f"\n==================================================================")
    print(f" [Matched-Budget Sweep] Quét ngân sách L_inf qua eps={eps_list}")
    print(f" Quy mô: N={n_eval} ảnh | So sánh: AQSM vs Task PGD vs Vanishing")
    print(f"==================================================================")

    attacks = ["aqsm", "task_pgd", "vanishing"]
    sweep_results: Dict[str, Dict[str, Any]] = {
        atk: {f"eps_{int(e*255)}": {} for e in eps_list}
        for atk in attacks
    }

    # Bảng lưu trữ dòng thô
    raw_rows: List[List[Any]] = []

    # 1. Đánh giá Clean Baseline
    print(">>> Đánh giá Clean Baseline...")
    clean_preds: List[Dict[str, np.ndarray]] = []
    with torch.no_grad():
        for start in range(0, n_eval, batch_size):
            end = min(n_eval, start + batch_size)
            out = model(clean_imgs[start:end])
            clean_preds.extend(postprocess_rtdetr_predictions(out.logits, out.pred_boxes, mode="hf_multilabel", max_dets=100))
    clean_metrics = evaluate_coco_predictions(clean_preds, gt_list)
    clean_map50 = float(clean_metrics["mAP@50"])
    clean_map95 = float(clean_metrics["mAP@50:95"])
    print(f">>> Clean Baseline: mAP@50 = {clean_map50*100:.2f}%, mAP@[50:95] = {clean_map95*100:.2f}%")

    # 2. Lặp qua từng mức ngân sách epsilon
    cache_dir = Path(__file__).resolve().parent.parent.parent / "edge_sec_pro" / "results" / "checkpoints" / "adv_cache"

    for eps in eps_list:
        eps_key = f"eps_{int(eps * 255)}"
        eps_num = f"{int(eps * 255)}/255"
        eps_int = int(round(eps * 255))
        print(f"\n>>> Đang quét ngân sách epsilon = {eps_num} ({eps:.4f})...")

        engine = DiagnosticAttackEngine(
            model=wrapper,
            eps=eps,
            alpha=eps / 8.0,
            num_steps=PGD_NUM_STEPS,
        )

        for atk in attacks:
            print(f"    -> Đang chuẩn bị mẫu và đánh giá đòn tấn công: {atk.upper()} (eps={eps_num})...")
            atk_preds: List[Dict[str, np.ndarray]] = []
            evicted_list: List[float] = []

            # Thử nạp cache nếu có
            cache_atk = "taskpgd" if atk in ["task_pgd", "taskpgd"] else atk
            cache_file = cache_dir / f"adv_cache_{cache_atk}_eps{eps_int}_n100.npz"
            if not cache_file.exists():
                cache_file = cache_dir / f"adv_cache_{cache_atk}_eps{eps_int}_n300.npz"

            if cache_file.exists():
                cached_deltas = np.load(cache_file)["delta_f16"][:n_eval].astype(np.float32)
                adv_all = torch.clamp(clean_imgs[:n_eval] + torch.from_numpy(cached_deltas).to(clean_imgs.device), 0.0, 1.0)
            else:
                adv_all = None

            for start in tqdm(range(0, n_eval, batch_size), desc=f"{atk} (eps={eps_num})", leave=False):
                end = min(n_eval, start + batch_size)
                c_sub = clean_imgs[start:end]

                if adv_all is not None:
                    a_sub = adv_all[start:end]
                else:
                    if atk == "aqsm":
                        a_sub, _ = engine.generate_aqsm(c_sub)
                    elif atk == "vanishing":
                        a_sub, _ = engine.generate_query_vanishing(c_sub)
                    else:
                        a_sub, _ = engine.generate_task_pgd(c_sub)

                # Suy luận và trích xuất chỉ số
                with torch.no_grad():
                    out = model(a_sub)
                    atk_preds.extend(postprocess_rtdetr_predictions(out.logits, out.pred_boxes, mode="hf_multilabel", max_dets=100))

                    if atk == "aqsm":
                        c_comp = extract_encoder_components(model, c_sub)
                        a_comp = extract_encoder_components(model, a_sub)
                        c_ind = c_comp["topk_ind"]
                        a_ind = a_comp["topk_ind"]
                        for b in range(end - start):
                            s_c = set(c_ind[b].cpu().tolist())
                            s_a = set(a_ind[b].cpu().tolist())
                            evicted_list.append(float(1.0 - len(s_c & s_a) / 300.0))

            m = evaluate_coco_predictions(atk_preds, gt_list)
            map50 = float(m["mAP@50"])
            map95 = float(m["mAP@50:95"])
            mean_evict = float(np.mean(evicted_list)) if evicted_list else 0.0

            sweep_results[atk][eps_key] = {
                "epsilon": eps,
                "epsilon_str": eps_num,
                "mAP@50": map50,
                "mAP@50:95": map95,
                "relative_drop_pct": float((clean_map50 - map50) / clean_map50 * 100.0),
                "mean_evicted_ratio": mean_evict,
            }

            raw_rows.append([
                atk,
                eps_num,
                f"{eps:.6f}",
                f"{map50:.6f}",
                f"{map95:.6f}",
                f"{(clean_map50 - map50) / clean_map50 * 100.0:.2f}",
                f"{mean_evict:.4f}",
            ])

            print(f"       Kết quả {atk.upper()}: mAP@50 = {map50*100:.2f}% (Drop: -{(clean_map50 - map50)/clean_map50*100:.1f}%) | Eviction: {mean_evict*100:.1f}%")

    # 3. Ghi dữ liệu thô ra file CSV phục vụ audit
    csv_path = RAW_CSV_DIR / "matched_budget_sweep_raw.csv"
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["attack", "epsilon_label", "epsilon_val", "mAP50", "mAP50_95", "relative_drop_pct", "mean_evicted_ratio"])
        for row in raw_rows:
            writer.writerow(row)
    print(f">>> Đã ghi nhận số liệu quét ngân sách: {csv_path}")

    # Ghi audit stream JSONL nếu có audit_logger
    if audit_logger is not None:
        for r in raw_rows:
            audit_logger.write_audit_record({
                "record_type": "SWEEP_EVAL",
                "attack": r[0],
                "epsilon_label": r[1],
                "epsilon_val": float(r[2]),
                "mAP50": float(r[3]),
                "mAP50_95": float(r[4]),
                "relative_drop_pct": float(r[5]),
                "mean_evicted_ratio": float(r[6]),
            })

    # 4. Lưu tổng hợp JSON
    final_output = {
        "clean_baseline": {
            "mAP@50": clean_map50,
            "mAP@50:95": clean_map95,
        },
        "sweep_results": sweep_results,
    }
    json_path = SUMMARY_JSON_DIR / "matched_budget_results.json"
    with open(json_path, mode="w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
    print(f">>> Đã lưu tổng hợp JSON: {json_path}")

    return final_output
