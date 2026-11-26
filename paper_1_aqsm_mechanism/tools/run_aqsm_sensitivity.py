# -*- coding: utf-8 -*-
"""
Module: paper_1_aqsm_mechanism.tools.run_aqsm_sensitivity
Mục đích: Khảo sát độ nhạy siêu tham số (Hyperparameter Sensitivity Sweep) cho AQSM:
  - top_k_fg in {5, 15, 30}
  - k_bg in {10, 30, 60}
  - bg_rank_min in {300, 500, 1000}
  - margin_logit (gamma) in {0.5, 1.0, 2.0}

Cấu hình mặc định quét theo từng trục quanh cấu hình chuẩn (one-at-a-time) hoặc
toàn bộ lưới (--full_grid) trên tập con COCO (mặc định max_samples=50),
xuất kết quả ra outputs/summary_json/aqsm_sensitivity_sweep.json.
"""

import argparse
import gc
import itertools
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from paper_1_aqsm_mechanism.core.attack_generator import DETRAttackGenerator
from paper_1_aqsm_mechanism.core.coco_gt import (
    evaluate_coco_predictions,
    postprocess_rtdetr_predictions,
)
from paper_1_aqsm_mechanism.core.dataset_loader import FlexibleCOCODataset
from paper_1_aqsm_mechanism.core.detr_wrapper import RTDETRModelWrapper
from paper_1_aqsm_mechanism.experiments.config import (
    COCO_ANN_FILE,
    COCO_VAL_DIR,
    OUTPUTS_DIR,
    configure_cpu_threads,
    detect_hardware_environment,
    get_device,
    get_img_size,
    get_model_tag,
    get_pgd_step_size,
    resolve_model_name,
    update_experiment_paths,
)
from paper_1_aqsm_mechanism.utils.io_helpers import save_json


DEFAULT_TOP_K_FG_GRID = [5, 15, 30]
DEFAULT_K_BG_GRID = [10, 30, 60]
DEFAULT_BG_RANK_MIN_GRID = [300, 500, 1000]
DEFAULT_MARGIN_LOGIT_GRID = [0.5, 1.0, 2.0]


def build_sweep_configs(full_grid: bool = False) -> List[Dict[str, Any]]:
    """
    Xây dựng danh sách cấu hình siêu tham số cần khảo sát.
    Nếu full_grid=False: quét từng trục quanh điểm chuẩn (15, 30, 500, 1.0).
    Nếu full_grid=True: quét toàn bộ tích Đề-các (3^4 = 81 cấu hình).
    """
    baseline = (15, 30, 500, 1.0)
    if full_grid:
        combos = list(
            itertools.product(
                DEFAULT_TOP_K_FG_GRID,
                DEFAULT_K_BG_GRID,
                DEFAULT_BG_RANK_MIN_GRID,
                DEFAULT_MARGIN_LOGIT_GRID,
            )
        )
    else:
        seen = set()
        combos: List[Tuple[int, int, int, float]] = [baseline]
        seen.add(baseline)
        for v in DEFAULT_TOP_K_FG_GRID:
            c = (v, baseline[1], baseline[2], baseline[3])
            if c not in seen:
                combos.append(c)
                seen.add(c)
        for v in DEFAULT_K_BG_GRID:
            c = (baseline[0], v, baseline[2], baseline[3])
            if c not in seen:
                combos.append(c)
                seen.add(c)
        for v in DEFAULT_BG_RANK_MIN_GRID:
            c = (baseline[0], baseline[1], v, baseline[3])
            if c not in seen:
                combos.append(c)
                seen.add(c)
        for v in DEFAULT_MARGIN_LOGIT_GRID:
            c = (baseline[0], baseline[1], baseline[2], v)
            if c not in seen:
                combos.append(c)
                seen.add(c)

    return [
        {
            "top_k_fg": int(k_fg),
            "k_bg": int(k_bg),
            "bg_rank_min": int(r_min),
            "margin_logit": float(gamma),
            "is_default": (k_fg, k_bg, r_min, gamma) == baseline,
        }
        for (k_fg, k_bg, r_min, gamma) in combos
    ]


def evaluate_aqsm_config(
    wrapper: RTDETRModelWrapper,
    attacker: DETRAttackGenerator,
    clean_proxy: Any,
    gt_list: List[Dict[str, Any]],
    n_eval: int,
    batch_size: int,
    device: torch.device,
    cfg: Dict[str, Any],
    steps: int,
    seed: int,
) -> Dict[str, Any]:
    """Chạy AQSM với một tổ hợp siêu tham số và đo mAP@50, mAP@50:95, recall."""
    preds = []
    t0 = time.time()
    for i in range(0, n_eval, batch_size):
        end_idx = min(i + batch_size, n_eval)
        b_clean = clean_proxy[i:end_idx].to(device)
        b_adv = attacker.generate_query_selection_attack(
            b_clean,
            steps=steps,
            margin_logit=cfg["margin_logit"],
            top_k_fg=cfg["top_k_fg"],
            k_bg=cfg["k_bg"],
            bg_rank_min=cfg["bg_rank_min"],
            bg_seed=seed + i,
        )
        with torch.no_grad():
            out_adv = wrapper(b_adv)
        preds.extend(
            postprocess_rtdetr_predictions(out_adv.logits, out_adv.pred_boxes, mode="hf_multilabel")
        )

    elapsed_s = time.time() - t0
    eval_res = evaluate_coco_predictions(preds, gt_list)
    return {
        **cfg,
        "mAP@50": round(float(eval_res["mAP@50"]), 2),
        "mAP@50:95": round(float(eval_res["mAP@50:95"]), 2),
        "recall_pct": round(float(eval_res["recall_pct"]), 2),
        "elapsed_sec": round(float(elapsed_s), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="AQSM Hyperparameter Sensitivity Sweep")
    parser.add_argument("--model", type=str, default="r18", help="Tên model hoặc alias (mặc định: r18)")
    parser.add_argument("--folder_image", type=str, default=str(COCO_VAL_DIR), help="Thư mục ảnh COCO")
    parser.add_argument("--annotations", type=str, default=str(COCO_ANN_FILE), help="File instances_val2017.json")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUTS_DIR), help="Thư mục xuất kết quả")
    parser.add_argument("--max_samples", type=int, default=50, help="Số ảnh đánh giá mỗi cấu hình (mặc định: 50)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size suy luận")
    parser.add_argument("--eps_255", type=int, default=16, help="Biên độ nhiễu L_inf trên thang 255")
    parser.add_argument("--steps", type=int, default=15, help="Số bước PGD")
    parser.add_argument("--seed", type=int, default=42, help="Seed cố định cho chọn nền ngẫu nhiên")
    parser.add_argument("--full_grid", action="store_true", help="Quét toàn bộ tích Đề-các 81 cấu hình thay vì one-at-a-time")
    parser.add_argument("--device", type=str, default="auto", help="Thiết bị (auto / cuda / cpu)")
    parser.add_argument("--threads", type=int, default=8, help="Số luồng CPU")
    args = parser.parse_args()

    configure_cpu_threads(args.threads)
    hw_info = detect_hardware_environment(preferred_device=args.device)
    device = get_device(preferred=args.device)
    batch_size = args.batch_size if args.batch_size is not None else hw_info["recommended_batch_size"]

    full_model_name = resolve_model_name(args.model)
    model_tag = get_model_tag(full_model_name)
    update_experiment_paths(
        folder_image=args.folder_image,
        annotations=args.annotations,
        output_dir=args.output_dir,
        model=full_model_name,
        pgd_steps=args.steps,
        eps_l_inf=float(args.eps_255) / 255.0,
    )

    wrapper = RTDETRModelWrapper(pretrained_model_name=full_model_name, device=device)
    wrapper.eval()

    dataset = FlexibleCOCODataset(
        folder_image=args.folder_image,
        annotations_file=args.annotations,
        img_size=get_img_size(),
        id2label=wrapper.model.config.id2label,
        lazy_tensors=True,
        device=device,
        max_samples=args.max_samples,
    )
    clean_proxy = dataset.tensors
    gt_list = dataset.gt_list
    n_eval = len(dataset)

    base_alpha = float(get_pgd_step_size())
    step_alpha = base_alpha if int(args.eps_255) >= 8 else min(base_alpha, 1.0 / 255.0)
    attacker = DETRAttackGenerator(
        wrapper,
        epsilon=float(args.eps_255) / 255.0,
        alpha=step_alpha,
        num_steps=args.steps,
    )

    # Đánh giá Clean baseline để đối chiếu
    clean_preds = []
    for i in range(0, n_eval, batch_size):
        end_idx = min(i + batch_size, n_eval)
        with torch.no_grad():
            out_c = wrapper(clean_proxy[i:end_idx].to(device))
        clean_preds.extend(
            postprocess_rtdetr_predictions(out_c.logits, out_c.pred_boxes, mode="hf_multilabel")
        )
    clean_eval = evaluate_coco_predictions(clean_preds, gt_list)

    configs = build_sweep_configs(full_grid=args.full_grid)
    print("=" * 80)
    print(f" [AQSM SENSITIVITY SWEEP] Model: {full_model_name} ({model_tag})")
    print(f" Samples: {n_eval} | Configs: {len(configs)} | Clean mAP@50: {clean_eval['mAP@50']:.2f}%")
    print("=" * 80)

    sweep_records: List[Dict[str, Any]] = []
    for cfg in tqdm(configs, desc="AQSM Sweep"):
        rec = evaluate_aqsm_config(
            wrapper=wrapper,
            attacker=attacker,
            clean_proxy=clean_proxy,
            gt_list=gt_list,
            n_eval=n_eval,
            batch_size=batch_size,
            device=device,
            cfg=cfg,
            steps=args.steps,
            seed=args.seed,
        )
        sweep_records.append(rec)
        print(
            f"  [k_fg={cfg['top_k_fg']}, k_bg={cfg['k_bg']}, r_min={cfg['bg_rank_min']}, "
            f"gamma={cfg['margin_logit']}] -> mAP@50: {rec['mAP@50']:.2f}%"
        )

    out_dir = Path(args.output_dir)
    summary_dir = out_dir / "summary_json"
    summary_dir.mkdir(parents=True, exist_ok=True)
    out_path = summary_dir / "aqsm_sensitivity_sweep.json"

    payload = {
        "task": "aqsm_hyperparameter_sensitivity_sweep",
        "model": full_model_name,
        "model_tag": model_tag,
        "n_samples": n_eval,
        "eps_255": args.eps_255,
        "steps": args.steps,
        "seed": args.seed,
        "full_grid": bool(args.full_grid),
        "clean_baseline": {
            "mAP@50": round(float(clean_eval["mAP@50"]), 2),
            "mAP@50:95": round(float(clean_eval["mAP@50:95"]), 2),
            "recall_pct": round(float(clean_eval["recall_pct"]), 2),
        },
        "sweep_results": sweep_records,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_json(payload, out_path)
    print(f"\n>>> Đã xuất kết quả sensitivity sweep: {out_path}")

    del wrapper, attacker
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
