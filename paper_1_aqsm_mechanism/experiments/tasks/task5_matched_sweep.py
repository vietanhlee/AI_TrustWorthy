# -*- coding: utf-8 -*-
"""
Module: paper_1_aqsm_mechanism.experiments.tasks.task5_matched_sweep
Mục đích: Thực hiện quét ngân sách nhiễu tương thích (Matched-Budget Perturbation Sweep)
theo chuẩn L_inf qua các mức eps in {4/255, 8/255, 16/255} (RQ4) -> Phục vụ Table 9 & Figure 7:
  - So sánh đối chiếu hiệu quả phá hoại mAP@50 giữa AQSM, Standard Task PGD, và Query Vanishing.
  - Phân tích động lực biến thiên của Query Eviction Ratio và Encoder Selection Margin theo ngân sách nhiễu.

Đặc tính chuẩn công nghiệp (Production Standards):
  - Nhận tham số linh hoạt qua create_base_parser (--folder_image, --annotations, --output_dir, --cache_dir, --max_samples).
  - Nạp ảnh theo lô lười (Lazy Batch Loading) chống tràn RAM khi chạy quy mô 5,000 ảnh.
  - Tự động ưu tiên nạp từ cache đã sinh, fallback sang sinh trực tiếp nếu thiếu.
  - Lưu trữ chi tiết kết quả thô ra CSV và tổng hợp thống kê ra JSON.
"""

import gc
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from tqdm import tqdm

WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from paper_1_aqsm_mechanism.core.adv_cache import load_cached_adv_images
from paper_1_aqsm_mechanism.core.attack_generator import DETRAttackGenerator
from paper_1_aqsm_mechanism.core.coco_gt import (
    evaluate_coco_predictions,
    postprocess_rtdetr_predictions,
)
from paper_1_aqsm_mechanism.core.dataset_loader import FlexibleCOCODataset
from paper_1_aqsm_mechanism.core.detr_wrapper import RTDETRModelWrapper
from paper_1_aqsm_mechanism.core.mechanism_analyzer import extract_encoder_components
from paper_1_aqsm_mechanism.experiments.audit_logger import ExperimentAuditLogger
from paper_1_aqsm_mechanism.experiments.config import (
    ADV_CACHE_DIR,
    MODEL_NAME,
    PGD_NUM_STEPS,
    PGD_STEP_SIZE,
    RANDOM_SEED,
    SWEEP_EPSILONS,
    configure_cpu_threads,
    detect_hardware_environment,
    get_device,
    get_raw_csv_dir,
    get_summary_json_dir,
    resolve_model_name,
    update_experiment_paths,
)
from paper_1_aqsm_mechanism.utils.cli import create_base_parser
from paper_1_aqsm_mechanism.utils.io_helpers import save_json, write_csv_rows


def run_matched_budget_sweep(
    wrapper: RTDETRModelWrapper,
    clean_imgs: Any,
    gt_list: List[Dict[str, Any]],
    image_ids: Sequence[int],
    epsilons: Optional[List[float]] = None,
    batch_size: int = 10,
    cache_dir: Optional[Path] = None,
    audit_logger: Any = None,
) -> Dict[str, Any]:
    """
    Quét ngân sách nhiễu trên tập ảnh kiểm thử và đo đạc mAP theo chuẩn COCO.
    """
    model = wrapper.model
    model.eval()
    device = wrapper.device
    n_eval = len(gt_list)
    eps_list = epsilons or SWEEP_EPSILONS
    cdir = Path(cache_dir) if cache_dir is not None else ADV_CACHE_DIR

    print("\n==================================================================")
    print(f" [Matched-Budget Sweep] Quét ngân sách L_inf qua eps={[f'{int(round(e*255))}/255' for e in eps_list]}")
    print(f" Quy mô: N={n_eval} ảnh | So sánh: AQSM vs Task PGD vs Vanishing")
    print("==================================================================")

    attacks = ["aqsm", "vanishing", "taskpgd"]
    sweep_results: Dict[str, Dict[str, Any]] = {
        atk: {f"eps_{int(round(e*255))}": {} for e in eps_list}
        for atk in attacks
    }

    raw_rows: List[List[Any]] = []

    print(">>> Đánh giá Clean Baseline...")
    clean_preds: List[Dict[str, np.ndarray]] = []
    with torch.no_grad():
        for start in tqdm(range(0, n_eval, batch_size), desc="Clean Eval"):
            end = min(n_eval, start + batch_size)
            c_sub = clean_imgs[start:end]
            if hasattr(c_sub, "to"):
                c_sub = c_sub.to(device)
            out = model(c_sub)
            clean_preds.extend(postprocess_rtdetr_predictions(out.logits, out.pred_boxes, mode="hf_multilabel", max_dets=100))

    clean_metrics = evaluate_coco_predictions(clean_preds, gt_list)
    clean_map50 = float(clean_metrics["mAP@50"])
    clean_map95 = float(clean_metrics["mAP@50:95"])
    print(f"    Clean Baseline: mAP@50 = {clean_map50:.2f}% | mAP@50:95 = {clean_map95:.2f}%\n")

    for eps in eps_list:
        eps_int = int(round(eps * 255.0))
        eps_num = f"{eps_int}/255"
        eps_key = f"eps_{eps_int}"
        print(f"\n---> Bắt đầu đo đạc tại ngân sách: eps = {eps_num} ({eps:.4f})")

        attacker = DETRAttackGenerator(
            model_wrapper=wrapper,
            epsilon=eps,
            alpha=PGD_STEP_SIZE if eps_int >= 8 else (1.0 / 255.0),
            num_steps=PGD_NUM_STEPS if eps_int >= 16 else min(PGD_NUM_STEPS, 10),
        )

        for atk in attacks:
            atk_preds: List[Dict[str, np.ndarray]] = []
            evicted_list: List[float] = []

            adv_all = None
            try:
                adv_all, _ = load_cached_adv_images(
                    attack=atk,
                    eps_num=eps_int,
                    n_eval=n_eval,
                    clean_imgs=clean_imgs,
                    expected_image_ids=image_ids,
                    cache_dir=cdir,
                    wrapper=wrapper,
                )
                print(f"      [Cache Hit] Đã nạp thành công cache cho {atk.upper()} (eps={eps_num})")
            except Exception:
                print(f"      [Live PGD] Sinh đối kháng trực tiếp {atk.upper()} (eps={eps_num}) on-the-fly...")

            with tqdm(total=n_eval, desc=f"  Eval {atk.upper()} ({eps_num})", unit="img") as pbar:
                for start in range(0, n_eval, batch_size):
                    end = min(n_eval, start + batch_size)
                    c_sub = clean_imgs[start:end]
                    if hasattr(c_sub, "to"):
                        c_sub = c_sub.to(device)

                    if adv_all is not None:
                        a_sub = adv_all[start:end]
                    else:
                        if atk == "aqsm":
                            a_sub = attacker.generate_query_selection_attack(c_sub)
                        elif atk == "vanishing":
                            a_sub = attacker.generate_vanishing_attack(c_sub)
                        else:
                            a_sub = attacker.generate_standard_task_pgd(c_sub)

                    if hasattr(a_sub, "to"):
                        a_sub = a_sub.to(device)

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
                                evicted_list.append(float(1.0 - len(s_c & s_a) / float(max(len(s_c), 1))))

                    pbar.update(end - start)

            m = evaluate_coco_predictions(atk_preds, gt_list)
            map50 = float(m["mAP@50"])
            map95 = float(m["mAP@50:95"])
            mean_evict = float(np.mean(evicted_list)) if evicted_list else float("nan")
            rel_drop = float((clean_map50 - map50) / max(clean_map50, 1e-4) * 100.0)

            sweep_results[atk][eps_key] = {
                "epsilon": eps,
                "epsilon_str": eps_num,
                "mAP@50": round(map50, 2),
                "mAP@50:95": round(map95, 2),
                "relative_drop_pct": round(rel_drop, 2),
                "mean_evicted_ratio": round(mean_evict, 4) if not np.isnan(mean_evict) else None,
            }

            raw_rows.append([
                atk,
                eps_num,
                f"{eps:.6f}",
                f"{map50:.6f}",
                f"{map95:.6f}",
                f"{rel_drop:.2f}",
                f"{mean_evict:.4f}" if not np.isnan(mean_evict) else "nan",
            ])

            print(f"       Kết quả {atk.upper()}: mAP@50 = {map50:.2f}% (Drop: -{rel_drop:.1f}%)" +
                  (f" | Eviction: {mean_evict*100:.1f}%" if not np.isnan(mean_evict) else ""))

    raw_csv_dir = get_raw_csv_dir()
    raw_csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = raw_csv_dir / "matched_budget_sweep_raw.csv"
    csv_header = ["attack", "epsilon_label", "epsilon_val", "mAP50", "mAP50_95", "relative_drop_pct", "mean_evicted_ratio"]
    write_csv_rows(csv_path, csv_header, raw_rows)
    print(f"\n>>> Đã ghi nhận số liệu quét ngân sách: {csv_path}")

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
                "mean_evicted_ratio": float(r[6]) if r[6] != "nan" else 0.0,
            })

    final_output = {
        "experiment": "Table 9 & Figure 8: Matched-Budget Perturbation Sweep",
        "n_images": n_eval,
        "model": getattr(wrapper, "pretrained_model_name", None) or resolve_model_name(),
        "device": str(device),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "clean_baseline": {
            "mAP@50": round(clean_map50, 2),
            "mAP@50:95": round(clean_map95, 2),
        },
        "sweep_results": sweep_results,
    }
    summary_json_dir = get_summary_json_dir()
    summary_json_dir.mkdir(parents=True, exist_ok=True)
    json_path = summary_json_dir / "matched_budget_sweep.json"
    save_json(final_output, json_path)
    save_json(final_output, summary_json_dir / "matched_budget_results.json")
    print(f">>> Đã lưu tổng hợp JSON Table 9: {json_path}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return final_output


def main():
    parser = create_base_parser("Matched-Budget Perturbation Sweep (Table 9, Figure 8)", task_name="Task 5")
    parser.add_argument("--epsilons", type=str, default="4,8,16", help="Danh sách epsilon /255 (ví dụ: '4,8,16')")
    args = parser.parse_args()

    chosen_model = resolve_model_name(getattr(args, "model", None))
    update_experiment_paths(
        folder_image=args.folder_image,
        annotations=args.annotations,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        model=chosen_model,
    )

    configure_cpu_threads(args.threads)
    hw_info = detect_hardware_environment(preferred_device=args.device)
    device = get_device(preferred=args.device)
    batch_size = args.batch_size if args.batch_size is not None else hw_info["recommended_batch_size"]

    audit_logger = ExperimentAuditLogger(task_name="task5_matched_sweep")

    wrapper = RTDETRModelWrapper(pretrained_model_name=chosen_model, device=device)
    wrapper.eval()

    dataset = FlexibleCOCODataset(
        folder_image=args.folder_image,
        annotations_file=args.annotations,
        img_size=640,
        id2label=wrapper.model.config.id2label,
        lazy_tensors=True,
        device=device,
        max_samples=args.max_samples,
    )

    clean_imgs = dataset.tensors
    gt_list = dataset.gt_list
    image_ids = dataset.image_ids

    audit_logger.log_metadata({
        "folder_image": str(args.folder_image),
        "total_images": len(dataset),
        "batch_size": batch_size,
        "threads": args.threads,
        "device": str(device),
        "hardware_info": hw_info,
        "random_seed": RANDOM_SEED,
        "model_name": chosen_model,
        "epsilons": args.epsilons,
        "max_samples": args.max_samples,
    })

    eps_floats = [float(e.strip()) / 255.0 for e in args.epsilons.split(",") if e.strip()]

    run_matched_budget_sweep(
        wrapper=wrapper,
        clean_imgs=clean_imgs,
        gt_list=gt_list,
        image_ids=image_ids,
        epsilons=eps_floats,
        batch_size=batch_size,
        cache_dir=Path(args.cache_dir),
        audit_logger=audit_logger,
    )

    audit_logger.close()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print("\n[HOÀN THÀNH TOÀN BỘ] Task 5: Matched-Budget Perturbation Sweep thành công!")


if __name__ == "__main__":
    main()
