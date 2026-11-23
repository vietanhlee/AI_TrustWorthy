from scipy import stats
# -*- coding: utf-8 -*-
"""
Module: paper_1_aqsm_mechanism.experiments.tasks.task1_causal_surgery
Mục đích: Task độc lập chạy phẫu thuật can thiệp nhân quả 2x2 Factorial Tensor-Swap (6 arms)
tại ranh giới Encoder -> Deformable Decoder trên RT-DETR cho các đòn tấn công:
AQSM, Query Vanishing, Standard Task PGD -> Phục vụ Table 3, Table 4, Figure 4.

Đặc tính chuẩn công nghiệp (Production Standards):
  - Nhận tham số linh hoạt qua create_base_parser (--folder_image, --annotations, --output_dir, --cache_dir, --max_samples).
  - Nạp ảnh theo lô lười (Lazy Batch Loading) chống tràn RAM khi chạy quy mô 5,000 ảnh.
  - Phân tích ANOVA lặp lại 2x2 chuẩn mực (Repeated-Measures ANOVA) và Paired Bootstrap 95% CI.
  - Ghi nhận raw CSV từng ảnh và summary JSON đầy đủ ngữ cảnh phục vụ điền bài báo.
"""

import argparse
import csv
import gc
import json
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from paper_1_aqsm_mechanism.core.adv_cache import load_cached_adv_images
from paper_1_aqsm_mechanism.core.coco_gt import (
    compute_per_image_coco_aps,
    evaluate_coco_from_match_cache,
    evaluate_coco_predictions,
    postprocess_rtdetr_predictions,
    precompute_coco_match_cache,
)
from paper_1_aqsm_mechanism.core.dataset_loader import FlexibleCOCODataset
from paper_1_aqsm_mechanism.core.detr_wrapper import RTDETRModelWrapper
from paper_1_aqsm_mechanism.core.mechanism_analyzer import (
    extract_encoder_components,
    run_decoder_from_components,
)
from paper_1_aqsm_mechanism.experiments.audit_logger import ExperimentAuditLogger
from paper_1_aqsm_mechanism.experiments.config import (
    MODEL_NAME,
    N_BOOTSTRAP,
    RANDOM_SEED,
    configure_cpu_threads,
    detect_hardware_environment,
    get_device,
    get_model_tag,
    get_raw_csv_dir,
    get_summary_json_dir,
    resolve_model_name,
    update_experiment_paths,
)
from paper_1_aqsm_mechanism.utils.cli import create_base_parser
from paper_1_aqsm_mechanism.utils.io_helpers import format_duration, save_json
from paper_1_aqsm_mechanism.utils.stats import (
    bootstrap_paired_diff,
    repeated_measures_2x2_anova,
)


def _gather_priors_by_indices(comp: Dict[str, Any], ind: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Trích xuất content query và reference coordinates từ một encoder component
    bằng tập chỉ mục chỉ định `ind` (phục vụ Arm C_index và Arm D_index).
    """
    ref_dim = comp["enc_outputs_coord_logits"].shape[-1]
    mem_dim = comp["output_memory"].shape[-1]
    ref_pts = comp["enc_outputs_coord_logits"].gather(
        dim=1, index=ind.unsqueeze(-1).expand(-1, -1, ref_dim)
    ).detach()
    tgt = comp["output_memory"].gather(
        dim=1, index=ind.unsqueeze(-1).expand(-1, -1, mem_dim)
    ).detach()
    return tgt, ref_pts


def verify_intervention_validity(comp_clean: Dict[str, Any], comp_adv: Dict[str, Any]) -> Dict[str, Any]:
    """
    Kiểm tra tính hợp lệ phân phối của can thiệp nhân quả (Intervention Validity).
    Xác nhận việc tráo đổi tensor giữa Clean và Adversarial không gây ra hiện tượng sụp đổ số học.
    """
    mem_clean = comp_clean["source_flatten"]
    mem_adv = comp_adv["source_flatten"]

    norm_clean = float(torch.linalg.norm(mem_clean).item())
    norm_adv = float(torch.linalg.norm(mem_adv).item())
    norm_ratio = norm_adv / max(norm_clean, 1e-12)

    ref_clean = torch.sigmoid(comp_clean["init_reference_points"])
    ref_adv = torch.sigmoid(comp_adv["init_reference_points"])

    has_nan = bool(
        torch.isnan(mem_clean).any()
        or torch.isnan(mem_adv).any()
        or torch.isnan(ref_clean).any()
        or torch.isnan(ref_adv).any()
    )
    has_inf = bool(
        torch.isinf(mem_clean).any()
        or torch.isinf(mem_adv).any()
        or torch.isinf(ref_clean).any()
        or torch.isinf(ref_adv).any()
    )

    ref_bounds_valid = bool(
        (ref_clean >= 0.0).all()
        and (ref_clean <= 1.0).all()
        and (ref_adv >= 0.0).all()
        and (ref_adv <= 1.0).all()
    )

    return {
        "valid": (not has_nan) and (not has_inf) and ref_bounds_valid,
        "norm_clean": norm_clean,
        "norm_adv": norm_adv,
        "norm_ratio": norm_ratio,
        "has_nan": has_nan,
        "has_inf": has_inf,
        "ref_bounds_valid": ref_bounds_valid,
    }


def run_causal_surgery(
    wrapper: RTDETRModelWrapper,
    clean_imgs: torch.Tensor,
    adv_imgs: torch.Tensor,
    gt_list: List[Dict[str, Any]],
    attack_name: str = "AQSM",
    batch_size: int = 10,
    n_boot: int = N_BOOTSTRAP,
    seed: int = RANDOM_SEED,
    audit_logger: Any = None,
) -> Dict[str, Any]:
    """
    Chạy phẫu thuật can thiệp nhân quả 6 nhánh trên N ảnh, ghi nhận chi tiết
    per-image metrics ra file CSV và tính bảng ANOVA 2x2.
    """
    model = wrapper.model
    model.eval()
    n_eval = len(gt_list)
    arms = ["Arm_A", "Arm_B", "Arm_C_full", "Arm_C_index", "Arm_D_full", "Arm_D_index"]

    preds_by_arm: Dict[str, List[Dict[str, np.ndarray]]] = {a: [] for a in arms}
    evicted_ratios: List[float] = []

    print(f"\n==================================================================")
    print(f" [Causal Surgery] Khởi chạy 6 nhánh can thiệp 2x2 trên N={n_eval} ảnh")
    print(f" Đòn tấn công: {attack_name}")
    print(f"==================================================================")
    t_start = time.time()

    with torch.no_grad():
        for start in tqdm(range(0, n_eval, batch_size), desc=f"Causal Swap ({attack_name})"):
            end = min(n_eval, start + batch_size)
            c_sub = clean_imgs[start:end]
            a_sub = adv_imgs[start:end]

            c_comp = extract_encoder_components(model, c_sub)
            a_comp = extract_encoder_components(model, a_sub)

            c_ind = c_comp["topk_ind"]
            a_ind = a_comp["topk_ind"]

            for b in range(end - start):
                s_c = set(c_ind[b].cpu().tolist())
                s_a = set(a_ind[b].cpu().tolist())
                evicted_ratios.append(float(1.0 - len(s_c & s_a) / 300.0))

            tgt_c_idx, ref_c_idx = _gather_priors_by_indices(a_comp, c_ind)
            tgt_a_idx, ref_a_idx = _gather_priors_by_indices(c_comp, a_ind)

            arm_inputs = {
                "Arm_A": (c_comp["target"], c_comp["source_flatten"], c_comp["init_reference_points"], c_comp),
                "Arm_B": (a_comp["target"], a_comp["source_flatten"], a_comp["init_reference_points"], a_comp),
                "Arm_C_full": (c_comp["target"], a_comp["source_flatten"], c_comp["init_reference_points"], a_comp),
                "Arm_C_index": (tgt_c_idx, a_comp["source_flatten"], ref_c_idx, a_comp),
                "Arm_D_full": (a_comp["target"], c_comp["source_flatten"], a_comp["init_reference_points"], c_comp),
                "Arm_D_index": (tgt_a_idx, c_comp["source_flatten"], ref_a_idx, c_comp),
            }

            for arm, (tgt, src, ref, comp) in arm_inputs.items():
                dec_out = run_decoder_from_components(
                    model=model,
                    target=tgt,
                    source_flatten=src,
                    init_reference_points=ref,
                    spatial_shapes=comp["spatial_shapes"],
                    spatial_shapes_list=comp["spatial_shapes_list"],
                    level_start_index=comp["level_start_index"],
                )
                batch_preds = postprocess_rtdetr_predictions(
                    dec_out["logits"], dec_out["pred_boxes"], mode="hf_multilabel", max_dets=100
                )
                preds_by_arm[arm].extend(batch_preds)

    elapsed = time.time() - t_start
    print(f">>> Hoàn thành suy luận 6 nhánh trong {elapsed:.2f}s ({elapsed/n_eval:.3f}s/ảnh)")

    point_maps: Dict[str, Dict[str, float]] = {}
    for arm in arms:
        eval_res = evaluate_coco_predictions(preds_by_arm[arm], gt_list)
        point_maps[arm] = {
            "mAP@50": float(eval_res["mAP@50"]),
            "mAP@50:95": float(eval_res["mAP@50:95"]),
        }

    per_image_aps: Dict[str, Dict[str, np.ndarray]] = {}
    for arm in arms:
        ap50_arr, ap95_arr = compute_per_image_coco_aps(preds_by_arm[arm], gt_list)
        per_image_aps[arm] = {
            "ap50": ap50_arr,
            "ap50_95": ap95_arr,
        }

    raw_csv_dir = get_raw_csv_dir()
    raw_csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = raw_csv_dir / f"causal_anova_per_image_{attack_name.lower()}.csv"
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_index", "image_id", "gt_box_count", "evicted_ratio",
            "arm_A_ap50", "arm_B_ap50", "arm_C_full_ap50", "arm_C_index_ap50", "arm_D_full_ap50", "arm_D_index_ap50",
            "arm_A_ap95", "arm_B_ap95", "arm_C_full_ap95", "arm_C_index_ap95", "arm_D_full_ap95", "arm_D_index_ap95",
        ])
        for i in range(n_eval):
            gt_boxes = len(gt_list[i].get("boxes_xyxy", []))
            img_id = gt_list[i].get("image_id", i)
            writer.writerow([
                i, img_id, gt_boxes, f"{evicted_ratios[i]:.4f}",
                f"{per_image_aps['Arm_A']['ap50'][i]:.6f}",
                f"{per_image_aps['Arm_B']['ap50'][i]:.6f}",
                f"{per_image_aps['Arm_C_full']['ap50'][i]:.6f}",
                f"{per_image_aps['Arm_C_index']['ap50'][i]:.6f}",
                f"{per_image_aps['Arm_D_full']['ap50'][i]:.6f}",
                f"{per_image_aps['Arm_D_index']['ap50'][i]:.6f}",
                f"{per_image_aps['Arm_A']['ap50_95'][i]:.6f}",
                f"{per_image_aps['Arm_B']['ap50_95'][i]:.6f}",
                f"{per_image_aps['Arm_C_full']['ap50_95'][i]:.6f}",
                f"{per_image_aps['Arm_C_index']['ap50_95'][i]:.6f}",
                f"{per_image_aps['Arm_D_full']['ap50_95'][i]:.6f}",
                f"{per_image_aps['Arm_D_index']['ap50_95'][i]:.6f}",
            ])
    print(f">>> Đã ghi nhận số liệu thô per-image: {csv_path}")

    if audit_logger is not None:
        for i in range(n_eval):
            gt_boxes = len(gt_list[i].get("boxes_xyxy", []))
            img_id = gt_list[i].get("image_id", i)
            audit_logger.write_audit_record({
                "record_type": "IMAGE_EVAL",
                "attack": attack_name,
                "image_index": i,
                "image_id": img_id,
                "gt_box_count": gt_boxes,
                "evicted_ratio": float(evicted_ratios[i]),
                "ap50_by_arm": {arm: float(per_image_aps[arm]["ap50"][i]) for arm in arms},
                "ap95_by_arm": {arm: float(per_image_aps[arm]["ap50_95"][i]) for arm in arms},
            })

    anova_full = repeated_measures_2x2_anova(
        y_00=per_image_aps["Arm_A"]["ap50"],
        y_11=per_image_aps["Arm_B"]["ap50"],
        y_10=per_image_aps["Arm_C_full"]["ap50"],
        y_01=per_image_aps["Arm_D_full"]["ap50"],
        n_boot=min(n_boot, 1000),
        seed=seed,
    )
    anova_index = repeated_measures_2x2_anova(
        y_00=per_image_aps["Arm_A"]["ap50"],
        y_11=per_image_aps["Arm_B"]["ap50"],
        y_10=per_image_aps["Arm_C_index"]["ap50"],
        y_01=per_image_aps["Arm_D_index"]["ap50"],
        n_boot=min(n_boot, 1000),
        seed=seed,
    )

    pairs = [
        ("Arm_D_full_minus_Arm_A", "Arm_D_full", "Arm_A"),
        ("Arm_D_index_minus_Arm_A", "Arm_D_index", "Arm_A"),
        ("Arm_C_full_minus_Arm_A", "Arm_C_full", "Arm_A"),
        ("Arm_B_minus_Arm_A", "Arm_B", "Arm_A"),
        ("Arm_D_full_minus_Arm_B", "Arm_D_full", "Arm_B"),
    ]
    boot_deltas: Dict[str, Dict[str, Any]] = {}
    rng = np.random.RandomState(seed)

    arm_caches = {arm: precompute_coco_match_cache(preds_by_arm[arm], gt_list) for arm in arms}
    d50_samples: Dict[str, List[float]] = {p: [] for p, _, _ in pairs}
    d95_samples: Dict[str, List[float]] = {p: [] for p, _, _ in pairs}

    for _ in tqdm(range(n_boot), desc=f"Bootstrap 95% CI ({attack_name})", leave=False):
        idx = rng.randint(0, n_eval, size=n_eval).tolist()
        boot_m = {arm: evaluate_coco_from_match_cache(arm_caches[arm], indices=idx) for arm in arms}
        for p_name, arm1, arm2 in pairs:
            d50_samples[p_name].append(boot_m[arm1]["mAP@50"] - boot_m[arm2]["mAP@50"])
            d95_samples[p_name].append(boot_m[arm1]["mAP@50:95"] - boot_m[arm2]["mAP@50:95"])

    for p_name, arm1, arm2 in pairs:
        d50_pt = point_maps[arm1]["mAP@50"] - point_maps[arm2]["mAP@50"]
        d95_pt = point_maps[arm1]["mAP@50:95"] - point_maps[arm2]["mAP@50:95"]
        ci50 = np.percentile(d50_samples[p_name], [2.5, 97.5]).tolist()
        ci95 = np.percentile(d95_samples[p_name], [2.5, 97.5]).tolist()
        boot_deltas[p_name] = {
            "point_delta_mAP@50": float(d50_pt),
            "ci95_mAP@50": [float(ci50[0]), float(ci50[1])],
            "point_delta_mAP@50_95": float(d95_pt),
            "ci95_mAP@50_95": [float(ci95[0]), float(ci95[1])],
        }

    # =========================================================================
    # BỔ SUNG KIỂM ĐỊNH PHI THAM SỐ (NON-PARAMETRIC CAUSAL INFERENCE)
    # Khắc phục triệt để floor effect & vi phạm giả định chuẩn của ANOVA
    # =========================================================================
    valid_mask = np.array([len(gt_list[i].get("boxes_xyxy", [])) > 0 for i in range(n_eval)])
    n_valid = int(np.sum(valid_mask))
    non_parametric_stats: Dict[str, Any] = {}
    
    contrast_definitions = [
        ("Arm_D_full_vs_Arm_B", "Arm_D_full", "Arm_B", "Causal Recovery of Clean Memory"),
        ("Arm_D_index_vs_Arm_A", "Arm_D_index", "Arm_A", "Eviction Penalty under Clean Memory"),
        ("Arm_C_full_vs_Arm_A", "Arm_C_full", "Arm_A", "Memory Damage under Clean Queries"),
        ("Arm_B_vs_Arm_A", "Arm_B", "Arm_A", "Total Joint Attack Damage"),
        ("Arm_D_index_vs_Arm_D_full", "Arm_D_index", "Arm_D_full", "Index vs Full Query Prior"),
    ]
    
    print(f"\n--- Phân tích Thống kê Phi tham số cấp ảnh độc lập (N={n_valid} ảnh hợp lệ) ---")
    for c_key, a1, a2, desc in contrast_definitions:
        diff_arr = (per_image_aps[a1]["ap50"][valid_mask] - per_image_aps[a2]["ap50"][valid_mask]) * 100.0
        mean_diff = float(np.mean(diff_arr))
        median_diff = float(np.median(diff_arr))
        std_diff = float(np.std(diff_arr, ddof=1))
        cohen_d = float(mean_diff / std_diff) if std_diff > 0 else 0.0
        
        # Paired Wilcoxon signed-rank test
        if np.any(diff_arr != 0):
            w_stat, w_p = stats.wilcoxon(diff_arr, alternative='two-sided')
            w_stat, w_p = float(w_stat), float(w_p)
        else:
            w_stat, w_p = 0.0, 1.0
            
        # Bootstrap BCa 95% Confidence Interval
        boot_means = []
        for _ in range(2000):
            s_idx = rng.randint(0, len(diff_arr), size=len(diff_arr))
            boot_means.append(np.mean(diff_arr[s_idx]))
        boot_means = np.sort(boot_means)
        ci_low = float(boot_means[int(2000 * 0.025)])
        ci_high = float(boot_means[int(2000 * 0.975)])
        
        non_parametric_stats[c_key] = {
            "description": desc,
            "mean_diff_pp": round(mean_diff, 2),
            "median_diff_pp": round(median_diff, 2),
            "std_diff": round(std_diff, 2),
            "cohen_d": round(cohen_d, 2),
            "paired_wilcoxon_stat": w_stat,
            "paired_wilcoxon_p_value": w_p,
            "bootstrap_95ci_pp": [round(ci_low, 2), round(ci_high, 2)],
        }
        print(f"  * {desc} ({a1} - {a2}): Mean = {mean_diff:+.2f} pp, Median = {median_diff:+.2f} pp, Cohen's d = {cohen_d:.2f}, Wilcoxon p = {w_p:.2e}")

    # Tính Per-image cell means cho từng nhánh can thiệp (Arm A, B, C_full, C_index, D_full, D_index)
    per_image_cell_means: Dict[str, Dict[str, float]] = {}
    for arm in arms:
        val_50 = per_image_aps[arm]["ap50"][valid_mask] * 100.0
        val_95 = per_image_aps[arm]["ap50_95"][valid_mask] * 100.0
        per_image_cell_means[arm] = {
            "ap50_mean": round(float(np.mean(val_50)), 2),
            "ap50_std": round(float(np.std(val_50, ddof=1)), 2) if len(val_50) > 1 else 0.0,
            "ap50_sem": round(float(np.std(val_50, ddof=1) / np.sqrt(len(val_50))), 2) if len(val_50) > 1 else 0.0,
            "ap50_median": round(float(np.median(val_50)), 2),
            "ap95_mean": round(float(np.mean(val_95)), 2),
            "ap95_std": round(float(np.std(val_95, ddof=1)), 2) if len(val_95) > 1 else 0.0,
            "ap95_sem": round(float(np.std(val_95, ddof=1) / np.sqrt(len(val_95))), 2) if len(val_95) > 1 else 0.0,
            "ap95_median": round(float(np.median(val_95)), 2),
        }

    arm_definitions = {
        "Arm_A": "Clean Baseline: Clean Memory M_clean, Clean Query Prior Q_clean (Top-300 clean)",
        "Arm_B": "Full Joint Attack: Adversarial Memory M_adv, Adversarial Query Prior Q_adv (Top-300 adv)",
        "Arm_C_full": "Intervention: Adversarial Memory M_adv, Clean Query Prior Q_clean (Top-300 clean content & coordinates)",
        "Arm_C_index": "Intervention: Adversarial Memory M_adv, Query Prior gathered from Adversarial Encoder using Clean Indices I_clean",
        "Arm_D_full": "Intervention: Clean Memory M_clean, Adversarial Query Prior Q_adv (Top-300 adv content & coordinates)",
        "Arm_D_index": "Intervention: Clean Memory M_clean, Query Prior gathered from Clean Encoder using Adversarial Indices I_adv",
    }

    results_summary: Dict[str, Any] = {
        "attack_name": attack_name,
        "n_evaluated_images": n_eval,
        "mean_evicted_ratio": float(np.mean(evicted_ratios)),
        "std_evicted_ratio": float(np.std(evicted_ratios)),
        "point_maps": point_maps,
        "per_image_cell_means": per_image_cell_means,
        "arm_definitions": arm_definitions,
        "anova_repeated_measures_full": anova_full,
        "anova_repeated_measures_index": anova_index,
        "bootstrap_paired_deltas": boot_deltas,
        "non_parametric_contrasts": non_parametric_stats,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    summary_json_dir = get_summary_json_dir()
    summary_json_dir.mkdir(parents=True, exist_ok=True)
    json_path = summary_json_dir / f"causal_swap_results_{attack_name.lower()}.json"
    save_json(results_summary, json_path)
    print(f">>> Đã lưu tổng hợp thống kê ANOVA: {json_path}")

    return results_summary


def main():
    parser = create_base_parser("Causal Surgery 2x2 Factorial Swap", task_name="Task 1")
    parser.add_argument("--attacks", type=str, default="aqsm,vanishing,taskpgd", help="Danh sách đòn tấn công cách nhau bởi dấu phẩy")
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

    audit_logger = ExperimentAuditLogger(task_name="task1_causal_surgery")

    # Nạp mô hình RT-DETR
    wrapper = RTDETRModelWrapper(pretrained_model_name=chosen_model, device=device)
    wrapper.eval()

    # Nạp dataset linh hoạt có hỗ trợ max_samples
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
        "max_samples": args.max_samples,
    })

    attack_list = [a.strip().lower() for a in args.attacks.split(",") if a.strip()]
    for atk in attack_list:
        audit_logger.logger.info(f"\n=======================================================")
        audit_logger.logger.info(f"=== Đang thực hiện Causal Surgery cho: {atk.upper()} (N={n_actual}) ===")
        audit_logger.logger.info(f"=======================================================")

        adv_imgs, _ = load_cached_adv_images(
            attack=atk,
            eps_num=16,
            n_eval=n_actual,
            clean_imgs=clean_imgs,
            expected_image_ids=image_ids,
            cache_dir=args.cache_dir,
            wrapper=wrapper,
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
        audit_logger.logger.info(f"Hoàn thành {atk.upper()}: ANOVA eta_p^2(Q)={eta_q:.4f}, eta_p^2(M)={eta_m:.4f}")

        del adv_imgs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    audit_logger.close()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print("\n[HOÀN THÀNH TOÀN BỘ] Task 1: Causal Surgery 2x2 Factorial Swap thành công!")


if __name__ == "__main__":
    main()
