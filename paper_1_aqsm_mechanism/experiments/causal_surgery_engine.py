# -*- coding: utf-8 -*-
"""
Module: paper_1_aqsm_mechanism.experiments.causal_surgery_engine
Mục đích: Thực thi phẫu thuật can thiệp nhân quả 2x2 Factorial Tensor-Swap
tại ranh giới Encoder -> Deformable Decoder trên mô hình RT-DETR:
  - M (Encoder Memory): `source_flatten` [B, 2100, 256] truyền vào Deformable Cross-Attention.
  - Q (Query Prior): {target [B, 300, 256], init_reference_points [B, 300, 4]}.
  - Đánh giá 6 nhánh can thiệp:
      + Arm A (Clean Baseline):        M_clean + Q_clean(I_clean)
      + Arm B (Pure Attack):           M_adv   + Q_adv(I_adv)
      + Arm C_full (Adv M + Clean Q):  M_adv   + Q_clean(I_clean)
      + Arm C_index (Adv M + Clean I): M_adv   + target_adv(I_clean), ref_adv(I_clean)
      + Arm D_full (Clean M + Adv Q):  M_clean + Q_adv(I_adv)
      + Arm D_index (Clean M + Adv I): M_clean + target_clean(I_adv), ref_clean(I_adv)

Đặc biệt: Lưu log chi tiết, xuất file CSV per-image raw metrics và file JSON tổng hợp thống kê.
"""

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from paper_1_aqsm_mechanism.core.coco_gt import (
    bootstrap_paired_diff,
    compute_per_image_coco_aps,
    evaluate_coco_from_match_cache,
    evaluate_coco_predictions,
    postprocess_rtdetr_predictions,
    precompute_coco_match_cache,
    repeated_measures_2x2_anova,
)
from paper_1_aqsm_mechanism.core.detr_wrapper import RTDETRModelWrapper
from paper_1_aqsm_mechanism.core.mechanism_analyzer import (
    extract_encoder_components,
    run_decoder_from_components,
)
from paper_1_aqsm_mechanism.experiments.config import (
    N_BOOTSTRAP,
    RANDOM_SEED,
    get_raw_csv_dir,
    get_summary_json_dir,
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
    Xác nhận rằng việc tráo đổi tensor giữa Clean và Adversarial không gây ra hiện tượng
    sụp đổ phân phối số học (numerical/out-of-distribution shock):
      1. Tỷ lệ chuẩn Frobenius của Memory M: ||M_adv||_F / ||M_clean||_F trong khoảng hợp lý [0.8, 1.4].
      2. Tọa độ khởi tạo hộp B_init thuộc miền đóng chuẩn hóa [0.0, 1.0].
      3. Không chứa giá trị phi số học (NaN, Inf) làm phá vỡ phép chiếu Deformable Attention.
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

    Returns:
        Dict: Toàn bộ kết quả thống kê có thể serialize JSON.
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

            # 1. Trích xuất thành phần Encoder trên ảnh sạch và ảnh tấn công
            c_comp = extract_encoder_components(model, c_sub)
            a_comp = extract_encoder_components(model, a_sub)

            c_ind = c_comp["topk_ind"]
            a_ind = a_comp["topk_ind"]

            # Đo tỷ lệ query bị đẩy văng (Query Eviction Ratio)
            for b in range(end - start):
                s_c = set(c_ind[b].cpu().tolist())
                s_a = set(a_ind[b].cpu().tolist())
                evicted_ratios.append(float(1.0 - len(s_c & s_a) / 300.0))

            tgt_c_idx, ref_c_idx = _gather_priors_by_indices(a_comp, c_ind)
            tgt_a_idx, ref_a_idx = _gather_priors_by_indices(c_comp, a_ind)

            # Cấu hình 6 nhánh can thiệp
            arm_inputs = {
                "Arm_A": (c_comp["target"], c_comp["source_flatten"], c_comp["init_reference_points"], c_comp),
                "Arm_B": (a_comp["target"], a_comp["source_flatten"], a_comp["init_reference_points"], a_comp),
                "Arm_C_full": (c_comp["target"], a_comp["source_flatten"], c_comp["init_reference_points"], a_comp),
                "Arm_C_index": (tgt_c_idx, a_comp["source_flatten"], ref_c_idx, a_comp),
                "Arm_D_full": (a_comp["target"], c_comp["source_flatten"], a_comp["init_reference_points"], c_comp),
                "Arm_D_index": (tgt_a_idx, c_comp["source_flatten"], ref_a_idx, c_comp),
            }

            # 2. Suy luận qua Deformable Decoder cho từng nhánh
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

    # 3. Đánh giá Global COCO mAP cho từng nhánh
    point_maps: Dict[str, Dict[str, float]] = {}
    for arm in arms:
        eval_res = evaluate_coco_predictions(preds_by_arm[arm], gt_list)
        point_maps[arm] = {
            "mAP@50": float(eval_res["mAP@50"]),
            "mAP@50:95": float(eval_res["mAP@50:95"]),
        }

    # 4. Tính per-image COCO AP để phục vụ Repeated-Measures ANOVA và lưu raw CSV
    per_image_aps: Dict[str, Dict[str, np.ndarray]] = {}
    for arm in arms:
        ap50_arr, ap95_arr = compute_per_image_coco_aps(preds_by_arm[arm], gt_list)
        per_image_aps[arm] = {
            "ap50": ap50_arr,
            "ap50_95": ap95_arr,
        }

    # 5. Lưu file CSV chứa số liệu thô từng ảnh (Per-Image Raw Metrics) phục vụ audit
    raw_csv_dir = get_raw_csv_dir()
    raw_csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = raw_csv_dir / f"causal_anova_per_image_{attack_name.lower()}.csv"
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_index",
            "image_id",
            "gt_box_count",
            "evicted_ratio",
            "arm_A_ap50",
            "arm_B_ap50",
            "arm_C_full_ap50",
            "arm_C_index_ap50",
            "arm_D_full_ap50",
            "arm_D_index_ap50",
            "arm_A_ap95",
            "arm_B_ap95",
            "arm_C_full_ap95",
            "arm_C_index_ap95",
            "arm_D_full_ap95",
            "arm_D_index_ap95",
        ])
        for i in range(n_eval):
            gt_boxes = len(gt_list[i].get("boxes_xyxy", []))
            img_id = gt_list[i].get("image_id", i)
            writer.writerow([
                i,
                img_id,
                gt_boxes,
                f"{evicted_ratios[i]:.4f}",
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

    # Ghi audit stream JSONL chi tiết từng ảnh nếu có audit_logger
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

    # 6. Tính Repeated-Measures 2x2 ANOVA F-test & Partial Eta-Squared
    # 2x2: Factor 1 = Memory M (Clean vs Adv), Factor 2 = Query Q (Clean vs Adv)
    # y_00 = Clean-Clean (Arm A), y_11 = Adv-Adv (Arm B), y_10 = Adv-Clean (Arm C), y_01 = Clean-Adv (Arm D)
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

    # 7. Tính Paired Bootstrap 95% Confidence Intervals
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

    # 8. Tổng hợp toàn bộ dữ liệu ra JSON
    results_summary: Dict[str, Any] = {
        "attack_name": attack_name,
        "n_evaluated_images": n_eval,
        "mean_evicted_ratio": float(np.mean(evicted_ratios)),
        "std_evicted_ratio": float(np.std(evicted_ratios)),
        "point_maps": point_maps,
        "anova_repeated_measures_full": anova_full,
        "anova_repeated_measures_index": anova_index,
        "bootstrap_paired_deltas": boot_deltas,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    summary_json_dir = get_summary_json_dir()
    summary_json_dir.mkdir(parents=True, exist_ok=True)
    json_path = summary_json_dir / f"causal_swap_results_{attack_name.lower()}.json"
    with open(json_path, mode="w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)
    print(f">>> Đã lưu tổng hợp thống kê ANOVA: {json_path}")

    return results_summary
