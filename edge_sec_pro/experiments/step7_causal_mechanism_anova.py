# -*- coding: utf-8 -*-
"""
Bước 7 (Nâng cấp N=300, COCO GT thật, AQSM & PGD Vanishing):
Thí nghiệm can thiệp nhân quả 2x2 (Encoder Memory M vs. Query Prior Q).

Làm rõ các tensor được hoán đổi tại ranh giới Encoder -> Deformable Decoder:
  - M (Encoder Memory): `source_flatten` shape [B, 2100, 256] (đặc trưng đa tỷ lệ S3/S4/S5 sau `decoder_input_proj`, truyền vào `encoder_hidden_states` của decoder cross-attention).
  - Q (Query Prior = {target, init_reference_points}):
      + `target` shape [B, 300, 256]: nội dung truy vấn trích từ `output_memory` tại 300 chỉ số `topk_ind`.
      + `init_reference_points` shape [B, 300, 4]: tọa độ tham chiếu [cx, cy, w, h] chưa sigmoid trích từ `enc_outputs_coord_logits` tại `topk_ind`.
  - 6 nhánh thí nghiệm:
      + Arm A (Clean Baseline):        M_clean + Q_clean(I_clean)
      + Arm B (Pure Attack):           M_adv   + Q_adv(I_adv)
      + Arm C_full (Adv M + Clean Q):  M_adv   + Q_clean(I_clean)
      + Arm C_index (Adv M + Clean I): M_adv   + target_adv(I_clean), ref_adv(I_clean)
      + Arm D_full (Clean M + Adv Q):  M_clean + Q_adv(I_adv)
      + Arm D_index (Clean M + Adv I): M_clean + target_clean(I_adv), ref_clean(I_adv)
"""

import argparse
import functools
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from tqdm import tqdm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

print = functools.partial(print, flush=True)

WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from edge_sec_pro.core.coco_gt import (
    bootstrap_paired_diff,
    compute_per_image_coco_aps,
    evaluate_coco_from_match_cache,
    evaluate_coco_predictions,
    load_coco_val2017_gt_cohort,
    postprocess_rtdetr_predictions,
    precompute_coco_match_cache,
    repeated_measures_2x2_anova,
)
from edge_sec_pro.core.config import (
    AQSM_LAMBDA_RANK,
    AQSM_RANKING_MARGIN,
    CHECKPOINTS_DIR,
    PGD_NUM_STEPS,
    PGD_STEP_SIZE,
    RESULTS_DIR,
)
from edge_sec_pro.core.metrics import make_serializable
from edge_sec_pro.experiments.step_generate_adv_cache import (
    cache_path,
    generate_single_cache,
    load_cached_adv_images,
)
from edge_sec_pro.transformer_monitors.detr_wrapper import RTDETRModelWrapper
from edge_sec_pro.transformer_monitors.rtdetr_mechanism_analyzer import (
    extract_encoder_components,
    run_decoder_from_components,
)


def _gather_priors_by_indices(comp: Dict[str, Any], ind: torch.Tensor):
    ref_dim = comp["enc_outputs_coord_logits"].shape[-1]
    mem_dim = comp["output_memory"].shape[-1]
    ref_pts = comp["enc_outputs_coord_logits"].gather(
        dim=1, index=ind.unsqueeze(-1).expand(-1, -1, ref_dim)
    ).detach()
    tgt = comp["output_memory"].gather(
        dim=1, index=ind.unsqueeze(-1).expand(-1, -1, mem_dim)
    ).detach()
    return tgt, ref_pts


def _bootstrap_global_coco_deltas(
    preds_by_arm: Dict[str, List[Dict[str, np.ndarray]]],
    gt_list: List[Dict[str, Any]],
    n_boot: int = 500,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Tính paired bootstrap 95% CI cho chênh lệch global COCO mAP@50 và mAP@50:95
    giữa các nhánh can thiệp trọng yếu.
    """
    n_img = len(gt_list)
    pairs = [
        ("Arm_B_minus_Arm_A", "Arm_B", "Arm_A"),
        ("Arm_C_full_minus_Arm_B", "Arm_C_full", "Arm_B"),
        ("Arm_C_index_minus_Arm_B", "Arm_C_index", "Arm_B"),
        ("Arm_D_full_minus_Arm_A", "Arm_D_full", "Arm_A"),
        ("Arm_D_index_minus_Arm_A", "Arm_D_index", "Arm_A"),
        ("Arm_D_full_minus_Arm_B", "Arm_D_full", "Arm_B"),
        ("Arm_D_full_minus_Arm_C_full", "Arm_D_full", "Arm_C_full"),
    ]
    point_maps = {
        arm: evaluate_coco_predictions(preds_by_arm[arm], gt_list)
        for arm in preds_by_arm
    }

    rng = np.random.RandomState(seed)
    boot_deltas: Dict[str, Dict[str, List[float]]] = {
        p_name: {"d50": [], "d50_95": []} for p_name, _, _ in pairs
    }

    needed_arms = sorted(set([a for _, a, _ in pairs] + [b for _, _, b in pairs]))
    arm_caches = {
        arm: precompute_coco_match_cache(preds_by_arm[arm], gt_list)
        for arm in needed_arms
    }
    for _ in tqdm(range(n_boot), desc="Bootstrap Global mAP", leave=False):
        idx = rng.randint(0, n_img, size=n_img).tolist()
        boot_m = {arm: evaluate_coco_from_match_cache(arm_caches[arm], indices=idx) for arm in needed_arms}
        for p_name, arm1, arm2 in pairs:
            boot_deltas[p_name]["d50"].append(boot_m[arm1]["mAP@50"] - boot_m[arm2]["mAP@50"])
            boot_deltas[p_name]["d50_95"].append(boot_m[arm1]["mAP@50:95"] - boot_m[arm2]["mAP@50:95"])

    summary: Dict[str, Any] = {}
    for p_name, arm1, arm2 in pairs:
        d50_pt = point_maps[arm1]["mAP@50"] - point_maps[arm2]["mAP@50"]
        d95_pt = point_maps[arm1]["mAP@50:95"] - point_maps[arm2]["mAP@50:95"]
        ci50 = np.percentile(boot_deltas[p_name]["d50"], [2.5, 97.5]).tolist()
        ci95 = np.percentile(boot_deltas[p_name]["d50_95"], [2.5, 97.5]).tolist()
        summary[p_name] = {
            "delta_mAP@50": float(d50_pt),
            "delta_mAP@50_ci95": [float(ci50[0]), float(ci50[1])],
            "delta_mAP@50:95": float(d95_pt),
            "delta_mAP@50:95_ci95": [float(ci95[0]), float(ci95[1])],
        }
    return summary


def run_causal_intervention_for_attack(
    attack_name: str,
    wrapper: RTDETRModelWrapper,
    clean_imgs: torch.Tensor,
    adv_imgs: torch.Tensor,
    gt_list: List[Dict[str, Any]],
    attack_meta: Dict[str, Any],
    batch_size: int = 10,
    n_boot_global: int = 500,
) -> Dict[str, Any]:
    """
    Thực thi đủ 6 nhánh can thiệp trên cùng tập N ảnh cho 1 đòn tấn công.
    """
    model = wrapper.model
    n_eval = len(gt_list)
    arms = ["Arm_A", "Arm_B", "Arm_C_full", "Arm_C_index", "Arm_D_full", "Arm_D_index"]
    preds_hf: Dict[str, List[Dict[str, np.ndarray]]] = {a: [] for a in arms}
    preds_qmax: Dict[str, List[Dict[str, np.ndarray]]] = {a: [] for a in arms}
    evicted_ratios: List[float] = []

    print(f"\n>>> [Causal Surgery] Đang chạy 6 nhánh trên N={n_eval} ảnh cho {attack_name.upper()}...")
    t0 = time.time()

    with torch.no_grad():
        for start in tqdm(range(0, n_eval, batch_size), desc=f"Causal {attack_name}"):
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
                evicted_ratios.append(1.0 - len(s_c & s_a) / 300.0)

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

            for arm_name, (tgt, mem, ref, shape_src) in arm_inputs.items():
                dec_out = run_decoder_from_components(
                    model=model,
                    target=tgt,
                    source_flatten=mem,
                    init_reference_points=ref,
                    spatial_shapes=shape_src["spatial_shapes"],
                    spatial_shapes_list=shape_src["spatial_shapes_list"],
                    level_start_index=shape_src["level_start_index"],
                    output_attentions=False,
                )
                preds_hf[arm_name].extend(
                    postprocess_rtdetr_predictions(
                        dec_out["logits"], dec_out["pred_boxes"], mode="hf_multilabel", max_dets=100
                    )
                )
                preds_qmax[arm_name].extend(
                    postprocess_rtdetr_predictions(
                        dec_out["logits"], dec_out["pred_boxes"], mode="per_query_max", max_dets=100
                    )
                )

            del c_comp, a_comp

    # 1. Global COCO mAP@50 & mAP@50:95
    global_coco: Dict[str, Any] = {}
    global_qmax: Dict[str, Any] = {}
    for arm_name in arms:
        global_coco[arm_name] = evaluate_coco_predictions(preds_hf[arm_name], gt_list)
        global_qmax[arm_name] = evaluate_coco_predictions(preds_qmax[arm_name], gt_list)

    base50 = max(global_coco["Arm_A"]["mAP@50"], 1e-6)
    base95 = max(global_coco["Arm_A"]["mAP@50:95"], 1e-6)
    for arm_name in arms:
        global_coco[arm_name]["relative_retention_50_pct"] = float(global_coco[arm_name]["mAP@50"] / base50 * 100.0)
        global_coco[arm_name]["relative_drop_50_pct"] = float((global_coco[arm_name]["mAP@50"] - base50) / base50 * 100.0)
        global_coco[arm_name]["relative_retention_50_95_pct"] = float(global_coco[arm_name]["mAP@50:95"] / base95 * 100.0)

    # Paired bootstrap 95% CI cho global COCO mAP deltas
    global_deltas_ci = _bootstrap_global_coco_deltas(
        preds_hf, gt_list, n_boot=n_boot_global, seed=42
    )

    # 2. Per-image AP@50 & AP@50:95 cho Repeated-Measures 2x2 ANOVA và Paired CI
    per_img_50: Dict[str, np.ndarray] = {}
    per_img_95: Dict[str, np.ndarray] = {}
    per_img_summary: Dict[str, Any] = {}
    for arm_name in arms:
        ap50, ap95 = compute_per_image_coco_aps(preds_hf[arm_name], gt_list)
        per_img_50[arm_name] = ap50
        per_img_95[arm_name] = ap95
        valid50 = ap50[~np.isnan(ap50)]
        valid95 = ap95[~np.isnan(ap95)]
        per_img_summary[arm_name] = {
            "n_valid_gt_images": int(len(valid50)),
            "mean_ap50": float(np.mean(valid50)),
            "std_ap50": float(np.std(valid50, ddof=1)),
            "mean_ap50_95": float(np.mean(valid95)),
            "std_ap50_95": float(np.std(valid95, ddof=1)),
        }

    # Chênh lệch theo từng ảnh (absolute AP% & relative % so với Arm A)
    per_image_deltas = {
        "Arm_B_minus_Arm_A": bootstrap_paired_diff(per_img_50["Arm_B"], per_img_50["Arm_A"]),
        "Arm_C_full_minus_Arm_B": bootstrap_paired_diff(per_img_50["Arm_C_full"], per_img_50["Arm_B"]),
        "Arm_C_index_minus_Arm_B": bootstrap_paired_diff(per_img_50["Arm_C_index"], per_img_50["Arm_B"]),
        "Arm_D_full_minus_Arm_A": bootstrap_paired_diff(per_img_50["Arm_D_full"], per_img_50["Arm_A"]),
        "Arm_D_index_minus_Arm_A": bootstrap_paired_diff(per_img_50["Arm_D_index"], per_img_50["Arm_A"]),
        "Arm_D_full_minus_Arm_B": bootstrap_paired_diff(per_img_50["Arm_D_full"], per_img_50["Arm_B"]),
    }

    anova_abs_ap50 = repeated_measures_2x2_anova(
        y_00=per_img_50["Arm_A"],
        y_11=per_img_50["Arm_B"],
        y_10=per_img_50["Arm_C_full"],
        y_01=per_img_50["Arm_D_full"],
    )
    anova_abs_ap50_95 = repeated_measures_2x2_anova(
        y_00=per_img_95["Arm_A"],
        y_11=per_img_95["Arm_B"],
        y_10=per_img_95["Arm_C_full"],
        y_01=per_img_95["Arm_D_full"],
    )

    elapsed = time.time() - t0
    print(f"  [OK] {attack_name.upper()} hoàn tất trong {elapsed:.1f}s (N_img={n_eval}, N_anova={anova_abs_ap50['N']}):")
    for a_name in arms:
        gc = global_coco[a_name]
        pi = per_img_summary[a_name]
        print(
            f"    - {a_name:<12}: Global mAP@50={gc['mAP@50']:.2f}% (50:95={gc['mAP@50:95']:.2f}%, "
            f"Ret={gc['relative_retention_50_pct']:.1f}%) | Per-Img Mean AP@50={pi['mean_ap50']:.2f}%"
        )
    me = anova_abs_ap50["marginal_effects"]
    sh_ss = anova_abs_ap50["attribution_shares"]["anova_condition_ss_share_pct"]
    sh_l1 = anova_abs_ap50["attribution_shares"]["l1_marginal_share_pct"]
    eta_p = anova_abs_ap50["attribution_shares"]["partial_eta_squared"]
    print(
        f"    -> 2x2 ANOVA (Per-Img AP@50): ME_M={me['ME_M']:+.2f} CI{me['ME_M_ci95']} | "
        f"ME_Q={me['ME_Q']:+.2f} CI{me['ME_Q_ci95']} | Inter={me['Interaction_MxQ']:+.2f} (p={anova_abs_ap50['anova_table']['Interaction_MxQ']['p']:.4f})"
    )
    print(
        f"    -> Tỷ lệ đóng góp: L1(M/Q/Int)=({sh_l1['Memory_M']:.1f}% / {sh_l1['Query_Q']:.1f}% / {sh_l1['Interaction_MxQ']:.1f}%) | "
        f"SS-Share=({sh_ss['Memory_M']:.1f}% / {sh_ss['Query_Q']:.1f}% / {sh_ss['Interaction_MxQ']:.1f}%) | "
        f"Partial eta^2(M={eta_p['Memory_M']:.3f}, Q={eta_p['Query_Q']:.3f})"
    )

    return {
        "attack": attack_name,
        "attack_meta": attack_meta,
        "n_eval": int(n_eval),
        "elapsed_sec": float(elapsed),
        "evicted_top300_ratio_mean": float(np.mean(evicted_ratios)),
        "evicted_top300_ratio_std": float(np.std(evicted_ratios, ddof=1)),
        "global_coco_map": global_coco,
        "global_coco_map_qmax": global_qmax,
        "global_coco_deltas_bootstrap_95ci": global_deltas_ci,
        "per_image_ap_summary": per_img_summary,
        "per_image_ap50_deltas": per_image_deltas,
        "anova_per_image_ap50": anova_abs_ap50,
        "anova_per_image_ap50_95": anova_abs_ap50_95,
    }


def run_step7(
    n_eval: int = 300,
    eps_num: int = 16,
    attacks: Sequence[str] = ("aqsm", "vanishing", "taskpgd"),
    batch_size: int = 10,
    threads: int = 4,
    n_boot_global: int = 500,
) -> Dict[str, Any]:
    print("=" * 90)
    print(f"BƯỚC 7: THÍ NGHIỆM CAN THIỆP NHÂN QUẢ 2x2 TRÊN COCO GT THẬT (N={n_eval}, eps={eps_num}/255)")
    print("=" * 90)

    torch.set_num_threads(threads)
    device = torch.device("cpu")
    wrapper = RTDETRModelWrapper(pretrained_model_name="PekingU/rtdetr_r18vd", device=device)
    wrapper.model.eval()

    cohort = load_coco_val2017_gt_cohort(
        n_cal=1200,
        n_test=600,
        eval_start=0,
        n_eval=n_eval,
        seed=42,
        id2label=wrapper.model.config.id2label,
        load_tensors=True,
    )
    clean_imgs = cohort["tensors"]
    gt_list = cohort["gt_list"]
    image_ids = cohort["image_ids"]

    tensor_spec = {
        "M_Encoder_Memory": "source_flatten [B, 2100, 256] (đầu ra decoder_input_proj trên S3/S4/S5, cấp cho encoder_hidden_states của Deformable Decoder)",
        "Q_Query_Prior": "{target [B, 300, 256], init_reference_points [B, 300, 4]} trích từ output_memory & enc_outputs_coord_logits tại topk_ind",
        "I_Selection_Indices": "topk_ind [B, 300] lấy top-300 theo max_c sigmoid(enc_outputs_class)",
        "arms_definition": {
            "Arm_A": "M_clean + target_clean[I_clean] + ref_clean[I_clean]",
            "Arm_B": "M_adv + target_adv[I_adv] + ref_adv[I_adv]",
            "Arm_C_full": "M_adv + target_clean[I_clean] + ref_clean[I_clean]",
            "Arm_C_index": "M_adv + target_adv[I_clean] + ref_adv[I_clean]",
            "Arm_D_full": "M_clean + target_adv[I_adv] + ref_adv[I_adv]",
            "Arm_D_index": "M_clean + target_clean[I_adv] + ref_clean[I_adv]",
        },
    }

    per_attack_results: Dict[str, Any] = {}
    for atk in attacks:
        if not cache_path(atk, eps_num, n_eval).exists():
            print(f"[Cache Miss] Sinh cache cho {atk} (eps={eps_num}/255, N={n_eval})...")
            generate_single_cache(
                attack=atk,
                eps_num=eps_num,
                n_eval=n_eval,
                batch_size=batch_size,
                steps=PGD_NUM_STEPS,
                alpha=PGD_STEP_SIZE,
                threads=threads,
            )
        adv_imgs, atk_meta = load_cached_adv_images(
            attack=atk,
            eps_num=eps_num,
            n_eval=n_eval,
            clean_imgs=clean_imgs,
            expected_image_ids=image_ids,
        )
        res_atk = run_causal_intervention_for_attack(
            attack_name=atk,
            wrapper=wrapper,
            clean_imgs=clean_imgs,
            adv_imgs=adv_imgs,
            gt_list=gt_list,
            attack_meta=atk_meta,
            batch_size=batch_size,
            n_boot_global=n_boot_global,
        )
        per_attack_results[atk] = res_atk

    payload = {
        "protocol": f"Repeated-Measures 2x2 Causal Surgery on Official COCO val2017 GT (N={n_eval}, eps={eps_num}/255, T={PGD_NUM_STEPS})",
        "n_eval": int(n_eval),
        "eps_num_255": int(eps_num),
        "tensor_spec": tensor_spec,
        "attacks": per_attack_results,
    }

    out_json = RESULTS_DIR / "rtdetr_factorial_anova_standardized.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(make_serializable(payload), f, indent=2)
    print(f"\n[OK] Đã xuất kết quả Causal Surgery N={n_eval}: {out_json}")

    ckpt_json = CHECKPOINTS_DIR / "res_step7_causal_query_swap.json"
    with open(ckpt_json, "w", encoding="utf-8") as f:
        json.dump(make_serializable(payload), f, indent=2)

    return payload


def main():
    parser = argparse.ArgumentParser(description="Step 7: 2x2 Causal Mechanism Isolation on COCO GT (N=300)")
    parser.add_argument("--n-eval", type=int, default=300)
    parser.add_argument("--eps-num", type=int, default=16)
    parser.add_argument("--attacks", nargs="+", default=["aqsm", "vanishing", "taskpgd"])
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--n-boot-global", type=int, default=500)
    parser.add_argument("--run-standardized-c1", action="store_true", help="Run the standardized Factorial Intervention 2x2 (C1) with pycocotools.COCOeval and repeated-measures ANOVA")
    args = parser.parse_args()

    if args.run_standardized_c1:
        c1_script_dir = WORKSPACE_DIR / "c1_c2_standardized_evaluation" / "src"
        if str(c1_script_dir) not in sys.path:
            sys.path.insert(0, str(c1_script_dir))
        from run_c1_factorial_intervention import main as run_c1_main
        run_c1_main()
    else:
        run_step7(
            n_eval=args.n_eval,
            eps_num=args.eps_num,
            attacks=args.attacks,
            batch_size=args.batch_size,
            threads=args.threads,
            n_boot_global=args.n_boot_global,
        )


if __name__ == "__main__":
    main()
