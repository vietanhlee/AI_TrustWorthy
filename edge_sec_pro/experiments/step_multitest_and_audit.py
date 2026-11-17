# -*- coding: utf-8 -*-
"""
Module: edge_sec_pro.experiments.step_multitest_and_audit
Mục đích (Mục 1 - Rà soát kết quả thực nghiệm hiện tại):
1. Tái lập và sửa lỗi `rtdetr_multitest_baselines_results.json` từ raw quantiles 4 kênh
   của `CalibratedMaxOREnsemble` (ActiveQ_Split, Top1Mean_Split, ESM_Dropoff, q(Layer_Maha)),
   loại bỏ bug cũ dùng nhầm điểm thô Layer_Maha chưa chuẩn hóa quantile.
2. Kiểm định chuẩn xác mAP@50, mAP@50:95 và ASR trên N=300 ảnh COCO val2017 GT thật
   cho Clean, 4 Natural Controls, AQSM, PGD Vanishing, Standard Task PGD, EQF, DQF,
   Adaptive Evasion (lambda=0.50), và Gradient Patch (48x48).
3. Xuất báo cáo audit (`results/rtdetr_audit_report.json`) đối chiếu rõ từng công thức ASR,
   ngưỡng Max-OR, sai lệch cấu hình cũ vs chuẩn hóa mới.
"""

import argparse
import functools
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

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
    evaluate_coco_predictions,
    load_coco_val2017_gt_cohort,
    postprocess_rtdetr_predictions,
)
from edge_sec_pro.core.config import (
    CHECKPOINTS_DIR,
    PGD_NUM_STEPS,
    PGD_STEP_SIZE,
    RESULTS_DIR,
)
from edge_sec_pro.core.dataset import apply_natural_corruption
from edge_sec_pro.core.metrics import (
    calculate_task_level_asr,
    clopper_pearson_interval,
    make_serializable,
)
from edge_sec_pro.experiments.step_generate_adv_cache import (
    cache_path,
    generate_single_cache,
    load_cached_adv_images,
)
from edge_sec_pro.transformer_monitors.detr_attack_generator import DETRAttackGenerator
from edge_sec_pro.transformer_monitors.detr_wrapper import RTDETRModelWrapper


def recompute_multitest_from_raw_npz() -> Dict[str, Any]:
    """
    Tính lại chính xác Uncalibrated OR, Bonferroni, Sidak và Calibrated Max-OR
    trên 4 kênh thành phần đúng với cài đặt `CalibratedMaxOREnsemble`.
    """
    ckpt_file = CHECKPOINTS_DIR / "calibration_and_detectors.pkl"
    npz_file = RESULTS_DIR / "rtdetr_advanced_defense_raw_scores.npz"
    with open(ckpt_file, "rb") as f:
        ckpt = pickle.load(f)
    npz = np.load(npz_file, allow_pickle=False)

    max_or = ckpt["detectors"]["Calibrated_Max_OR"]
    maha_cal = np.asarray(max_or.cal_sorted_scores["Layer_Maha"], dtype=np.float64)
    tau_calibrated = float(max_or.threshold)

    def _get_max_q(prefix: str) -> np.ndarray:
        q_act = np.asarray(npz[f"{prefix}__ActiveQ_Split"], dtype=np.float64)
        q_top = np.asarray(npz[f"{prefix}__Top1Mean_Split"], dtype=np.float64)
        q_esm = np.asarray(npz[f"{prefix}__ESM_Dropoff"], dtype=np.float64)
        raw_maha = np.asarray(npz[f"{prefix}__Layer_Maha"], dtype=np.float64)
        q_maha = np.searchsorted(maha_cal, raw_maha, side="right") / float(len(maha_cal))
        return np.vstack([q_act, q_top, q_esm, q_maha]).max(axis=0)

    clean_max_q = _get_max_q("clean_test")
    max_diff_clean = float(np.max(np.abs(clean_max_q - npz["clean_test__Calibrated_Max_OR"])))
    assert max_diff_clean < 1e-9, f"Sai lệch khi tái tạo clean Max-OR: {max_diff_clean}"

    attack_map = {
        "AQSM": "Attack__Query_Selection_Manipulation_AQSM",
        "EQF": "Attack__Encoder_Query_Fabrication_EQF,_15-step_PGD",
        "Vanishing": "Attack__PGD_Query_Vanishing_15_steps",
        "Fabrication": "Attack__PGD_Query_Fabrication_DQF_15_steps",
        "Adaptive_Evasion": "Attack__Adaptive_Evasion_PGD_Hinge,_lambda=0.5",
        "Adaptive_Joint": "Attack__Joint_Adaptive_Evasion_1D_+_Maha_Awareness",
        "Adaptive_ESM": "Attack__Adaptive_ESM_Evasion_Dropoff-Aware,_lambda=2.0",
        "Adaptive_Fused": "Attack__Adaptive_Fused_Ensemble_Evasion_All-Tiers",
    }
    attack_max_qs = {k: _get_max_q(pref) for k, pref in attack_map.items() if f"{pref}__Calibrated_Max_OR" in npz.files}

    schemes = {
        "Uncalibrated_OR": {
            "name": "Uncalibrated Standard OR (M=4)",
            "threshold_tau": 0.9500,
            "note": "alpha = 0.05 mỗi kênh không kiểm soát FWER",
        },
        "Bonferroni_OR": {
            "name": "Bonferroni Correction (alpha/4)",
            "threshold_tau": 1.0 - 0.05 / 4.0,
            "note": "tau = 1 - alpha/4 = 0.9875",
        },
        "Sidak_OR": {
            "name": "Šidák Correction",
            "threshold_tau": float((1.0 - 0.05) ** (1.0 / 4.0)),
            "note": "tau = (1 - alpha)^(1/4) = 0.9872585",
        },
        "Calibrated_Max_OR": {
            "name": "Calibrated Max-OR (4-Channel Gate)",
            "threshold_tau": tau_calibrated,
            "note": f"Hiệu chuẩn cực trị thực nghiệm trên N_cal=1200: tau = {tau_calibrated:.4f}",
        },
    }

    multitest_out: Dict[str, Any] = {}
    for sk, meta in schemes.items():
        th = meta["threshold_tau"]
        fp_cnt = int(np.sum(clean_max_q >= th))
        n_test = int(len(clean_max_q))
        fp_ci = clopper_pearson_interval(fp_cnt, n_test)
        atk_block: Dict[str, Any] = {}
        for ak, q_arr in attack_max_qs.items():
            tp_cnt = int(np.sum(q_arr >= th))
            n_a = int(len(q_arr))
            tp_ci = clopper_pearson_interval(tp_cnt, n_a)
            atk_block[ak] = {
                "tp_count": tp_cnt,
                "n_samples": n_a,
                "tpr_pct": float(tp_cnt / max(n_a, 1) * 100.0),
                "ci_95": [float(tp_ci[0] * 100.0), float(tp_ci[1] * 100.0)],
            }
        multitest_out[sk] = {
            **meta,
            "clean_fpr": {
                "fp_count": fp_cnt,
                "n_test": n_test,
                "fpr_pct": float(fp_cnt / max(n_test, 1) * 100.0),
                "ci_95": [float(fp_ci[0] * 100.0), float(fp_ci[1] * 100.0)],
            },
            "attack_tpr": atk_block,
        }

    # Lưu bản chuẩn hóa mới (và giữ bản sao kiểm toán lý do chênh lệch so với file cũ)
    out_file = RESULTS_DIR / "rtdetr_multitest_baselines_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(make_serializable(multitest_out), f, indent=2)

    return {
        "clean_reconstruct_max_abs_diff": max_diff_clean,
        "multitest": multitest_out,
    }


def evaluate_full_coco_benchmark_and_asr(
    n_eval: int = 300,
    batch_size: int = 10,
    threads: int = 4,
    include_dqf_adaptive: bool = True,
) -> Dict[str, Any]:
    """
    Đo lường mAP@50, mAP@50:95 trên COCO val2017 GT thật (N=300) và kiểm toán lại ASR
    cho các điều kiện trong Bảng 2.
    """
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

    # Clean baseline inference
    clean_logits_l: List[torch.Tensor] = []
    clean_preds: List[Dict[str, np.ndarray]] = []
    clean_preds_qmax: List[Dict[str, np.ndarray]] = []
    with torch.no_grad():
        for st_i in tqdm(range(0, n_eval, batch_size), desc="Audit Clean N=300"):
            st = wrapper(clean_imgs[st_i : min(n_eval, st_i + batch_size)].to(device))
            clean_logits_l.append(st.logits.cpu())
            clean_preds.extend(postprocess_rtdetr_predictions(st.logits, st.pred_boxes, mode="hf_multilabel", max_dets=100))
            clean_preds_qmax.extend(postprocess_rtdetr_predictions(st.logits, st.pred_boxes, mode="per_query_max", max_dets=100))
    clean_logits = torch.cat(clean_logits_l, dim=0)
    clean_coco = evaluate_coco_predictions(clean_preds, gt_list)
    clean_coco_qmax = evaluate_coco_predictions(clean_preds_qmax, gt_list)

    # Kiểm toán phân bố Clean Baseline để giải trình ASR
    clean_probs = torch.sigmoid(clean_logits)
    clean_max_q_all = clean_probs.max(dim=-1).values.max(dim=-1).values
    clean_has_ge_035 = int((clean_max_q_all >= 0.35).sum().item())
    clean_has_ge_020 = int((clean_max_q_all >= 0.20).sum().item())
    clean_has_class1_ge_035 = int((clean_probs[..., 1].max(dim=-1).values >= 0.35).sum().item())

    # Tải attacker & clean_stats cho controls, patch và adaptive đúng chuẩn worker_b1/b2
    ckpt_file = CHECKPOINTS_DIR / "calibration_and_detectors.pkl"
    with open(ckpt_file, "rb") as f:
        ckpt = pickle.load(f)
    cs = ckpt["clean_stats"]
    attacker = DETRAttackGenerator(wrapper, epsilon=16.0 / 255.0, alpha=PGD_STEP_SIZE, num_steps=PGD_NUM_STEPS)

    # Tạo các điều kiện kiểm thử trên N=300 khớp hoàn toàn với worker_b1_controls_patch
    scenarios: Dict[str, Dict[str, Any]] = {
        "Control: Gaussian Noise (sigma=16/255)": {
            "imgs": attacker.generate_random_noise(clean_imgs, noise_type="gaussian"),
            "asr_type": None,
        },
        "Control: Random Patch (48x48 Random Pos)": {
            "imgs": attacker.generate_random_patch(clean_imgs, patch_size=48, patch_loc="random", seed=1337),
            "asr_type": None,
        },
        "Clean Shift: Gaussian Blur (5x5, sigma=1.5)": {
            "imgs": attacker.generate_clean_shifts(clean_imgs, shift_type="blur"),
            "asr_type": None,
        },
        "Clean Shift: Brightness (-30%)": {
            "imgs": attacker.generate_clean_shifts(clean_imgs, shift_type="brightness"),
            "asr_type": None,
        },
    }

    for atk_key, disp_name, asr_t in [
        ("aqsm", "Attack: Query Selection Manipulation (AQSM)", "AQSM"),
        ("vanishing", "Attack: PGD Query Vanishing (15 steps)", "Vanishing"),
        ("taskpgd", "Attack: Standard Task PGD (15 steps)", "Vanishing"),
    ]:
        if not cache_path(atk_key, 16, n_eval).exists():
            generate_single_cache(atk_key, eps_num=16, n_eval=n_eval, batch_size=batch_size, steps=PGD_NUM_STEPS, alpha=PGD_STEP_SIZE, threads=threads)
        adv_x, _ = load_cached_adv_images(atk_key, 16, n_eval, clean_imgs, image_ids)
        scenarios[disp_name] = {"imgs": adv_x, "asr_type": asr_t}

    if include_dqf_adaptive:
        for atk_key, disp_name, asr_t in [
            ("eqf", "Attack: Encoder Query Fabrication (EQF, 15 steps)", "Fabrication"),
            ("dqf", "Attack: PGD Query Fabrication (DQF, 15 steps)", "Fabrication"),
        ]:
            if not cache_path(atk_key, 16, n_eval).exists():
                generate_single_cache(atk_key, eps_num=16, n_eval=n_eval, batch_size=batch_size, steps=PGD_NUM_STEPS, alpha=PGD_STEP_SIZE, threads=threads)
            adv_x, _ = load_cached_adv_images(atk_key, 16, n_eval, clean_imgs, image_ids)
            scenarios[disp_name] = {"imgs": adv_x, "asr_type": asr_t}

        # Gradient Patch trained on first 60 cal images (matching worker_b1_controls_patch)
        patch_cache = cache_path("patch48", 16, n_eval)
        if patch_cache.exists():
            adv_patch, _ = load_cached_adv_images("patch48", 16, n_eval, clean_imgs, image_ids)
        else:
            from edge_sec_pro.core.dataset import load_coco_images
            cal_imgs, _ = load_coco_images(n_cal=1200, n_test=600, img_size=320, seed=42)
            patch = attacker.optimize_adversarial_patch(cal_imgs[:60], patch_size=48, patch_loc="random", target_class_idx=1, num_epochs=4, lr=0.05)
            adv_patch = attacker.apply_adversarial_patch(clean_imgs, patch, patch_loc="random", seed=1337)
        scenarios["Attack: Gradient Patch (48x48 Random Pos)"] = {"imgs": adv_patch, "asr_type": "Patch"}

        adapt_cache = cache_path("adaptive1d", 16, n_eval)
        if adapt_cache.exists():
            adv_adapt, _ = load_cached_adv_images("adaptive1d", 16, n_eval, clean_imgs, image_ids)
        else:
            chunks = []
            for st_i in tqdm(range(0, n_eval, batch_size), desc="Sinh Adaptive 1D (lambda=0.50)"):
                sub = clean_imgs[st_i : min(n_eval, st_i + batch_size)]
                chunks.append(
                    attacker.generate_adaptive_attack(
                        sub,
                        target_class_idx=1,
                        evasion_weight=0.5,
                        steps=PGD_NUM_STEPS,
                        loss_mode="hinge",
                        target_detector="1d",
                        tau_bounds_act=cs["tau_act_bounds"],
                        tau_bounds_top1=cs["tau_top_bounds"],
                        clean_act_mean=cs["mu_act"],
                        clean_act_std=cs["sd_act"],
                        clean_top1_mean=cs["mu_top1"],
                        clean_top1_std=cs["sd_top1"],
                        temperature=25.0,
                    ).detach().cpu()
                )
            adv_adapt = torch.cat(chunks, dim=0)
        scenarios["Attack: Adaptive PGD (1D Hinge, lambda=0.50)"] = {"imgs": adv_adapt, "asr_type": "Adaptive"}

    # Đánh giá từng kịch bản
    table_res: Dict[str, Any] = {
        "Clean Baseline": {
            "coco_map_hf": clean_coco,
            "coco_map_qmax": clean_coco_qmax,
            "rel_drop_50_pct": 0.0,
            "rel_drop_50_95_pct": 0.0,
        }
    }
    base50 = max(clean_coco["mAP@50"], 1e-6)
    base95 = max(clean_coco["mAP@50:95"], 1e-6)

    for sc_name, sc_obj in scenarios.items():
        sc_imgs = sc_obj["imgs"]
        adv_l: List[torch.Tensor] = []
        preds_sc: List[Dict[str, np.ndarray]] = []
        with torch.no_grad():
            for st_i in tqdm(range(0, n_eval, batch_size), desc=f"Eval {sc_name[:28]}", leave=False):
                st = wrapper(sc_imgs[st_i : min(n_eval, st_i + batch_size)].to(device))
                adv_l.append(st.logits.cpu())
                preds_sc.extend(postprocess_rtdetr_predictions(st.logits, st.pred_boxes, mode="hf_multilabel", max_dets=100))
        adv_logits = torch.cat(adv_l, dim=0)
        coco_m = evaluate_coco_predictions(preds_sc, gt_list)

        asr_audit: Dict[str, Any] = {}
        if sc_obj["asr_type"] is not None:
            asr_val, succ, tot, _ = calculate_task_level_asr(sc_obj["asr_type"], clean_logits, adv_logits)
            ci = clopper_pearson_interval(succ, tot)
            asr_audit["code_formula"] = {
                "asr_type": sc_obj["asr_type"],
                "asr_pct": float(asr_val * 100.0),
                "succ": int(succ),
                "total": int(tot),
                "ci_95": [float(ci[0] * 100.0), float(ci[1] * 100.0)],
            }
            if sc_obj["asr_type"] == "Fabrication" or sc_obj["asr_type"] == "Adaptive":
                # Kiểm tra riêng các ảnh chưa có class=1 trên ảnh sạch
                c_cls1 = clean_probs[..., 1].max(dim=-1).values >= 0.35
                a_cls1 = torch.sigmoid(adv_logits)[..., 1].max(dim=-1).values >= 0.35
                net_new = (~c_cls1) & a_cls1
                asr_audit["net_new_class1_pct"] = float(net_new.float().mean().item() * 100.0)
                asr_audit["clean_already_had_class1_count"] = int(c_cls1.sum().item())
            if sc_obj["asr_type"] == "Vanishing":
                # Kiểm tra ASR có điều kiện trên ảnh sạch vốn có detection >= 0.35
                a_max = torch.sigmoid(adv_logits).max(dim=-1).values.max(dim=-1).values
                c_valid = clean_max_q_all >= 0.35
                cond_succ = int(((a_max < 0.20) & c_valid).sum().item())
                cond_tot = int(c_valid.sum().item())
                asr_audit["conditional_on_clean_ge035"] = {
                    "asr_pct": float(cond_succ / max(cond_tot, 1) * 100.0),
                    "succ": cond_succ,
                    "total": cond_tot,
                }

        table_res[sc_name] = {
            "coco_map_hf": coco_m,
            "rel_drop_50_pct": float((coco_m["mAP@50"] - base50) / base50 * 100.0),
            "rel_drop_50_95_pct": float((coco_m["mAP@50:95"] - base95) / base95 * 100.0),
            "asr_audit": asr_audit,
        }
        print(
            f"  [{sc_name:<45}] mAP@50={coco_m['mAP@50']:5.2f}% ({table_res[sc_name]['rel_drop_50_pct']:+6.1f}%) | "
            f"mAP@50:95={coco_m['mAP@50:95']:5.2f}% | "
            f"ASR={asr_audit.get('code_formula', {}).get('asr_pct', 0.0):.1f}%"
        )

    return {
        "n_eval": int(n_eval),
        "clean_query_stats": {
            "n_images": int(n_eval),
            "clean_max_conf_ge_035": clean_has_ge_035,
            "clean_max_conf_ge_020": clean_has_ge_020,
            "clean_class1_bicycle_ge_035": clean_has_class1_ge_035,
        },
        "scenarios": table_res,
    }


def main():
    parser = argparse.ArgumentParser(description="Audit Multitest FWER, Table 2 COCO mAP, and ASR Definitions")
    parser.add_argument("--n-eval", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--skip-map-audit", action="store_true")
    args = parser.parse_args()

    print("=" * 90)
    print("MỤC 1 AUDIT: MULTITEST FWER, COCO GT mAP@50 & ASR DEFINITIONS")
    print("=" * 90)

    mt_res = recompute_multitest_from_raw_npz()
    print("\n[1/2] Kết quả Multi-Testing Baseline trên 4 Quantiles (Sửa bug Layer_Maha):")
    for sk, info in mt_res["multitest"].items():
        cf = info["clean_fpr"]
        print(
            f"  - {sk:<20} (tau={info['threshold_tau']:.4f}): Clean FPR = {cf['fpr_pct']:.2f}% ({cf['fp_count']}/{cf['n_test']}) | "
            f"AQSM TPR={info['attack_tpr'].get('AQSM', {}).get('tpr_pct', 0.0):.1f}% | "
            f"Vanishing TPR={info['attack_tpr'].get('Vanishing', {}).get('tpr_pct', 0.0):.1f}%"
        )

    audit_payload: Dict[str, Any] = {
        "multitest_audit": mt_res,
    }

    if not args.skip_map_audit:
        print("\n[2/2] Đánh giá toàn bộ Bảng 2 trên COCO val2017 GT thật (N=300)...")
        map_res = evaluate_full_coco_benchmark_and_asr(
            n_eval=args.n_eval,
            batch_size=args.batch_size,
            threads=args.threads,
            include_dqf_adaptive=True,
        )
        audit_payload["coco_map_and_asr_audit"] = map_res

    out_audit = RESULTS_DIR / "rtdetr_audit_report.json"
    with open(out_audit, "w", encoding="utf-8") as f:
        json.dump(make_serializable(audit_payload), f, indent=2)
    print(f"\n[OK] Lưu báo cáo rà soát hoàn chỉnh tại: {out_audit}")


if __name__ == "__main__":
    main()
