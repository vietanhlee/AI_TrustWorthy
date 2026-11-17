# -*- coding: utf-8 -*-
"""
Script: step_matched_map_sweep.py
Mục đích: Thực nghiệm đối chứng Matched-mAP giữa AQSM và Standard Task PGD:
1. Quét Task PGD ở các mức ngân sách nhiễu siêu nhỏ:
   eps in [0.5/255, 1.0/255, 2.0/255] trên N=100 ảnh COCO GT thật.
2. Tìm điểm mà Task PGD có cùng mức suy giảm mAP như AQSM (mAP@50 ~ 8% và ~ 2%).
3. Đối chiếu tính lén lút (Detection TPR của các Guardrail Monitors) và cơ chế lỗi:
   - Bùng nổ số lượng false positive boxes
   - Phân bố nhiễu FG vs BG
   - Tỷ lệ trùng khớp Top-300
   - FG Spatial Hit Rate và Attention Entropy
"""

import os
import sys
import json
import time
import pickle
from pathlib import Path
from typing import Dict, List, Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
from tqdm import tqdm

torch.set_num_threads(10)

WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from edge_sec_pro.core.coco_gt import (
    load_coco_val2017_gt_cohort,
    postprocess_rtdetr_predictions,
    evaluate_coco_predictions,
)
from edge_sec_pro.core.config import CHECKPOINTS_DIR, RESULTS_DIR
from edge_sec_pro.transformer_monitors.detr_wrapper import RTDETRModelWrapper
from edge_sec_pro.transformer_monitors.detr_attack_generator import DETRAttackGenerator
from edge_sec_pro.transformer_monitors.rtdetr_advanced_features import RTDETRDeepFeatureExtractor
from edge_sec_pro.transformer_monitors.rtdetr_mechanism_analyzer import (
    capture_deformable_attention_internals,
    compute_token_fg_masks,
    _summarize_attention_state,
    _sigmoid_np,
)
import pickle


def run_matched_map_experiment(n_eval: int = 100, batch_size: int = 10) -> Dict[str, Any]:
    print("=" * 85)
    print(f"THỰC NGHIỆM MATCHED-mAP: TASK PGD vs AQSM TRÊN N={n_eval} ẢNH COCO GT")
    print("=" * 85)

    device = torch.device("cpu")
    wrapper = RTDETRModelWrapper(pretrained_model_name="PekingU/rtdetr_r18vd", device=device)
    model = wrapper.model
    model.eval()

    # 1. Nạp COCO Ground Truth Cohort N=100
    cohort = load_coco_val2017_gt_cohort(
        n_cal=1200,
        n_test=600,
        eval_start=0,
        n_eval=n_eval,
        seed=42,
        id2label=model.config.id2label,
        load_tensors=True,
    )
    clean_imgs = cohort["tensors"]
    hf_labels = cohort["hf_labels"]
    gt_list = cohort["gt_list"]
    gt_boxes_batch = [gt["boxes_xyxy"] for gt in gt_list]

    # Nạp calibration checkpoints để chấm điểm guardrail
    ckpt_path = CHECKPOINTS_DIR / "calibration_and_detectors.pkl"
    with open(ckpt_path, "rb") as f:
        ckpt_data = pickle.load(f)
    cal_thresholds = ckpt_data["cal_thresholds"]
    cal_detectors = ckpt_data["detectors"]
    extractor = RTDETRDeepFeatureExtractor(conf_threshold=0.35, top_k_queries=10, projection_dim=64)

    # 2. Đánh giá Clean Baseline trên N=100
    print("[1/3] Đánh giá Clean Baseline N=100...")
    clean_preds = []
    clean_feats = []
    with torch.no_grad():
        for start in range(0, n_eval, batch_size):
            sub_x = clean_imgs[start : start + batch_size]
            st = wrapper(sub_x)
            clean_preds.extend(
                postprocess_rtdetr_predictions(st.logits, st.pred_boxes, mode="hf_multilabel", max_dets=100)
            )
            clean_feats.extend(extractor.extract_from_state(st))

    clean_map = evaluate_coco_predictions(clean_preds, gt_list)
    print(f"  Clean N={n_eval}: mAP@50={clean_map['mAP@50']:.2f}%, mAP@50:95={clean_map['mAP@50:95']:.2f}%")

    # 3. Quét Task PGD tại eps in [0.5, 1.0, 2.0]/255
    task_epsilons = [0.5 / 255.0, 1.0 / 255.0]
    task_results: Dict[str, Any] = {}

    print("\n[2/3] Sinh và đánh giá Standard Task PGD tại eps in [0.5/255, 1.0/255, 2.0/255]...")
    for eps_val in task_epsilons:
        eps_str = f"{eps_val*255.0:.1f}/255"
        alpha = min(2.0 / 255.0, eps_val / 2.0)
        attacker = DETRAttackGenerator(wrapper, epsilon=eps_val, alpha=alpha, num_steps=15)

        adv_imgs_l = []
        desc = f"Task PGD eps={eps_str}"
        t0 = time.time()
        for start in tqdm(range(0, n_eval, batch_size), desc=desc):
            end = min(n_eval, start + batch_size)
            sub_x = clean_imgs[start:end]
            sub_lbl = hf_labels[start:end]
            adv_sub = attacker.generate_standard_task_pgd(sub_x, hf_labels=sub_lbl, steps=15, alpha=alpha)
            adv_imgs_l.append(adv_sub.detach().cpu())

        adv_imgs = torch.cat(adv_imgs_l, dim=0)
        delta = adv_imgs - clean_imgs
        actual_linf = float(delta.abs().max().item())
        mean_l2 = float(torch.norm(delta.view(n_eval, -1), dim=1).mean().item())

        # Đánh giá predictions và guardrails
        preds_adv = []
        feats_adv = []
        attn_records = []
        active_boxes_cnt = []

        with torch.no_grad():
            for start in range(0, n_eval, batch_size):
                end = min(n_eval, start + batch_size)
                sub_adv = adv_imgs[start:end]
                gt_sub = gt_boxes_batch[start:end]

                with capture_deformable_attention_internals(model) as cap:
                    st = wrapper(sub_adv)

                preds_adv.extend(
                    postprocess_rtdetr_predictions(st.logits, st.pred_boxes, mode="hf_multilabel", max_dets=100)
                )
                feats_adv.extend(extractor.extract_from_state(st))

                probs = torch.sigmoid(st.logits)
                scores, _ = probs.max(dim=-1)
                for b_i in range(end - start):
                    act_cnt = int((scores[b_i] >= 0.35).sum().item())
                    active_boxes_cnt.append(act_cnt)

                attn_sum = _summarize_attention_state(cap, st.logits.cpu().numpy(), gt_sub)["per_image"]
                attn_records.extend(attn_sum)

        coco_m = evaluate_coco_predictions(preds_adv, gt_list)

        # Monitor TPR
        mon_tpr = {}
        for ch_name, det in cal_detectors.items():
            sc = det.score(feats_adv)
            th = cal_thresholds[ch_name]
            tpr = float(np.mean(np.asarray(sc) >= th) * 100.0)
            mon_tpr[ch_name] = tpr

        fg_hit_all = float(np.nanmean([r["mean_fg_spatial_hit"] for r in attn_records]) * 100.0)
        fg_hit_act = float(np.nanmean([r["mean_fg_spatial_hit_active_q"] for r in attn_records]) * 100.0)
        entropy_mean = float(np.nanmean([r["mean_entropy_norm"] for r in attn_records]))
        mean_active = float(np.mean(active_boxes_cnt))

        task_results[eps_str] = {
            "epsilon": eps_val,
            "max_linf": actual_linf,
            "mean_l2": mean_l2,
            "mAP@50": coco_m["mAP@50"],
            "mAP@75": coco_m["mAP@75"],
            "mAP@50:95": coco_m["mAP@50:95"],
            "mean_active_boxes": mean_active,
            "fg_spatial_hit_all": fg_hit_all,
            "fg_spatial_hit_act": fg_hit_act,
            "attn_entropy_norm": entropy_mean,
            "monitor_tpr": mon_tpr,
            "elapsed_sec": time.time() - t0,
        }

        print(
            f"  [Task PGD eps={eps_str:<8}] mAP@50={coco_m['mAP@50']:5.2f}% | "
            f"mAP@50:95={coco_m['mAP@50:95']:5.2f}% | "
            f"Boxes={mean_active:4.1f} | "
            f"FG Hit={fg_hit_all:4.1f}% | "
            f"Max-OR TPR={mon_tpr['Calibrated_Max_OR']:5.1f}% | "
            f"ActiveQ TPR={mon_tpr['ActiveQ_Split']:5.1f}% | "
            f"ESM TPR={mon_tpr['ESM_Dropoff']:5.1f}%"
        )

    # 4. Tải kết quả AQSM tương ứng từ kết quả đã có
    with open(RESULTS_DIR / "rtdetr_mechanism_analysis_results.json", "r", encoding="utf-8") as f:
        mech_data = json.load(f)["epsilon_budget_comparison"]

    full_output = {
        "n_eval": n_eval,
        "clean_baseline": clean_map,
        "task_pgd_sub_epsilon_sweep": task_results,
        "aqsm_sweep_reference": {
            "4/255": mech_data["4/255"]["aqsm"],
            "8/255": mech_data["8/255"]["aqsm"],
            "16/255": mech_data["16/255"]["aqsm"] if "16/255" in mech_data else None,
        },
        "standard_taskpgd_reference": {
            "4/255": mech_data["4/255"]["taskpgd"],
            "8/255": mech_data["8/255"]["taskpgd"],
        }
    }

    out_file = RESULTS_DIR / "rtdetr_matched_map_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2, ensure_ascii=False)
    print(f"\n--> Đã lưu kết quả Matched-mAP: {out_file}")

    return full_output


if __name__ == "__main__":
    run_matched_map_experiment(n_eval=50, batch_size=10)
