# -*- coding: utf-8 -*-
"""
Script: step_prior_ablation_misrouting.py
Mục đích: Thực nghiệm Ablation phân rã vi mô 3 thành phần của Query Prior:
  1. Query Selection Indices (I)
  2. Query Content Feature (Q_content = target)
  3. Initial Reference Boxes (B_init = init_reference_points)

Nhằm xác định thành phần nào thực sự gây ra hiện tượng Spatial Target Misrouting
ở Deformable Decoder của RT-DETR dưới đòn tấn công AQSM.

Các nhánh can thiệp (trên nền Memory sạch M_clean):
  - Arm A (Clean Baseline):         M_clean, I_clean, Q_clean(I_clean), B_clean(I_clean)
  - Arm B (Full AQSM):             M_adv,   I_adv,   Q_adv(I_adv),     B_adv(I_adv)
  - Arm D_full (Full Prior Adv):   M_clean, I_adv,   Q_adv(I_adv),     B_adv(I_adv)
  - Arm D_index (Chỉ tráo chỉ số): M_clean, I_adv,   Q_clean(I_adv),   B_clean(I_adv)
  - Arm D_box (Chỉ tráo B_init):   M_clean, I_clean, Q_clean(I_clean), B_adv(I_clean)
  - Arm D_feat (Chỉ tráo Q_feat):  M_clean, I_clean, Q_adv(I_clean),   B_clean(I_clean)
  - Arm D_misrouted_box:           M_clean, I_clean, Q_clean(I_clean), B_adv(I_adv)

Đo lường trên N=300 ảnh COCO val2017 GT thật:
  - COCO mAP@50 và mAP@50:95
  - Spatial FG Hit Rate (All queries & Active queries)
  - Reference Point Shift (L1, L2)
  - Sampling Location Shift (L1, L2)
  - Normalized Attention Entropy
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
from tqdm import tqdm

torch.set_num_threads(4)

WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from edge_sec_pro.core.coco_gt import (
    load_coco_val2017_gt_cohort,
    postprocess_rtdetr_predictions,
    evaluate_coco_predictions,
    precompute_coco_match_cache,
    evaluate_coco_from_match_cache,
)
from edge_sec_pro.core.config import CHECKPOINTS_DIR, RESULTS_DIR
from edge_sec_pro.transformer_monitors.detr_wrapper import RTDETRModelWrapper
from edge_sec_pro.transformer_monitors.rtdetr_mechanism_analyzer import (
    extract_encoder_components,
    run_decoder_from_components,
    capture_deformable_attention_internals,
    _summarize_attention_state,
    _sigmoid_np,
)


def _gather_priors(comp: Dict[str, Any], ind: torch.Tensor):
    ref_dim = comp["enc_outputs_coord_logits"].shape[-1]
    mem_dim = comp["output_memory"].shape[-1]
    ref_pts = comp["enc_outputs_coord_logits"].gather(
        dim=1, index=ind.unsqueeze(-1).expand(-1, -1, ref_dim)
    ).detach()
    tgt = comp["output_memory"].gather(
        dim=1, index=ind.unsqueeze(-1).expand(-1, -1, mem_dim)
    ).detach()
    return tgt, ref_pts


def run_ablation_experiment(n_eval: int = 300, batch_size: int = 10) -> Dict[str, Any]:
    print("=" * 80)
    print(f"THỰC NGHIỆM ABLATION PHÂN RÃ QUERY PRIOR TRÊN RT-DETR (N={n_eval} ảnh)")
    print("=" * 80)

    device = torch.device("cpu")
    wrapper = RTDETRModelWrapper(pretrained_model_name="PekingU/rtdetr_r18vd", device=device)
    model = wrapper.model
    model.eval()

    # 1. Nạp COCO Ground Truth Cohort
    print("[1/4] Nạp dữ liệu COCO GT Cohort...")
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
    gt_list = cohort["gt_list"]
    gt_boxes_batch = [gt["boxes_xyxy"] for gt in gt_list]

    # 2. Nạp Adversarial Cache AQSM (eps=16/255)
    print("[2/4] Nạp Adversarial Cache AQSM eps=16/255...")
    cache_path = CHECKPOINTS_DIR / "adv_cache" / f"adv_cache_aqsm_eps16_n{n_eval}.npz"
    if not cache_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file cache: {cache_path}")
    cache_data = np.load(cache_path)
    delta_f16 = cache_data["delta_f16"]
    adv_imgs = torch.clamp(clean_imgs + torch.from_numpy(delta_f16.astype(np.float32)), 0.0, 1.0)

    # Định nghĩa 7 nhánh ablation
    arms = [
        "Arm_A",               # Clean Baseline
        "Arm_B",               # Full AQSM
        "Arm_D_full",          # Clean Memory + Full Adv Prior
        "Arm_D_index",         # Clean Memory + Adv Index (Content & Box from Clean at I_adv)
        "Arm_D_box",           # Clean Memory + Clean Index + Clean Content + Adv Box at I_clean
        "Arm_D_feat",          # Clean Memory + Clean Index + Adv Content at I_clean + Clean Box
        "Arm_D_misrouted_box", # Clean Memory + Clean Index + Clean Content + Adv Box at I_adv
    ]

    preds_by_arm: Dict[str, List[Dict[str, np.ndarray]]] = {a: [] for a in arms}
    attn_records_by_arm: Dict[str, List[Dict[str, Any]]] = {a: [] for a in arms}

    print("[3/4] Chạy can thiệp và hook Deformable Attention trên từng batch...")
    t0 = time.time()

    for start in tqdm(range(0, n_eval, batch_size), desc="Ablation Priors"):
        end = min(n_eval, start + batch_size)
        c_sub = clean_imgs[start:end]
        a_sub = adv_imgs[start:end]
        gt_sub = gt_boxes_batch[start:end]

        with torch.no_grad():
            c_comp = extract_encoder_components(model, c_sub)
            a_comp = extract_encoder_components(model, a_sub)

            c_ind = c_comp["topk_ind"]
            a_ind = a_comp["topk_ind"]

            # Priors chéo
            tgt_clean_at_adv_idx, ref_clean_at_adv_idx = _gather_priors(c_comp, a_ind)
            tgt_adv_at_clean_idx, ref_adv_at_clean_idx = _gather_priors(a_comp, c_ind)

            arm_configs = {
                "Arm_A": (c_comp["target"], c_comp["source_flatten"], c_comp["init_reference_points"]),
                "Arm_B": (a_comp["target"], a_comp["source_flatten"], a_comp["init_reference_points"]),
                "Arm_D_full": (a_comp["target"], c_comp["source_flatten"], a_comp["init_reference_points"]),
                "Arm_D_index": (tgt_clean_at_adv_idx, c_comp["source_flatten"], ref_clean_at_adv_idx),
                "Arm_D_box": (c_comp["target"], c_comp["source_flatten"], ref_adv_at_clean_idx),
                "Arm_D_feat": (tgt_adv_at_clean_idx, c_comp["source_flatten"], c_comp["init_reference_points"]),
                "Arm_D_misrouted_box": (c_comp["target"], c_comp["source_flatten"], a_comp["init_reference_points"]),
            }

            for arm_name, (tgt, mem, ref) in arm_configs.items():
                with capture_deformable_attention_internals(model) as cap:
                    dec_out = run_decoder_from_components(
                        model=model,
                        target=tgt,
                        source_flatten=mem,
                        init_reference_points=ref,
                        spatial_shapes=c_comp["spatial_shapes"],
                        spatial_shapes_list=c_comp["spatial_shapes_list"],
                        level_start_index=c_comp["level_start_index"],
                        output_attentions=False,
                    )

                logits_np = dec_out["logits"].cpu().numpy()
                boxes_np = dec_out["pred_boxes"].cpu().numpy()

                # Lưu predictions cho mAP
                preds_by_arm[arm_name].extend(
                    postprocess_rtdetr_predictions(
                        dec_out["logits"], dec_out["pred_boxes"], mode="hf_multilabel", max_dets=100
                    )
                )

                # Tổng hợp thống kê attention
                attn_summary = _summarize_attention_state(cap, logits_np, gt_sub)["per_image"]
                for b_idx in range(len(attn_summary)):
                    init_ref_np = torch.sigmoid(ref[b_idx]).cpu().numpy()
                    clean_ref_np = torch.sigmoid(c_comp["init_reference_points"][b_idx]).cpu().numpy()
                    ref_shift_l2 = float(np.mean(np.linalg.norm(init_ref_np[:, :2] - clean_ref_np[:, :2], axis=-1)))
                    ref_shift_l1 = float(np.mean(np.abs(init_ref_np[:, :2] - clean_ref_np[:, :2])))

                    rec = attn_summary[b_idx]
                    rec["ref_shift_l2"] = ref_shift_l2
                    rec["ref_shift_l1"] = ref_shift_l1
                    attn_records_by_arm[arm_name].append(rec)

    elapsed = time.time() - t0
    print(f"Hoàn thành suy luận sau {elapsed:.1f}s.")

    # 4. Đánh giá COCO mAP và Thống kê Deformable Attention
    print("[4/4] Tính toán mAP và tổng hợp chỉ số không gian...")
    results: Dict[str, Any] = {"elapsed_sec": elapsed, "n_eval": n_eval, "arms": {}}

    for arm_name in arms:
        coco_map = evaluate_coco_predictions(preds_by_arm[arm_name], gt_list)
        recs = attn_records_by_arm[arm_name]

        fg_hit_all = float(np.nanmean([r["mean_fg_spatial_hit"] for r in recs]))
        fg_hit_act = float(np.nanmean([r["mean_fg_spatial_hit_active_q"] for r in recs]))
        entropy_mean = float(np.nanmean([r["mean_entropy_norm"] for r in recs]))
        dispersion_mean = float(np.nanmean([r["mean_dispersion_radius"] for r in recs]))
        top1_weight_mean = float(np.nanmean([r["mean_top1_weight"] for r in recs]))
        ref_shift_l2_mean = float(np.nanmean([r["ref_shift_l2"] for r in recs]))
        ref_shift_l1_mean = float(np.nanmean([r["ref_shift_l1"] for r in recs]))
        active_boxes_mean = float(np.nanmean([r["n_active_queries"] for r in recs]))

        results["arms"][arm_name] = {
            "mAP@50": coco_map["mAP@50"],
            "mAP@75": coco_map["mAP@75"],
            "mAP@50:95": coco_map["mAP@50:95"],
            "fg_spatial_hit_all": fg_hit_all * 100.0,
            "fg_spatial_hit_act": fg_hit_act * 100.0,
            "attn_entropy_norm": entropy_mean,
            "sampling_dispersion": dispersion_mean,
            "top1_weight_mean": top1_weight_mean,
            "ref_shift_l2": ref_shift_l2_mean,
            "ref_shift_l1": ref_shift_l1_mean,
            "mean_active_boxes": active_boxes_mean,
        }

        print(
            f"  {arm_name:<20}: mAP@50={coco_map['mAP@50']:5.2f}% | "
            f"mAP@50:95={coco_map['mAP@50:95']:5.2f}% | "
            f"FG Hit All={fg_hit_all*100:5.1f}% | "
            f"FG Hit Act={fg_hit_act*100:5.1f}% | "
            f"Ref Shift L2={ref_shift_l2_mean:.4f} | "
            f"Entropy={entropy_mean:.4f}"
        )

    out_file = RESULTS_DIR / "rtdetr_prior_ablation_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n--> Đã xuất kết quả ablation: {out_file}")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ablation and Controlled Spatial Misrouting on RT-DETR Priors")
    parser.add_argument("--n-eval", type=int, default=300, help="Number of images to evaluate")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size")
    parser.add_argument("--run-controlled-c2", action="store_true", help="Run the standardized Controlled Spatial Misrouting (C2) with 5 seeds and exact post-clipping L2 match")
    args = parser.parse_args()

    if args.run_controlled_c2:
        c2_script_dir = WORKSPACE_DIR / "c1_c2_standardized_evaluation" / "src"
        if str(c2_script_dir) not in sys.path:
            sys.path.insert(0, str(c2_script_dir))
        from run_c2_controlled_misrouting import main as run_c2_main
        run_c2_main()
    else:
        run_ablation_experiment(n_eval=args.n_eval, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
