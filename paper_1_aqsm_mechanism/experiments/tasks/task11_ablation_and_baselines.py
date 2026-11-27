# -*- coding: utf-8 -*-
"""
Module: paper_1_aqsm_mechanism.experiments.tasks.task11_ablation_and_baselines
Mục đích: Thực nghiệm mở rộng theo phản biện chuyên sâu (Reviewer Major Revision):
  1. Ablation từng thành phần hàm mất mát AQSM:
     (L_suppr, L_rank, L_elev, L_suppr+L_rank, L_suppr+L_elev, L_rank+L_elev, Full Tripartite)
     -> Xuất summary_json/tripartite_ablation_results.json (tương thích 100% Table 1).
  2. Nhóm đối chứng định lượng Gaussian Noise (sigma = 16/255, matched cohort).
  3. Đối chứng thực nghiệm tấn công Encoder Self-Attention (cf. Jayasundara et al. [27]).
  4. Khoảng tin cậy Bootstrap 95% cấp từng ảnh trên Pooled COCO mAP@50 & mAP@50:95.
"""

import gc
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import torch
from tqdm import tqdm

WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from paper_1_aqsm_mechanism.core.attack_generator import DETRAttackGenerator
from paper_1_aqsm_mechanism.core.coco_gt import (
    compute_pooled_coco_bootstrap_ci,
    evaluate_coco_predictions,
    postprocess_rtdetr_predictions,
)
from paper_1_aqsm_mechanism.core.dataset_loader import FlexibleCOCODataset
from paper_1_aqsm_mechanism.core.detr_wrapper import RTDETRModelWrapper
from paper_1_aqsm_mechanism.experiments.audit_logger import ExperimentAuditLogger
from paper_1_aqsm_mechanism.experiments.config import (
    RANDOM_SEED,
    configure_cpu_threads,
    detect_hardware_environment,
    get_device,
    get_epsilon_l_inf,
    get_img_size,
    get_pgd_num_steps,
    get_pgd_step_size,
    get_summary_json_dir,
    resolve_model_name,
    update_experiment_paths,
)
from paper_1_aqsm_mechanism.utils.cli import create_base_parser
from paper_1_aqsm_mechanism.utils.io_helpers import save_json


def _evaluate_condition(
    wrapper: RTDETRModelWrapper,
    clean_imgs: Any,
    gt_list: List[Dict[str, Any]],
    generator_fn: Optional[Callable[[torch.Tensor], torch.Tensor]],
    desc: str,
    batch_size: int = 8,
    n_boot: int = 500,
    seed: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """Đánh giá thống nhất một kịch bản (Clean / Loss Ablation / Gaussian Noise / Attention Attack [27])."""
    model = wrapper.model
    model.eval()
    device = wrapper.device
    is_sigmoid = bool(getattr(wrapper, "is_sigmoid", True))
    pp_mode = "hf_multilabel" if is_sigmoid else "detr_softmax"
    n_q_model = int(getattr(wrapper, "num_queries", 300) or 300)
    n_eval = len(gt_list)

    preds_all: List[Dict[str, np.ndarray]] = []
    ovl_300: List[float] = []
    evict_15: List[float] = []
    mem_cos: List[float] = []
    delta_p_list: List[float] = []
    delta_h_list: List[float] = []
    active_boxes_list: List[float] = []

    t0 = time.time()
    for start in tqdm(range(0, n_eval, batch_size), desc=desc):
        end = min(n_eval, start + batch_size)
        b_clean = clean_imgs[start:end]
        if hasattr(b_clean, "to"):
            b_clean = b_clean.to(device)

        if generator_fn is None:
            b_eval = b_clean
        else:
            b_eval = generator_fn(b_clean)

        with torch.no_grad():
            c_out = model(pixel_values=b_clean, return_dict=True)
            a_out = c_out if generator_fn is None else model(pixel_values=b_eval, return_dict=True)

            preds_all.extend(
                postprocess_rtdetr_predictions(a_out.logits, a_out.pred_boxes, mode=pp_mode)
            )
            probs_dec = torch.sigmoid(a_out.logits) if is_sigmoid else torch.softmax(a_out.logits, dim=-1)[..., :-1]
            act_cnt = (probs_dec.max(dim=-1).values >= 0.35).sum(dim=-1).cpu().numpy()
            active_boxes_list.extend(act_cnt.tolist())

            # Telemetry Encoder Selection & Memory Cosine
            c_enc = getattr(c_out, "enc_outputs_class", None)
            a_enc = getattr(a_out, "enc_outputs_class", None)
            if c_enc is not None and a_enc is not None:
                c_p = torch.sigmoid(c_enc).max(dim=-1).values
                a_p = torch.sigmoid(a_enc).max(dim=-1).values
                k_sel = min(n_q_model, c_p.shape[1])
                k_top15 = min(15, k_sel)
                c_topk = torch.topk(c_p, k=k_sel, dim=-1).indices.cpu().numpy()
                a_topk = torch.topk(a_p, k=k_sel, dim=-1).indices.cpu().numpy()
                for b_i in range(c_topk.shape[0]):
                    set_c = set(c_topk[b_i].tolist())
                    set_a = set(a_topk[b_i].tolist())
                    ovl_300.append(len(set_c & set_a) / float(max(1, k_sel)) * 100.0)
                    top15_c = c_topk[b_i][:k_top15]
                    evict_15.append(float(np.mean(~np.isin(top15_c, a_topk[b_i]))) * 100.0)

            c_mem = getattr(c_out, "encoder_last_hidden_state", None)
            a_mem = getattr(a_out, "encoder_last_hidden_state", None)
            if c_mem is not None and a_mem is not None:
                if isinstance(c_mem, (list, tuple)):
                    cm = torch.cat([t.flatten(2).transpose(1, 2) if t.ndim == 4 else t for t in c_mem], dim=1)
                    am = torch.cat([t.flatten(2).transpose(1, 2) if t.ndim == 4 else t for t in a_mem], dim=1)
                else:
                    cm, am = c_mem, a_mem
                cs = torch.nn.functional.cosine_similarity(cm, am, dim=-1).mean(dim=-1).cpu().numpy()
                mem_cos.extend(cs.tolist())

            c_ref = getattr(c_out, "init_reference_points", None)
            a_ref = getattr(a_out, "init_reference_points", None)
            if c_ref is not None and a_ref is not None:
                cr = torch.sigmoid(c_ref)[..., :2].cpu().numpy()
                ar = torch.sigmoid(a_ref)[..., :2].cpu().numpy()
                for b_i in range(cr.shape[0]):
                    dp = float(np.mean(np.linalg.norm(ar[b_i] - cr[b_i], axis=-1)))
                    cmat = cdist(cr[b_i], ar[b_i], metric="euclidean")
                    r_ind, c_ind = linear_sum_assignment(cmat)
                    dh = float(np.mean(cmat[r_ind, c_ind]))
                    delta_p_list.append(dp)
                    delta_h_list.append(dh)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    eval_res = evaluate_coco_predictions(preds_all, gt_list)
    boot_res = compute_pooled_coco_bootstrap_ci(preds_all, gt_list, n_boot=n_boot, seed=seed)

    return {
        "n_images": int(n_eval),
        "mAP@50": round(float(eval_res["mAP@50"]), 2),
        "mAP@50_ci95": boot_res["mAP@50_ci95"],
        "mAP@50:95": round(float(eval_res["mAP@50:95"]), 2),
        "mAP@50:95_ci95": boot_res["mAP@50:95_ci95"],
        "recall_pct": round(float(eval_res.get("recall_pct", 0.0)), 2),
        "top300_overlap_pct": round(float(np.mean(ovl_300)), 2) if ovl_300 else None,
        "evicted_top15_fg_pct": round(float(np.mean(evict_15)), 2) if evict_15 else None,
        "encoder_memory_cosine": round(float(np.mean(mem_cos)), 4) if mem_cos else None,
        "ref_drift_delta_p": round(float(np.mean(delta_p_list)), 4) if delta_p_list else None,
        "hungarian_drift_delta_h": round(float(np.mean(delta_h_list)), 4) if delta_h_list else None,
        "active_boxes_mean": round(float(np.mean(active_boxes_list)), 2) if active_boxes_list else 0.0,
        "active_boxes_std": round(float(np.std(active_boxes_list, ddof=1)) if len(active_boxes_list) > 1 else 0.0, 2),
        "elapsed_seconds": round(time.time() - t0, 2),
    }


def run_ablation_and_baselines(
    wrapper: RTDETRModelWrapper,
    dataset: Any,
    batch_size: int = 8,
    n_boot: int = 500,
    steps: Optional[int] = None,
    eps: Optional[float] = None,
    alpha: Optional[float] = None,
    seed: int = RANDOM_SEED,
    audit_logger: Any = None,
) -> Dict[str, Any]:
    """
    Chạy đầy đủ chuỗi thực nghiệm bổ sung theo phản biện:
      - Tripartite Loss Ablation (L_suppr, L_rank, L_elev và các tổ hợp cặp).
      - Gaussian Noise Control định lượng (sigma=epsilon, clipped in [-epsilon, +epsilon]).
      - Encoder Self-Attention Attack Baseline (cf. Jayasundara et al. [27]).
      - Pooled mAP Image-level 95% Bootstrap Confidence Intervals.
    """
    eff_steps = int(steps) if steps is not None else int(get_pgd_num_steps())
    eff_eps = float(eps) if eps is not None else float(get_epsilon_l_inf())
    eff_alpha = float(alpha) if alpha is not None else float(get_pgd_step_size())
    attacker = DETRAttackGenerator(wrapper, epsilon=eff_eps, alpha=eff_alpha, num_steps=eff_steps)

    clean_imgs = dataset.tensors
    gt_list = dataset.gt_list

    # 1. Clean Baseline + Pooled Bootstrap CI
    clean_res = _evaluate_condition(
        wrapper=wrapper,
        clean_imgs=clean_imgs,
        gt_list=gt_list,
        generator_fn=None,
        desc="Clean Baseline (CI)",
        batch_size=batch_size,
        n_boot=n_boot,
        seed=seed,
    )
    clean_map50 = max(clean_res["mAP@50"], 1e-6)

    # 2. Tripartite AQSM Loss Ablation (tương thích 100% schema Table 1 trong generate_publication_tables.py)
    ablation_configs = [
        ("L_suppr", 1.0, 0.0, 0.0, "Foreground confidence suppression only"),
        ("L_rank", 0.0, 1.2, 0.0, "Margin-ranking inversion only"),
        ("L_elev", 0.0, 0.0, 0.5, "Distal background elevation only"),
        ("L_suppr+L_rank", 1.0, 1.2, 0.0, "Suppression + margin ranking"),
        ("L_suppr+L_elev", 1.0, 0.0, 0.5, "Suppression + distal elevation"),
        ("L_rank+L_elev", 0.0, 1.2, 0.5, "Dual-boundary ranking + elevation"),
        ("Full_Tripartite", 1.0, 1.2, 0.5, "Compound Eviction & Misrouting"),
    ]
    variants_out: Dict[str, Any] = {}
    for v_key, l_sup, l_rnk, l_elv, mode_note in ablation_configs:
        def _make_gen(ls=l_sup, lr=l_rnk, le=l_elv):
            return lambda x: attacker.generate_query_selection_attack(
                x,
                steps=eff_steps,
                lambda_suppr=ls,
                lambda_rank=lr,
                lambda_elev=le,
                bg_seed=seed,
            )

        res_v = _evaluate_condition(
            wrapper=wrapper,
            clean_imgs=clean_imgs,
            gt_list=gt_list,
            generator_fn=_make_gen(),
            desc=f"Ablation {v_key}",
            batch_size=batch_size,
            n_boot=n_boot,
            seed=seed,
        )
        res_v["relative_drop_pct"] = round(((clean_res["mAP@50"] - res_v["mAP@50"]) / clean_map50) * 100.0, 2)
        res_v["failure_mode"] = mode_note
        res_v["weights"] = {"lambda_suppr": l_sup, "lambda_rank": l_rnk, "lambda_elev": l_elv}
        variants_out[v_key] = res_v

    tripartite_json = {
        "experiment": "Table 1: Tripartite AQSM Loss Component Ablation",
        "model": wrapper.pretrained_model_name,
        "n_images": len(gt_list),
        "epsilon_l_inf": eff_eps,
        "pgd_steps": eff_steps,
        "clean_baseline": clean_res,
        "variants": variants_out,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # 3. Quantitative Gaussian Noise Control (sigma = eps = 16/255)
    gaussian_res = _evaluate_condition(
        wrapper=wrapper,
        clean_imgs=clean_imgs,
        gt_list=gt_list,
        generator_fn=lambda x: attacker.generate_random_noise(x, noise_type="gaussian", sigma=eff_eps, seed=seed),
        desc=f"Gaussian Noise (sigma={eff_eps*255:.0f}/255)",
        batch_size=batch_size,
        n_boot=n_boot,
        seed=seed,
    )
    gaussian_res["relative_drop_pct"] = round(((clean_res["mAP@50"] - gaussian_res["mAP@50"]) / clean_map50) * 100.0, 2)

    # 4. Encoder Self-Attention Attack Baseline (cf. Jayasundara et al. [27])
    enc_attn_res = _evaluate_condition(
        wrapper=wrapper,
        clean_imgs=clean_imgs,
        gt_list=gt_list,
        generator_fn=lambda x: attacker.generate_encoder_attention_attack(x, steps=eff_steps),
        desc="Encoder Self-Attention [27]",
        batch_size=batch_size,
        n_boot=n_boot,
        seed=seed,
    )
    enc_attn_res["relative_drop_pct"] = round(((clean_res["mAP@50"] - enc_attn_res["mAP@50"]) / clean_map50) * 100.0, 2)

    # 5. Pixel-Space M/Q Decoupled Controls (Không OOD tensor-swap):
    #    - Memory-Isolated Query Attack (giữ nguyên Memory cosine, chỉ phá Query Selection)
    #    - Pure Memory Attack (chỉ phá Memory cosine, không có loss xếp hạng Query)
    mem_isolated_res = _evaluate_condition(
        wrapper=wrapper,
        clean_imgs=clean_imgs,
        gt_list=gt_list,
        generator_fn=lambda x: attacker.generate_memory_isolated_query_attack(x, steps=eff_steps, bg_seed=seed),
        desc="Memory-Isolated Query",
        batch_size=batch_size,
        n_boot=n_boot,
        seed=seed,
    )
    mem_isolated_res["relative_drop_pct"] = round(((clean_res["mAP@50"] - mem_isolated_res["mAP@50"]) / clean_map50) * 100.0, 2)

    pure_mem_res = _evaluate_condition(
        wrapper=wrapper,
        clean_imgs=clean_imgs,
        gt_list=gt_list,
        generator_fn=lambda x: attacker.generate_pure_memory_attack(x, steps=eff_steps),
        desc="Pure Memory Disruption",
        batch_size=batch_size,
        n_boot=n_boot,
        seed=seed,
    )
    pure_mem_res["relative_drop_pct"] = round(((clean_res["mAP@50"] - pure_mem_res["mAP@50"]) / clean_map50) * 100.0, 2)

    combined_summary = {
        "experiment": "Task 11: Tripartite Loss Ablation, Gaussian Noise Control, M/Q Pixel Controls & Encoder Attention Baseline [27]",
        "model": wrapper.pretrained_model_name,
        "n_images": len(gt_list),
        "resolution": f"{get_img_size()}x{get_img_size()}",
        "epsilon_l_inf": eff_eps,
        "pgd_steps": eff_steps,
        "n_bootstrap_resamples": n_boot,
        "clean_baseline": clean_res,
        "tripartite_loss_ablation": variants_out,
        "gaussian_noise_control": gaussian_res,
        "encoder_attention_baseline_27": enc_attn_res,
        "memory_isolated_query_control": mem_isolated_res,
        "pure_memory_control": pure_mem_res,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    sdir = get_summary_json_dir()
    sdir.mkdir(parents=True, exist_ok=True)
    save_json(tripartite_json, sdir / "tripartite_ablation_results.json")
    save_json(combined_summary, sdir / "ablation_and_baselines_summary.json")
    print(f">>> Đã lưu Tripartite Ablation JSON ra: {sdir / 'tripartite_ablation_results.json'}")
    print(f">>> Đã lưu Summary Baseline & Bootstrap CI JSON ra: {sdir / 'ablation_and_baselines_summary.json'}")

    if audit_logger is not None:
        audit_logger.write_audit_record({
            "record_type": "ABLATION_AND_BASELINES",
            "model": wrapper.pretrained_model_name,
            "n_images": len(gt_list),
            "clean_map50": clean_res["mAP@50"],
            "gaussian_noise_map50": gaussian_res["mAP@50"],
            "encoder_attention_27_map50": enc_attn_res["mAP@50"],
        })

    return combined_summary


def main():
    parser = create_base_parser("AQSM Loss Ablation & Reviewer Baselines (Task 11)", task_name="Task 11")
    parser.add_argument("--n_boot", type=int, default=500, help="Số lần bootstrap CI theo ảnh trên pooled mAP (mặc định: 500)")
    parser.add_argument("--pgd_steps", type=int, default=None, help="Số bước lặp PGD (mặc định theo config)")
    args = parser.parse_args()

    chosen_model = resolve_model_name(getattr(args, "model", None))
    update_experiment_paths(
        folder_image=args.folder_image,
        annotations=args.annotations,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        model=chosen_model,
        pgd_steps=args.pgd_steps,
    )
    configure_cpu_threads(args.threads)
    hw_info = detect_hardware_environment(preferred_device=args.device)
    device = get_device(preferred=args.device)
    batch_size = args.batch_size if args.batch_size is not None else hw_info["recommended_batch_size"]

    audit_logger = ExperimentAuditLogger(task_name="task11_ablation_and_baselines")
    wrapper = RTDETRModelWrapper(pretrained_model_name=chosen_model, device=device)
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

    audit_logger.log_metadata({
        "folder_image": str(args.folder_image),
        "total_images": len(dataset),
        "batch_size": batch_size,
        "device": str(device),
        "model_name": chosen_model,
        "n_boot": args.n_boot,
    })

    run_ablation_and_baselines(
        wrapper=wrapper,
        dataset=dataset,
        batch_size=batch_size,
        n_boot=args.n_boot,
        steps=args.pgd_steps,
        audit_logger=audit_logger,
    )

    audit_logger.close()
    del dataset, wrapper
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print("\n[HOÀN THÀNH TOÀN BỘ] Task 11: Ablation & Baselines thành công!")


if __name__ == "__main__":
    main()
