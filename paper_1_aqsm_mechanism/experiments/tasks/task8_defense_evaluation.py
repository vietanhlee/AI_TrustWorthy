# -*- coding: utf-8 -*-
"""
Module: paper_1_aqsm_mechanism.experiments.tasks.task8_defense_evaluation
Mục đích: CLI điều phối Task 8 - Đánh giá cơ chế phòng thủ Encoder Selection Monitor (ESM)
và Calibrated Guardrail trên tập COCO val2017 -> Phục vụ Table 11 & Figure 9:
  1. Hiệu chuẩn FPR danh định 5.0% trên N_cal ảnh sạch đầu tiên (Zero Data Leakage).
  2. Đo đạc tỷ lệ phát hiện (TPR @ 5% FPR) trước AQSM, Query Vanishing, Task PGD.
  3. Đánh giá khả năng chống chịu tấn công thích ứng (White-box Adaptive PGD theo Carlini-Tramèr).
  4. Xuất summary JSON và raw CSV phục vụ kiểm toán và vẽ biểu đồ trong bài báo.
"""

import gc
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from paper_1_aqsm_mechanism.core.adv_cache import load_cached_adv_images
from paper_1_aqsm_mechanism.core.coco_gt import (
    evaluate_coco_predictions,
    postprocess_rtdetr_predictions,
)
from paper_1_aqsm_mechanism.core.dataset_loader import FlexibleCOCODataset
from paper_1_aqsm_mechanism.core.defense_engine import (
    ActiveBoxCountMonitor,
    EncoderSelectionMonitor,
    SpatialConsistencyMonitor,
    evaluate_defense_cohort,
)
from paper_1_aqsm_mechanism.core.hardening_prototype import (
    evaluate_sqa_empirical_ablation,
    evaluate_sqa_hardening,
    save_hardening_results,
)
from paper_1_aqsm_mechanism.core.detr_wrapper import RTDETRModelWrapper
from paper_1_aqsm_mechanism.experiments.audit_logger import ExperimentAuditLogger
from paper_1_aqsm_mechanism.experiments.config import (
    MODEL_NAME,
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


def _compute_scm_differentiable_penalty(
    init_ref_boxes: torch.Tensor,
    pred_boxes: torch.Tensor,
    disp_bounds: Tuple[float, float],
    drift_bounds: Tuple[float, float],
) -> torch.Tensor:
    """Tính hàm phạt khả vi giữ spatial dispersion và reference drift trong biên ảnh sạch đã hiệu chuẩn."""
    ref_sig = (
        torch.sigmoid(init_ref_boxes)
        if (init_ref_boxes.max() > 1.5 or init_ref_boxes.min() < -0.5)
        else torch.clamp(init_ref_boxes, 0.0, 1.0)
    )
    pred_sig = (
        torch.sigmoid(pred_boxes)
        if (pred_boxes.max() > 1.5 or pred_boxes.min() < -0.5)
        else torch.clamp(pred_boxes, 0.0, 1.0)
    )
    mean_center = ref_sig[..., :2].mean(dim=1, keepdim=True)
    disp = torch.sqrt(torch.sum((ref_sig[..., :2] - mean_center) ** 2, dim=-1) + 1e-8).mean(dim=1)
    drift = torch.sqrt(torch.sum((pred_sig[..., :2] - ref_sig[..., :2]) ** 2, dim=-1) + 1e-8).mean(dim=1)

    d_lo, d_hi = disp_bounds
    r_lo, r_hi = drift_bounds
    pen_disp = F.relu(d_lo - disp) + F.relu(disp - d_hi)
    pen_drift = F.relu(r_lo - drift) + F.relu(drift - r_hi)
    return ((pen_disp / max(abs(d_hi - d_lo), 1e-3)) ** 2 + (pen_drift / max(abs(r_hi - r_lo), 1e-3)) ** 2).mean()


def run_adaptive_monitor_suite_test(
    wrapper: RTDETRModelWrapper,
    clean_imgs: Any,
    target_mode: str,
    esm_monitor: Optional[EncoderSelectionMonitor] = None,
    scm_monitor: Optional[SpatialConsistencyMonitor] = None,
    box_monitor: Optional[ActiveBoxCountMonitor] = None,
    n_samples: int = 50,
    lambdas: Sequence[float] = (0.5, 1.0, 2.0),
    batch_size: int = 4,
    gt_list: Optional[List[Dict[str, Any]]] = None,
    hf_labels: Optional[List[Dict[str, torch.Tensor]]] = None,
    steps: Optional[int] = None,
    eps: Optional[float] = None,
    alpha: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Thực nghiệm White-Box Adaptive Evasion thực thụ nhắm vào:
      - target_mode == 'scm': Né tránh SpatialConsistencyMonitor.
      - target_mode == 'bcm': Né tránh ActiveBoxCountMonitor (soft-clamping cardinality penalty).
      - target_mode == 'ensemble': Né tránh đồng thời cả 3 kênh (ESM + SCM + BCM Max-OR Union).

    Báo cáo đầy đủ TPR bị phát hiện thật, Raw Bypass (%), Attack Effectiveness (%) và
    Joint Conditioned Evasion Success (%) (= không bị phát hiện VÀ đồng thời làm sụp đổ phát hiện)
    cũng như Post-Attack mAP@50 (%).
    """
    model = wrapper.model
    device = wrapper.device
    is_sigmoid = bool(getattr(wrapper, "is_sigmoid", True))
    pp_mode = "hf_multilabel" if is_sigmoid else "detr_softmax"
    n_q_model = max(1, int(getattr(wrapper, "num_queries", 300) or 300))
    n_eval = min(n_samples, len(clean_imgs))
    n_steps = int(steps) if steps is not None else int(get_pgd_num_steps())
    eps_val = float(eps) if eps is not None else float(get_epsilon_l_inf())
    step_alpha = float(alpha) if alpha is not None else float(get_pgd_step_size())

    # Chuẩn bị ngưỡng hiệu chuẩn từ chính các monitor đã fit
    target_drop = float(getattr(esm_monitor, "cal_dropoff_median", 35.0)) if esm_monitor else 35.0
    target_margin = float(getattr(esm_monitor, "cal_margin_median", 0.05)) if esm_monitor else 0.05
    disp_bounds = (
        (float(scm_monitor.disp_mon.lower_thresh), float(scm_monitor.disp_mon.upper_thresh))
        if scm_monitor is not None and scm_monitor.is_fitted
        else (0.15, 0.35)
    )
    drift_bounds = (
        (float(scm_monitor.drift_mon.lower_thresh), float(scm_monitor.drift_mon.upper_thresh))
        if scm_monitor is not None and scm_monitor.is_fitted
        else (0.005, 0.08)
    )
    if box_monitor is not None and box_monitor.is_fitted:
        bcm_low = float(box_monitor.twosided_monitor.lower_thresh)
        bcm_high = float(box_monitor.twosided_monitor.upper_thresh)
        bcm_target = float(np.median(box_monitor.cal_counts)) if box_monitor.cal_counts.size > 0 else 8.0
    else:
        bcm_low, bcm_high, bcm_target = 2.0, 20.0, 8.0

    results: Dict[str, Any] = {}
    print(f"\n[Adaptive {target_mode.upper()} Evasion] Đánh giá trên N={n_eval} ảnh (batch_size={batch_size}, steps={n_steps})...")

    for lam in lambdas:
        all_detected: List[int] = []
        all_effective: List[int] = []
        all_joint: List[int] = []
        adaptive_preds_eval: List[Dict[str, Any]] = []
        active_box_counts: List[float] = []
        t_start = time.time()

        for b_start in range(0, n_eval, batch_size):
            b_end = min(n_eval, b_start + batch_size)
            sub_clean = clean_imgs[b_start:b_end]
            if hasattr(sub_clean, "to"):
                sub_clean = sub_clean.to(device)

            with torch.no_grad():
                clean_ref_out = model(sub_clean)
                c_probs = torch.sigmoid(clean_ref_out.logits) if is_sigmoid else torch.softmax(clean_ref_out.logits, dim=-1)[..., :-1]
                clean_max_p, _ = c_probs.max(dim=-1)
                clean_top_confs = clean_max_p.max(dim=-1).values.cpu().numpy()
                k_fg = min(15, clean_max_p.shape[1])
                clean_top_idx = torch.topk(clean_max_p, k=k_fg, dim=-1).indices  # [B, k_fg]
                has_enc_c = hasattr(clean_ref_out, "enc_outputs_class") and clean_ref_out.enc_outputs_class is not None
                clean_enc_p = (
                    torch.sigmoid(clean_ref_out.enc_outputs_class).max(dim=-1).values
                    if has_enc_c
                    else clean_max_p
                )
                clean_enc_top_idx = torch.topk(clean_enc_p, k=min(15, clean_enc_p.shape[1]), dim=-1).indices

            sub_hf = None
            if hf_labels is not None:
                sub_hf = [
                    {
                        "class_labels": hf_labels[idx]["class_labels"].to(device),
                        "boxes": hf_labels[idx]["boxes"].to(device),
                    }
                    for idx in range(b_start, b_end)
                    if idx < len(hf_labels) and len(hf_labels[idx]["boxes"]) > 0
                ]
                if len(sub_hf) != (b_end - b_start):
                    sub_hf = None

            adv = sub_clean.clone().detach()
            for _ in range(n_steps):
                adv_var = adv.clone().detach().requires_grad_(True)
                mask = torch.ones((adv_var.shape[0], adv_var.shape[2], adv_var.shape[3]), device=device, dtype=torch.long)
                if sub_hf is not None and target_mode == "bcm":
                    out = model(pixel_values=adv_var, pixel_mask=mask, labels=sub_hf, return_dict=True)
                else:
                    out = model(pixel_values=adv_var, pixel_mask=mask, return_dict=True)

                dec_probs = torch.sigmoid(out.logits) if is_sigmoid else torch.softmax(out.logits, dim=-1)[..., :-1]
                dec_max_p, _ = dec_probs.max(dim=-1)  # [B, N_q]

                # 1. Task failure loss (giữ phá hủy object thật)
                if sub_hf is not None and getattr(out, "loss", None) is not None and target_mode == "bcm":
                    task_loss = -out.loss  # maximize detection loss via gradient descent step (-alpha * grad)
                    # Wait: step below is adv = adv - step_alpha * grad_sign, so minimize (-loss) maximizes Task loss!
                else:
                    # Dìm các clean foreground slots đã biết ở cả encoder và decoder
                    sel_dec = torch.gather(dec_max_p, 1, clean_top_idx)
                    task_loss = (sel_dec ** 2).mean()
                    if hasattr(out, "enc_outputs_class") and out.enc_outputs_class is not None:
                        enc_max_p = torch.sigmoid(out.enc_outputs_class).max(dim=-1).values
                        sel_enc = torch.gather(enc_max_p, 1, clean_enc_top_idx)
                        task_loss = task_loss + (sel_enc ** 2).mean()

                # 2. Evasion loss tương ứng theo target_mode
                evasion_loss = torch.zeros((), device=device)

                if target_mode in ("esm", "ensemble") and esm_monitor is not None:
                    has_enc = hasattr(out, "enc_outputs_class") and out.enc_outputs_class is not None
                    enc_logits = out.enc_outputs_class if has_enc else out.logits
                    probs_e = torch.sigmoid(enc_logits) if (has_enc or is_sigmoid) else torch.softmax(enc_logits, dim=-1)[..., :-1]
                    max_p_e, _ = probs_e.max(dim=-1)
                    k_top_n = min(n_q_model + 1, max_p_e.shape[1])
                    k_nq_idx = min(n_q_model - 1, k_top_n - 1)
                    topk_vals, _ = torch.topk(max_p_e, k=k_top_n, dim=-1)
                    diff_drop = topk_vals[:, 0] / (topk_vals[:, k_nq_idx].clamp(min=1e-4) + 1e-4)
                    next_v = topk_vals[:, k_nq_idx + 1] if topk_vals.shape[1] > k_nq_idx + 1 else torch.zeros_like(topk_vals[:, k_nq_idx])
                    diff_marg = topk_vals[:, k_nq_idx] - next_v
                    evasion_loss = evasion_loss + F.mse_loss(diff_drop, torch.full_like(diff_drop, target_drop)) + 10.0 * F.mse_loss(diff_marg, torch.full_like(diff_marg, target_margin))

                if target_mode in ("scm", "ensemble"):
                    ref_b = out.init_reference_points if (hasattr(out, "init_reference_points") and out.init_reference_points is not None) else out.pred_boxes
                    evasion_loss = evasion_loss + _compute_scm_differentiable_penalty(ref_b, out.pred_boxes, disp_bounds, drift_bounds)

                if target_mode in ("bcm", "ensemble"):
                    # Soft cardinality penalty kẹp số active boxes quanh sạch trong [bcm_low, bcm_high]
                    soft_cnt = torch.sigmoid(25.0 * (dec_max_p - 0.35)).sum(dim=-1)
                    pen_lo = F.relu((bcm_low + 0.5) - soft_cnt)
                    pen_hi = F.relu(soft_cnt - (bcm_high - 0.5))
                    pen_mid = ((soft_cnt - bcm_target) / max(bcm_high - bcm_low, 2.0)) ** 2
                    evasion_loss = evasion_loss + (pen_lo ** 2 + pen_hi ** 2 + 0.25 * pen_mid).mean()

                total_loss = task_loss + float(lam) * evasion_loss

                model.zero_grad()
                if adv_var.grad is not None:
                    adv_var.grad.zero_()
                total_loss.backward()

                with torch.no_grad():
                    grad_sign = adv_var.grad.sign() if adv_var.grad is not None else torch.zeros_like(adv_var)
                    adv = adv - step_alpha * grad_sign
                    delta = torch.clamp(adv - sub_clean, min=-eps_val, max=eps_val)
                    adv = torch.clamp(sub_clean + delta, min=0.0, max=1.0).detach()

            with torch.no_grad():
                final_out = model(adv)
                b_len = adv.shape[0]
                det_flags = np.zeros(b_len, dtype=bool)
                if target_mode in ("esm", "ensemble") and esm_monitor is not None:
                    f_lgt = final_out.enc_outputs_class if (hasattr(final_out, "enc_outputs_class") and final_out.enc_outputs_class is not None) else final_out.logits
                    det_flags |= (esm_monitor.predict(f_lgt.cpu()) == 1)
                if target_mode in ("scm", "ensemble") and scm_monitor is not None:
                    f_ref = final_out.init_reference_points if (hasattr(final_out, "init_reference_points") and final_out.init_reference_points is not None) else final_out.pred_boxes
                    det_flags |= (scm_monitor.predict(f_ref.cpu(), final_out.pred_boxes.cpu()) == 1)
                if target_mode in ("bcm", "ensemble") and box_monitor is not None:
                    b_cnts = box_monitor.count_active_boxes(final_out.logits.cpu())
                    active_box_counts.extend(b_cnts.tolist())
                    det_flags |= (box_monitor.twosided_monitor.predict(b_cnts) == 1)

                if gt_list is not None:
                    adaptive_preds_eval.extend(
                        postprocess_rtdetr_predictions(final_out.logits, final_out.pred_boxes, mode=pp_mode)
                    )

                adv_p = torch.sigmoid(final_out.logits) if is_sigmoid else torch.softmax(final_out.logits, dim=-1)[..., :-1]
                adv_max_p, _ = adv_p.max(dim=-1)
                adv_top_confs = adv_max_p.max(dim=-1).values.cpu().numpy()

                for b_i in range(b_len):
                    is_det = bool(det_flags[b_i])
                    if target_mode == "bcm" and gt_list is not None and len(adaptive_preds_eval) > 0:
                        # Với BCM, kẻ tấn công có thể giữ ~8 dummy boxes >= 0.35 ở vùng nền, nên đo effectiveness bằng per-image AP collapse (< 50% clean AP hoặc conf < 0.35)
                        img_idx = b_start + b_i
                        gt_b = gt_list[img_idx].get("boxes_xyxy", np.zeros((0, 4), dtype=np.float32)) if img_idx < len(gt_list) else np.zeros((0, 4), dtype=np.float32)
                        pr_b = adaptive_preds_eval[-b_len + b_i]
                        # Kiểm tra xem mọi box active còn trúng IoU>=0.5 với bất kỳ GT nào không
                        act_mask = pr_b["scores"] >= 0.35
                        if len(gt_b) > 0 and np.any(act_mask):
                            from paper_1_aqsm_mechanism.core.mechanism_analyzer import _box_iou_np
                            max_iou = float(np.max(_box_iou_np(pr_b["boxes"][act_mask], gt_b)))
                            is_eff = bool(max_iou < 0.50 or adv_top_confs[b_i] < clean_top_confs[b_i] * 0.5)
                        else:
                            is_eff = bool(adv_top_confs[b_i] < 0.35 or len(gt_b) == 0)
                    else:
                        is_eff = bool(adv_top_confs[b_i] < 0.35 or adv_top_confs[b_i] < clean_top_confs[b_i] * 0.5)

                    all_detected.append(int(is_det))
                    all_effective.append(int(is_eff))
                    all_joint.append(int((not is_det) and is_eff))

            del adv, sub_clean
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        tpr_det = float(np.mean(all_detected)) * 100.0 if all_detected else 0.0
        raw_bypass = 100.0 - tpr_det
        eff_rate = float(np.mean(all_effective)) * 100.0 if all_effective else 0.0
        joint_rate = float(np.mean(all_joint)) * 100.0 if all_joint else 0.0
        post_map50 = None
        if gt_list is not None and adaptive_preds_eval:
            ev_map = evaluate_coco_predictions(adaptive_preds_eval, list(gt_list[:len(adaptive_preds_eval)]))
            post_map50 = round(float(ev_map.get("mAP@50", 0.0)), 2)

        elapsed = time.time() - t_start
        rec_l: Dict[str, Any] = {
            "evasion_weight_lambda": float(lam),
            "target_monitor": target_mode,
            "n_evaluated_samples": int(n_eval),
            "detected_tpr_pct": round(tpr_det, 2),
            "raw_monitor_bypass_pct": round(raw_bypass, 2),
            "attack_effective_rate_pct": round(eff_rate, 2),
            "joint_evasion_success_pct": round(joint_rate, 2),
            "post_attack_map50_pct": post_map50,
            "elapsed_seconds": round(elapsed, 2),
        }
        if active_box_counts:
            rec_l["mean_active_boxes"] = round(float(np.mean(active_box_counts)), 2)
        results[f"lambda_{lam}"] = rec_l
        m_str = f" | Post-mAP@50={post_map50:.2f}%" if post_map50 is not None else ""
        print(
            f"   + [{target_mode.upper()}] lam={lam}: TPR={tpr_det:.1f}% | Bypass={raw_bypass:.1f}% | Joint={joint_rate:.1f}%{m_str} ({elapsed:.1f}s)"
        )

    return results


def run_adaptive_evasion_test(
    wrapper: RTDETRModelWrapper,
    clean_imgs: Any,
    esm_monitor: EncoderSelectionMonitor,
    n_samples: int = 15,
    lambdas: List[float] = [0.2, 0.5, 1.0, 2.0],
    batch_size: int = 4,
    gt_list: Optional[List[Dict[str, Any]]] = None,
    steps: Optional[int] = None,
    eps: Optional[float] = None,
    alpha: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Thực nghiệm tấn công thích ứng White-Box Adaptive PGD (Carlini-Tramèr):
    Kẻ tấn công vừa dìm detection mAP vừa tối ưu để né tránh ngưỡng phát hiện của ESM.
    Hỗ trợ xử lý theo lô (mini-batching) chống tràn VRAM khi chạy quy mô lớn (lên tới 1,000 ảnh).

    Đặc biệt điều kiện hóa Evasion Success (Conditioned Evasion Success):
    Một mẫu chỉ được coi là Né tránh Thành công Thực sự (True Evasion) khi và chỉ khi:
    1. Vượt qua bộ giám sát ESM (anomaly_score < threshold, tức detected = 0).
    2. ĐỒNG THỜI đánh sập thành công khả năng phát hiện của detector (confidence dìm sâu hoặc mAP sụp đổ).
    Nếu chỉ né được monitor mà không làm sụp detector thì attack thất bại!
    """
    model = wrapper.model
    device = wrapper.device
    is_sigmoid = bool(getattr(wrapper, "is_sigmoid", True))
    pp_mode = "hf_multilabel" if is_sigmoid else "detr_softmax"
    n_q_model = max(1, int(getattr(wrapper, "num_queries", 300) or 300))
    n_eval = min(n_samples, len(clean_imgs))
    target_drop = float(getattr(esm_monitor, "cal_dropoff_median", 35.0))
    target_margin = float(getattr(esm_monitor, "cal_margin_median", 0.05))
    n_steps = int(steps) if steps is not None else int(get_pgd_num_steps())
    eps_val = float(eps) if eps is not None else float(get_epsilon_l_inf())
    step_alpha = float(alpha) if alpha is not None else float(get_pgd_step_size())

    results = {}
    print(f"\n[Adaptive Evasion] Đang đánh giá tấn công thích ứng né tránh ESM trên N={n_eval} ảnh (batch_size={batch_size}, steps={n_steps})...")

    for lam in lambdas:
        all_monitor_preds = []
        all_attack_effective = []
        all_conditioned_evasion = []
        adaptive_preds_eval = []
        t_start = time.time()
        for b_start in range(0, n_eval, batch_size):
            b_end = min(n_eval, b_start + batch_size)
            sub_clean = clean_imgs[b_start:b_end]
            if hasattr(sub_clean, "to"):
                sub_clean = sub_clean.to(device)

            # Tính điểm tự tin tối đa trên ảnh sạch làm mốc so sánh
            with torch.no_grad():
                clean_ref_out = model(sub_clean)
                clean_ref_probs = torch.sigmoid(clean_ref_out.logits) if is_sigmoid else torch.softmax(clean_ref_out.logits, dim=-1)[..., :-1]
                clean_max_p, _ = clean_ref_probs.max(dim=-1)
                clean_top_confs = clean_max_p.max(dim=-1).values.cpu().numpy()  # [B]

            adv = sub_clean.clone().detach()

            # Chu trình PGD chuẩn hóa theo cấu hình
            for _ in range(n_steps):
                adv_var = adv.clone().detach().requires_grad_(True)
                mask = torch.ones((adv_var.shape[0], adv_var.shape[2], adv_var.shape[3]), device=device, dtype=torch.long)
                out = model(pixel_values=adv_var, pixel_mask=mask)
                has_enc = hasattr(out, "enc_outputs_class") and out.enc_outputs_class is not None
                enc_logits = out.enc_outputs_class if has_enc else out.logits
                probs = torch.sigmoid(enc_logits) if (has_enc or is_sigmoid) else torch.softmax(enc_logits, dim=-1)[..., :-1]
                max_probs, _ = probs.max(dim=-1)

                # Task Loss (Dìm điểm các query có xác suất cao)
                k = min(15, max_probs.shape[1])
                top_probs, _ = torch.topk(max_probs, k=k, dim=-1)
                task_loss = (top_probs ** 2).mean()

                # Evasion Loss (Né tránh đồng thời cả Dropoff và Margin theo đúng nguyên lý Max-OR)
                n_cand = max_probs.shape[1]
                k_top_n = min(n_q_model + 1, n_cand)
                k_nq_idx = min(n_q_model - 1, k_top_n - 1)
                topk_vals, _ = torch.topk(max_probs, k=k_top_n, dim=-1)
                top1_val = topk_vals[:, 0]
                top_nq_val = topk_vals[:, k_nq_idx]
                top_nq_next_val = topk_vals[:, k_nq_idx + 1] if topk_vals.shape[1] > k_nq_idx + 1 else torch.zeros_like(top_nq_val)

                diff_dropoff = top1_val / (top_nq_val.clamp(min=1e-4) + 1e-4)
                diff_margin = top_nq_val - top_nq_next_val

                loss_drop = F.mse_loss(diff_dropoff, torch.full_like(diff_dropoff, target_drop))
                loss_margin = F.mse_loss(diff_margin, torch.full_like(diff_margin, target_margin))
                evasion_loss = loss_drop + 10.0 * loss_margin

                total_loss = task_loss + lam * evasion_loss

                model.zero_grad()
                if adv_var.grad is not None:
                    adv_var.grad.zero_()
                total_loss.backward()

                with torch.no_grad():
                    grad_sign = adv_var.grad.sign() if adv_var.grad is not None else torch.zeros_like(adv_var)
                    adv = adv - step_alpha * grad_sign
                    delta = torch.clamp(adv - sub_clean, min=-eps_val, max=eps_val)
                    adv = torch.clamp(sub_clean + delta, min=0.0, max=1.0).detach()

            with torch.no_grad():
                final_out = model(adv)
                final_logits = final_out.enc_outputs_class if hasattr(final_out, "enc_outputs_class") and final_out.enc_outputs_class is not None else final_out.logits
                preds = esm_monitor.predict(final_logits.cpu())
                all_monitor_preds.extend(preds)

                if gt_list is not None:
                    adaptive_preds_eval.extend(
                        postprocess_rtdetr_predictions(final_out.logits, final_out.pred_boxes, mode=pp_mode)
                    )

                # Đo đạc tính hiệu quả của đòn tấn công (detection suppression effectiveness)
                adv_final_probs = torch.sigmoid(final_out.logits) if is_sigmoid else torch.softmax(final_out.logits, dim=-1)[..., :-1]
                adv_max_p, _ = adv_final_probs.max(dim=-1)
                adv_top_confs = adv_max_p.max(dim=-1).values.cpu().numpy()

                for b_i in range(len(preds)):
                    is_detected = bool(preds[b_i] == 1)
                    # Tấn công hiệu quả khi conf tối đa bị dìm < 0.35 hoặc giảm > 50% so với clean
                    is_effective = bool(adv_top_confs[b_i] < 0.35 or adv_top_confs[b_i] < clean_top_confs[b_i] * 0.5)
                    # Conditioned Evasion: Không bị phát hiện VÀ tấn công vẫn hiệu quả!
                    is_conditioned = bool((not is_detected) and is_effective)

                    all_attack_effective.append(int(is_effective))
                    all_conditioned_evasion.append(int(is_conditioned))

            del adv, sub_clean
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        tpr_evade = float(np.mean(all_monitor_preds)) * 100.0 if len(all_monitor_preds) > 0 else 0.0
        raw_bypass = 100.0 - tpr_evade
        eff_rate = float(np.mean(all_attack_effective)) * 100.0 if len(all_attack_effective) > 0 else 0.0
        cond_evade = float(np.mean(all_conditioned_evasion)) * 100.0 if len(all_conditioned_evasion) > 0 else 0.0
        post_map50 = None
        if gt_list is not None and len(adaptive_preds_eval) > 0:
            ev_map = evaluate_coco_predictions(adaptive_preds_eval, list(gt_list[:len(adaptive_preds_eval)]))
            post_map50 = round(float(ev_map.get("mAP@50", 0.0)), 2)

        elapsed = time.time() - t_start
        results[f"lambda_{lam}"] = {
            "evasion_weight_lambda": lam,
            "n_evaluated_samples": n_eval,
            "detected_tpr_pct": round(tpr_evade, 2),
            "evasion_success_pct": round(cond_evade, 2),
            "raw_monitor_bypass_pct": round(raw_bypass, 2),
            "attack_effective_rate_pct": round(eff_rate, 2),
            "conditioned_evasion_success_pct": round(cond_evade, 2),
            "post_attack_map50_pct": post_map50,
            "elapsed_seconds": round(elapsed, 2),
        }
        map_str = f" | Post-mAP@50 = {post_map50:.2f}%" if post_map50 is not None else ""
        print(f"   + Lambda = {lam:3.1f} (N={n_eval}) | Detected TPR = {tpr_evade:5.1f}% | Raw Bypass = {raw_bypass:5.1f}% | Conditioned Evasion = {cond_evade:5.1f}%{map_str} ({elapsed:.1f}s)")

    return results

def run_defense_evaluation(
    wrapper: RTDETRModelWrapper,
    dataset: Any,
    cal_ratio: float = 0.3,
    target_fpr: float = 0.05,
    n_adaptive: int = 50,
    cache_dir: Optional[Path] = None,
    audit_logger: Any = None,
    run_sqa_ablation: bool = False,
    batch_size: int = 4,
) -> Dict[str, Any]:
    """
    Thực hiện đánh giá toàn diện cơ chế phòng thủ ESM trên tập ảnh kiểm thử.
    """
    clean_imgs = dataset.tensors
    image_ids = dataset.image_ids
    n_actual = len(dataset)
    device = wrapper.device

    print("\n==================================================================")
    print(" [Defense Evaluation Engine] Đánh giá Hệ thống Phòng thủ ESM & Guardrail")
    print(f" N={n_actual} ảnh COCO val2017 | Target FPR={target_fpr*100:.1f}%")
    print("==================================================================")

    # Nạp các tập mẫu đối kháng
    adv_dict = {}
    for atk in ["aqsm", "vanishing", "taskpgd"]:
        adv_imgs, _ = load_cached_adv_images(
            attack=atk,
            eps_num=16,
            n_eval=n_actual,
            clean_imgs=clean_imgs,
            expected_image_ids=image_ids,
            cache_dir=cache_dir,
            wrapper=wrapper,
            steps=get_pgd_num_steps(),
            hf_labels=getattr(dataset, "hf_labels", None),
        )
        adv_dict[atk] = adv_imgs

    # Xác định n_cal theo tỷ lệ hoặc tối đa 1000 ảnh (bảo đảm 1 <= n_cal < n_actual)
    n_total = n_actual
    if n_total < 2:
        raise ValueError(f"Task 8 yêu cầu tối thiểu 2 ảnh để chia calibration/test, nhận được n_actual={n_total}")
    n_cal = min(1000, max(50, int(n_actual * cal_ratio)))
    if n_actual <= 100:
        n_cal = max(1, n_actual // 2)
    n_cal = min(n_cal, n_total - 1)
    n_test = n_total - n_cal

    # Trích xuất một lượt duy nhất toàn bộ đặc trưng cho ESM, SCM và ActiveBoxCountMonitor
    bsz = max(1, int(batch_size))
    is_sigmoid = bool(getattr(wrapper, "is_sigmoid", True))
    n_q_model = int(getattr(wrapper, "num_queries", 300) or 300)
    print(f"[Defense Eval] Trích xuất một lượt đặc trưng hiệu chuẩn trên {n_cal} ảnh sạch...")
    cal_logits = []
    cal_refs = []
    cal_preds_b = []
    cal_dec_logits = []
    with torch.no_grad():
        for i in range(0, n_cal, bsz):
            end_i = min(i + bsz, n_cal)
            b = clean_imgs[i : end_i]
            if hasattr(b, "to"):
                b = b.to(device)
            out = wrapper.model(b)
            lgt = out.enc_outputs_class if hasattr(out, "enc_outputs_class") and out.enc_outputs_class is not None else out.logits
            r = out.init_reference_points if hasattr(out, "init_reference_points") and out.init_reference_points is not None else out.pred_boxes
            cal_logits.append(lgt.cpu())
            cal_refs.append(r.cpu())
            cal_preds_b.append(out.pred_boxes.cpu())
            cal_dec_logits.append(out.logits.cpu())

    esm_fitted = EncoderSelectionMonitor(target_fpr=target_fpr, num_queries=n_q_model)
    esm_fitted.fit(cal_logits)

    print("\n[Defense Eval] Đánh giá Spatial-Consistency Monitor (SCM) & Box-Count Watchdog...")
    scm = SpatialConsistencyMonitor(target_fpr=target_fpr)
    scm.fit(cal_refs, cal_preds_b)

    box_monitor = ActiveBoxCountMonitor(target_fpr=target_fpr, conf_threshold=0.35, is_sigmoid=is_sigmoid)
    box_monitor.fit(cal_dec_logits)

    # Đánh giá ESM/SCM/Box trên Clean Test (FPR + Ensemble Union Bound)
    print(f"[Defense Eval] Đánh giá FPR trên {n_test} ảnh sạch độc lập...")
    test_clean_enc = []
    test_clean_refs = []
    test_clean_preds_b = []
    test_clean_dec_logits = []
    with torch.no_grad():
        for i in range(0, n_test, bsz):
            start_idx = n_cal + i
            end_idx = min(start_idx + bsz, n_total)
            b = clean_imgs[start_idx : end_idx]
            if hasattr(b, "to"):
                b = b.to(device)
            out = wrapper.model(b)
            lgt = out.enc_outputs_class if hasattr(out, "enc_outputs_class") and out.enc_outputs_class is not None else out.logits
            r = out.init_reference_points if hasattr(out, "init_reference_points") and out.init_reference_points is not None else out.pred_boxes
            test_clean_enc.append(lgt.cpu())
            test_clean_refs.append(r.cpu())
            test_clean_preds_b.append(out.pred_boxes.cpu())
            test_clean_dec_logits.append(out.logits.cpu())

    test_clean_enc_cat = torch.cat(test_clean_enc, dim=0)
    esm_clean_scores = esm_fitted.score(test_clean_enc_cat)
    esm_clean_flags = esm_clean_scores >= esm_fitted.fused_thresh
    esm_clean_fpr = float(np.mean(esm_clean_flags)) * 100.0

    scm_clean_scores = scm.score(torch.cat(test_clean_refs, dim=0), torch.cat(test_clean_preds_b, dim=0))
    scm_clean_flags = scm_clean_scores >= scm.fused_thresh
    scm_clean_fpr = float(np.mean(scm_clean_flags)) * 100.0
    test_clean_dec_cat = torch.cat(test_clean_dec_logits, dim=0)
    box_clean_counts = box_monitor.count_active_boxes(test_clean_dec_cat)
    box_clean_flags = box_monitor.twosided_monitor.predict(box_clean_counts) > 0
    union_clean_fpr = float(np.mean(esm_clean_flags | scm_clean_flags | box_clean_flags)) * 100.0

    defense_results: Dict[str, Any] = {
        "target_fpr_pct": target_fpr * 100.0,
        "empirical_clean_fpr_pct": round(esm_clean_fpr, 2),
        "clean_mean_anomaly_score": round(float(np.mean(esm_clean_scores)), 4),
        "n_cal": n_cal,
        "n_test": n_test,
        "fused_threshold": round(esm_fitted.fused_thresh, 4),
        "union_bound_pct": round(union_clean_fpr, 2),
        "attacks": {},
    }

    scm_results = {
        "target_fpr_pct": target_fpr * 100.0,
        "empirical_clean_fpr_pct": round(scm_clean_fpr, 2),
        "fused_threshold": round(scm.fused_thresh, 4),
        "attacks": {},
    }

    import sklearn.metrics as skm
    test_adv_dec_logits_dict: Dict[str, torch.Tensor] = {}
    for atk_name, adv_full in adv_dict.items():
        test_adv_enc = []
        test_adv_refs = []
        test_adv_preds_b = []
        test_adv_dec_lgt = []
        with torch.no_grad():
            for i in range(0, n_test, bsz):
                start_idx = n_cal + i
                end_idx = min(start_idx + bsz, n_total)
                b = adv_full[start_idx : end_idx]
                if hasattr(b, "to"):
                    b = b.to(device)
                out = wrapper.model(b)
                lgt = out.enc_outputs_class if hasattr(out, "enc_outputs_class") and out.enc_outputs_class is not None else out.logits
                r = out.init_reference_points if hasattr(out, "init_reference_points") and out.init_reference_points is not None else out.pred_boxes
                test_adv_enc.append(lgt.cpu())
                test_adv_refs.append(r.cpu())
                test_adv_preds_b.append(out.pred_boxes.cpu())
                test_adv_dec_lgt.append(out.logits.cpu())

        # Đánh giá ESM trên cùng batch đã suy luận
        scores_adv_esm = esm_fitted.score(torch.cat(test_adv_enc, dim=0))
        tpr_esm = float(np.mean(scores_adv_esm >= esm_fitted.fused_thresh)) * 100.0
        all_lbls = np.concatenate([np.zeros(n_test), np.ones(n_test)])
        try:
            auroc_esm = float(skm.roc_auc_score(all_lbls, np.concatenate([esm_clean_scores, scores_adv_esm])))
        except Exception:
            auroc_esm = 0.5000
        defense_results["attacks"][atk_name] = {
            "tpr_detection_rate_pct": round(tpr_esm, 2),
            "auroc": round(auroc_esm, 4),
            "mean_anomaly_score": round(float(np.mean(scores_adv_esm)), 4),
        }
        print(f"   + [ESM] {atk_name.upper():12s}: TPR = {tpr_esm:5.2f}% | AUROC = {auroc_esm:.4f}")

        test_adv_dec_logits_dict[atk_name] = torch.cat(test_adv_dec_lgt, dim=0)
        scores_adv_scm = scm.score(torch.cat(test_adv_refs, dim=0), torch.cat(test_adv_preds_b, dim=0))
        tpr_scm = float(np.mean(scores_adv_scm >= scm.fused_thresh)) * 100.0
        all_scs = np.concatenate([scm_clean_scores, scores_adv_scm])
        try:
            auroc_scm = float(skm.roc_auc_score(all_lbls, all_scs))
        except Exception:
            auroc_scm = 0.5000

        scm_results["attacks"][atk_name] = {
            "tpr_detection_rate_pct": round(tpr_scm, 2),
            "auroc": round(auroc_scm, 4),
            "mean_anomaly_score": round(float(np.mean(scores_adv_scm)), 4),
        }
        print(f"   + [SCM] {atk_name.upper():12s}: TPR = {tpr_scm:5.2f}% | AUROC = {auroc_scm:.4f}")

    defense_results["spatial_consistency_monitor"] = scm_results
    box_count_eval = box_monitor.evaluate_cohort(
        torch.cat(test_clean_dec_logits, dim=0),
        test_adv_dec_logits_dict,
    )
    defense_results["box_count_monitor"] = box_count_eval
    print(
        f"   + [BoxWatchdog] Clean FPR(upper/2-sided)={box_count_eval['clean_fpr_upper_pct']:.2f}% / {box_count_eval['clean_fpr_twosided_pct']:.2f}% | "
        f"Clean boxes={box_count_eval['clean_box_stats']['mean']:.2f}"
    )
    for atk_k, b_info in box_count_eval["attacks"].items():
        print(
            f"     - {atk_k.upper():10s}: boxes={b_info['box_stats']['mean']:5.2f} +- {b_info['box_stats']['std']:4.2f} "
            f"(p95={b_info['box_stats']['p95']:.1f}, max={b_info['box_stats']['max']:.0f}) | "
            f"TPR(upper)={b_info['tpr_upper_pct']:5.2f}% | TPR(2-sided)={b_info['tpr_twosided_pct']:5.2f}% | AUROC(2-sided)={b_info['auroc_twosided']:.4f}"
        )

    # Đánh giá tấn công thích ứng White-Box Adaptive Evasion
    avail_test = n_actual - n_cal if (n_actual - n_cal) > 0 else n_actual
    n_adapt = min(n_adaptive, avail_test)
    adapt_start = n_cal if (n_actual - n_cal) > 0 else 0
    test_adaptive_imgs = clean_imgs[adapt_start : adapt_start + n_adapt]
    adapt_gt = dataset.gt_list[adapt_start : adapt_start + n_adapt] if hasattr(dataset, "gt_list") else None
    adaptive_results = run_adaptive_evasion_test(
        wrapper=wrapper,
        clean_imgs=test_adaptive_imgs,
        esm_monitor=esm_fitted,
        n_samples=len(test_adaptive_imgs),
        lambdas=[0.2, 0.5, 1.0, 2.0],
        batch_size=bsz,
        gt_list=adapt_gt,
    )
    defense_results["adaptive_evasion"] = adaptive_results

    # Bổ sung thực nghiệm White-Box Adaptive Evasion thật cho SCM, ActiveBoxCountMonitor (BCM) & Joint Ensemble
    adapt_hf = dataset.hf_labels[adapt_start : adapt_start + n_adapt] if getattr(dataset, "hf_labels", None) is not None else None
    defense_results["adaptive_scm_evasion"] = run_adaptive_monitor_suite_test(
        wrapper=wrapper,
        clean_imgs=test_adaptive_imgs,
        target_mode="scm",
        scm_monitor=scm,
        n_samples=len(test_adaptive_imgs),
        lambdas=[0.5, 1.0, 2.0],
        batch_size=bsz,
        gt_list=adapt_gt,
        hf_labels=adapt_hf,
    )
    defense_results["adaptive_bcm_evasion"] = run_adaptive_monitor_suite_test(
        wrapper=wrapper,
        clean_imgs=test_adaptive_imgs,
        target_mode="bcm",
        box_monitor=box_monitor,
        n_samples=len(test_adaptive_imgs),
        lambdas=[0.5, 1.0, 2.0],
        batch_size=bsz,
        gt_list=adapt_gt,
        hf_labels=adapt_hf,
    )
    defense_results["adaptive_ensemble_evasion"] = run_adaptive_monitor_suite_test(
        wrapper=wrapper,
        clean_imgs=test_adaptive_imgs,
        target_mode="ensemble",
        esm_monitor=esm_fitted,
        scm_monitor=scm,
        box_monitor=box_monitor,
        n_samples=len(test_adaptive_imgs),
        lambdas=[1.0, 2.0],
        batch_size=bsz,
        gt_list=adapt_gt,
        hf_labels=adapt_hf,
    )

    # Đánh giá kiến trúc gia cố SQA (Section 6.4 - Table 12)
    print("\n[Defense Eval] Đánh giá Đề xuất Gia cố Kiến trúc SQA (Section 6.4 - Table 12)...")
    sdir_sqa = get_summary_json_dir()
    has_sqa_provenance = (sdir_sqa / "hardening_evaluation_results.json").exists() or (
        sdir_sqa / "sqa_controls_and_adaptive_summary.json"
    ).exists()
    need_live_sqa = run_sqa_ablation or (not has_sqa_provenance)
    empirical_ablation = None
    if need_live_sqa and hasattr(dataset, "gt_list") and len(dataset.gt_list) > 0 and "aqsm" in adv_dict:
        try:
            print("   [Task 8] Đang thực hiện Live GPU Forward Pass cho SQA Ablation...")
            empirical_ablation = evaluate_sqa_empirical_ablation(
                model_wrapper=wrapper,
                clean_imgs=clean_imgs,
                adv_imgs=adv_dict["aqsm"],
                gt_list=dataset.gt_list,
                batch_size=bsz,
                adv_taskpgd_imgs=adv_dict.get("taskpgd"),
                adaptive_steps=get_pgd_num_steps(),
                n_adaptive_samples=n_adapt,
            )
        except Exception as e:
            print(f"   [CẢNH BÁO] Không thể chạy live SQA ablation ({e}).")

    if empirical_ablation is not None or has_sqa_provenance:
        sqa_results = evaluate_sqa_hardening(
            empirical_ablation=empirical_ablation,
            summary_dir=sdir_sqa,
            model_name=wrapper.pretrained_model_name,
        )
    else:
        print("   [CẢNH BÁO] Chưa có SQA provenance và live ablation không khả dụng; ghi kết quả trạng thái pending.")
        sqa_results = {
            "experiment": "Section 7.2: Empirical Validation of Architectural Hardening via Static Query Augmentation",
            "model": wrapper.pretrained_model_name,
            "status": "pending_sqa_controls_run",
            "results": {},
        }
    defense_results["sqa_hardening"] = sqa_results

    summary_json_dir = get_summary_json_dir()
    summary_json_dir.mkdir(parents=True, exist_ok=True)

    # Lưu kết quả JSON Table 11
    summary_path = summary_json_dir / "defense_esm_evaluation.json"
    save_json(defense_results, summary_path)
    print(f"\n>>> Đã lưu kết quả thực nghiệm phòng thủ Table 11 ra: {summary_path}")

    # Lưu alias tương thích ngược defense_evaluation_summary.json
    summary_path_alias = summary_json_dir / "defense_evaluation_summary.json"
    save_json(defense_results, summary_path_alias)
    print(f">>> Đã lưu alias Table 11 ra: {summary_path_alias}")

    hardening_path = summary_json_dir / "hardening_evaluation_results.json"
    save_hardening_results(sqa_results, hardening_path)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return defense_results


def main():
    parser = create_base_parser("Empirical Defense & Adaptive Evasion Evaluation (Table 11)", task_name="Task 8")
    parser.add_argument("--cal_ratio", type=float, default=0.3, help="Tỷ lệ tập calibration không chồng lấn (mặc định 30%%)")
    parser.add_argument("--n_adaptive", type=int, default=1000, help="Số lượng ảnh kiểm thử tấn công thích ứng (mặc định 1000 để chạy toàn bộ tập test)")
    parser.add_argument("--run_sqa_ablation", action="store_true", help="Chạy trực tiếp Forward Pass thực tế qua Deformable Decoder trên GPU để đánh giá SQA và các controls")
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

    audit_logger = ExperimentAuditLogger(task_name="task8_defense_evaluation")
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
        "threads": args.threads,
        "device": str(device),
        "hardware_info": hw_info,
        "random_seed": RANDOM_SEED,
        "model_name": chosen_model,
        "defense_name": "EncoderSelectionMonitor (ESM)",
        "max_samples": args.max_samples,
    })

    defense_results = run_defense_evaluation(
        wrapper=wrapper,
        dataset=dataset,
        cal_ratio=args.cal_ratio,
        target_fpr=0.05,
        n_adaptive=args.n_adaptive,
        cache_dir=args.cache_dir,
        audit_logger=audit_logger,
        run_sqa_ablation=args.run_sqa_ablation,
        batch_size=batch_size,
    )

    audit_logger.log_step(
        step_name="Defense_Evaluation_Complete",
        step_idx=1,
        total_steps=1,
        metrics={
            "clean_fpr": defense_results["empirical_clean_fpr_pct"],
            "aqsm_tpr": defense_results["attacks"]["aqsm"]["tpr_detection_rate_pct"],
            "vanishing_tpr": defense_results["attacks"]["vanishing"]["tpr_detection_rate_pct"],
            "task_pgd_tpr": defense_results["attacks"]["taskpgd"]["tpr_detection_rate_pct"],
        },
    )
    audit_logger.close()
    del dataset, wrapper
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    print("\n[HOÀN THÀNH TOÀN BỘ] Task 8: Defense Evaluation thành công!")


if __name__ == "__main__":
    main()
