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
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from paper_1_aqsm_mechanism.core.adv_cache import load_cached_adv_images
from paper_1_aqsm_mechanism.core.dataset_loader import FlexibleCOCODataset
from paper_1_aqsm_mechanism.core.defense_engine import (
    EncoderSelectionMonitor,
    evaluate_defense_cohort,
)
from paper_1_aqsm_mechanism.core.detr_wrapper import RTDETRModelWrapper
from paper_1_aqsm_mechanism.experiments.audit_logger import ExperimentAuditLogger
from paper_1_aqsm_mechanism.experiments.config import (
    MODEL_NAME,
    RANDOM_SEED,
    configure_cpu_threads,
    detect_hardware_environment,
    get_device,
    get_summary_json_dir,
    update_experiment_paths,
)
from paper_1_aqsm_mechanism.utils.cli import create_base_parser
from paper_1_aqsm_mechanism.utils.io_helpers import save_json


def run_adaptive_evasion_test(
    wrapper: RTDETRModelWrapper,
    clean_imgs: Any,
    esm_monitor: EncoderSelectionMonitor,
    n_samples: int = 15,
    lambdas: List[float] = [0.2, 0.5, 1.0, 2.0],
    batch_size: int = 4,
) -> Dict[str, Any]:
    """
    Thực nghiệm tấn công thích ứng White-Box Adaptive PGD (Carlini-Tramèr):
    Kẻ tấn công vừa dìm detection mAP vừa tối ưu để né tránh ngưỡng phát hiện của ESM.
    Hỗ trợ xử lý theo lô (mini-batching) chống tràn VRAM khi chạy quy mô lớn (lên tới 1,000 ảnh).
    """
    model = wrapper.model
    device = wrapper.device
    n_eval = min(n_samples, len(clean_imgs))
    target_drop = float(getattr(esm_monitor, "cal_dropoff_median", 35.0))
    target_margin = float(getattr(esm_monitor, "cal_margin_median", 0.05))

    results = {}
    print(f"\n[Adaptive Evasion] Đang đánh giá tấn công thích ứng né tránh ESM trên N={n_eval} ảnh (batch_size={batch_size})...")

    for lam in lambdas:
        all_preds = []
        t_start = time.time()
        for b_start in range(0, n_eval, batch_size):
            b_end = min(n_eval, b_start + batch_size)
            sub_clean = clean_imgs[b_start:b_end]
            if hasattr(sub_clean, "to"):
                sub_clean = sub_clean.to(device)

            adv = sub_clean.clone().detach()
            step_alpha = 2.0 / 255.0
            eps = 16.0 / 255.0

            # Chu trình PGD 15 bước chuẩn hóa (khớp với Stationary Attack)
            for _ in range(15):
                adv_var = adv.clone().detach().requires_grad_(True)
                mask = torch.ones((adv_var.shape[0], adv_var.shape[2], adv_var.shape[3]), device=device, dtype=torch.long)
                out = model(pixel_values=adv_var, pixel_mask=mask)
                enc_logits = out.enc_outputs_class if hasattr(out, "enc_outputs_class") and out.enc_outputs_class is not None else out.logits
                probs = torch.sigmoid(enc_logits)
                max_probs, _ = probs.max(dim=-1)

                # Task Loss (Dìm điểm các query có xác suất cao)
                k = min(15, max_probs.shape[1])
                top_probs, _ = torch.topk(max_probs, k=k, dim=-1)
                task_loss = (top_probs ** 2).mean()

                # Evasion Loss (Né tránh đồng thời cả Dropoff và Margin theo đúng nguyên lý Max-OR)
                topk_vals, _ = torch.topk(max_probs, k=min(301, max_probs.shape[1]), dim=-1)
                top1_val = topk_vals[:, 0]
                top300_val = topk_vals[:, 299]
                top301_val = topk_vals[:, 300] if topk_vals.shape[1] > 300 else torch.zeros_like(top300_val)

                diff_dropoff = top1_val / (top300_val.clamp(min=1e-4) + 1e-4)
                diff_margin = top300_val - top301_val

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
                    delta = torch.clamp(adv - sub_clean, min=-eps, max=eps)
                    adv = torch.clamp(sub_clean + delta, min=0.0, max=1.0).detach()

            with torch.no_grad():
                final_out = model(adv)
                final_logits = final_out.enc_outputs_class if hasattr(final_out, "enc_outputs_class") and final_out.enc_outputs_class is not None else final_out.logits
                preds = esm_monitor.predict(final_logits.cpu())
                all_preds.extend(preds)

            del adv, sub_clean
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        tpr_evade = float(np.mean(all_preds)) * 100.0 if len(all_preds) > 0 else 0.0
        elapsed = time.time() - t_start
        results[f"lambda_{lam}"] = {
            "evasion_weight_lambda": lam,
            "n_evaluated_samples": n_eval,
            "detected_tpr_pct": round(tpr_evade, 2),
            "evasion_success_pct": round(100.0 - tpr_evade, 2),
            "elapsed_seconds": round(elapsed, 2),
        }
        print(f"   + Lambda = {lam:3.1f} (N={n_eval}) | Detection TPR = {tpr_evade:5.1f}% | Evasion Success = {100.0 - tpr_evade:5.1f}% ({elapsed:.1f}s)")

    return results

def run_defense_evaluation(
    wrapper: RTDETRModelWrapper,
    dataset: Any,
    cal_ratio: float = 0.3,
    target_fpr: float = 0.05,
    n_adaptive: int = 50,
    cache_dir: Optional[Path] = None,
    audit_logger: Any = None,
) -> Dict[str, Any]:
    """
    Thực hiện đánh giá toàn diện cơ chế phòng thủ ESM trên tập ảnh kiểm thử.
    """
    clean_imgs = dataset.tensors
    image_ids = dataset.image_ids
    n_actual = len(dataset)
    device = wrapper.device

    print("\n==================================================================")
    print(f" [Defense Evaluation Engine] Đánh giá Hệ thống Phòng thủ ESM & Guardrail")
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
        )
        adv_dict[atk] = adv_imgs

    # Xác định n_cal theo tỷ lệ hoặc tối đa 1000 ảnh
    n_cal = min(1000, max(50, int(n_actual * cal_ratio)))
    if n_actual <= 100:
        n_cal = max(10, n_actual // 2)

    # Đánh giá TPR @ 5% FPR trên Clean vs Adversarial
    defense_results = evaluate_defense_cohort(
        model_wrapper=wrapper,
        clean_tensors=clean_imgs,
        adv_tensors_dict=adv_dict,
        target_fpr=target_fpr,
        n_cal=n_cal,
    )

    # Khởi tạo ESM đã fit trên n_cal ảnh đầu để test adaptive evasion
    cal_logits = []
    with torch.no_grad():
        for i in range(0, n_cal, 10):
            end_i = min(i + 10, n_cal)
            b = clean_imgs[i : end_i]
            if hasattr(b, "to"):
                b = b.to(device)
            out = wrapper.model(b)
            lgt = out.enc_outputs_class if hasattr(out, "enc_outputs_class") and out.enc_outputs_class is not None else out.logits
            cal_logits.append(lgt.cpu())

    esm_fitted = EncoderSelectionMonitor(target_fpr=target_fpr)
    esm_fitted.fit(cal_logits)

    # Đánh giá tấn công thích ứng White-Box Adaptive Evasion
    avail_test = n_actual - n_cal if (n_actual - n_cal) > 0 else n_actual
    n_adapt = min(n_adaptive, avail_test)
    test_adaptive_imgs = clean_imgs[n_cal : n_cal + n_adapt]
    adaptive_results = run_adaptive_evasion_test(
        wrapper=wrapper,
        clean_imgs=test_adaptive_imgs,
        esm_monitor=esm_fitted,
        n_samples=len(test_adaptive_imgs),
        lambdas=[0.2, 0.5, 1.0, 2.0],
        batch_size=4,
    )
    defense_results["adaptive_evasion"] = adaptive_results

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

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return defense_results


def main():
    parser = create_base_parser("Empirical Defense & Adaptive Evasion Evaluation (Table 11)", task_name="Task 8")
    parser.add_argument("--cal_ratio", type=float, default=0.3, help="Tỷ lệ tập calibration không chồng lấn (mặc định 30%%)")
    parser.add_argument("--n_adaptive", type=int, default=1000, help="Số lượng ảnh kiểm thử tấn công thích ứng (mặc định 1000 để chạy toàn bộ tập test)")
    args = parser.parse_args()

    update_experiment_paths(
        folder_image=args.folder_image,
        annotations=args.annotations,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
    )
    configure_cpu_threads(args.threads)
    hw_info = detect_hardware_environment(preferred_device=args.device)
    device = get_device(preferred=args.device)
    batch_size = args.batch_size if args.batch_size is not None else hw_info["recommended_batch_size"]

    audit_logger = ExperimentAuditLogger(task_name="task8_defense_evaluation")
    wrapper = RTDETRModelWrapper(pretrained_model_name=MODEL_NAME, device=device)
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

    audit_logger.log_metadata({
        "folder_image": str(args.folder_image),
        "total_images": len(dataset),
        "batch_size": batch_size,
        "threads": args.threads,
        "device": str(device),
        "hardware_info": hw_info,
        "random_seed": RANDOM_SEED,
        "model_name": MODEL_NAME,
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
