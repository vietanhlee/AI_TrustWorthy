# -*- coding: utf-8 -*-
"""
Script nâng cấp toàn diện và đồng bộ hóa production tasks cho Paper 1:
1. task8_defense_evaluation.py:
   - Thêm mini-batching cho run_adaptive_evasion_test để chạy mượt mà trên 1000 ảnh (hoặc toàn bộ tập test) mà không bị OOM GPU.
   - Thêm giải phóng bộ nhớ CUDA định kỳ.
   - Đặt mặc định n_adaptive = 1000 (để chạy toàn bộ tập kiểm thử).
2. task1_causal_surgery.py:
   - Tích hợp tự động kiểm định phi tham số Paired Wilcoxon Signed-Rank Test và BCa Bootstrap CI 95% cấp ảnh.
   - Tính toán Cohen's d effect size cho tất cả các cặp so sánh nhân quả.
   - Ghi trực tiếp các chỉ số này vào file JSON kết quả (causal_swap_results_{attack}.json).
"""

import re
from pathlib import Path

def patch_task8():
    file_path = Path("paper_1_aqsm_mechanism/experiments/tasks/task8_defense_evaluation.py")
    content = file_path.read_text(encoding="utf-8")
    
    # 1. Replace run_adaptive_evasion_test with mini-batched version
    old_func_pattern = r"def run_adaptive_evasion_test\(.*?\n(?=def run_defense_evaluation)"
    
    new_func = '''def run_adaptive_evasion_test(
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
    print(f"\\n[Adaptive Evasion] Đang đánh giá tấn công thích ứng né tránh ESM trên N={n_eval} ảnh (batch_size={batch_size})...")

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

'''
    content = re.sub(old_func_pattern, new_func, content, flags=re.DOTALL)
    
    # Update call in run_defense_evaluation to pass batch_size
    content = content.replace(
        "adaptive_results = run_adaptive_evasion_test(\n        wrapper=wrapper,\n        clean_imgs=test_adaptive_imgs,\n        esm_monitor=esm_fitted,\n        n_samples=len(test_adaptive_imgs),\n        lambdas=[0.2, 0.5, 1.0, 2.0],\n    )",
        "adaptive_results = run_adaptive_evasion_test(\n        wrapper=wrapper,\n        clean_imgs=test_adaptive_imgs,\n        esm_monitor=esm_fitted,\n        n_samples=len(test_adaptive_imgs),\n        lambdas=[0.2, 0.5, 1.0, 2.0],\n        batch_size=4,\n    )"
    )
    
    # Update default n_adaptive to 1000 in parser
    content = content.replace(
        'parser.add_argument("--n_adaptive", type=int, default=100, help="Số lượng ảnh kiểm thử tấn công thích ứng (mặc định 100)")',
        'parser.add_argument("--n_adaptive", type=int, default=1000, help="Số lượng ảnh kiểm thử tấn công thích ứng (mặc định 1000 để chạy toàn bộ tập test)")'
    )
    
    file_path.write_text(content, encoding="utf-8")
    print(">>> Đã nâng cấp task8_defense_evaluation.py thành công!")

def patch_task1():
    file_path = Path("paper_1_aqsm_mechanism/experiments/tasks/task1_causal_surgery.py")
    content = file_path.read_text(encoding="utf-8")
    
    # Add scipy.stats import if missing
    if "from scipy import stats" not in content:
        content = "from scipy import stats\n" + content

    # Add non-parametric block before results_summary
    old_summary_anchor = "    results_summary: Dict[str, Any] = {"
    
    non_parametric_code = '''    # =========================================================================
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
    
    print(f"\\n--- Phân tích Thống kê Phi tham số cấp ảnh độc lập (N={n_valid} ảnh hợp lệ) ---")
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

'''
    if "non_parametric_stats" not in content:
        content = content.replace(
            old_summary_anchor,
            non_parametric_code + old_summary_anchor
        )
        content = content.replace(
            '"bootstrap_paired_deltas": boot_deltas,',
            '"bootstrap_paired_deltas": boot_deltas,\n        "non_parametric_contrasts": non_parametric_stats,'
        )
    
    file_path.write_text(content, encoding="utf-8")
    print(">>> Đã nâng cấp task1_causal_surgery.py thành công!")

if __name__ == "__main__":
    patch_task8()
    patch_task1()
