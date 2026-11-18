# -*- coding: utf-8 -*-
"""
Bước 3: Đánh giá Tấn công Adversarial Patch (48x48) & Quét Kích Thước (Sweep 16, 32, 48, 64)
Bao gồm:
1. Huấn luyện (Optimize) Gradient Patch bằng Adam EoT (Expectation over Transformation).
   -> Ghi vết chi tiết từng Epoch / Step (Loss, Target Confidence, LR, Elapsed time)
      ra file CSV lịch sử huấn luyện (phục vụ vẽ biểu đồ học tập mà không cần chạy lại).
2. Đánh giá đòn tấn công Patch 48x48 trên tập test độc lập (N=300) ghép cặp với Clean Test.
3. Quét kích thước Patch (16x16, 32x32, 48x48, 64x64) để làm rõ Spatial Attention Gap (K=4).
4. Xuất bảng kiểm toán từng mẫu (Sample Audit CSV) và JSON metrics kèm tọa độ đường cong ROC.
"""

import os
import sys
import time
import pickle
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
from tqdm import tqdm

# Đảm bảo import đúng workspace
EXPERIMENTS_DIR = Path(__file__).resolve().parent
PAPER3_DIR = EXPERIMENTS_DIR.parent
WORKSPACE_DIR = PAPER3_DIR.parent

if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))
if str(WORKSPACE_DIR / "edge_sec_pro") not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR / "edge_sec_pro"))

from paper_3_defense_guardrail.experiments.common_exp import (
    load_coco_images,
    clopper_pearson_interval,
    compute_auroc_and_bootstrap_ci,
    calculate_task_level_asr,
    make_serializable,
    RTDETRModelWrapper,
    RTDETRDeepFeatureExtractor,
    DETRAttackGenerator,
    CHECKPOINTS_DIR,
)

from paper_3_defense_guardrail.experiments.exp_logger import (
    setup_experiment_logger,
    MetricHistoryTracker,
    SampleAuditTracker,
    save_audit_json,
    compute_roc_coordinates,
    compute_confusion_stats,
    PAPER3_DIR,
    RESULTS_DIR,
)


def train_patch_with_logging(
    wrapper: RTDETRModelWrapper,
    cal_images: torch.Tensor,
    patch_size: int = 48,
    target_class_idx: int = 1,
    num_epochs: int = 4,
    lr: float = 0.05,
    history_tracker: MetricHistoryTracker = None,
    logger = None,
) -> torch.Tensor:
    """
    Huấn luyện Adversarial Patch với tính năng ghi log chi tiết từng epoch và step.
    Ghi vết trực tiếp ra file CSV thông qua history_tracker.
    """
    device = wrapper.device
    B_cal, C, H, W = cal_images.shape
    patch = torch.full((1, C, patch_size, patch_size), 0.5, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([patch], lr=lr)

    batch_size = min(16, B_cal)
    num_batches = max(1, B_cal // batch_size)
    total_steps = num_epochs * num_batches
    rng = np.random.RandomState(42)

    step_counter = 0
    if logger:
        logger.info(f"Bắt đầu tối ưu Gradient Patch ({patch_size}x{patch_size}) trên {B_cal} ảnh Calibration:")
        logger.info(f"   -> Epochs = {num_epochs}, Batch Size = {batch_size}, LR = {lr}, Total Steps = {total_steps}")

    for epoch in range(1, num_epochs + 1):
        perm = torch.randperm(B_cal)
        epoch_losses = []

        for b_idx in tqdm(range(num_batches), desc=f"  Tối ưu Patch Epoch {epoch}/{num_epochs}", unit="batch", dynamic_ncols=True):
            step_counter += 1
            indices = perm[b_idx * batch_size : (b_idx + 1) * batch_size]
            batch_imgs = cal_images[indices].clone().to(device)
            cur_B = len(indices)

            # Áp dụng Expectation over Transformation (EoT): dán patch tại tọa độ ngẫu nhiên
            for b in range(cur_B):
                max_y = max(0, H - patch_size)
                max_x = max(0, W - patch_size)
                py = int(rng.randint(0, max_y + 1)) if max_y > 0 else 0
                px = int(rng.randint(0, max_x + 1)) if max_x > 0 else 0
                batch_imgs[b, :, py : py + patch_size, px : px + patch_size] = patch[0]

            outputs = wrapper.model(pixel_values=batch_imgs, return_dict=True)
            logits = outputs.logits  # [B, N, C_out]

            target_prob = torch.sigmoid(logits[..., target_class_idx])
            k_top = min(3, target_prob.shape[1])
            top_probs, _ = torch.topk(target_prob, k=k_top, dim=-1)
            loss = -torch.log(top_probs + 1e-6).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                patch.clamp_(0.0, 1.0)

            loss_val = float(loss.item())
            epoch_losses.append(loss_val)

            # Ghi vết từng bước huấn luyện
            if history_tracker:
                mean_target_conf = float(top_probs.mean().item())
                history_tracker.log_step(
                    epoch=epoch,
                    step=step_counter,
                    total_steps=total_steps,
                    loss=loss_val,
                    task_loss=loss_val,
                    lr=lr,
                    extra_metrics={"mean_target_confidence": round(mean_target_conf, 4)},
                    logger=logger if (b_idx == num_batches - 1 or step_counter % 2 == 0) else None,
                )

        if logger:
            logger.info(f"   [Epoch {epoch}/{num_epochs} Done] Mean Loss = {np.mean(epoch_losses):.5f}")

    return patch.detach()


def run_step3():
    logger, log_file = setup_experiment_logger("step3_patch_attack")
    logger.info("BƯỚC 3: ĐÁNH GIÁ ADVERSARIAL PATCH (48x48) & QUÉT KÍCH THƯỚC (SWEEP 16..64)")

    ckpt_file = CHECKPOINTS_DIR / "calibration_and_detectors.pkl"
    if not ckpt_file.exists():
        err_msg = f"Chưa chạy Bước 1! Không tìm thấy checkpoint: {ckpt_file}"
        logger.error(err_msg)
        raise FileNotFoundError(err_msg)

    with open(ckpt_file, "rb") as f:
        ckpt = pickle.load(f)

    detectors = ckpt["detectors"]
    cal_thresholds = ckpt["cal_thresholds"]
    clean_test_scores = {k: np.array(v) for k, v in ckpt["clean_test_scores"].items()}

    device = torch.device("cpu")
    logger.info(f"Khởi tạo phần cứng: Device = {device}, PyTorch Threads = {torch.get_num_threads()}")

    wrapper = RTDETRModelWrapper(pretrained_model_name="PekingU/rtdetr_r18vd", device=device)
    wrapper.model.eval()
    extractor = RTDETRDeepFeatureExtractor(conf_threshold=0.35, top_k_queries=10, projection_dim=64)
    attacker = DETRAttackGenerator(wrapper, epsilon=16.0 / 255.0, alpha=2.0 / 255.0, num_steps=15)

    cal_imgs, test_imgs = load_coco_images(n_cal=1200, n_test=600, img_size=320, seed=42)
    N_eval = 300
    eval_imgs = test_imgs[:N_eval]
    logger.info(f"Nạp dữ liệu COCO: N_cal={len(cal_imgs)}, N_test_eval={N_eval} (ghép cặp 1-1 với clean test).")

    # 1. Tối ưu hóa Gradient Patch 48x48 với Tracking chi tiết
    logger.info("\n1. BẮT ĐẦU HUẤN LUYỆN ADVERSARIAL PATCH 48x48 (Adam EoT trên 60 ảnh Calib)...")
    history_tracker = MetricHistoryTracker("step3_patch_optimization", output_dir=RESULTS_DIR)
    t0_opt = time.time()
    patch_48 = train_patch_with_logging(
        wrapper=wrapper,
        cal_images=cal_imgs[:60],
        patch_size=48,
        target_class_idx=1,
        num_epochs=4,
        lr=0.05,
        history_tracker=history_tracker,
        logger=logger,
    )
    t_opt_dur = time.time() - t0_opt
    history_csv_path = history_tracker.save_csv()
    logger.info(f"   -> Hoàn tất huấn luyện patch 48x48 trong {t_opt_dur:.2f}s.")
    logger.info(f"   -> Lịch sử từng epoch/step đã lưu tại: {history_csv_path}")

    # 2. Đánh giá trên tập kiểm thử độc lập (N=300)
    logger.info(f"\n2. ĐÁNH GIÁ TẤN CÔNG PATCH 48x48 TRÊN {N_eval} ẢNH KIỂM THỬ ĐỘC LẬP...")
    adv_patch_imgs = attacker.apply_adversarial_patch(eval_imgs, patch_48, patch_loc="random", seed=1337)

    batch_size = 20
    adv_feats = []
    adv_logits_l = []
    clean_logits_l = []

    for i in tqdm(range(0, N_eval, batch_size), desc="  Trích xuất Patch 48x48", unit="batch", dynamic_ncols=True):
        b_clean = eval_imgs[i : i + batch_size].to(device)
        b_adv = adv_patch_imgs[i : i + batch_size].to(device)
        with torch.no_grad():
            st_c = wrapper(b_clean)
            clean_logits_l.append(st_c.logits.cpu())
            st_a = wrapper(b_adv)
            adv_feats.extend(extractor.extract_from_state(st_a))
            adv_logits_l.append(st_a.logits.cpu())
            del st_c, st_a

    clean_logits = torch.cat(clean_logits_l, dim=0)
    adv_logits = torch.cat(adv_logits_l, dim=0)

    asr, succ_cnt, valid_cnt, succ_mask = calculate_task_level_asr("Patch", clean_logits, adv_logits, target_class_idx=1)
    asr_ci = clopper_pearson_interval(succ_cnt, valid_cnt)
    logger.info(f"   -> Task-level Attack Success Rate (ASR) = {asr*100:5.1f}% [{asr_ci[0]*100:4.1f}%, {asr_ci[1]*100:4.1f}%] ({succ_cnt}/{valid_cnt})")

    # Đánh giá các bộ giám sát
    adv_scores = {name: det.score(adv_feats) for name, det in detectors.items()}
    method_metrics: Dict[str, Any] = {}
    roc_curves: Dict[str, Any] = {}

    logger.info("\n   KẾT QUẢ ĐÁNH GIÁ BỘ GIÁM SÁT TRƯỚC ĐÒN GRADIENT PATCH (48x48):")
    logger.info("   " + "-" * 90)
    logger.info(f"   {'Detector':<20} | {'AUROC':<10} | {'95% AUROC CI':<20} | {'TPR @ 5% FPR':<14} | {'95% Clopper-Pearson CI':<24}")
    logger.info("   " + "-" * 90)

    for m_name, det in detectors.items():
        s_neg = clean_test_scores[m_name]
        s_pos = adv_scores[m_name]
        labels = np.concatenate([np.zeros(len(s_neg)), np.ones(len(s_pos))])
        all_scores = np.concatenate([s_neg, s_pos])

        auroc, auroc_ci = compute_auroc_and_bootstrap_ci(labels, all_scores)
        roc_coords = compute_roc_coordinates(labels, all_scores, num_points=100)
        roc_curves[m_name] = roc_coords

        if m_name in ["ActiveQ_Split", "Top1Mean_Split", "ESM_Dropoff", "ESM_Margin"]:
            preds = det.predict(adv_feats)
        else:
            preds = (s_pos >= cal_thresholds[m_name]).astype(int)

        tp_cnt = int(np.sum(preds))
        tpr = tp_cnt / float(N_eval)
        tpr_ci = clopper_pearson_interval(tp_cnt, N_eval)

        # Tính Confusion Matrix
        y_true_patch = np.ones(N_eval, dtype=int)
        conf = compute_confusion_stats(y_true_patch, preds)

        method_metrics[m_name] = {
            "auroc": round(float(auroc), 4),
            "auroc_ci": [round(float(auroc_ci[0]), 4), round(float(auroc_ci[1]), 4)],
            "tpr": round(float(tpr), 5),
            "tpr_ci": [round(float(tpr_ci[0]), 5), round(float(tpr_ci[1]), 5)],
            "tp_count": tp_cnt,
            "total_samples": N_eval,
            "threshold": round(float(cal_thresholds[m_name]), 4),
            "confusion": conf,
        }

        ci_auroc_str = f"[{auroc_ci[0]:.3f}, {auroc_ci[1]:.3f}]"
        ci_tpr_str = f"[{tpr_ci[0]*100:4.1f}%, {tpr_ci[1]*100:4.1f}%]"
        logger.info(f"   {m_name:<20} | {auroc:5.3f}      | {ci_auroc_str:<20} | {tpr*100:5.1f}%        | {ci_tpr_str:<24}")

    logger.info("   " + "-" * 90)

    # 3. Ghi vết từng mẫu (Sample Audit CSV)
    audit_tracker = SampleAuditTracker("step3_patch_attack", output_dir=RESULTS_DIR)
    for idx in range(N_eval):
        s_dict = {name: float(adv_scores[name][idx]) for name in detectors}
        p_dict = {
            name: int(det.predict([adv_feats[idx]])[0]) if name in ["ActiveQ_Split", "Top1Mean_Split", "ESM_Dropoff", "ESM_Margin"]
            else int(adv_scores[name][idx] >= cal_thresholds[name])
            for name, det in detectors.items()
        }
        is_succ = int(succ_mask[idx]) if succ_mask is not None and idx < len(succ_mask) else -1
        audit_tracker.record_sample(
            sample_idx=idx,
            condition="gradient_patch_48x48",
            ground_truth=1,
            scores=s_dict,
            preds=p_dict,
            is_attack_success=is_succ,
            extra_info={
                "active_query_soft": float(adv_feats[idx].active_query_soft),
                "query_top1_mean": float(adv_feats[idx].query_top1_mean),
                "enc_dropoff": float(adv_feats[idx].enc_dropoff),
                "enc_margin": float(adv_feats[idx].enc_margin),
            }
        )
    samples_csv_path = audit_tracker.save_csv()
    logger.info(f"   -> Đã lưu bảng kiểm toán từng mẫu (Sample Audit CSV): {samples_csv_path}")

    # 4. Quét kích thước Patch (Sweep 16, 32, 48, 64)
    logger.info("\n3. BẮT ĐẦU QUÉT KÍCH THƯỚC PATCH (SWEEP SIZES = [16, 32, 48, 64])...")
    sweep_sizes = [16, 32, 48, 64]
    sweep_results = {}
    N_sweep = 100
    sweep_imgs = eval_imgs[:N_sweep]

    for s in sweep_sizes:
        logger.info(f"   -> Đang đánh giá kích thước patch {s}x{s} (N={N_sweep})...")
        p_s = attacker.optimize_adversarial_patch(cal_imgs[:40], patch_size=s, num_epochs=2)
        adv_s = attacker.apply_adversarial_patch(sweep_imgs, p_s, patch_loc="random", seed=1337)

        s_feats = []
        for i in tqdm(range(0, N_sweep, batch_size), desc=f"    Trích xuất Patch {s}x{s}", unit="batch", dynamic_ncols=True, leave=False):
            b = adv_s[i : i + batch_size].to(device)
            with torch.no_grad():
                st = wrapper(b)
                s_feats.extend(extractor.extract_from_state(st))
                del st

        s_scores = {name: det.score(s_feats) for name, det in detectors.items()}
        s_tprs = {}
        for m_name, det in detectors.items():
            if m_name in ["ActiveQ_Split", "Top1Mean_Split", "ESM_Dropoff", "ESM_Margin"]:
                p = det.predict(s_feats)
            else:
                p = (s_scores[m_name] >= cal_thresholds[m_name]).astype(int)
            s_tprs[m_name] = round(float(np.mean(p)), 5)

        area_pct = round((s * s) / (320 * 320) * 100.0, 2)
        sweep_results[f"size_{s}"] = {
            "patch_dimension": s,
            "area_percentage": area_pct,
            "tpr_per_detector": s_tprs,
        }
        logger.info(f"      Size {s}x{s} ({area_pct}% diện tích) -> Calibrated Max-OR TPR = {s_tprs['Calibrated_Max_OR']*100:4.1f}% | Layer Maha = {s_tprs['Layer_Maha']*100:4.1f}%")

    # 5. Xuất kết quả JSON chuẩn
    res_json_paper3 = RESULTS_DIR / "step3_patch_attack_metrics.json"
    res_json_global = CHECKPOINTS_DIR / "res_step3_patch_attack.json"

    export_payload = {
        "experiment": "step3_patch_attack",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {"device": str(device), "num_threads": torch.get_num_threads()},
        "eval_cohort_size": N_eval,
        "patch_main_size": 48,
        "task_asr": {
            "asr": round(float(asr), 5),
            "ci_95": [round(float(asr_ci[0]), 5), round(float(asr_ci[1]), 5)],
            "success_count": int(succ_cnt),
            "valid_count": int(valid_cnt),
        },
        "detectors_eval": method_metrics,
        "patch_size_sweep": sweep_results,
        "roc_curves": roc_curves,
        "artifacts": {
            "optimization_history_csv": str(history_csv_path),
            "samples_csv": str(samples_csv_path),
            "log_file": str(log_file),
        }
    }

    save_audit_json(res_json_paper3, export_payload)
    save_audit_json(res_json_global, export_payload)

    logger.info(f"\n-> Đã lưu Audit JSON (Paper 3) : {res_json_paper3}")
    logger.info(f"-> Đã đồng bộ JSON toàn dự án  : {res_json_global}")
    logger.info("\n>>> BƯỚC 3 ĐÃ HOÀN TẤT THÀNH CÔNG VỚI ĐẦY ĐỦ DỮ LIỆU KIỂM TOÁN! <<<")


if __name__ == "__main__":
    run_step3()
