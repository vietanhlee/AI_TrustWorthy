# -*- coding: utf-8 -*-
"""
Bước 4: Đánh giá Tấn công Adversarial Query Selection Manipulation (AQSM) & Tuyến phòng thủ ESM
Bao gồm:
1. Đánh giá hiện tượng Bỏ đói thông tin (Information Starvation) tại cổng Encoder.
2. Đo đạc độ lệch phân phối của bộ chỉ số ESM (ESM_Dropoff và ESM_Margin).
3. Đánh giá tỷ lệ phát hiện (TPR @ 5% FPR) và AUROC của toàn bộ các bộ giám sát (ActiveQ, Top1Mean, ESM, Maha, Max-OR).
4. Xuất log chi tiết, file CSV mẫu kiểm toán và file JSON metrics kèm tọa độ ROC.
"""

import os
import sys
import time
import pickle
from pathlib import Path
from typing import Dict, Any, List
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
    SampleAuditTracker,
    save_audit_json,
    compute_roc_coordinates,
    compute_confusion_stats,
    PAPER3_DIR,
    RESULTS_DIR,
)


def run_step4():
    logger, log_file = setup_experiment_logger("step4_aqsm_and_esm")
    logger.info("BƯỚC 4: ĐÁNH GIÁ AQSM & HIỆU NĂNG ENCODER SELECTION MONITOR (ESM) (N=300)")

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

    _, test_imgs = load_coco_images(n_cal=1200, n_test=600, img_size=640, seed=42)
    N_pgd = 300
    eval_imgs = test_imgs[:N_pgd]
    logger.info(f"Nạp dữ liệu COCO: Lấy N={N_pgd} ảnh kiểm thử độc lập.")

    # 1. Clean Baseline Forward Pass
    logger.info(f"\n1. Chạy Clean Forward Pass trên {N_pgd} ảnh...")
    batch_size = 15
    clean_logits_l = []
    clean_feats = []
    t0_clean = time.time()

    for i in range(0, N_pgd, batch_size):
        b = eval_imgs[i : i + batch_size].to(device)
        with torch.no_grad():
            st = wrapper(b)
            clean_feats.extend(extractor.extract_from_state(st))
            clean_logits_l.append(st.logits.cpu())
            del st
    clean_logits = torch.cat(clean_logits_l, dim=0)
    logger.info(f"   -> Clean Pass hoàn tất trong {time.time() - t0_clean:.2f}s.")

    # 2. Sinh hoặc Nạp đòn tấn công AQSM
    adv_cache_file = CHECKPOINTS_DIR / "adv_cache" / f"adv_cache_aqsm_eps16_n{N_pgd}.npz"
    if adv_cache_file.exists():
        logger.info(f"\n2. Tìm thấy Precomputed Adversarial Cache tại: {adv_cache_file.name}")
        data_cache = np.load(adv_cache_file, allow_pickle=False)
        delta = torch.from_numpy(data_cache["delta_f16"].astype(np.float32))
        adv_aqsm_imgs = torch.clamp(eval_imgs + delta, 0.0, 1.0)
        logger.info(f"   -> Đã nạp thành công {len(adv_aqsm_imgs)} ảnh AQSM đối kháng từ cache.")
    else:
        logger.info(f"\n2. Sinh đòn AQSM (Ranking Inversion Loss, 15 steps PGD, N={N_pgd})...")
        t0_gen = time.time()
        adv_aqsm_list = []
        for i in tqdm(range(0, N_pgd, batch_size), desc="  Sinh đòn AQSM"):
            sub_imgs = eval_imgs[i : i + batch_size]
            adv_sub = attacker.generate_query_selection_attack(sub_imgs, steps=15, margin=0.25, lambda_rank=1.2)
            adv_aqsm_list.append(adv_sub)
        adv_aqsm_imgs = torch.cat(adv_aqsm_list, dim=0)
        logger.info(f"   -> Sinh AQSM hoàn tất trong {time.time() - t0_gen:.1f}s.")

    # 3. Đánh giá trạng thái dưới đòn AQSM
    logger.info("\n3. Đánh giá trạng thái suy luận dưới đòn AQSM...")
    adv_feats = []
    adv_logits_l = []
    t0_eval = time.time()

    for i in range(0, N_pgd, batch_size):
        b_adv = adv_aqsm_imgs[i : i + batch_size].to(device)
        with torch.no_grad():
            st = wrapper(b_adv)
            adv_feats.extend(extractor.extract_from_state(st))
            adv_logits_l.append(st.logits.cpu())
            del st
    adv_logits = torch.cat(adv_logits_l, dim=0)
    logger.info(f"   -> Forward pass AQSM hoàn tất trong {time.time() - t0_eval:.2f}s.")

    asr, succ_cnt, valid_cnt, succ_mask = calculate_task_level_asr("AQSM", clean_logits, adv_logits)
    asr_ci = clopper_pearson_interval(succ_cnt, valid_cnt)
    logger.info(f"   -> Task-level Attack Success Rate (ASR) = {asr*100:5.1f}% [{asr_ci[0]*100:4.1f}%, {asr_ci[1]*100:4.1f}%] ({succ_cnt}/{valid_cnt})")

    # Thống kê phân phối đặc trưng ESM
    clean_drop_vals = [f.enc_dropoff for f in clean_feats]
    adv_drop_vals = [f.enc_dropoff for f in adv_feats]
    clean_marg_vals = [f.enc_margin for f in clean_feats]
    adv_marg_vals = [f.enc_margin for f in adv_feats]

    logger.info("\n   THỐNG KÊ PHÂN PHỐI ĐẶC TRƯNG ESM (ENCODER SELECTION MONITOR):")
    logger.info(f"   * S_dropoff : Clean Mean = {np.mean(clean_drop_vals):.2f} ± {np.std(clean_drop_vals):.2f}  --->  AQSM Mean = {np.mean(adv_drop_vals):.2f} ± {np.std(adv_drop_vals):.2f}")
    logger.info(f"   * S_margin  : Clean Mean = {np.mean(clean_marg_vals):.4f} ± {np.std(clean_marg_vals):.4f}  --->  AQSM Mean = {np.mean(adv_marg_vals):.4f} ± {np.std(adv_marg_vals):.4f}")

    # 4. Đánh giá tỷ lệ phát hiện (TPR) của từng Detector
    adv_scores = {name: det.score(adv_feats) for name, det in detectors.items()}
    method_metrics: Dict[str, Any] = {}
    roc_curves: Dict[str, Any] = {}

    logger.info("\n   KẾT QUẢ ĐÁNH GIÁ BỘ GIÁM SÁT TRƯỚC ĐÒN AQSM:")
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
        tpr = tp_cnt / float(N_pgd)
        tpr_ci = clopper_pearson_interval(tp_cnt, N_pgd)

        y_true_aqsm = np.ones(N_pgd, dtype=int)
        conf = compute_confusion_stats(y_true_aqsm, preds)

        method_metrics[m_name] = {
            "auroc": round(float(auroc), 4),
            "auroc_ci": [round(float(auroc_ci[0]), 4), round(float(auroc_ci[1]), 4)],
            "tpr": round(float(tpr), 5),
            "tpr_ci": [round(float(tpr_ci[0]), 5), round(float(tpr_ci[1]), 5)],
            "tp_count": tp_cnt,
            "total_samples": N_pgd,
            "threshold": round(float(cal_thresholds[m_name]), 4),
            "confusion": conf,
        }

        ci_auroc_str = f"[{auroc_ci[0]:.3f}, {auroc_ci[1]:.3f}]"
        ci_tpr_str = f"[{tpr_ci[0]*100:4.1f}%, {tpr_ci[1]*100:4.1f}%]"
        logger.info(f"   {m_name:<20} | {auroc:5.3f}      | {ci_auroc_str:<20} | {tpr*100:5.1f}%        | {ci_tpr_str:<24}")

    logger.info("   " + "-" * 90)

    # 5. Ghi vết từng mẫu (Sample Audit CSV)
    audit_tracker = SampleAuditTracker("step4_aqsm_and_esm", output_dir=RESULTS_DIR)
    for idx in range(N_pgd):
        s_dict = {name: float(adv_scores[name][idx]) for name in detectors}
        p_dict = {
            name: int(det.predict([adv_feats[idx]])[0]) if name in ["ActiveQ_Split", "Top1Mean_Split", "ESM_Dropoff", "ESM_Margin"]
            else int(adv_scores[name][idx] >= cal_thresholds[name])
            for name, det in detectors.items()
        }
        is_succ = int(succ_mask[idx]) if succ_mask is not None and idx < len(succ_mask) else -1
        audit_tracker.record_sample(
            sample_idx=idx,
            condition="aqsm_eps16",
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

    # 6. Xuất Audit JSON
    res_json_paper3 = RESULTS_DIR / "step4_aqsm_esm_metrics.json"
    res_json_global = CHECKPOINTS_DIR / "res_step4_aqsm_esm.json"

    export_payload = {
        "experiment": "step4_aqsm_and_esm",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {"device": str(device), "num_threads": torch.get_num_threads()},
        "eval_cohort_size": N_pgd,
        "task_asr": {
            "asr": round(float(asr), 5),
            "ci_95": [round(float(asr_ci[0]), 5), round(float(asr_ci[1]), 5)],
            "success_count": int(succ_cnt),
            "valid_count": int(valid_cnt),
        },
        "esm_feature_distributions": {
            "clean_dropoff_mean": round(float(np.mean(clean_drop_vals)), 4),
            "clean_dropoff_std": round(float(np.std(clean_drop_vals)), 4),
            "adv_dropoff_mean": round(float(np.mean(adv_drop_vals)), 4),
            "adv_dropoff_std": round(float(np.std(adv_drop_vals)), 4),
            "clean_margin_mean": round(float(np.mean(clean_marg_vals)), 4),
            "clean_margin_std": round(float(np.std(clean_marg_vals)), 4),
            "adv_margin_mean": round(float(np.mean(adv_marg_vals)), 4),
            "adv_margin_std": round(float(np.std(adv_marg_vals)), 4),
        },
        "detectors_eval": method_metrics,
        "roc_curves": roc_curves,
        "artifacts": {
            "samples_csv": str(samples_csv_path),
            "log_file": str(log_file),
        }
    }

    save_audit_json(res_json_paper3, export_payload)
    save_audit_json(res_json_global, export_payload)

    logger.info(f"-> Đã lưu Audit JSON (Paper 3) : {res_json_paper3}")
    logger.info(f"-> Đã đồng bộ JSON toàn dự án  : {res_json_global}")
    logger.info("\n>>> BƯỚC 4 ĐÃ HOÀN TẤT THÀNH CÔNG VỚI ĐẦY ĐỦ DỮ LIỆU KIỂM TOÁN! <<<")


if __name__ == "__main__":
    run_step4()
