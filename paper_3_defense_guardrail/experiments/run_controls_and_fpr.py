# -*- coding: utf-8 -*-
"""
Bước 2: Đánh giá Tỷ lệ Báo động Oan (FPR Stability) Dưới Biến Dạng Tự Nhiên & Controls
Bao gồm:
1. Control: Nhiễu hạt Gaussian (sigma = 16/255) (N=300)
2. Control: Random Cutout Patch (48x48 vị trí ngẫu nhiên) (N=300)
3. Clean Shift: Gaussian Blur (5x5, sigma = 1.5) (N=300)
4. Clean Shift: Brightness Shift (-30% độ sáng) (N=300)
Đánh giá độ ổn định báo động sai (FPR) của Calibrated Max-OR so với các kênh đơn lẻ.
Xuất log chi tiết, file CSV mẫu kiểm toán và file JSON metrics.
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

from edge_sec_pro.experiments.common_exp import (
    load_coco_images,
    clopper_pearson_interval,
    compute_auroc_and_bootstrap_ci,
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
    compute_confusion_stats,
    PAPER3_DIR,
    RESULTS_DIR,
)


def run_step2():
    logger, log_file = setup_experiment_logger("step2_controls_and_fpr")
    logger.info("BƯỚC 2: ĐÁNH GIÁ ĐỘ ỔN ĐỊNH BÁO ĐỘNG OAN (FPR STABILITY) TRÊN CONTROLS VÀ NATURAL SHIFTS")

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
    logger.info(f"Đã nạp {len(detectors)} detectors đã hiệu chuẩn từ: {ckpt_file}")

    device = torch.device("cpu")
    logger.info(f"Khởi tạo phần cứng: Device = {device}, PyTorch Threads = {torch.get_num_threads()}")

    wrapper = RTDETRModelWrapper(pretrained_model_name="PekingU/rtdetr_r18vd", device=device)
    wrapper.model.eval()
    extractor = RTDETRDeepFeatureExtractor(conf_threshold=0.35, top_k_queries=10, projection_dim=64)
    attacker = DETRAttackGenerator(wrapper, epsilon=16.0 / 255.0, alpha=2.0 / 255.0, num_steps=15)

    # Nạp 300 ảnh độc lập từ test set để làm controls và natural shifts
    N_eval = 300
    logger.info(f"Nạp dữ liệu COCO: Lấy N={N_eval} ảnh độc lập từ test set làm cohort đánh giá...")
    _, test_imgs = load_coco_images(n_cal=1200, n_test=600, img_size=320, seed=42)
    eval_imgs = test_imgs[:N_eval]

    controls: List[Tuple[str, str, torch.Tensor]] = [
        ("gaussian_noise", "Control: Gaussian Noise (sigma=16/255)", attacker.generate_random_noise(eval_imgs, noise_type="gaussian")),
        ("random_patch", "Control: Random Patch (48x48 Random Pos)", attacker.generate_random_patch(eval_imgs, patch_size=48, patch_loc="random", seed=1337)),
        ("gaussian_blur", "Clean Shift: Gaussian Blur (5x5, sigma=1.5)", attacker.generate_clean_shifts(eval_imgs, shift_type="blur")),
        ("brightness_shift", "Clean Shift: Brightness (-30%)", attacker.generate_clean_shifts(eval_imgs, shift_type="brightness")),
    ]

    results_summary: Dict[str, Any] = {}
    audit_tracker = SampleAuditTracker("step2_controls_and_fpr", output_dir=RESULTS_DIR)
    batch_size = 20

    logger.info(f"\nBắt đầu đánh giá lần lượt {len(controls)} kịch bản biến dạng (N={N_eval} mỗi kịch bản)...")

    for cond_tag, c_name, c_imgs in controls:
        logger.info("=" * 80)
        logger.info(f"ĐANG ĐÁNH GIÁ: {c_name} (N = {len(c_imgs)})")
        logger.info("=" * 80)

        t0_cond = time.time()
        c_feats = []
        num_batches = int(np.ceil(len(c_imgs) / batch_size))

        for b_idx in range(num_batches):
            b = c_imgs[b_idx * batch_size : (b_idx + 1) * batch_size].to(device)
            with torch.no_grad():
                st = wrapper(b)
                c_feats.extend(extractor.extract_from_state(st))
                del st

        elapsed_feats = time.time() - t0_cond
        logger.info(f"   -> Trích xuất đặc trưng hoàn tất trong {elapsed_feats:.2f}s ({elapsed_feats/len(c_imgs)*1000:.1f} ms/ảnh).")

        c_scores = {name: det.score(c_feats) for name, det in detectors.items()}
        method_metrics: Dict[str, Any] = {}

        logger.info(f"   {'Detector':<20} | {'FPR (%)':<10} | {'95% Clopper-Pearson CI':<24} | {'AUROC':<10} | {'Số Báo Động':<12}")
        logger.info("   " + "-" * 85)

        for m_name, det in detectors.items():
            s_neg = clean_test_scores[m_name]
            s_pos = c_scores[m_name]
            labels = np.concatenate([np.zeros(len(s_neg)), np.ones(len(s_pos))])
            all_scores = np.concatenate([s_neg, s_pos])
            auroc, auroc_ci = compute_auroc_and_bootstrap_ci(labels, all_scores)

            if m_name in ["ActiveQ_Split", "Top1Mean_Split", "ESM_Dropoff", "ESM_Margin"]:
                preds = det.predict(c_feats)
            else:
                preds = (s_pos >= cal_thresholds[m_name]).astype(int)

            fa_cnt = int(np.sum(preds))
            fpr = fa_cnt / float(len(c_imgs))
            fpr_ci = clopper_pearson_interval(fa_cnt, len(c_imgs))
            conf = compute_confusion_stats(np.zeros(len(c_imgs), dtype=int), preds)

            method_metrics[m_name] = {
                "fpr": round(float(fpr), 5),
                "fpr_ci": [round(float(fpr_ci[0]), 5), round(float(fpr_ci[1]), 5)],
                "fa_count": fa_cnt,
                "total_samples": len(c_imgs),
                "auroc": round(float(auroc), 4),
                "auroc_ci": [round(float(auroc_ci[0]), 4), round(float(auroc_ci[1]), 4)],
                "threshold": round(float(cal_thresholds[m_name]), 4),
                "confusion": conf,
            }

            ci_str = f"[{fpr_ci[0]*100:4.1f}%, {fpr_ci[1]*100:4.1f}%]"
            logger.info(f"   {m_name:<20} | {fpr*100:5.1f}%    | {ci_str:<24} | {auroc:5.3f}      | {fa_cnt:3d}/{len(c_imgs)}")

        # Ghi nhận chi tiết từng mẫu vào Audit CSV
        for idx in range(len(c_imgs)):
            s_dict = {name: float(c_scores[name][idx]) for name in detectors}
            if cond_tag in ["gaussian_noise", "random_patch", "gaussian_blur", "brightness_shift"]:
                p_dict = {
                    name: int(det.predict([c_feats[idx]])[0]) if name in ["ActiveQ_Split", "Top1Mean_Split", "ESM_Dropoff", "ESM_Margin"]
                    else int(c_scores[name][idx] >= cal_thresholds[name])
                    for name, det in detectors.items()
                }
            audit_tracker.record_sample(
                sample_idx=idx,
                condition=cond_tag,
                ground_truth=0,  # Negative control (không phải tấn công đối kháng)
                scores=s_dict,
                preds=p_dict,
                is_attack_success=0,
                extra_info={
                    "active_query_soft": float(c_feats[idx].active_query_soft),
                    "query_top1_mean": float(c_feats[idx].query_top1_mean),
                    "enc_dropoff": float(c_feats[idx].enc_dropoff),
                    "enc_margin": float(c_feats[idx].enc_margin),
                }
            )

        results_summary[cond_tag] = {
            "condition_name": c_name,
            "sample_count": len(c_imgs),
            "metrics": method_metrics,
        }

    # Xuất file CSV kiểm toán
    samples_csv_path = audit_tracker.save_csv()
    logger.info(f"\n-> Đã lưu bảng kiểm toán từng mẫu (Sample Audit CSV): {samples_csv_path}")

    # Tính Macro Mean FPR
    logger.info("\n" + "=" * 85)
    logger.info("TỔNG HỢP MACRO MEAN FPR TRÊN TOÀN BỘ CONTROLS VÀ NATURAL SHIFTS:")
    logger.info("-" * 85)
    macro_fprs = {}
    for m_name in detectors:
        all_fprs = [results_summary[cond]["metrics"][m_name]["fpr"] for cond in results_summary]
        macro_mean = float(np.mean(all_fprs))
        macro_fprs[m_name] = round(macro_mean, 5)
        logger.info(f"   * {m_name:<20} : Macro Mean FPR = {macro_mean*100:5.2f}%")
    logger.info("=" * 85)

    # Xuất Audit JSON
    res_json_paper3 = RESULTS_DIR / "step2_controls_fpr_metrics.json"
    res_json_global = CHECKPOINTS_DIR / "res_step2_controls_fpr.json"

    export_payload = {
        "experiment": "step2_controls_and_fpr",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {"device": str(device), "num_threads": torch.get_num_threads()},
        "eval_cohort_size": N_eval,
        "macro_mean_fpr": macro_fprs,
        "conditions": results_summary,
        "artifacts": {
            "samples_csv": str(samples_csv_path),
            "log_file": str(log_file),
        }
    }

    save_audit_json(res_json_paper3, export_payload)
    save_audit_json(res_json_global, export_payload)

    logger.info(f"-> Đã lưu Audit JSON (Paper 3) : {res_json_paper3}")
    logger.info(f"-> Đã đồng bộ JSON toàn dự án  : {res_json_global}")
    logger.info("\n>>> BƯỚC 2 ĐÃ HOÀN TẤT THÀNH CÔNG VỚI ĐẦY ĐỦ DỮ LIỆU KIỂM TOÁN! <<<")


if __name__ == "__main__":
    run_step2()
