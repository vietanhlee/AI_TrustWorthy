# -*- coding: utf-8 -*-
"""
Bước 1: Trích xuất đặc trưng Calibration (N=1200) & Fit hệ thống Giám sát Đa tầng (M=4)
Đánh giá tỷ lệ báo động sai (FPR) trên tập Clean Test độc lập (N=600) (Zero data leakage).
Xuất đầy đủ log chi tiết, checkpoint pkl, CSV kiểm toán từng mẫu và JSON metrics.
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
    make_serializable,
    RTDETRModelWrapper,
    RTDETRDeepFeatureExtractor,
    SingleFeatureBaseline,
    LayerwiseMahalanobisDetector,
    CalibratedMaxOREnsemble,
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


def run_step1():
    logger, log_file = setup_experiment_logger("step1_calibration")
    logger.info("BƯỚC 1: TRÍCH XUẤT CALIBRATION & FIT HỆ THỐNG GIÁM SÁT ĐA TẦNG (M=4)")

    device = torch.device("cpu")
    logger.info(f"Khởi tạo phần cứng: Device = {device}, PyTorch Threads = {torch.get_num_threads()}")

    # 1. Nạp mô hình RT-DETR
    logger.info("1. Nạp mô hình RT-DETR-R18VD pretrained (PekingU/rtdetr_r18vd)...")
    wrapper = RTDETRModelWrapper(pretrained_model_name="PekingU/rtdetr_r18vd", device=device)
    wrapper.model.eval()
    extractor = RTDETRDeepFeatureExtractor(conf_threshold=0.35, top_k_queries=10, projection_dim=64)
    logger.info("   -> Mô hình và bộ trích xuất đặc trưng (projection_dim=64) đã sẵn sàng.")

    # 2. Phân chia tập dữ liệu nghiêm ngặt
    N_cal, N_test = 1200, 600
    img_size = 640
    seed = 42
    logger.info(f"2. Nạp dữ liệu COCO val2017: N_cal={N_cal}, N_test={N_test}, Size={img_size}x{img_size}, Seed={seed} (Zero Data Leakage)...")
    t0_data = time.time()
    cal_imgs, test_imgs = load_coco_images(n_cal=N_cal, n_test=N_test, img_size=img_size, seed=seed)
    logger.info(f"   -> Nạp {len(cal_imgs)} ảnh calib và {len(test_imgs)} ảnh test hoàn tất trong {time.time() - t0_data:.2f}s.")

    # 3. Trích xuất đặc trưng tập Calibration
    batch_size = 20
    logger.info(f"3. Bắt đầu trích xuất đặc trưng trên {N_cal} ảnh Calibration (Batch Size = {batch_size})...")
    cal_features = []
    t0_cal = time.time()
    num_batches_cal = int(np.ceil(N_cal / batch_size))

    cal_batches = list(range(0, N_cal, batch_size))
    with tqdm(total=N_cal, desc="  Calibration Feature Extraction", unit="img", dynamic_ncols=True) as pbar:
        for b_idx, i in enumerate(cal_batches):
            b = cal_imgs[i : i + batch_size].to(device)
            with torch.no_grad():
                st = wrapper(b)
                batch_feats = extractor.extract_from_state(st)
                cal_features.extend(batch_feats)
                del st
            pbar.update(len(b))
            if (b_idx + 1) % 10 == 0 or (b_idx + 1) == num_batches_cal:
                elapsed = time.time() - t0_cal
                speed = (i + len(b)) / elapsed
                logger.info(f"   [Batch {b_idx+1:2d}/{num_batches_cal:2d}] Đã xử lý {i + len(b):4d}/{N_cal} ảnh | Tốc độ: {speed:5.1f} ảnh/s")

    t_cal = time.time() - t0_cal
    logger.info(f"   -> Hoàn tất Calibration trong {t_cal:.2f}s ({t_cal/N_cal*1000:.1f} ms/ảnh).")

    # Phân tích thống kê phân phối sạch
    clean_act = np.array([f.active_query_ratio for f in cal_features], dtype=np.float64)
    clean_top1 = np.array([f.query_top1_mean for f in cal_features], dtype=np.float64)
    clean_drop = np.array([f.enc_dropoff for f in cal_features], dtype=np.float64)
    clean_marg = np.array([f.enc_margin for f in cal_features], dtype=np.float64)

    logger.info(f"   -> THỐNG KÊ PHÂN PHỐI SẠCH CHUẨN (N_cal = {N_cal}):")
    logger.info(f"      * ActiveQ Soft : Mean={np.mean(clean_act):.4f} ± {np.std(clean_act):.4f} | Bounds 95% = [{np.percentile(clean_act, 2.5):.4f}, {np.percentile(clean_act, 97.5):.4f}]")
    logger.info(f"      * Top1Mean     : Mean={np.mean(clean_top1):.4f} ± {np.std(clean_top1):.4f} | Bounds 95% = [{np.percentile(clean_top1, 2.5):.4f}, {np.percentile(clean_top1, 97.5):.4f}]")
    logger.info(f"      * ESM Dropoff  : Mean={np.mean(clean_drop):.4f} ± {np.std(clean_drop):.4f} | Bounds 95% = [{np.percentile(clean_drop, 2.5):.4f}, {np.percentile(clean_drop, 97.5):.4f}]")
    logger.info(f"      * ESM Margin   : Mean={np.mean(clean_marg):.4f} ± {np.std(clean_marg):.4f} | Bounds 95% = [{np.percentile(clean_marg, 2.5):.4f}, {np.percentile(clean_marg, 97.5):.4f}]")

    # 4. Huấn luyện (Fit) hệ thống giám sát phân tầng
    logger.info("\n4. Huấn luyện hệ thống giám sát phân tầng (Target nominal FPR = 0.05)...")
    detectors = {
        "ActiveQ_Split": SingleFeatureBaseline("active_query_soft", tail="two_sided_split", target_fpr=0.05),
        "Top1Mean_Split": SingleFeatureBaseline("query_top1_mean", tail="two_sided_split", target_fpr=0.05),
        "ESM_Dropoff": SingleFeatureBaseline("enc_dropoff", tail="two_sided_split", target_fpr=0.05),
        "ESM_Margin": SingleFeatureBaseline("enc_margin", tail="two_sided_split", target_fpr=0.05),
        "Layer_Maha": LayerwiseMahalanobisDetector(shrinkage=0.05, target_fpr=0.05),
        "Calibrated_Max_OR": CalibratedMaxOREnsemble(target_fpr=0.05),
    }

    cal_thresholds = {}
    for name, det in detectors.items():
        t0_fit = time.time()
        det.fit(cal_features)
        cal_thresholds[name] = float(det.threshold)
        logger.info(f"   + [{name:<18}] Fit xong ({time.time() - t0_fit:.3f}s) | Ngưỡng Calib @ 5.0% FPR: {det.threshold:.4f}")

    # 5. Đánh giá kiểm toán trên tập Clean Test Set độc lập (N=600)
    logger.info(f"\n5. Trích xuất và đánh giá trên {N_test} ảnh Clean Test độc lập...")
    test_features = []
    t0_test = time.time()
    num_batches_test = int(np.ceil(N_test / batch_size))

    test_batches = list(range(0, N_test, batch_size))
    with tqdm(total=N_test, desc="  Clean Test Feature Extraction", unit="img", dynamic_ncols=True) as pbar:
        for b_idx, i in enumerate(test_batches):
            b = test_imgs[i : i + batch_size].to(device)
            with torch.no_grad():
                st = wrapper(b)
                test_features.extend(extractor.extract_from_state(st))
                del st
            pbar.update(len(b))
            if (b_idx + 1) % 10 == 0 or (b_idx + 1) == num_batches_test:
                logger.info(f"   [Clean Test Batch {b_idx+1:2d}/{num_batches_test:2d}] Đã xử lý {i + len(b):4d}/{N_test} ảnh")

    logger.info(f"   -> Trích xuất Clean Test hoàn tất trong {time.time() - t0_test:.2f}s.")

    # Tính toán điểm và dự đoán
    test_scores = {name: det.score(test_features) for name, det in detectors.items()}
    test_preds = {}
    for name, det in detectors.items():
        if name in ["ActiveQ_Split", "Top1Mean_Split", "ESM_Dropoff", "ESM_Margin"]:
            test_preds[name] = det.predict(test_features)
        else:
            test_preds[name] = (test_scores[name] >= det.threshold).astype(int)

    # 6. Ghi vết từng mẫu (Sample-Level Audit Tracker)
    audit_tracker = SampleAuditTracker("step1_calibration", output_dir=RESULTS_DIR)
    for idx in range(N_test):
        s_dict = {name: float(test_scores[name][idx]) for name in detectors}
        p_dict = {name: int(test_preds[name][idx]) for name in detectors}
        audit_tracker.record_sample(
            sample_idx=idx,
            condition="clean_test",
            ground_truth=0,
            scores=s_dict,
            preds=p_dict,
            is_attack_success=0,
            extra_info={
                "active_query_soft": float(test_features[idx].active_query_soft),
                "query_top1_mean": float(test_features[idx].query_top1_mean),
                "enc_dropoff": float(test_features[idx].enc_dropoff),
                "enc_margin": float(test_features[idx].enc_margin),
            }
        )
    samples_csv_path = audit_tracker.save_csv()
    logger.info(f"   -> Đã lưu bảng kiểm toán từng mẫu (Sample Audit CSV): {samples_csv_path}")

    # Đánh giá FPR & Thống kê nhầm lẫn
    test_fprs = {}
    confusion_stats = {}
    y_true_clean = np.zeros(N_test, dtype=int)

    logger.info("\n6. KẾT QUẢ ĐÁNH GIÁ TỶ LỆ BÁO ĐỘNG OAN (CLEAN TEST FPR):")
    logger.info("-" * 85)
    logger.info(f"{'Tên Bộ Giám Sát':<22} | {'Clean FPR (%)':<15} | {'95% Clopper-Pearson CI':<24} | {'Số Báo Động / Tổng':<18}")
    logger.info("-" * 85)

    for name, det in detectors.items():
        fa_cnt = int(np.sum(test_preds[name]))
        fpr = fa_cnt / float(N_test)
        ci = clopper_pearson_interval(fa_cnt, N_test)
        test_fprs[name] = (fpr, ci, fa_cnt)
        confusion_stats[name] = compute_confusion_stats(y_true_clean, test_preds[name])

        ci_str = f"[{ci[0]*100:4.1f}%, {ci[1]*100:4.1f}%]"
        ratio_str = f"{fa_cnt:3d} / {N_test}"
        logger.info(f"{name:<22} | {fpr*100:5.1f}%          | {ci_str:<24} | {ratio_str:<18}")

    logger.info("-" * 85)

    # 7. Lưu Checkpoint nhị phân pkl và Audit JSON
    checkpoint_file = CHECKPOINTS_DIR / "calibration_and_detectors.pkl"
    logger.info(f"\n7. Lưu Checkpoint nhị phân tại: {checkpoint_file}")
    with open(checkpoint_file, "wb") as f:
        pickle.dump({
            "N_cal": N_cal,
            "N_test": N_test,
            "detectors": detectors,
            "cal_thresholds": cal_thresholds,
            "clean_test_scores": {k: v.tolist() for k, v in test_scores.items()},
            "clean_test_fprs": test_fprs,
            "clean_stats": {
                "mu_act": float(np.mean(clean_act)),
                "sd_act": float(np.std(clean_act)),
                "mu_top1": float(np.mean(clean_top1)),
                "sd_top1": float(np.std(clean_top1)),
                "tau_act_bounds": (float(np.percentile(clean_act, 2.5)), float(np.percentile(clean_act, 97.5))),
                "tau_top_bounds": (float(np.percentile(clean_top1, 2.5)), float(np.percentile(clean_top1, 97.5))),
            }
        }, f)

    # Xuất JSON kết quả chuẩn cho Paper 3 và toàn dự án
    res_json_paper3 = RESULTS_DIR / "step1_calibration_metrics.json"
    res_json_global = CHECKPOINTS_DIR / "res_step1_calibration.json"

    export_payload = {
        "experiment": "step1_calibration",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {"device": str(device), "num_threads": torch.get_num_threads()},
        "dataset": {"N_cal": N_cal, "N_test": N_test, "img_size": img_size, "seed": seed},
        "cal_thresholds": cal_thresholds,
        "clean_stats": {
            "mu_act": float(np.mean(clean_act)),
            "sd_act": float(np.std(clean_act)),
            "mu_top1": float(np.mean(clean_top1)),
            "sd_top1": float(np.std(clean_top1)),
            "bounds_act_95": [float(np.percentile(clean_act, 2.5)), float(np.percentile(clean_act, 97.5))],
            "bounds_top1_95": [float(np.percentile(clean_top1, 2.5)), float(np.percentile(clean_top1, 97.5))],
            "bounds_dropoff_95": [float(np.percentile(clean_drop, 2.5)), float(np.percentile(clean_drop, 97.5))],
            "bounds_margin_95": [float(np.percentile(clean_marg, 2.5)), float(np.percentile(clean_marg, 97.5))],
        },
        "clean_test_fprs": {
            k: {
                "fpr": round(float(v[0]), 5),
                "ci_95": [round(float(v[1][0]), 5), round(float(v[1][1]), 5)],
                "false_alarm_count": int(v[2]),
                "total_samples": N_test,
                "confusion": confusion_stats[k]
            }
            for k, v in test_fprs.items()
        },
        "artifacts": {
            "checkpoint_pkl": str(checkpoint_file),
            "samples_csv": str(samples_csv_path),
            "log_file": str(log_file),
        }
    }

    save_audit_json(res_json_paper3, export_payload)
    save_audit_json(res_json_global, export_payload)

    logger.info(f"   -> Đã lưu Audit JSON (Paper 3) : {res_json_paper3}")
    logger.info(f"   -> Đã đồng bộ JSON toàn dự án  : {res_json_global}")
    logger.info("\n>>> BƯỚC 1 ĐÃ HOÀN TẤT THÀNH CÔNG VỚI ĐẦY ĐỦ DỮ LIỆU KIỂM TOÁN! <<<")


if __name__ == "__main__":
    run_step1()
