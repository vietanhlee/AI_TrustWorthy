# -*- coding: utf-8 -*-
"""
Bước 9: Đánh giá Đối sánh với các Baseline Phòng thủ Kinh điển (Defensive Baselines)
Bao gồm:
1. High-Frequency Total Variation (TV) Energy Detector
2. Input Reconstruction qua JPEG Compression Difference (Quality=75)
3. Đề xuất Calibrated Max-OR Guardrail
Đánh giá trên: Clean Test, AQSM Attack (eps=16/255), Gaussian Noise (sigma=16), Gaussian Blur (5x5).
Xuất log chi tiết, file CSV mẫu kiểm toán và file JSON metrics.
"""

import os
import io
import sys
import time
import pickle
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
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


def compute_total_variation(images: torch.Tensor) -> np.ndarray:
    """Tính High-Frequency Total Variation (TV) trên batch ảnh [B, C, H, W]."""
    diff_h = torch.abs(images[:, :, 1:, :] - images[:, :, :-1, :]).mean(dim=[1, 2, 3])
    diff_w = torch.abs(images[:, :, :, 1:] - images[:, :, :, :-1]).mean(dim=[1, 2, 3])
    tv = (diff_h + diff_w).cpu().numpy()
    return tv


def compute_jpeg_diff(images: torch.Tensor, quality: int = 75) -> np.ndarray:
    """Tính L1 Reconstruction Error sau khi nén JPEG quality=75."""
    B = images.shape[0]
    diffs = []
    imgs_np = (images.permute(0, 2, 3, 1).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    for i in range(B):
        pil_img = Image.fromarray(imgs_np[i])
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        jpeg_img = np.array(Image.open(buf)).astype(np.float32) / 255.0
        orig_img = imgs_np[i].astype(np.float32) / 255.0
        diff = np.mean(np.abs(orig_img - jpeg_img))
        diffs.append(diff)
    return np.array(diffs)


def run_experiment():
    logger, log_file = setup_experiment_logger("step9_defensive_baselines")
    logger.info("BƯỚC 9: ĐÁNH GIÁ ĐỐI SÁNH DEFENSIVE BASELINES (TV, JPEG-DIFF VS CALIBRATED MAX-OR)")

    device = torch.device("cpu")
    logger.info(f"Khởi tạo phần cứng: Device = {device}, PyTorch Threads = {torch.get_num_threads()}")

    wrapper = RTDETRModelWrapper(pretrained_model_name="PekingU/rtdetr_r18vd", device=device)
    wrapper.model.eval()
    extractor = RTDETRDeepFeatureExtractor(conf_threshold=0.35, top_k_queries=10, projection_dim=64)
    attacker = DETRAttackGenerator(wrapper, epsilon=16.0 / 255.0, alpha=2.0 / 255.0, num_steps=15)

    # Nạp Checkpoint detector nếu có
    ckpt_file = CHECKPOINTS_DIR / "calibration_and_detectors.pkl"
    max_or_detector = None
    cal_thresh_max_or = None
    if ckpt_file.exists():
        with open(ckpt_file, "rb") as f:
            ckpt = pickle.load(f)
        max_or_detector = ckpt["detectors"].get("Calibrated_Max_OR", None)
        cal_thresh_max_or = ckpt["cal_thresholds"].get("Calibrated_Max_OR", None)
        logger.info(f"Đã nạp Calibrated Max-OR Detector từ {ckpt_file.name} (Ngưỡng = {cal_thresh_max_or})")

    # 1. Nạp dữ liệu calibration và test
    N_cal, N_eval = 100, 25
    cal_imgs, test_imgs = load_coco_images(n_cal=N_cal, n_test=N_eval, img_size=640, seed=42)
    cal_imgs = cal_imgs.to(device)
    test_imgs = test_imgs.to(device)
    logger.info(f"Nạp dữ liệu: N_cal = {N_cal}, N_eval = {N_eval}")

    # 2. Hiệu chuẩn ngưỡng 5% FPR trên Clean Calibration
    logger.info(f"\n1. Hiệu chuẩn ngưỡng danh định 5.0% FPR trên {N_cal} ảnh Calibration...")
    cal_tv = compute_total_variation(cal_imgs)
    cal_jpeg = compute_jpeg_diff(cal_imgs)

    tau_tv = float(np.percentile(cal_tv, 95.0))
    tau_jpeg = float(np.percentile(cal_jpeg, 95.0))
    logger.info(f"   + Ngưỡng TV Energy (alpha=5%): {tau_tv:.5f}")
    logger.info(f"   + Ngưỡng JPEG Diff  (alpha=5%): {tau_jpeg:.5f}")

    # 3. Tạo các tập dữ liệu thử nghiệm
    logger.info(f"\n2. Tạo các tập dữ liệu đánh giá đối kháng và biến dạng tự nhiên (N={N_eval})...")
    # AQSM Attack
    adv_aqsm = attacker.generate_query_selection_attack(test_imgs, steps=15, margin=0.25, lambda_rank=1.2)
    # Gaussian noise
    noise = torch.randn_like(test_imgs) * (16.0 / 255.0)
    corrupt_noise = torch.clamp(test_imgs + noise, 0.0, 1.0)
    # Gaussian blur
    corrupt_blur = F.avg_pool2d(test_imgs, kernel_size=5, stride=1, padding=2)

    datasets: Dict[str, Tuple[torch.Tensor, int]] = {
        "Clean Test": (test_imgs, 0),
        "AQSM Attack (eps=16)": (adv_aqsm, 1),
        "Gaussian Noise (sigma=16)": (corrupt_noise, 0),
        "Gaussian Blur (5x5)": (corrupt_blur, 0),
    }

    results: Dict[str, Any] = {}
    audit_tracker = SampleAuditTracker("step9_defensive_baselines", output_dir=RESULTS_DIR)

    logger.info("\n3. Bắt đầu đánh giá đối sánh hiệu năng phát hiện và độ trễ thực thi:")
    logger.info("-" * 95)
    logger.info(f"{'Tập Đánh Giá':<28} | {'TV Detector (%)':<18} | {'JPEG-diff (%)':<18} | {'Calibrated Max-OR (%)':<22} | {'Trễ (ms)':<10}")
    logger.info("-" * 95)

    for cond_name, (data_tensor, gt_label) in tqdm(datasets.items(), desc="  Đánh giá Defensive Baselines", unit="cond", dynamic_ncols=True):
        # Đo độ trễ và điểm TV
        t0_tv = time.time()
        tv_scores = compute_total_variation(data_tensor)
        tv_ms = (time.time() - t0_tv) * 1000.0 / N_eval

        # Đo độ trễ và điểm JPEG-diff
        t0_jpeg = time.time()
        jpeg_scores = compute_jpeg_diff(data_tensor)
        jpeg_ms = (time.time() - t0_jpeg) * 1000.0 / N_eval

        # Đo độ trễ và điểm Calibrated Max-OR
        max_or_scores = np.zeros(N_eval)
        max_or_ms = 0.0
        if max_or_detector is not None:
            t0_or = time.time()
            with torch.no_grad():
                st = wrapper(data_tensor)
                feats = extractor.extract_from_state(st)
                max_or_scores = max_or_detector.score(feats)
                del st
            max_or_ms = (time.time() - t0_or) * 1000.0 / N_eval

        preds_tv = (tv_scores >= tau_tv).astype(int)
        preds_jpeg = (jpeg_scores >= tau_jpeg).astype(int)
        preds_or = (max_or_scores >= cal_thresh_max_or).astype(int) if cal_thresh_max_or is not None else np.zeros(N_eval, dtype=int)

        rate_tv = float(np.mean(preds_tv)) * 100.0
        rate_jpeg = float(np.mean(preds_jpeg)) * 100.0
        rate_or = float(np.mean(preds_or)) * 100.0

        ci_tv = clopper_pearson_interval(int(np.sum(preds_tv)), N_eval)
        ci_jpeg = clopper_pearson_interval(int(np.sum(preds_jpeg)), N_eval)
        ci_or = clopper_pearson_interval(int(np.sum(preds_or)), N_eval)

        results[cond_name] = {
            "ground_truth_class": gt_label,
            "sample_count": N_eval,
            "tv_detector": {
                "trigger_rate_pct": round(rate_tv, 2),
                "ci_95": [round(float(ci_tv[0] * 100), 2), round(float(ci_tv[1] * 100), 2)],
                "threshold": round(tau_tv, 5),
                "latency_per_frame_ms": round(tv_ms, 2),
            },
            "jpeg_diff": {
                "trigger_rate_pct": round(rate_jpeg, 2),
                "ci_95": [round(float(ci_jpeg[0] * 100), 2), round(float(ci_jpeg[1] * 100), 2)],
                "threshold": round(tau_jpeg, 5),
                "latency_per_frame_ms": round(jpeg_ms, 2),
            },
            "calibrated_max_or": {
                "trigger_rate_pct": round(rate_or, 2),
                "ci_95": [round(float(ci_or[0] * 100), 2), round(float(ci_or[1] * 100), 2)],
                "threshold": round(cal_thresh_max_or, 5) if cal_thresh_max_or else 0.0,
                "latency_per_frame_ms": round(max_or_ms, 2),
            }
        }

        logger.info(f"{cond_name:<28} | {rate_tv:>6.1f}%{'':<11} | {rate_jpeg:>6.1f}%{'':<11} | {rate_or:>6.1f}%{'':<15} | {tv_ms:>5.2f} ms")

        # Ghi vết từng mẫu ra Audit CSV
        for idx in range(N_eval):
            audit_tracker.record_sample(
                sample_idx=idx,
                condition=cond_name,
                ground_truth=gt_label,
                scores={
                    "TV_Energy": float(tv_scores[idx]),
                    "JPEG_Diff": float(jpeg_scores[idx]),
                    "Calibrated_Max_OR": float(max_or_scores[idx]),
                },
                preds={
                    "TV_Energy": int(preds_tv[idx]),
                    "JPEG_Diff": int(preds_jpeg[idx]),
                    "Calibrated_Max_OR": int(preds_or[idx]),
                },
                is_attack_success=1 if (gt_label == 1 and preds_or[idx] == 0) else 0,
            )

    logger.info("-" * 95)
    samples_csv_path = audit_tracker.save_csv()
    logger.info(f"\n-> Đã lưu bảng kiểm toán từng mẫu (Sample Audit CSV): {samples_csv_path}")

    # Xuất Audit JSON
    res_json_paper3 = RESULTS_DIR / "step9_defensive_baselines_metrics.json"
    res_json_global = CHECKPOINTS_DIR / "res_step9_defensive_baselines.json"

    export_payload = {
        "experiment": "step9_defensive_baselines",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {"device": str(device), "num_threads": torch.get_num_threads()},
        "eval_cohort_size": N_eval,
        "calibration_size": N_cal,
        "results": results,
        "artifacts": {
            "samples_csv": str(samples_csv_path),
            "log_file": str(log_file),
        }
    }

    save_audit_json(res_json_paper3, export_payload)
    save_audit_json(res_json_global, export_payload)

    logger.info(f"-> Đã lưu Audit JSON (Paper 3) : {res_json_paper3}")
    logger.info(f"-> Đã đồng bộ JSON toàn dự án  : {res_json_global}")
    logger.info("\n>>> BƯỚC 9 ĐÃ HOÀN TẤT THÀNH CÔNG VỚI ĐẦY ĐỦ DỮ LIỆU KIỂM TOÁN! <<<")


if __name__ == "__main__":
    run_experiment()
