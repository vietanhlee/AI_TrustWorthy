# -*- coding: utf-8 -*-
"""
Bước 8: Đo lường Thời gian Thực thi (Inference Latency & Runtime Overhead Profile)
Bao gồm:
1. Warmup hệ thống và CPU cache (15 lượt).
2. Đo đạc độ trễ Model Base RT-DETR (50 lần lặp).
3. Đo đạc chi phí trích xuất đặc trưng trung gian (Intermediate Tensor Extraction).
4. Đo đạc chi phí tính điểm từng kênh và cổng Calibrated Max-OR (Decision Scoring).
5. Đo đạc tổng thời gian suy luận kết hợp và tính toán tỷ lệ phụ trội (Relative Overhead %).
Xuất log chi tiết, file CSV lịch sử 50 lượt đo và file JSON metrics.
"""

import os
import sys
import time
import pickle
import platform
import psutil
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import torch

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
    make_serializable,
    RTDETRModelWrapper,
    RTDETRDeepFeatureExtractor,
    CHECKPOINTS_DIR,
)

from paper_3_defense_guardrail.experiments.exp_logger import (
    setup_experiment_logger,
    MetricHistoryTracker,
    save_audit_json,
    PAPER3_DIR,
    RESULTS_DIR,
)


def run_step8():
    logger, log_file = setup_experiment_logger("step8_edge_latency")
    logger.info("BƯỚC 8: ĐO LƯỜNG THỜI GIAN THỰC THI (EDGE-CLASS CPU EXECUTION PROFILE)")

    ckpt_file = CHECKPOINTS_DIR / "calibration_and_detectors.pkl"
    if not ckpt_file.exists():
        err_msg = f"Chưa chạy Bước 1! Không tìm thấy checkpoint: {ckpt_file}"
        logger.error(err_msg)
        raise FileNotFoundError(err_msg)

    with open(ckpt_file, "rb") as f:
        ckpt = pickle.load(f)

    detectors = ckpt["detectors"]

    device = torch.device("cpu")
    num_threads = torch.get_num_threads()
    logger.info(f"Khởi tạo phần cứng: Device = {device}, PyTorch Threads = {num_threads}")
    logger.info(f"CPU Model: {platform.processor()} | Cores: {psutil.cpu_count(logical=False)} Phys / {psutil.cpu_count(logical=True)} Log")

    wrapper = RTDETRModelWrapper(pretrained_model_name="PekingU/rtdetr_r18vd", device=device)
    wrapper.model.eval()
    extractor = RTDETRDeepFeatureExtractor(conf_threshold=0.35, top_k_queries=10, projection_dim=64)

    dummy_tensor = torch.rand(1, 3, 640, 640, device=device)

    # 1. Khởi động (Warmup)
    logger.info("\n1. Khởi động CPU cache (Warmup 15 lượt)...")
    for _ in range(15):
        with torch.no_grad():
            st = wrapper(dummy_tensor)
            _ = extractor.extract_from_state(st)
            del st
    logger.info("   -> Hoàn tất Warmup.")

    # 2. Đo đạc chi tiết qua N_runs = 50 lần lặp
    N_runs = 50
    logger.info(f"\n2. Bắt đầu đo đạc hồ sơ độ trễ ({N_runs} lần lặp độc lập trên ảnh 640x640)...")
    history_tracker = MetricHistoryTracker("step8_latency_runs", output_dir=RESULTS_DIR)

    lat_base = []
    lat_extract = []
    lat_score_or = []
    lat_total_guard = []
    lat_combined = []

    from tqdm import tqdm

    for run_i in tqdm(range(1, N_runs + 1), desc="  Đo độ trễ CPU (Latency Profile)", unit="run", dynamic_ncols=True):
        # Base Model
        t0 = time.perf_counter()
        with torch.no_grad():
            st = wrapper(dummy_tensor)
        t1 = time.perf_counter()
        d_base = (t1 - t0) * 1000.0

        # Tensor Extraction
        t0 = time.perf_counter()
        feats = extractor.extract_from_state(st)
        t1 = time.perf_counter()
        d_extract = (t1 - t0) * 1000.0

        # Decision Scoring (Max-OR)
        t0 = time.perf_counter()
        _ = detectors["Calibrated_Max_OR"].score(feats)
        t1 = time.perf_counter()
        d_score = (t1 - t0) * 1000.0

        d_guard = d_extract + d_score
        d_comb = d_base + d_guard

        lat_base.append(d_base)
        lat_extract.append(d_extract)
        lat_score_or.append(d_score)
        lat_total_guard.append(d_guard)
        lat_combined.append(d_comb)

        history_tracker.log_step(
            epoch=1,
            step=run_i,
            total_steps=N_runs,
            loss=0.0,
            extra_metrics={
                "lat_base_ms": round(d_base, 3),
                "lat_extract_ms": round(d_extract, 3),
                "lat_score_or_ms": round(d_score, 3),
                "lat_guard_total_ms": round(d_guard, 3),
                "lat_combined_ms": round(d_comb, 3),
            },
            logger=logger if (run_i % 10 == 0 or run_i == N_runs) else None,
        )

        del st, feats

    runs_csv_path = history_tracker.save_csv()
    logger.info(f"   -> Đã lưu lịch sử 50 lượt đo độ trễ ra CSV: {runs_csv_path}")

    # 3. Tổng hợp phân tích thống kê
    mean_base, std_base = float(np.mean(lat_base)), float(np.std(lat_base))
    mean_ext, std_ext = float(np.mean(lat_extract)), float(np.std(lat_extract))
    mean_sc, std_sc = float(np.mean(lat_score_or)), float(np.std(lat_score_or))
    mean_guard, std_guard = float(np.mean(lat_total_guard)), float(np.std(lat_total_guard))
    mean_comb, std_comb = float(np.mean(lat_combined)), float(np.std(lat_combined))

    overhead_ratio = (mean_guard / mean_base) * 100.0
    fps_base = 1000.0 / mean_base
    fps_comb = 1000.0 / mean_comb

    logger.info("\n" + "=" * 90)
    logger.info("TỔNG HỢP HỒ SƠ HIỆU NĂNG THỜI GIAN THỰC (EDGE LATENCY PROFILE):")
    logger.info("-" * 90)
    logger.info(f"1. RT-DETR Base Model Inference : {mean_base:6.2f} ± {std_base:5.2f} ms ({fps_base:5.2f} FPS)")
    logger.info(f"2. Intermediate Tensor Extract : {mean_ext:6.2f} ± {std_ext:5.2f} ms ({mean_ext/mean_base*100:4.2f}% base)")
    logger.info(f"3. Decision Scoring (Max-OR)   : {mean_sc:6.3f} ± {std_sc:5.3f} ms ({mean_sc/mean_base*100:4.2f}% base)")
    logger.info(f"4. Proposed Guardrail Total    : {mean_guard:6.2f} ± {std_guard:5.2f} ms (+{overhead_ratio:4.2f}% OVERHEAD)")
    logger.info(f"5. Combined Forward Pass       : {mean_comb:6.2f} ± {std_comb:5.2f} ms ({fps_comb:5.2f} FPS)")
    logger.info("=" * 90)

    # 4. Xuất Audit JSON
    res_json_paper3 = RESULTS_DIR / "step8_edge_latency_metrics.json"
    res_json_global = CHECKPOINTS_DIR / "res_step8_edge_latency.json"

    export_payload = {
        "experiment": "step8_edge_latency",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform_info": {
            "processor": platform.processor(),
            "python_version": sys.version.split()[0],
            "pytorch_threads": num_threads,
            "device": str(device),
        },
        "num_runs": N_runs,
        "input_resolution": "640x640",
        "latency_metrics": {
            "rtdetr_base_ms": {"mean": round(mean_base, 2), "std": round(std_base, 2), "fps": round(fps_base, 2)},
            "tensor_extraction_ms": {"mean": round(mean_ext, 2), "std": round(std_ext, 2)},
            "decision_scoring_ms": {"mean": round(mean_sc, 4), "std": round(std_sc, 4)},
            "total_guardrail_ms": {"mean": round(mean_guard, 2), "std": round(std_guard, 2)},
            "combined_inference_ms": {"mean": round(mean_comb, 2), "std": round(std_comb, 2), "fps": round(fps_comb, 2)},
            "relative_overhead_percentage": round(overhead_ratio, 2),
        },
        "artifacts": {
            "runs_csv": str(runs_csv_path),
            "log_file": str(log_file),
        }
    }

    save_audit_json(res_json_paper3, export_payload)
    save_audit_json(res_json_global, export_payload)

    logger.info(f"\n-> Đã lưu Audit JSON (Paper 3) : {res_json_paper3}")
    logger.info(f"-> Đã đồng bộ JSON toàn dự án  : {res_json_global}")
    logger.info("\n>>> BƯỚC 8 ĐÃ HOÀN TẤT THÀNH CÔNG VỚI ĐẦY ĐỦ DỮ LIỆU KIỂM TOÁN! <<<")


if __name__ == "__main__":
    run_step8()
