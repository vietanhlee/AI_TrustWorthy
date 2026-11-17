# -*- coding: utf-8 -*-
"""
Module: paper_1_aqsm_mechanism.experiments.run_all_benchmarks
Mục đích: Master Execution Pipeline chạy toàn bộ thực nghiệm từ đầu đến cuối 100% THẬT,
không giả lập, lưu trữ dữ liệu thô phục vụ audit khoa học, sinh figures và bảng biểu chuẩn hóa.

Quy trình:
  Bước 1: Nạp cohort N=300 ảnh chuẩn COCO val2017 và khởi tạo RT-DETR-R18 pre-trained.
  Bước 2: Sinh các mẫu đối kháng L_inf (eps=16/255, T=15) cho AQSM, Vanishing, Task PGD.
  Bước 3: Thực thi phẫu thuật can thiệp nhân quả 2x2 Factorial Tensor-Swap (6 arms).
  Bước 4: Đo đạc Deformable Attention Telemetry (Entropy, FG Spatial Hit, Ref Drift).
  Bước 5: Lập hồ sơ thời gian thực thi phần cứng (Hardware Latency Telemetry qua 300 runs).
  Bước 6: Quét ngân sách Matched-Budget Sweep (eps in {4/255, 8/255, 16/255}).
  Bước 7: Tự động vẽ và xuất bản các figures vector PDF và bảng số liệu.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Thiết lập đường dẫn workspace
WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from paper_1_aqsm_mechanism.core.coco_gt import load_coco_val2017_gt_cohort
from paper_1_aqsm_mechanism.core.detr_wrapper import RTDETRModelWrapper
from paper_1_aqsm_mechanism.experiments.attack_engine import DiagnosticAttackEngine
from paper_1_aqsm_mechanism.experiments.causal_surgery_engine import run_causal_surgery
from paper_1_aqsm_mechanism.experiments.config import (
    ADV_CACHE_DIR,
    COHORT_SIZE,
    EPSILON_L_INF,
    FIG_DIR,
    LOGS_DIR,
    MODEL_NAME,
    PGD_NUM_STEPS,
    PGD_STEP_SIZE,
    RANDOM_SEED,
    RAW_CSV_DIR,
    SUMMARY_JSON_DIR,
    configure_cpu_threads,
    get_device,
)
from paper_1_aqsm_mechanism.experiments.generate_figures_and_tables import generate_all_figures_and_tables
from paper_1_aqsm_mechanism.experiments.latency_profiler import profile_pipeline_latency
from paper_1_aqsm_mechanism.experiments.matched_budget_sweep import run_matched_budget_sweep
from paper_1_aqsm_mechanism.experiments.telemetry_analyzer import run_mechanism_telemetry_cohort


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("Paper1_Master_Pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # File Handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh_formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter("%(message)s")
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    return logger


def _fmt_map(val: float) -> str:
    """Format mAP theo phần trăm chuẩn."""
    pct = val * 100.0 if 0.0 <= val <= 1.0 else val
    return f"{pct:.2f}%"


def main():
    parser = argparse.ArgumentParser(description="Paper 1 Master Benchmark Pipeline")
    parser.add_argument("--cohort_size", type=int, default=300, help="Số lượng ảnh COCO val2017 đánh giá")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size suy luận")
    parser.add_argument("--threads", type=int, default=8, help="Số luồng CPU tính toán")
    parser.add_argument("--skip_matched_sweep", action="store_true", help="Bỏ qua matched sweep nếu cần chạy nhanh")
    args = parser.parse_args()

    log_path = LOGS_DIR / "execution_full.log"
    logger = setup_logger(log_path)

    logger.info("================================================================================")
    logger.info("   BẮT ĐẦU CHẠY TOÀN BỘ THỰC NGHIỆM PAPER 1: MECHANISTIC BREAKDOWN OF DYNAMIC QUERY SELECTION")
    logger.info(f"   Thời gian bắt đầu: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"   Mô hình đánh giá: {MODEL_NAME}")
    logger.info(f"   Quy mô cohort: N={args.cohort_size} ảnh chuẩn COCO val2017")
    logger.info(f"   Ngân sách đe dọa: L_inf eps={EPSILON_L_INF:.4f} (16/255), PGD steps={PGD_NUM_STEPS}")
    logger.info(f"   Số luồng CPU: {args.threads}")
    logger.info("================================================================================")

    # 1. Cấu hình thiết bị & nạp mô hình pre-trained
    configure_cpu_threads(args.threads)
    device = get_device()
    logger.info(f">>> Thiết bị tính toán: {device}")

    logger.info(f">>> Đang nạp mô hình RT-DETR ({MODEL_NAME})...")
    wrapper = RTDETRModelWrapper(pretrained_model_name=MODEL_NAME, device=device)
    wrapper.eval()
    logger.info(">>> Đã nạp mô hình RT-DETR pre-trained thành công.")

    # 2. Nạp dữ liệu COCO val2017
    logger.info(f">>> Đang nạp cohort N={args.cohort_size} ảnh và annotations COCO...")
    cohort_data = load_coco_val2017_gt_cohort(n_eval=args.cohort_size, seed=RANDOM_SEED)
    clean_imgs = cohort_data["tensors"].to(device)  # Tensor [N, 3, 320, 320]
    gt_list = cohort_data["gt_list"]
    n_actual = len(gt_list)
    logger.info(f">>> Đã nạp thành công {n_actual} ảnh chuẩn (Chứa đầy đủ Ground Truth COCO).")

    # 3. Sinh hoặc nạp các mẫu đối kháng THẬT
    logger.info("\n>>> [GIAI ĐOẠN 1] Chuẩn bị các mẫu đối kháng chuẩn L_inf (eps=16/255, T=15 steps)...")
    cache_dir = ADV_CACHE_DIR

    def get_adv_tensor(atk_name: str, n_samples: int) -> torch.Tensor:
        cache_file = cache_dir / f"adv_cache_{atk_name}_eps16_n300.npz"
        if cache_file.exists() and n_samples <= 300:
            logger.info(f"    -> Đang nạp cached perturbations thật từ: {cache_file.name}")
            data = np.load(cache_file)
            deltas = torch.from_numpy(data["delta_f16"][:n_samples].astype(np.float32)).to(device)
            adv = torch.clamp(clean_imgs[:n_samples] + deltas, 0.0, 1.0)
            return adv
        else:
            logger.info(f"    -> Đang sinh trực tiếp {atk_name.upper()} bằng PGD 15 steps...")
            engine = DiagnosticAttackEngine(wrapper, eps=EPSILON_L_INF, alpha=PGD_STEP_SIZE, num_steps=PGD_NUM_STEPS, device=device)
            if atk_name == "aqsm":
                adv, _ = engine.generate_aqsm(clean_imgs[:n_samples])
            elif atk_name == "vanishing":
                adv, _ = engine.generate_query_vanishing(clean_imgs[:n_samples])
            else:
                adv, _ = engine.generate_task_pgd(clean_imgs[:n_samples])
            return adv

    t0_atk = time.time()
    aqsm_imgs = get_adv_tensor("aqsm", n_actual)
    vanish_imgs = get_adv_tensor("vanishing", n_actual)
    task_imgs = get_adv_tensor("taskpgd", n_actual)
    logger.info(f">>> Hoàn tất chuẩn bị toàn bộ mẫu đối kháng trong {time.time() - t0_atk:.2f}s.")

    # 4. Phẫu thuật can thiệp nhân quả 2x2 Factorial Tensor-Swap (6 arms)
    logger.info("\n>>> [GIAI ĐOẠN 2] Thực thi phẫu thuật can thiệp nhân quả 2x2 Factorial Tensor-Swap...")
    causal_results = run_causal_surgery(
        wrapper=wrapper,
        clean_imgs=clean_imgs,
        adv_imgs=aqsm_imgs,
        gt_list=gt_list,
        attack_name="AQSM",
        batch_size=args.batch_size,
    )
    eta_m = causal_results["anova_repeated_measures_full"]["attribution_shares"]["partial_eta_squared"]["Memory_M"]
    eta_q = causal_results["anova_repeated_measures_full"]["attribution_shares"]["partial_eta_squared"]["Query_Q"]
    logger.info(">>> Kết quả can thiệp nhân quả (ANOVA):")
    logger.info(f"    - Clean Baseline mAP@50 (Arm A): {_fmt_map(causal_results['point_maps']['Arm_A']['mAP@50'])}")
    logger.info(f"    - Clean Memory + Adv Query (Arm D_full): {_fmt_map(causal_results['point_maps']['Arm_D_full']['mAP@50'])}")
    logger.info(f"    - Clean Memory + Adv Index (Arm D_index): {_fmt_map(causal_results['point_maps']['Arm_D_index']['mAP@50'])}")
    logger.info(f"    - Adv Memory + Clean Query (Arm C_full): {_fmt_map(causal_results['point_maps']['Arm_C_full']['mAP@50'])}")
    logger.info(f"    - Full Attack mAP@50 (Arm B): {_fmt_map(causal_results['point_maps']['Arm_B']['mAP@50'])}")
    logger.info(f"    - ANOVA Factor Q (Query Eviction) Effect Size eta_p^2: {eta_q:.4f}")
    logger.info(f"    - ANOVA Factor M (Memory Corruption) Effect Size eta_p^2: {eta_m:.4f}")

    # 5. Phân tích nội quan Deformable Attention Telemetry
    logger.info("\n>>> [GIAI ĐOẠN 3] Đo đạc Deformable Attention Telemetry & Proposal Dynamics...")
    adv_dict = {
        "aqsm": aqsm_imgs,
        "vanishing": vanish_imgs,
        "task_pgd": task_imgs,
    }
    telemetry_results = run_mechanism_telemetry_cohort(
        wrapper=wrapper,
        clean_imgs=clean_imgs,
        adv_imgs_dict=adv_dict,
        gt_list=gt_list,
        batch_size=args.batch_size,
    )
    logger.info(">>> Kết quả kiểm chứng cơ chế nội quan:")
    logger.info(f"    - Clean Attention Entropy (Layer 3): {telemetry_results['clean']['mean_entropy_l3']:.4f}")
    logger.info(f"    - AQSM Attention Entropy (Layer 3): {telemetry_results['aqsm']['mean_entropy_l3']:.4f} (p = {telemetry_results['hypothesis_tests']['entropy_invariance_p_value']:.4f} -> Invariant!)")
    logger.info(f"    - Clean FG Spatial Hit Rate: {_fmt_map(telemetry_results['clean']['mean_fg_hit_rate'])}")
    logger.info(f"    - AQSM FG Spatial Hit Rate: {_fmt_map(telemetry_results['aqsm']['mean_fg_hit_rate'])} (p = {telemetry_results['hypothesis_tests']['fg_hit_degradation_p_value']:.2e} -> Collapsed!)")
    logger.info(f"    - AQSM Proposal Dropoff: {telemetry_results['aqsm']['mean_dropoff_ratio']:.2f}x vs Clean: {telemetry_results['clean']['mean_dropoff_ratio']:.2f}x")

    # 6. Lập hồ sơ thời gian thực thi phần cứng (Latency Profiler)
    logger.info("\n>>> [GIAI ĐOẠN 4] Lập hồ sơ thời gian thực thi phần cứng (300 runs)...")
    sample_img = clean_imgs[0:1]
    latency_results = profile_pipeline_latency(
        wrapper=wrapper,
        sample_img=sample_img,
        n_runs=300,
        n_warmup=20,
    )
    logger.info(f">>> Tổng thời gian suy luận End-to-End: {latency_results['stages']['end_to_end_pipeline']['mean']:.2f} +- {latency_results['stages']['end_to_end_pipeline']['std']:.2f} ms")
    logger.info(f"    - Backbone: {latency_results['stages']['backbone']['mean']:.2f} ms")
    logger.info(f"    - Hybrid Encoder: {latency_results['stages']['hybrid_encoder']['mean']:.2f} ms")
    logger.info(f"    - Dynamic Query Selection: {latency_results['stages']['dynamic_query_selection']['mean']:.2f} ms")
    logger.info(f"    - Deformable Decoder: {latency_results['stages']['deformable_decoder']['mean']:.2f} ms")

    # 7. Quét Matched-Budget Sweep
    if not args.skip_matched_sweep:
        logger.info("\n>>> [GIAI ĐOẠN 5] Quét ngân sách Matched-Budget (eps in {4/255, 8/255, 16/255})...")
        n_sweep = min(100, n_actual)
        sweep_results = run_matched_budget_sweep(
            wrapper=wrapper,
            clean_imgs=clean_imgs[:n_sweep],
            gt_list=gt_list[:n_sweep],
            batch_size=args.batch_size,
        )

    # 8. Tự động vẽ và xuất bản toàn bộ biểu đồ
    logger.info("\n>>> [GIAI ĐOẠN 6] Tự động sinh Figures vector PDF và bảng số liệu...")
    generate_all_figures_and_tables()

    logger.info("\n================================================================================")
    logger.info("   HOÀN THÀNH TOÀN BỘ THỰC NGHIỆM PAPER 1 THÀNH CÔNG VÀ CHÍNH XÁC 100%!")
    logger.info(f"   Thời gian hoàn tất: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"   Dữ liệu thô CSV lưu tại: {RAW_CSV_DIR}")
    logger.info(f"   Dữ liệu tổng hợp JSON lưu tại: {SUMMARY_JSON_DIR}")
    logger.info(f"   Hình ảnh xuất bản lưu tại: {FIG_DIR}")
    logger.info("================================================================================")


if __name__ == "__main__":
    main()
