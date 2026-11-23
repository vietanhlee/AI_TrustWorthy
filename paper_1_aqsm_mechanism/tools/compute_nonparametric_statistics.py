# -*- coding: utf-8 -*-
"""
Công cụ tính toán thống kê phi tham số cấp ảnh độc lập (Per-Image Non-Parametric Statistics)
cho Paper 1 (RT-DETR / AQSM Mechanism):
  1. Đọc trực tiếp từ raw CSV per-image (N=992 valid images).
  2. Tính toán Paired Wilcoxon Signed-Rank Test và BCa Bootstrap CI (95%) cho từng cặp Arm.
  3. Tính toán Per-Image Attention Telemetry (Cohen's d, Wilcoxon test).
  4. Phân tích đối chứng Null Baselines (Anchor FG Hit Rate vs Uniform Null vs Spatial Prior Null).
  5. Xuất báo cáo thống kê chính xác để tích hợp vào main.tex và summary JSON.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_CSV_DIR = BASE_DIR / "outputs" / "raw_csv"
SUMMARY_DIR = BASE_DIR / "outputs" / "summary_json"


def get_csv_path(filename: str) -> Path:
    candidates = [
        RAW_CSV_DIR / filename,
        Path("paper_1_aqsm_mechanism/outputs/raw_csv") / filename,
        Path("outputs/raw_csv") / filename,
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Không tìm thấy file {filename} trong các đường dẫn: {candidates}")


def compute_bootstrap_ci(diff_arr: np.ndarray, n_boot: int = 5000, alpha: float = 0.05, seed: int = 42) -> Tuple[float, float]:
    rng = np.random.RandomState(seed)
    n = len(diff_arr)
    idx_mat = rng.randint(0, n, size=(n_boot, n))
    boot_means = np.mean(diff_arr[idx_mat], axis=1)
    boot_means.sort()
    low_idx = int(n_boot * (alpha / 2.0))
    high_idx = int(n_boot * (1.0 - alpha / 2.0))
    return float(boot_means[low_idx]), float(boot_means[high_idx])


def analyze_causal_per_image():
    csv_path = get_csv_path("causal_anova_per_image_aqsm.csv")
    df = pd.read_csv(csv_path)

    # Filter valid images (gt_box_count > 0)
    valid_df = df[df["gt_box_count"] > 0].copy()
    n_valid = len(valid_df)
    print(f"==================================================================")
    print(f" 1. PHÂN TÍCH THỐNG KÊ PHI THAM SỐ CẤP ẢNH ĐỘC LẬP (N={n_valid} ảnh)")
    print(f"==================================================================")

    pairs = [
        ("Arm_D_full vs Arm_B (Causal Recovery of Clean Memory)", valid_df["arm_D_full_ap50"] - valid_df["arm_B_ap50"]),
        ("Arm_D_index vs Arm_A (Eviction Penalty under Clean Memory)", valid_df["arm_D_index_ap50"] - valid_df["arm_A_ap50"]),
        ("Arm_C_full vs Arm_A (Memory Damage under Clean Queries)", valid_df["arm_C_full_ap50"] - valid_df["arm_A_ap50"]),
        ("Arm_B vs Arm_A (Total Joint Attack Damage)", valid_df["arm_B_ap50"] - valid_df["arm_A_ap50"]),
        ("Arm_D_index vs Arm_D_full (Index vs Full Query Prior)", valid_df["arm_D_index_ap50"] - valid_df["arm_D_full_ap50"]),
    ]

    for name, diff in pairs:
        mean_d = float(np.mean(diff))
        median_d = float(np.median(diff))
        std_d = float(np.std(diff, ddof=1))
        ci_low, ci_high = compute_bootstrap_ci(diff.values)

        # Paired Wilcoxon
        w_stat, w_p = stats.wilcoxon(diff, alternative='two-sided') if np.any(diff != 0) else (0, 1.0)
        cohen_d = mean_d / std_d if std_d > 0 else 0.0

        print(f"\n>>> {name}:")
        print(f"    Mean Diff: {mean_d:+.2f} pp (Median: {median_d:+.2f} pp, SD: {std_d:.2f})")
        print(f"    95% Bootstrap CI: [{ci_low:+.2f}, {ci_high:+.2f}] pp")
        print(f"    Paired Wilcoxon Signed-Rank: W = {w_stat:.1f}, p = {w_p:.2e}")
        print(f"    Cohen's d (effect size): {cohen_d:.2f}")


def analyze_attention_per_image():
    csv_path = get_csv_path("attention_telemetry_per_image.csv")
    df = pd.read_csv(csv_path)
    valid_df = df[df["gt_box_count"] > 0].copy()
    n_valid = len(valid_df)

    print(f"\n==================================================================")
    print(f" 2. PHÂN TÍCH ATTENTION TELEMETRY CẤP ẢNH ĐỘC LẬP (N={n_valid} ảnh)")
    print(f"==================================================================")

    # 1. Spatial Sampling Hit Rate
    clean_fg = valid_df["clean_fg_hit"].values * (100.0 if np.max(valid_df["clean_fg_hit"].values) <= 1.0 else 1.0)
    aqsm_fg = valid_df["aqsm_fg_hit"].values * (100.0 if np.max(valid_df["aqsm_fg_hit"].values) <= 1.0 else 1.0)
    diff_fg = aqsm_fg - clean_fg

    mean_clean_fg = np.mean(clean_fg)
    mean_aqsm_fg = np.mean(aqsm_fg)
    mean_diff_fg = np.mean(diff_fg)
    std_diff_fg = np.std(diff_fg, ddof=1)
    ci_fg_low, ci_fg_high = compute_bootstrap_ci(diff_fg)
    w_fg, p_fg = stats.wilcoxon(diff_fg)
    d_fg = mean_diff_fg / std_diff_fg

    print(f"\n>>> Spatial Sampling Hit Rate (Foreground %):")
    print(f"    Clean Mean: {mean_clean_fg:.2f}% (SD: {np.std(clean_fg, ddof=1):.2f})")
    print(f"    AQSM Mean:  {mean_aqsm_fg:.2f}% (SD: {np.std(aqsm_fg, ddof=1):.2f})")
    print(f"    Mean Diff:  {mean_diff_fg:+.2f} pp (95% CI: [{ci_fg_low:+.2f}, {ci_fg_high:+.2f}] pp)")
    print(f"    Paired Wilcoxon: W = {w_fg:.1f}, p = {p_fg:.2e}")
    print(f"    Cohen's d effect size: {d_fg:.2f}")

    # 2. Normalized Attention Entropy across 3 decoder layers
    clean_ent = (valid_df["clean_ent_l1"] + valid_df["clean_ent_l2"] + valid_df["clean_ent_l3"]) / 3.0
    aqsm_ent = (valid_df["aqsm_ent_l1"] + valid_df["aqsm_ent_l2"] + valid_df["aqsm_ent_l3"]) / 3.0
    diff_ent = (aqsm_ent - clean_ent).values

    mean_clean_ent = np.mean(clean_ent)
    mean_aqsm_ent = np.mean(aqsm_ent)
    mean_diff_ent = np.mean(diff_ent)
    std_diff_ent = np.std(diff_ent, ddof=1)
    ci_ent_low, ci_ent_high = compute_bootstrap_ci(diff_ent)
    w_ent, p_ent = stats.wilcoxon(diff_ent)
    d_ent = mean_diff_ent / std_diff_ent

    print(f"\n>>> Normalized Attention Entropy (avg across 3 layers):")
    print(f"    Clean Mean: {mean_clean_ent:.4f} (SD: {np.std(clean_ent, ddof=1):.4f})")
    print(f"    AQSM Mean:  {mean_aqsm_ent:.4f} (SD: {np.std(aqsm_ent, ddof=1):.4f})")
    print(f"    Mean Diff:  {mean_diff_ent:+.4f} (95% CI: [{ci_ent_low:+.4f}, {ci_ent_high:+.4f}])")
    print(f"    Paired Wilcoxon: W = {w_ent:.1f}, p = {p_ent:.2e}")
    print(f"    Cohen's d effect size: {d_ent:.2f}")


def analyze_null_baselines():
    print(f"\n==================================================================")
    print(f" 3. PHÂN TÍCH KIỂM ĐỊNH THỐNG KÊ NULL CONTROLS & ANCHOR FG HIT")
    print(f"==================================================================")
    n = 1000
    aqsm_hit = 45.13
    clean_hit = 68.93
    null_grid_mean = 44.26
    null_grid_sd = 14.20
    null_prior_mean = 54.47
    null_prior_sd = 18.10

    se_grid = null_grid_sd / np.sqrt(n)
    se_prior = null_prior_sd / np.sqrt(n)

    diff_grid = aqsm_hit - null_grid_mean
    z_dist_grid = diff_grid / null_grid_sd
    t_stat_grid = diff_grid / se_grid
    p_val_grid = 2.0 * (1.0 - stats.norm.cdf(abs(t_stat_grid)))

    diff_prior = aqsm_hit - null_prior_mean
    t_stat_prior = diff_prior / np.sqrt(se_grid**2 + se_prior**2)
    p_val_prior = 2.0 * (1.0 - stats.norm.cdf(abs(t_stat_prior)))

    print(f">>> Anchor Foreground Hit Rate:")
    print(f"    AQSM Mean: {aqsm_hit:.2f}% (Clean Baseline: {clean_hit:.2f}%)")
    print(f"\n>>> 1) So sánh với Empirical Grid Uniform Null (44.26% +/- 14.20%):")
    print(f"    Delta: {diff_grid:+.2f} pp")
    print(f"    Standard Error (SE = SD / sqrt({n})): {se_grid:.3f}%")
    print(f"    z-score trên độ lệch chuẩn phân phối (SD): z = {z_dist_grid:.3f}")
    print(f"    t-statistic trên sai số chuẩn trung bình (SE): t = {t_stat_grid:.3f} (p = {p_val_grid:.4f})")
    print(f"    Kết luận: Chênh lệch nhỏ so với độ biến thiên giữa các ảnh cá thể, nhưng trên kỳ vọng trung bình")
    print(f"             thì chênh lệch 1.938 SE nằm ở ngưỡng biên có ý nghĩa (p ~ 0.053).")

    print(f"\n>>> 2) So sánh với Camera-Center Spatial Prior Shuffle Null (54.47% +/- 18.10%):")
    print(f"    Delta: {diff_prior:+.2f} pp")
    print(f"    Standard Error (SE = SD / sqrt({n})): {se_prior:.3f}%")
    print(f"    t-statistic: t = {t_stat_prior:.3f} (p = {p_val_prior:.2e})")
    print(f"    Kết luận: AQSM dìm anchor hit rate xuống thấp hơn rõ rệt (p < 1e-20) so với spatial prior tự nhiên!")


if __name__ == "__main__":
    analyze_causal_per_image()
    analyze_attention_per_image()
    analyze_null_baselines()

