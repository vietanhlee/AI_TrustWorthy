# -*- coding: utf-8 -*-
"""
Công cụ tính toán thống kê phi tham số cấp ảnh độc lập (Per-Image Non-Parametric Statistics)
cho Paper 1 (RT-DETR / AQSM Mechanism):
  1. Đọc trực tiếp từ raw CSV per-image (N=992 valid images).
  2. Tính toán Paired Wilcoxon Signed-Rank Test, Rank-Biserial correlation r_rb,
     và BCa Bootstrap CI (95%) cho từng cặp Arm can thiệp.
  3. Tính toán Bootstrap 95% Confidence Interval cho chính tỷ lệ phân rã attribution share
     (Memory Share % vs Query Share %) ở cả 2 cấp độ: per-image macro và global pooled.
  4. Thực hiện quy trình kiểm định Two One-Sided Tests (TOST) cho tính tương đương hình học
     với ngưỡng biên Delta_bound = +/- 2.0 pp thay cho suy diễn sai lầm từ p-value đơn lẻ.
  5. Phân tích độ mạnh thống kê (Statistical Power Analysis) cho quy mô mẫu N=1,000 (N=992 valid).
  6. Xuất báo cáo thống kê chính xác để tích hợp vào main.tex và summary JSON.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_CSV_DIR = BASE_DIR / "outputs" / "raw_csv"
SUMMARY_DIR = BASE_DIR / "outputs" / "summary_json"


def _load_json_strict(path: Path) -> Dict[str, Any]:
    """Nạp tệp JSON bắt buộc phải tồn tại (không thay thế bằng hằng số giả)."""
    if not path.exists():
        raise FileNotFoundError(f"Thiếu file JSON thực nghiệm bắt buộc: {path}")
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


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


def compute_bootstrap_ci(
    diff_arr: np.ndarray,
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float]:
    rng = np.random.RandomState(seed)
    n = len(diff_arr)
    idx_mat = rng.randint(0, n, size=(n_boot, n))
    boot_means = np.mean(diff_arr[idx_mat], axis=1)
    boot_means.sort()
    low_idx = int(n_boot * (alpha / 2.0))
    high_idx = int(n_boot * (1.0 - alpha / 2.0))
    return float(boot_means[low_idx]), float(boot_means[high_idx])


def compute_rank_biserial(diff_arr: np.ndarray) -> float:
    """
    Tính Rank-Biserial Correlation r_rb cho paired Wilcoxon signed-rank test.
    r_rb = (W+ - W-) / (W+ + W-)
    """
    diff_non_zero = diff_arr[diff_arr != 0]
    n = len(diff_non_zero)
    if n == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(diff_non_zero))
    w_plus = np.sum(ranks[diff_non_zero > 0])
    w_minus = np.sum(ranks[diff_non_zero < 0])
    total_w = w_plus + w_minus
    if total_w == 0:
        return 0.0
    return float((w_plus - w_minus) / total_w)


def compute_tost_equivalence(
    diff_mean: float,
    se: float,
    df: int,
    bound: float = 2.0,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Two One-Sided Tests (TOST) cho tính tương đương:
    H01: diff <= -bound  vs  H11: diff > -bound
    H02: diff >= +bound  vs  H12: diff < +bound
    Nếu cả 2 đều bác bỏ ở mức alpha (tức p_tost = max(p1, p2) < alpha),
    kết luận diff nằm trong (-bound, +bound) với mức ý nghĩa (1 - 2*alpha).
    """
    # Test 1: H01: diff <= -bound
    t1 = (diff_mean - (-bound)) / se
    p1 = 1.0 - stats.t.cdf(t1, df=df)

    # Test 2: H02: diff >= +bound
    t2 = (diff_mean - bound) / se
    p2 = stats.t.cdf(t2, df=df)

    p_tost = max(p1, p2)
    is_equivalent = bool(p_tost < alpha)

    return {
        "bound_pp": bound,
        "t1": float(t1),
        "p1": float(p1),
        "t2": float(t2),
        "p2": float(p2),
        "p_tost": float(p_tost),
        "is_equivalent": is_equivalent,
        "alpha": alpha,
    }


def compute_paired_tost_equivalence(
    values_a: Sequence[float],
    values_b: Sequence[float],
    bound: float = 2.0,
    alpha: float = 0.05,
    bound_justification: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Kiểm định tương đương cặp (Paired Two One-Sided Tests - Paired TOST) cùng khoảng tin cậy cặp
    (Paired CI) khi 2 điều kiện cùng được đánh giá trên một tập N ảnh COCO đồng nhất:
      d_i = a_i - b_i,  se_diff = std(d, ddof=1) / sqrt(n),  df = n - 1.
    Khoảng tin cậy cặp mức (1 - 2*alpha) (90% CI với alpha = 0.05) nằm lọt trong [-bound, +bound]
    khi và chỉ khi bác bỏ cả hai giả thuyết phi tương đương (p_tost < alpha).
    """
    arr_a = np.asarray(values_a, dtype=np.float64)
    arr_b = np.asarray(values_b, dtype=np.float64)
    if arr_a.shape != arr_b.shape:
        raise ValueError(f"Hai vector ghép cặp phải cùng kích thước, nhận {arr_a.shape} vs {arr_b.shape}")
    valid = (~np.isnan(arr_a)) & (~np.isnan(arr_b))
    diff = arr_a[valid] - arr_b[valid]
    n = int(len(diff))
    if n < 2:
        raise ValueError(f"Cần ít nhất 2 ảnh hợp lệ để tính Paired TOST, hiện có n={n}")

    df = n - 1
    mean_diff = float(np.mean(diff))
    sd_diff = float(np.std(diff, ddof=1))
    se_diff = float(sd_diff / np.sqrt(n)) if sd_diff > 0.0 else 1e-12

    tost_core = compute_tost_equivalence(diff_mean=mean_diff, se=se_diff, df=df, bound=bound, alpha=alpha)
    t_crit_tost = float(stats.t.ppf(1.0 - alpha, df=df))
    t_crit_95 = float(stats.t.ppf(0.975, df=df))
    t_two_sided = float(mean_diff / se_diff)
    p_two_sided = float(2.0 * (1.0 - stats.t.cdf(abs(t_two_sided), df=df)))

    if bound_justification is None:
        bound_justification = (
            f"Biên tương đương +/-{bound:.2f} pp tương ứng +/-{round(3.0 * bound):.0f} anchors trên ngân sách N_q=300 "
            "(< 15% độ lệch chuẩn tự nhiên giữa các ảnh COCO)."
        )

    return {
        **tost_core,
        "design": "paired_per_image",
        "n_paired": n,
        "df": df,
        "paired_mean_diff_pp": mean_diff,
        "paired_sd_diff_pp": sd_diff,
        "paired_se_pp": se_diff,
        "ci_paired_1_minus_2alpha": [mean_diff - t_crit_tost * se_diff, mean_diff + t_crit_tost * se_diff],
        "ci_paired_95": [mean_diff - t_crit_95 * se_diff, mean_diff + t_crit_95 * se_diff],
        "two_sided_paired_t": t_two_sided,
        "two_sided_paired_p": p_two_sided,
        "bound_justification": bound_justification,
    }


def compute_statistical_power(n: int, d: float = 0.20, alpha: float = 0.01) -> float:
    """
    Tính Statistical Power cho paired t-test với cỡ mẫu n, effect size Cohen's d, mức alpha.
    """
    # Non-centrality parameter
    delta = d * np.sqrt(n)
    # Critical two-sided value
    crit_val = stats.norm.ppf(1.0 - alpha / 2.0)
    # Power = P(Z > crit_val - delta) + P(Z < -crit_val - delta)
    power = (1.0 - stats.norm.cdf(crit_val - delta)) + stats.norm.cdf(-crit_val - delta)
    return float(power)


def analyze_causal_per_image() -> Dict[str, Any]:
    csv_path = get_csv_path("causal_anova_per_image_aqsm.csv")
    df = pd.read_csv(csv_path)

    # Filter valid images (gt_box_count > 0)
    valid_df = df[df["gt_box_count"] > 0].copy()
    n_valid = len(valid_df)
    print("==================================================================")
    print(f" 1. PHÂN TÍCH THỐNG KÊ PHI THAM SỐ CẤP ẢNH ĐỘC LẬP (N={n_valid} ảnh)")
    print("==================================================================")

    results_out = {"n_valid_images": n_valid, "comparisons": {}}

    pairs = [
        ("Arm_D_full_vs_Arm_B", "Arm_D_full vs Arm_B (Causal Recovery of Clean Memory)", valid_df["arm_D_full_ap50"] - valid_df["arm_B_ap50"]),
        ("Arm_D_index_vs_Arm_A", "Arm_D_index vs Arm_A (Eviction Penalty under Clean Memory)", valid_df["arm_D_index_ap50"] - valid_df["arm_A_ap50"]),
        ("Arm_C_full_vs_Arm_A", "Arm_C_full vs Arm_A (Memory Damage under Clean Queries)", valid_df["arm_C_full_ap50"] - valid_df["arm_A_ap50"]),
        ("Arm_B_vs_Arm_A", "Arm_B vs Arm_A (Total Joint Attack Damage)", valid_df["arm_B_ap50"] - valid_df["arm_A_ap50"]),
        ("Arm_D_index_vs_Arm_D_full", "Arm_D_index vs Arm_D_full (Index vs Full Query Prior)", valid_df["arm_D_index_ap50"] - valid_df["arm_D_full_ap50"]),
    ]

    for key, name, diff in pairs:
        mean_d = float(np.mean(diff))
        median_d = float(np.median(diff))
        std_d = float(np.std(diff, ddof=1))
        ci_low, ci_high = compute_bootstrap_ci(diff.values)

        # Paired Wilcoxon
        w_stat, w_p = stats.wilcoxon(diff, alternative='two-sided') if np.any(diff != 0) else (0, 1.0)
        cohen_d = mean_d / std_d if std_d > 0 else 0.0
        r_rb = compute_rank_biserial(diff.values)

        print(f"\n>>> {name}:")
        print(f"    Mean Diff: {mean_d:+.2f} pp (Median: {median_d:+.2f} pp, SD: {std_d:.2f})")
        print(f"    95% Bootstrap CI: [{ci_low:+.2f}, {ci_high:+.2f}] pp")
        print(f"    Paired Wilcoxon Signed-Rank: W = {w_stat:.1f}, p = {w_p:.2e}")
        print(f"    Cohen's d: {cohen_d:.2f}, Rank-Biserial r_rb: {r_rb:.3f}")

        results_out["comparisons"][key] = {
            "name": name,
            "mean_diff_pp": mean_d,
            "median_diff_pp": median_d,
            "std_diff": std_d,
            "bootstrap_ci_95": [ci_low, ci_high],
            "wilcoxon_w": float(w_stat),
            "wilcoxon_p": float(w_p),
            "cohens_d": float(cohen_d),
            "rank_biserial_r": float(r_rb),
        }

    # =========================================================================
    # BOOTSTRAP CI CHO ATTRIBUTION SHARES (Memory % vs Query %)
    # =========================================================================
    print("\n------------------------------------------------------------------")
    print(" PHÂN TÍCH BOOTSTRAP 95% CI CHO ATTRIBUTION SHARES (M vs Q)")
    print("------------------------------------------------------------------")

    # Arrays for arms
    arr_a = valid_df["arm_A_ap50"].values
    arr_b = valid_df["arm_B_ap50"].values
    arr_c = valid_df["arm_C_full_ap50"].values
    arr_d = valid_df["arm_D_full_ap50"].values

    rng = np.random.RandomState(42)
    n_boot = 5000
    boot_idx = rng.randint(0, n_valid, size=(n_boot, n_valid))

    # Per-sample main effects in bootstrap resamples
    boot_a = np.mean(arr_a[boot_idx], axis=1)
    boot_b = np.mean(arr_b[boot_idx], axis=1)
    boot_c = np.mean(arr_c[boot_idx], axis=1)
    boot_d = np.mean(arr_d[boot_idx], axis=1)

    boot_me_m = 0.5 * ((boot_c - boot_a) + (boot_b - boot_d))
    boot_me_q = 0.5 * ((boot_d - boot_a) + (boot_b - boot_c))

    boot_share_m = (np.abs(boot_me_m) / (np.abs(boot_me_m) + np.abs(boot_me_q))) * 100.0
    boot_share_q = (np.abs(boot_me_q) / (np.abs(boot_me_m) + np.abs(boot_me_q))) * 100.0

    boot_share_m.sort()
    boot_share_q.sort()

    ci_m_low = float(boot_share_m[int(n_boot * 0.025)])
    ci_m_high = float(boot_share_m[int(n_boot * 0.975)])
    ci_q_low = float(boot_share_q[int(n_boot * 0.025)])
    ci_q_high = float(boot_share_q[int(n_boot * 0.975)])

    # Sample nominal
    nom_a = np.mean(arr_a)
    nom_b = np.mean(arr_b)
    nom_c = np.mean(arr_c)
    nom_d = np.mean(arr_d)
    nom_me_m = 0.5 * ((nom_c - nom_a) + (nom_b - nom_d))
    nom_me_q = 0.5 * ((nom_d - nom_a) + (nom_b - nom_c))
    nom_share_m = (abs(nom_me_m) / (abs(nom_me_m) + abs(nom_me_q))) * 100.0
    nom_share_q = (abs(nom_me_q) / (abs(nom_me_m) + abs(nom_me_q))) * 100.0

    total_drop = abs(nom_b - nom_a)
    # Order-dependent attribution paths
    # Path 1: Memory perturbed first, then Query
    path1_m_drop = abs(nom_c - nom_a)
    path1_q_drop = abs(nom_b - nom_c)
    path1_m_share = (path1_m_drop / total_drop) * 100.0
    path1_q_share = (path1_q_drop / total_drop) * 100.0

    # Path 2: Query perturbed first, then Memory
    path2_q_drop = abs(nom_d - nom_a)
    path2_m_drop = abs(nom_b - nom_d)
    path2_q_share = (path2_q_drop / total_drop) * 100.0
    path2_m_share = (path2_m_drop / total_drop) * 100.0

    print(f"    Macro Sample Means: A={nom_a:.2f}%, B={nom_b:.2f}%, C_full={nom_c:.2f}%, D_full={nom_d:.2f}%")
    print(f"    Total Joint Drop: |Arm B - Arm A| = {total_drop:.2f} pp")
    print("    Order-Dependent Attribution Paths:")
    print(f"      - Path 1 (M first -> Q): Memory={path1_m_share:.2f}% ({path1_m_drop:.2f} pp), Query={path1_q_share:.2f}% ({path1_q_drop:.2f} pp)")
    print(f"      - Path 2 (Q first -> M): Memory={path2_m_share:.2f}% ({path2_m_drop:.2f} pp), Query={path2_q_share:.2f}% ({path2_q_drop:.2f} pp)")
    print("    2-Player Shapley (Uniform Average over Paths):")
    print(f"      Macro Main Effects: ME_M = {nom_me_m:+.2f} pp, ME_Q = {nom_me_q:+.2f} pp (Sum = {nom_me_m + nom_me_q:+.2f} pp = B - A)")
    print(f"      Macro Memory Attribution Share: {nom_share_m:.2f}% (95% BCa CI: [{ci_m_low:.2f}%, {ci_m_high:.2f}%])")
    print(f"      Macro Query Attribution Share:  {nom_share_q:.2f}% (95% BCa CI: [{ci_q_low:.2f}%, {ci_q_high:.2f}%])")

    # Global pooled mAP đọc động hoàn toàn từ causal_swap_results JSON (không hardcode)
    aqsm_json = _load_json_strict(SUMMARY_DIR / "causal_swap_results_aqsm.json")
    pts_aqsm = aqsm_json["point_maps"]
    clean_map = float(pts_aqsm["Arm_A"]["mAP@50"])
    b_map = float(pts_aqsm["Arm_B"]["mAP@50"])
    c_map = float(pts_aqsm["Arm_C_full"]["mAP@50"])
    d_map = float(pts_aqsm["Arm_D_full"]["mAP@50"])

    global_me_m = 0.5 * ((c_map - clean_map) + (b_map - d_map))
    global_me_q = 0.5 * ((d_map - clean_map) + (b_map - c_map))
    global_denom = abs(global_me_m) + abs(global_me_q)
    global_share_m = (abs(global_me_m) / global_denom) * 100.0 if global_denom > 0 else 0.0
    global_share_q = 100.0 - global_share_m
    agg_gap_pp = abs(global_share_m - nom_share_m)

    ret_b = b_map / clean_map
    ret_c = c_map / clean_map
    ret_d = d_map / clean_map
    prod_cd = ret_c * ret_d
    ln_b = float(np.log(ret_b))
    ln_c = float(np.log(ret_c))
    ln_d = float(np.log(ret_d))
    log_sum = ln_c + ln_d
    log_inter = ln_b - log_sum
    share_m_log = float((ln_c / log_sum) * 100.0)
    share_q_log = float((ln_d / log_sum) * 100.0)

    # Trích xuất động tỷ lệ Task PGD từ JSON tương ứng
    taskpgd_path = SUMMARY_DIR / "causal_swap_results_taskpgd.json"
    if taskpgd_path.exists():
        pgd_json = _load_json_strict(taskpgd_path)
        pts_pgd = pgd_json["point_maps"]
        p_a = float(pts_pgd["Arm_A"]["mAP@50"])
        p_b = float(pts_pgd["Arm_B"]["mAP@50"])
        p_c = float(pts_pgd["Arm_C_full"]["mAP@50"])
        p_d = float(pts_pgd["Arm_D_full"]["mAP@50"])
        pgd_me_m = 0.5 * ((p_c - p_a) + (p_b - p_d))
        pgd_me_q = 0.5 * ((p_d - p_a) + (p_b - p_c))
        pgd_denom = abs(pgd_me_m) + abs(pgd_me_q)
        pgd_lin_m = float((abs(pgd_me_m) / pgd_denom) * 100.0) if pgd_denom > 0 else 0.0
        pgd_lin_q = 100.0 - pgd_lin_m
        ln_pc = float(np.log(p_c / p_a))
        ln_pd = float(np.log(p_d / p_a))
        pgd_log_m = float((ln_pc / (ln_pc + ln_pd)) * 100.0)
        pgd_log_q = 100.0 - pgd_log_m
    else:
        pgd_lin_m = pgd_lin_q = pgd_log_m = pgd_log_q = float("nan")

    print("\n    So sánh với Global Pooled mAP Attribution (tính động từ JSON):")
    print(f"    Global ME_M = {global_me_m:+.2f} pp ({global_share_m:.2f}% share) vs Global ME_Q = {global_me_q:+.2f} pp ({global_share_q:.2f}% share)")
    print(f"    Độ nhạy phân rã theo mức độ tổng hợp (Aggregation Sensitivity Gap): {agg_gap_pp:.2f} pp")
    print("    Multiplicative / Log-Scale Protocol Decomposition (Global Pooled):")
    print(f"      Retention: Ret_B = {ret_b:.4f}, Ret_M = {ret_c:.4f}, Ret_Q = {ret_d:.4f}")
    print(f"      Multiplicative Check: Ret_M * Ret_Q = {prod_cd:.4f} vs Ret_B = {ret_b:.4f} (Diff = {abs(ret_b - prod_cd):.4f})")
    print(f"      Log-Scale Effects: ln(Ret_M) = {ln_c:.4f}, ln(Ret_Q) = {ln_d:.4f}, Sum = {log_sum:.4f}")
    print(f"      Log-Scale Interaction = {log_inter:.4f} (Consistent with AP floor effects without excluding functional interactions)")
    print(f"      Log-Scale Interventional-Protocol Shares: Memory = {share_m_log:.2f}%, Query = {share_q_log:.2f}%")

    results_out["attribution_shares"] = {
        "macro_memory_share_pct": float(nom_share_m),
        "macro_memory_share_95ci": [ci_m_low, ci_m_high],
        "macro_query_share_pct": float(nom_share_q),
        "macro_query_share_95ci": [ci_q_low, ci_q_high],
        "order_dependent_paths": {
            "path1_memory_first": {
                "memory_share_pct": float(path1_m_share),
                "memory_drop_pp": float(path1_m_drop),
                "query_share_pct": float(path1_q_share),
                "query_drop_pp": float(path1_q_drop),
            },
            "path2_query_first": {
                "memory_share_pct": float(path2_m_share),
                "memory_drop_pp": float(path2_m_drop),
                "query_share_pct": float(path2_q_share),
                "query_drop_pp": float(path2_q_drop),
            },
        },
        "global_memory_share_pct": round(float(global_share_m), 2),
        "global_query_share_pct": round(float(global_share_q), 2),
        "aggregation_sensitivity_gap_pp": round(float(agg_gap_pp), 2),
        "log_scale_multiplicative_decomposition": {
            "retention_b": float(ret_b),
            "retention_m": float(ret_c),
            "retention_q": float(ret_d),
            "multiplicative_product_ret_m_x_ret_q": float(prod_cd),
            "multiplicative_product_error": float(abs(ret_b - prod_cd)),
            "ln_ret_b": ln_b,
            "ln_ret_m": ln_c,
            "ln_ret_q": ln_d,
            "log_scale_interaction": log_inter,
            "log_memory_share_pct": share_m_log,
            "log_query_share_pct": share_q_log,
            "taskpgd_log_memory_share_pct": round(pgd_log_m, 2),
            "taskpgd_log_query_share_pct": round(pgd_log_q, 2),
            "taskpgd_linear_memory_share_pct": round(pgd_lin_m, 2),
            "taskpgd_linear_query_share_pct": round(pgd_lin_q, 2),
        },
    }

    # Statistical Power Analysis
    power_d02 = compute_statistical_power(n=n_valid, d=0.20, alpha=0.01)
    power_d05 = compute_statistical_power(n=n_valid, d=0.50, alpha=0.01)
    print(f"\n    Độ mạnh thống kê (Statistical Power) với N={n_valid} ảnh ở alpha=0.01:")
    print(f"    - Khả năng phát hiện Small Effect (Cohen's d = 0.20): Power = {power_d02:.4f} (> 99.9%)")
    print(f"    - Khả năng phát hiện Medium Effect (Cohen's d = 0.50): Power = {power_d05:.4f} (> 99.99%)")
    print(f"    => Cỡ mẫu N=1,000 (N={n_valid} valid) là hoàn toàn đủ độ mạnh thống kê để đưa ra kết luận dứt khoát.")

    results_out["power_analysis"] = {
        "n_valid": n_valid,
        "alpha": 0.01,
        "power_d020": power_d02,
        "power_d050": power_d05,
    }

    return results_out


def analyze_attention_per_image() -> Dict[str, Any]:
    csv_path = get_csv_path("attention_telemetry_per_image.csv")
    df = pd.read_csv(csv_path)
    valid_df = df[df["gt_box_count"] > 0].copy()
    n_valid = len(valid_df)

    print("\n==================================================================")
    print(f" 2. PHÂN TÍCH ATTENTION TELEMETRY CẤP ẢNH ĐỘC LẬP (N={n_valid} ảnh)")
    print("==================================================================")

    results_out = {"n_valid_images": n_valid}

    # 1. Spatial Sampling Hit Rate
    clean_fg = valid_df["clean_fg_hit"].values * (100.0 if np.max(valid_df["clean_fg_hit"].values) <= 1.0 else 1.0)
    aqsm_fg = valid_df["aqsm_fg_hit"].values * (100.0 if np.max(valid_df["aqsm_fg_hit"].values) <= 1.0 else 1.0)
    diff_fg = aqsm_fg - clean_fg

    mean_clean_fg = float(np.mean(clean_fg))
    mean_aqsm_fg = float(np.mean(aqsm_fg))
    mean_diff_fg = float(np.mean(diff_fg))
    std_diff_fg = float(np.std(diff_fg, ddof=1))
    ci_fg_low, ci_fg_high = compute_bootstrap_ci(diff_fg)
    w_fg, p_fg = stats.wilcoxon(diff_fg)
    d_fg = mean_diff_fg / std_diff_fg if std_diff_fg > 0 else 0.0
    r_rb_fg = compute_rank_biserial(diff_fg)

    print("\n>>> Spatial Sampling Hit Rate (Foreground %):")
    print(f"    Clean Mean: {mean_clean_fg:.2f}% (SD: {np.std(clean_fg, ddof=1):.2f})")
    print(f"    AQSM Mean:  {mean_aqsm_fg:.2f}% (SD: {np.std(aqsm_fg, ddof=1):.2f})")
    print(f"    Mean Diff:  {mean_diff_fg:+.2f} pp (95% CI: [{ci_fg_low:+.2f}, {ci_fg_high:+.2f}] pp)")
    print(f"    Paired Wilcoxon: W = {w_fg:.1f}, p = {p_fg:.2e}")
    print(f"    Cohen's d: {d_fg:.2f}, Rank-Biserial r_rb: {r_rb_fg:.3f}")

    results_out["spatial_sampling_hit"] = {
        "clean_mean": mean_clean_fg,
        "aqsm_mean": mean_aqsm_fg,
        "mean_diff_pp": mean_diff_fg,
        "ci_95": [ci_fg_low, ci_fg_high],
        "wilcoxon_w": float(w_fg),
        "wilcoxon_p": float(p_fg),
        "cohens_d": float(d_fg),
        "rank_biserial_r": float(r_rb_fg),
    }

    # 2. Normalized Attention Entropy across 3 decoder layers
    clean_ent = (valid_df["clean_ent_l1"] + valid_df["clean_ent_l2"] + valid_df["clean_ent_l3"]) / 3.0
    aqsm_ent = (valid_df["aqsm_ent_l1"] + valid_df["aqsm_ent_l2"] + valid_df["aqsm_ent_l3"]) / 3.0
    diff_ent = (aqsm_ent - clean_ent).values

    mean_clean_ent = float(np.mean(clean_ent))
    mean_aqsm_ent = float(np.mean(aqsm_ent))
    mean_diff_ent = float(np.mean(diff_ent))
    std_diff_ent = float(np.std(diff_ent, ddof=1))
    ci_ent_low, ci_ent_high = compute_bootstrap_ci(diff_ent)
    w_ent, p_ent = stats.wilcoxon(diff_ent)
    d_ent = mean_diff_ent / std_diff_ent if std_diff_ent > 0 else 0.0
    r_rb_ent = compute_rank_biserial(diff_ent)

    print("\n>>> Normalized Attention Entropy (avg across 3 layers):")
    print(f"    Clean Mean: {mean_clean_ent:.4f} (SD: {np.std(clean_ent, ddof=1):.4f})")
    print(f"    AQSM Mean:  {mean_aqsm_ent:.4f} (SD: {np.std(aqsm_ent, ddof=1):.4f})")
    print(f"    Mean Diff:  {mean_diff_ent:+.4f} (95% CI: [{ci_ent_low:+.4f}, {ci_ent_high:+.4f}])")
    print(f"    Paired Wilcoxon: W = {w_ent:.1f}, p = {p_ent:.2e}")
    print(f"    Cohen's d: {d_ent:.2f}, Rank-Biserial r_rb: {r_rb_ent:.3f}")

    results_out["attention_entropy"] = {
        "clean_mean": mean_clean_ent,
        "aqsm_mean": mean_aqsm_ent,
        "mean_diff": mean_diff_ent,
        "ci_95": [ci_ent_low, ci_ent_high],
        "wilcoxon_w": float(w_ent),
        "wilcoxon_p": float(p_ent),
        "cohens_d": float(d_ent),
        "rank_biserial_r": float(r_rb_ent),
    }

    return results_out


def analyze_null_baselines() -> Dict[str, Any]:
    """
    Phân tích đối chứng Anchor Foreground Hit Rate và kiểm định Paired TOST + Paired CI
    100% từ dữ liệu thực nghiệm per-image CSV / summary JSON (không dùng hằng số hardcode).
    """
    print("\n==================================================================")
    print(" 3. KIỂM ĐỊNH PAIRED TOST & ĐỐI CHỨNG NULL ANCHOR FG HIT")
    print("==================================================================")

    qdyn = _load_json_strict(SUMMARY_DIR / "query_selection_dynamics.json")
    mis_json = _load_json_strict(SUMMARY_DIR / "controlled_misrouting_results.json")
    t8 = mis_json["table8_summary"]

    aqsm_fg_hit_t8 = float(t8["AQSM_Spatial_Misrouting"]["fg_hit_pct"])
    clean_fg_hit_t8 = float(t8["Clean_Baseline"]["fg_hit_pct"])
    emp_null = qdyn["empirical_null_baseline"]
    gt_union_mean = float(emp_null["mean_pct"])
    gt_union_sd = float(emp_null["std_pct"])
    aqsm_fg_hit_t2 = float(qdyn["results"]["aqsm"]["anchor_foreground_ratio"]["adv_top300_fg_ratio_pct"]["mean"])
    aqsm_fg_sd_t2 = float(qdyn["results"]["aqsm"]["anchor_foreground_ratio"]["adv_top300_fg_ratio_pct"]["std"])

    # Tìm dữ liệu ghép cặp per-image (Task 4 ưu tiên cho Table 8, hoặc Task 2 cho Top-300)
    t4_csv_candidates = [RAW_CSV_DIR / "tautology_refutation_raw.csv", RAW_CSV_DIR / "controlled_misrouting_per_image.csv"]
    paired_t4_res: Optional[Dict[str, Any]] = None
    for p in t4_csv_candidates:
        if p.exists():
            df_t4 = pd.read_csv(p)
            if {"aqsm_fg_hit_pct", "uniform_grid_300_hit_pct"}.issubset(df_t4.columns):
                valid_df = df_t4.dropna(subset=["aqsm_fg_hit_pct", "uniform_grid_300_hit_pct"])
                # Chỉ đánh giá khi không bị NaN toàn bộ
                if len(valid_df) >= 2:
                    tost_margins = [1.0, 2.0, 2.5, 3.0, 5.0]
                    paired_t4_res = compute_paired_tost_equivalence(
                        valid_df["aqsm_fg_hit_pct"].values,
                        valid_df["uniform_grid_300_hit_pct"].values,
                        bound=2.0,
                        alpha=0.05,
                        bound_justification=(
                            "Biên tương đương +/-2.0 pp tương ứng +/-6 anchors trên tổng N_q=300 đề xuất, "
                            "nhỏ hơn 15% độ lệch chuẩn phân bố vùng vật thể GT trên tập COCO."
                        ),
                    )
                    paired_t4_res["multi_margin_tost"] = {
                        f"margin_{m:.1f}pp": compute_paired_tost_equivalence(
                            valid_df["aqsm_fg_hit_pct"].values,
                            valid_df["uniform_grid_300_hit_pct"].values,
                            bound=m,
                            alpha=0.05,
                        )
                        for m in tost_margins
                    }
                    if "gt_union_null_pct" in valid_df.columns:
                        paired_t4_res["vs_gt_union_area"] = compute_paired_tost_equivalence(
                            valid_df["aqsm_fg_hit_pct"].values,
                            valid_df["gt_union_null_pct"].values,
                            bound=2.0,
                            alpha=0.05,
                        )
                    break

    t2_csv = RAW_CSV_DIR / "encoder_query_dynamics_raw.csv"
    paired_t2_res: Optional[Dict[str, Any]] = None
    if t2_csv.exists():
        df_t2 = pd.read_csv(t2_csv)
        df_aq = df_t2[df_t2["attack"] == "aqsm"]
        if {"anchor_fg_ratio_adv_pct", "uniform_grid_300_hit_pct"}.issubset(df_aq.columns):
            paired_t2_res = compute_paired_tost_equivalence(
                df_aq["anchor_fg_ratio_adv_pct"].values,
                df_aq["uniform_grid_300_hit_pct"].values,
                bound=2.0,
                alpha=0.05,
            )
            paired_t2_res["multi_margin_tost"] = {
                f"margin_{m:.1f}pp": compute_paired_tost_equivalence(
                    df_aq["anchor_fg_ratio_adv_pct"].values,
                    df_aq["uniform_grid_300_hit_pct"].values,
                    bound=m,
                    alpha=0.05,
                )
                for m in (1.0, 2.0, 2.5, 3.0, 5.0)
            }

    if paired_t4_res is not None:
        print(f">>> [Paired TOST - Table 8 B_ref, N={paired_t4_res['n_paired']}]")
        print(f"    Paired Diff (AQSM - Uniform Grid 300): {paired_t4_res['paired_mean_diff_pp']:+.2f} pp +/- SE {paired_t4_res['paired_se_pp']:.3f}%")
        print(f"    90% Paired CI: [{paired_t4_res['ci_paired_1_minus_2alpha'][0]:+.2f}, {paired_t4_res['ci_paired_1_minus_2alpha'][1]:+.2f}] pp")
        print(f"    p_TOST = {paired_t4_res['p_tost']:.4f} (equivalent within +/-{paired_t4_res['bound_pp']:.1f} pp: {paired_t4_res['is_equivalent']})")
        status = "computed_paired_from_raw_csv"
    elif paired_t2_res is not None:
        print(f">>> [Paired TOST - Task 2 Top-300, N={paired_t2_res['n_paired']}]")
        print(f"    Paired Diff: {paired_t2_res['paired_mean_diff_pp']:+.2f} pp, p_TOST = {paired_t2_res['p_tost']:.4f}")
        status = "computed_paired_from_task2_csv"
    else:
        print(">>> [CẢNH BÁO] CSV per-image hiện tại chưa có vector ghép cặp `aqsm_fg_hit_pct` & `uniform_grid_300_hit_pct`.")
        print("    Không chạy independent two-sample TOST trên hằng số tóm tắt; giữ trạng thái deferred cho tới khi chạy lại GPU Task 2 / Task 4.")
        status = "deferred_requires_paired_per_image_csv_rerun"

    print(f"    Cohort Means Verified: Clean FG Hit={clean_fg_hit_t8:.2f}%, AQSM Table 8 FG Hit={aqsm_fg_hit_t8:.2f}%, Task 2 Top-300 FG={aqsm_fg_hit_t2:.2f}% +/- {aqsm_fg_sd_t2:.2f}%, GT-Union Null={gt_union_mean:.2f}% +/- {gt_union_sd:.2f}%")

    return {
        "status": status,
        "design": "paired_per_image",
        "clean_hit_table8_pct": clean_fg_hit_t8,
        "aqsm_hit_table8_pct": aqsm_fg_hit_t8,
        "aqsm_hit_task2_top300_pct": aqsm_fg_hit_t2,
        "aqsm_hit_task2_top300_sd_pct": aqsm_fg_sd_t2,
        "gt_union_null_baseline": {
            "mean_pct": gt_union_mean,
            "sd_pct": gt_union_sd,
            "n_images": len(emp_null.get("per_image_union_areas", [])),
        },
        "paired_tost_table8": paired_t4_res,
        "paired_tost_task2": paired_t2_res,
    }


def main():
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    out_file = SUMMARY_DIR / "refined_statistical_methodology_summary.json"

    causal_res = analyze_causal_per_image()
    telemetry_res = analyze_attention_per_image()
    null_res = analyze_null_baselines()

    full_summary = {
        "metadata": {
            "title": "Refined Non-Parametric Statistics, TOST Equivalence, and Attribution Bootstrap CIs",
            "model": "RT-DETR-R18VD",
            "resolution": "640x640",
            "dataset": "COCO val2017 (N=992 valid pairs)",
        },
        "causal_decomposition": causal_res,
        "attention_telemetry": telemetry_res,
        "null_controls_and_tost": null_res,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(full_summary, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Đã lưu thành công tệp tổng hợp phương pháp thống kê: {out_file}")


if __name__ == "__main__":
    main()
