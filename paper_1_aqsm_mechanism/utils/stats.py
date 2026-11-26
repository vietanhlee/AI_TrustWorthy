# -*- coding: utf-8 -*-
"""
Module: paper_1_aqsm_mechanism.utils.stats
Mục đích: Cung cấp các thuật toán phân tích thống kê suy luận chuyên sâu phục vụ
kiểm chứng giả thuyết khoa học cho Paper 1:
  1. Phân tích phương sai lặp lại hai yếu tố: 2x2 Repeated-Measures ANOVA (Factor M x Factor Q).
  2. Ước lượng khoảng tin cậy 95% Confidence Interval bằng Paired Bootstrap resampling.
  3. Kiểm định phi tham số Wilcoxon signed-rank test.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from scipy import stats


def bootstrap_paired_diff(
    vals_a: np.ndarray,
    vals_b: np.ndarray,
    n_boot: int = 5000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Tính trung bình chênh lệch ghép cặp (vals_a - vals_b), độ lệch chuẩn,
    khoảng tin cậy 95% Bootstrap CI và p-value kiểm định Wilcoxon signed-rank.
    """
    vals_a = np.asarray(vals_a)
    vals_b = np.asarray(vals_b)
    mask = (~np.isnan(vals_a)) & (~np.isnan(vals_b))
    diff = np.asarray(vals_a[mask] - vals_b[mask], dtype=np.float64)
    n = len(diff)
    if n == 0:
        return {"n": 0, "mean_diff": 0.0, "std_diff": 0.0, "ci_95": [0.0, 0.0], "wilcoxon_p": 1.0}

    mean_d = float(np.mean(diff))
    std_d = float(np.std(diff, ddof=1)) if n > 1 else 0.0
    rng = np.random.RandomState(seed)
    idx_mat = rng.randint(0, n, size=(n_boot, n))
    boot_means = diff[idx_mat].mean(axis=1)
    ci = np.percentile(boot_means, [2.5, 97.5]).tolist()
    _, p_wilcoxon = stats.wilcoxon(diff) if np.any(diff != 0) else (0.0, 1.0)
    return {
        "n": int(n),
        "mean_diff": mean_d,
        "std_diff": std_d,
        "ci_95": [float(ci[0]), float(ci[1])],
        "wilcoxon_p": float(p_wilcoxon),
    }


def repeated_measures_2x2_anova(
    y_00: np.ndarray,
    y_11: np.ndarray,
    y_10: np.ndarray,
    y_01: np.ndarray,
    n_boot: int = 10000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Thực hiện phân tích phương sai Repeated-Measures 2x2 chuẩn mực cho:
      - Factor M (Encoder Memory: 0=Clean, 1=Adv)
      - Factor Q (Query Prior: 0=Clean, 1=Adv)
    với Y_00=Arm A, Y_01=Arm D_full, Y_10=Arm C_full, Y_11=Arm B.

    Tách bạch rõ:
      1. Hiệu ứng biên ME(M), ME(Q), Inter(M x Q) kèm 95% Bootstrap CI.
      2. Tỷ lệ L1 trên giá trị tuyệt đối của hiệu ứng biên vs Tổng suy giảm Y_11 - Y_00.
      3. Bảng ANOVA chuẩn (SS, MS, F, p-value, eta^2_conditions, partial eta^2_p).
    """
    y_00 = np.asarray(y_00)
    y_11 = np.asarray(y_11)
    y_10 = np.asarray(y_10)
    y_01 = np.asarray(y_01)

    valid = (~np.isnan(y_00)) & (~np.isnan(y_11)) & (~np.isnan(y_10)) & (~np.isnan(y_01))
    ya = np.asarray(y_00[valid], dtype=np.float64)
    yb = np.asarray(y_11[valid], dtype=np.float64)
    yc = np.asarray(y_10[valid], dtype=np.float64)
    yd = np.asarray(y_01[valid], dtype=np.float64)
    n = len(ya)
    if n < 2:
        raise ValueError(f"Số mẫu hợp lệ không đủ cho ANOVA (N={n})")

    # Hiệu ứng biên theo từng ảnh
    vec_me_M = 0.5 * ((yc - ya) + (yb - yd))
    vec_me_Q = 0.5 * ((yd - ya) + (yb - yc))
    vec_inter = 0.5 * ((yb - yc) - (yd - ya))
    vec_total = yb - ya  # Đúng bằng vec_me_M + vec_me_Q

    mean_me_M = float(np.mean(vec_me_M))
    mean_me_Q = float(np.mean(vec_me_Q))
    mean_inter = float(np.mean(vec_inter))
    mean_total = float(np.mean(vec_total))

    rng = np.random.RandomState(seed)
    idx_mat = rng.randint(0, n, size=(n_boot, n))
    ci_M = np.percentile(vec_me_M[idx_mat].mean(axis=1), [2.5, 97.5]).tolist()
    ci_Q = np.percentile(vec_me_Q[idx_mat].mean(axis=1), [2.5, 97.5]).tolist()
    ci_Inter = np.percentile(vec_inter[idx_mat].mean(axis=1), [2.5, 97.5]).tolist()
    ci_Total = np.percentile(vec_total[idx_mat].mean(axis=1), [2.5, 97.5]).tolist()

    # Ma trận Y [N, 2, 2] với axis 1 = M (0,1), axis 2 = Q (0,1)
    y_mat = np.zeros((n, 2, 2), dtype=np.float64)
    y_mat[:, 0, 0] = ya
    y_mat[:, 0, 1] = yd
    y_mat[:, 1, 0] = yc
    y_mat[:, 1, 1] = yb

    grand_mean = np.mean(y_mat)
    cell_means = np.mean(y_mat, axis=0)
    m_means = np.mean(y_mat, axis=(0, 2))
    q_means = np.mean(y_mat, axis=(0, 1))

    ss_M = float(2 * n * np.sum((m_means - grand_mean) ** 2))
    ss_Q = float(2 * n * np.sum((q_means - grand_mean) ** 2))
    ss_inter = float(n * np.sum((cell_means - m_means[:, None] - q_means[None, :] + grand_mean) ** 2))

    # Sai số nội chủ thể cho thiết kế repeated-measures 2x2
    subj_mean = np.mean(y_mat, axis=(1, 2), keepdims=True)
    ss_subjects = float(4 * np.sum((subj_mean - grand_mean) ** 2))

    m_subj = np.mean(y_mat, axis=2)  # [N, 2]
    ss_err_M = float(2 * np.sum((m_subj - m_means[None, :] - subj_mean[:, :, 0] + grand_mean) ** 2))

    q_subj = np.mean(y_mat, axis=1)  # [N, 2]
    ss_err_Q = float(2 * np.sum((q_subj - q_means[None, :] - subj_mean[:, 0, :] + grand_mean) ** 2))

    resid_inter = (
        y_mat
        - m_subj[:, :, None]
        - q_subj[:, None, :]
        - cell_means[None, :, :]
        + m_means[None, :, None]
        + q_means[None, None, :]
        + subj_mean
        - grand_mean
    )
    ss_err_inter = float(np.sum(resid_inter ** 2))
    ss_total_all = float(np.sum((y_mat - grand_mean) ** 2))

    df_eff = 1
    df_err = n - 1
    f_M = (ss_M / df_eff) / max(ss_err_M / df_err, 1e-12)
    f_Q = (ss_Q / df_eff) / max(ss_err_Q / df_err, 1e-12)
    f_inter = (ss_inter / df_eff) / max(ss_err_inter / df_err, 1e-12)

    p_M = float(1.0 - stats.f.cdf(f_M, df_eff, df_err))
    p_Q = float(1.0 - stats.f.cdf(f_Q, df_eff, df_err))
    p_inter = float(1.0 - stats.f.cdf(f_inter, df_eff, df_err))

    ss_cond = max(ss_M + ss_Q + ss_inter, 1e-12)
    l1_sum = max(abs(mean_me_M) + abs(mean_me_Q) + abs(mean_inter), 1e-12)
    signed_denom = mean_total if abs(mean_total) > 1e-12 else 1.0

    return {
        "N": int(n),
        "df": [df_eff, df_err],
        "marginal_effects": {
            "ME_M": mean_me_M,
            "ME_M_ci95": [float(ci_M[0]), float(ci_M[1])],
            "ME_Q": mean_me_Q,
            "ME_Q_ci95": [float(ci_Q[0]), float(ci_Q[1])],
            "Interaction_MxQ": mean_inter,
            "Interaction_ci95": [float(ci_Inter[0]), float(ci_Inter[1])],
            "Total_Y11_minus_Y00": mean_total,
            "Total_ci95": [float(ci_Total[0]), float(ci_Total[1])],
        },
        "attribution_shares": {
            "signed_fraction_of_total_drop_pct": {
                "Memory_M": float(mean_me_M / signed_denom * 100.0),
                "Query_Q": float(mean_me_Q / signed_denom * 100.0),
                "note": "Y_11 - Y_00 = ME_M + ME_Q (tổng đúng 100%)",
            },
            "l1_marginal_share_pct": {
                "Memory_M": float(abs(mean_me_M) / l1_sum * 100.0),
                "Query_Q": float(abs(mean_me_Q) / l1_sum * 100.0),
                "Interaction_MxQ": float(abs(mean_inter) / l1_sum * 100.0),
                "note": "|ME| / (|ME_M| + |ME_Q| + |Inter|)",
            },
            "anova_condition_ss_share_pct": {
                "Memory_M": float(ss_M / ss_cond * 100.0),
                "Query_Q": float(ss_Q / ss_cond * 100.0),
                "Interaction_MxQ": float(ss_inter / ss_cond * 100.0),
                "note": "SS_factor / (SS_M + SS_Q + SS_inter)",
            },
            "partial_eta_squared": {
                "Memory_M": float(ss_M / max(ss_M + ss_err_M, 1e-12)),
                "Query_Q": float(ss_Q / max(ss_Q + ss_err_Q, 1e-12)),
                "Interaction_MxQ": float(ss_inter / max(ss_inter + ss_err_inter, 1e-12)),
            },
        },
        "anova_table": {
            "SS_subjects": ss_subjects,
            "SS_total": ss_total_all,
            "Memory_M": {"SS": ss_M, "SS_err": ss_err_M, "F": float(f_M), "p": p_M},
            "Query_Q": {"SS": ss_Q, "SS_err": ss_err_Q, "F": float(f_Q), "p": p_Q},
            "Interaction_MxQ": {"SS": ss_inter, "SS_err": ss_err_inter, "F": float(f_inter), "p": p_inter},
        },
    }


def _safe_finite_float(val: Any) -> Optional[float]:
    """Chuyển đổi giá trị sang float hữu hạn an toàn cho chuẩn JSON nghiêm ngặt."""
    if val is None:
        return None
    f_val = float(val)
    return f_val if np.isfinite(f_val) else None


def _compute_rank_biserial_paired(diff: np.ndarray) -> float:
    """
    Tính hệ số tương quan hạng nhị phân (matched-pairs rank-biserial correlation)
    từ hiệu ghép cặp trên từng ảnh, bỏ qua các cặp có hiệu bằng 0 theo quy ước Wilcoxon.
    """
    nonzero = diff[diff != 0.0]
    if nonzero.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero))
    total_rank_sum = float(np.sum(ranks))
    if total_rank_sum <= 0.0:
        return 0.0
    w_pos = float(np.sum(ranks[nonzero > 0.0]))
    w_neg = float(np.sum(ranks[nonzero < 0.0]))
    return float((w_pos - w_neg) / total_rank_sum)


def summarize_across_seed_stability(
    seed_values: Sequence[float],
    baseline_value: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Tier 1 (Across-Seed Stability): Tổng hợp phân bố qua K seed với
    Mean, Std, Min, Max, Range và Seed Signal-to-Noise Ratio (SNR_seed).
    """
    arr = np.asarray(seed_values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n_seeds = int(arr.size)
    if n_seeds == 0:
        return {
            "n_seeds": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "range": None,
            "median": None,
            "cv_pct": None,
            "zero_seed_variance": True,
            "baseline_value": _safe_finite_float(baseline_value),
            "signed_diff_mean": None,
            "abs_effect_mean": None,
            "snr_seed": None,
        }

    mean_v = float(np.mean(arr))
    std_v = float(np.std(arr, ddof=1)) if n_seeds > 1 else 0.0
    min_v = float(np.min(arr))
    max_v = float(np.max(arr))
    range_v = float(max_v - min_v)
    median_v = float(np.median(arr))
    cv_pct = float(abs(std_v / mean_v) * 100.0) if abs(mean_v) > 1e-12 else None
    zero_var = bool(std_v <= 1e-12)

    base_clean = _safe_finite_float(baseline_value)
    signed_diff = float(mean_v - base_clean) if base_clean is not None else None
    abs_effect = float(abs(signed_diff)) if signed_diff is not None else None
    # Không xuất Infinity ra JSON khi độ lệch chuẩn qua các seed bằng 0.
    snr_seed = (
        float(abs_effect / std_v)
        if (abs_effect is not None and not zero_var)
        else None
    )

    return {
        "n_seeds": n_seeds,
        "mean": mean_v,
        "std": std_v,
        "min": min_v,
        "max": max_v,
        "range": range_v,
        "median": median_v,
        "cv_pct": cv_pct,
        "zero_seed_variance": zero_var,
        "baseline_value": base_clean,
        "signed_diff_mean": signed_diff,
        "abs_effect_mean": abs_effect,
        "snr_seed": snr_seed,
    }


def compute_inter_seed_consistency(
    per_image_vectors_by_seed: Dict[Any, Sequence[float]],
) -> Dict[str, Any]:
    """
    Tier 2b (Inter-Seed Image-Level Consistency): Đánh giá mức độ nhất quán thứ tự
    và đồng thuận tuyệt đối giữa các seed trên tập N ảnh hợp lệ dùng chung.
    Tính Spearman rho, Pearson r từng cặp seed và Two-Way Agreement ICC(2, 1).
    """
    seed_keys = list(per_image_vectors_by_seed.keys())
    k_seeds = len(seed_keys)
    if k_seeds < 2:
        return {
            "n_seeds": k_seeds,
            "n_valid_images": 0,
            "pairwise_spearman": {},
            "pairwise_pearson": {},
            "mean_pairwise_spearman_rho": None,
            "min_pairwise_spearman_rho": None,
            "mean_pairwise_pearson_r": None,
            "min_pairwise_pearson_r": None,
            "icc_2_1": None,
        }

    cols: List[np.ndarray] = [
        np.asarray(per_image_vectors_by_seed[k], dtype=np.float64) for k in seed_keys
    ]
    min_len = min(c.shape[0] for c in cols)
    if min_len == 0:
        return {
            "n_seeds": k_seeds,
            "n_valid_images": 0,
            "pairwise_spearman": {},
            "pairwise_pearson": {},
            "mean_pairwise_spearman_rho": None,
            "min_pairwise_spearman_rho": None,
            "mean_pairwise_pearson_r": None,
            "min_pairwise_pearson_r": None,
            "icc_2_1": None,
        }

    mat = np.column_stack([c[:min_len] for c in cols])
    valid_mask = np.all(np.isfinite(mat), axis=1)
    mat = mat[valid_mask]
    n_valid = int(mat.shape[0])

    if n_valid < 2:
        return {
            "n_seeds": k_seeds,
            "n_valid_images": n_valid,
            "pairwise_spearman": {},
            "pairwise_pearson": {},
            "mean_pairwise_spearman_rho": None,
            "min_pairwise_spearman_rho": None,
            "mean_pairwise_pearson_r": None,
            "min_pairwise_pearson_r": None,
            "icc_2_1": None,
        }

    pairwise_spearman: Dict[str, Optional[float]] = {}
    pairwise_pearson: Dict[str, Optional[float]] = {}
    spearman_vals: List[float] = []
    pearson_vals: List[float] = []

    for i in range(k_seeds):
        for j in range(i + 1, k_seeds):
            pair_name = f"{seed_keys[i]}_vs_{seed_keys[j]}"
            vi = mat[:, i]
            vj = mat[:, j]
            const_i = bool(np.std(vi) <= 1e-12)
            const_j = bool(np.std(vj) <= 1e-12)
            if const_i and const_j:
                rho = 1.0 if np.allclose(vi, vj) else None
                pear = 1.0 if np.allclose(vi, vj) else None
            elif const_i or const_j:
                rho = None
                pear = None
            else:
                rho_res = stats.spearmanr(vi, vj)
                pear_res = stats.pearsonr(vi, vj)
                rho = _safe_finite_float(rho_res.statistic)
                pear = _safe_finite_float(pear_res.statistic)

            pairwise_spearman[pair_name] = rho
            pairwise_pearson[pair_name] = pear
            if rho is not None:
                spearman_vals.append(rho)
            if pear is not None:
                pearson_vals.append(pear)

    # Two-way random-effects absolute agreement single-measure ICC(2, 1)
    grand_mean = float(np.mean(mat))
    row_means = np.mean(mat, axis=1)
    col_means = np.mean(mat, axis=0)

    ss_r = float(k_seeds * np.sum((row_means - grand_mean) ** 2))
    ss_c = float(n_valid * np.sum((col_means - grand_mean) ** 2))
    ss_tot = float(np.sum((mat - grand_mean) ** 2))
    ss_e = max(0.0, ss_tot - ss_r - ss_c)

    ms_r = ss_r / float(n_valid - 1)
    ms_c = ss_c / float(k_seeds - 1)
    ms_e = ss_e / float((n_valid - 1) * (k_seeds - 1))

    denom = ms_r + (k_seeds - 1) * ms_e + (k_seeds * (ms_c - ms_e)) / float(n_valid)
    if ss_tot <= 1e-15:
        icc_val: Optional[float] = 1.0
    elif denom <= 1e-15:
        icc_val = None
    else:
        icc_val = _safe_finite_float((ms_r - ms_e) / denom)

    return {
        "n_seeds": k_seeds,
        "n_valid_images": n_valid,
        "pairwise_spearman": pairwise_spearman,
        "pairwise_pearson": pairwise_pearson,
        "mean_pairwise_spearman_rho": float(np.mean(spearman_vals)) if spearman_vals else None,
        "min_pairwise_spearman_rho": float(np.min(spearman_vals)) if spearman_vals else None,
        "mean_pairwise_pearson_r": float(np.mean(pearson_vals)) if pearson_vals else None,
        "min_pairwise_pearson_r": float(np.min(pearson_vals)) if pearson_vals else None,
        "icc_2_1": icc_val,
    }


def evaluate_multiseed_statistical_evidence(
    baseline_per_image: Sequence[float],
    seed_per_image_map: Dict[Any, Sequence[float]],
    seed_scalar_summary: Optional[Dict[Any, float]] = None,
    clean_scalar_baseline: Optional[float] = None,
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Bộ kiểm định thống kê 2 tầng cho thực nghiệm đa seed (tránh kiểm định sai với K=3):
    - Tier 1 (Across-Seed Stability): Mean, Std, [Min, Max], SNR_seed trên độ đo tổng hợp
      cũng như trung bình ghép cặp theo ảnh qua từng seed.
    - Tier 2 (Within-Seed Per-Image Inference & Consistency):
      (a) Conjunction / Worst-case seed inference: Wilcoxon signed-rank + Bootstrap 95% CI
          + rank-biserial r_rb riêng cho TỪNG seed, báo cáo seed yếu nhất (worst-case).
      (b) Inter-seed image-level consistency: Spearman rho từng cặp seed + ICC(2, 1).
      (c) Cluster-pooled per-image mean across seeds vs Clean baseline.
    """
    base_arr = np.asarray(baseline_per_image, dtype=np.float64)
    seed_keys = list(seed_per_image_map.keys())

    # Tier 1a: Trên scalar tổng hợp (ví dụ dataset mAP@50 mỗi seed) nếu được cung cấp
    scalar_tier1: Optional[Dict[str, Any]] = None
    if seed_scalar_summary:
        ordered_scalars = [seed_scalar_summary[k] for k in seed_keys if k in seed_scalar_summary]
        if not ordered_scalars:
            ordered_scalars = list(seed_scalar_summary.values())
        scalar_tier1 = summarize_across_seed_stability(
            ordered_scalars, baseline_value=clean_scalar_baseline
        )

    # Tier 2a: Chạy kiểm định ghép cặp theo ảnh độc lập cho TỪNG seed
    per_seed_paired: Dict[str, Dict[str, Any]] = {}
    per_seed_mean_deltas: List[float] = []
    worst_seed_id: Optional[str] = None
    worst_tuple: Optional[Tuple[float, float]] = None  # (largest p, -abs(mean_diff))

    for s_key in seed_keys:
        s_str = str(s_key)
        s_arr = np.asarray(seed_per_image_map[s_key], dtype=np.float64)
        n_pair = min(base_arr.shape[0], s_arr.shape[0])
        b_sub = base_arr[:n_pair]
        s_sub = s_arr[:n_pair]
        valid = np.isfinite(b_sub) & np.isfinite(s_sub)
        b_v = b_sub[valid]
        s_v = s_sub[valid]

        if b_v.size == 0:
            res = {
                "n": 0,
                "mean_diff": None,
                "std_diff": None,
                "ci_95": [None, None],
                "wilcoxon_p": None,
                "rank_biserial_r": None,
            }
        else:
            boot_res = bootstrap_paired_diff(s_v, b_v, n_boot=n_boot, seed=seed)
            r_rb = _compute_rank_biserial_paired(s_v - b_v)
            res = {
                "n": int(boot_res["n"]),
                "mean_diff": _safe_finite_float(boot_res["mean_diff"]),
                "std_diff": _safe_finite_float(boot_res["std_diff"]),
                "ci_95": [
                    _safe_finite_float(boot_res["ci_95"][0]),
                    _safe_finite_float(boot_res["ci_95"][1]),
                ],
                "wilcoxon_p": _safe_finite_float(boot_res["wilcoxon_p"]),
                "rank_biserial_r": _safe_finite_float(r_rb),
            }
            per_seed_mean_deltas.append(float(boot_res["mean_diff"]))

            # Xác định worst-case seed: p-value lớn nhất, tie-break theo |mean_diff| nhỏ nhất
            cand = (float(boot_res["wilcoxon_p"]), -abs(float(boot_res["mean_diff"])))
            if worst_tuple is None or cand > worst_tuple:
                worst_tuple = cand
                worst_seed_id = s_str

        per_seed_paired[s_str] = res

    # Tier 1b: Độ ổn định của trung bình chênh lệch theo ảnh qua các seed
    per_image_delta_tier1 = summarize_across_seed_stability(
        per_seed_mean_deltas, baseline_value=0.0
    )

    valid_seed_results = [
        v for v in per_seed_paired.values() if v.get("wilcoxon_p") is not None
    ]
    if valid_seed_results and worst_seed_id is not None:
        worst_res = per_seed_paired[worst_seed_id]
        worst_p = max(float(v["wilcoxon_p"]) for v in valid_seed_results)
        min_abs_diff = min(abs(float(v["mean_diff"])) for v in valid_seed_results)
        min_abs_rrb = min(abs(float(v["rank_biserial_r"])) for v in valid_seed_results)
        all_p05 = all(float(v["wilcoxon_p"]) < 0.05 for v in valid_seed_results)
        all_p001 = all(float(v["wilcoxon_p"]) < 0.001 for v in valid_seed_results)
        all_p1e10 = all(float(v["wilcoxon_p"]) < 1e-10 for v in valid_seed_results)
        worst_ci = worst_res["ci_95"]
    else:
        worst_p = None
        min_abs_diff = None
        min_abs_rrb = None
        all_p05 = False
        all_p001 = False
        all_p1e10 = False
        worst_ci = [None, None]

    # Tier 2b: Inter-seed image-level consistency
    consistency = compute_inter_seed_consistency(seed_per_image_map)

    # Tier 2c: Cluster-pooled per-image mean across seeds vs baseline
    pooled_paired: Dict[str, Any] = {
        "n": 0,
        "mean_diff": None,
        "std_diff": None,
        "ci_95": [None, None],
        "wilcoxon_p": None,
        "rank_biserial_r": None,
    }
    if seed_keys and base_arr.size > 0:
        min_len = min(
            [base_arr.shape[0]]
            + [np.asarray(seed_per_image_map[k]).shape[0] for k in seed_keys]
        )
        if min_len > 0:
            mat = np.column_stack(
                [np.asarray(seed_per_image_map[k], dtype=np.float64)[:min_len] for k in seed_keys]
            )
            b_clip = base_arr[:min_len]
            has_any_seed = np.any(np.isfinite(mat), axis=1) & np.isfinite(b_clip)
            if np.any(has_any_seed):
                pooled_vec = np.nanmean(mat[has_any_seed], axis=1)
                b_pool = b_clip[has_any_seed]
                boot_pool = bootstrap_paired_diff(
                    pooled_vec, b_pool, n_boot=n_boot, seed=seed
                )
                pooled_paired = {
                    "n": int(boot_pool["n"]),
                    "mean_diff": _safe_finite_float(boot_pool["mean_diff"]),
                    "std_diff": _safe_finite_float(boot_pool["std_diff"]),
                    "ci_95": [
                        _safe_finite_float(boot_pool["ci_95"][0]),
                        _safe_finite_float(boot_pool["ci_95"][1]),
                    ],
                    "wilcoxon_p": _safe_finite_float(boot_pool["wilcoxon_p"]),
                    "rank_biserial_r": _safe_finite_float(
                        _compute_rank_biserial_paired(pooled_vec - b_pool)
                    ),
                }

    return {
        "tier1_scalar_stability": scalar_tier1,
        "tier1_per_image_delta_stability": per_image_delta_tier1,
        "tier2_per_seed_paired_tests": per_seed_paired,
        "tier2_worst_seed_summary": {
            "worst_seed_id": worst_seed_id,
            "worst_seed_max_wilcoxon_p": worst_p,
            "worst_seed_min_abs_mean_diff": min_abs_diff,
            "worst_seed_min_abs_rank_biserial_r": min_abs_rrb,
            "worst_seed_ci_95": worst_ci,
            "all_seeds_p_lt_005": all_p05,
            "all_seeds_p_lt_001": all_p001,
            "all_seeds_p_lt_1e10": all_p1e10,
        },
        "tier2_inter_seed_consistency": consistency,
        "tier2_cluster_pooled_paired": pooled_paired,
    }
