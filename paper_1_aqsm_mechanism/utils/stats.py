# -*- coding: utf-8 -*-
"""
Module: paper_1_aqsm_mechanism.utils.stats
Mục đích: Cung cấp các thuật toán phân tích thống kê suy luận chuyên sâu phục vụ
kiểm chứng giả thuyết khoa học cho Paper 1:
  1. Phân tích phương sai lặp lại hai yếu tố: 2x2 Repeated-Measures ANOVA (Factor M x Factor Q).
  2. Ước lượng khoảng tin cậy 95% Confidence Interval bằng Paired Bootstrap resampling.
  3. Kiểm định phi tham số Wilcoxon signed-rank test.
"""

from typing import Any, Dict, Optional, Sequence, Union
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
