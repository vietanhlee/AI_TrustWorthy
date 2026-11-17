# -*- coding: utf-8 -*-
"""
Bước 11: Kiểm định Giả thuyết Ghép cặp McNemar (Paired McNemar Hypothesis Tests)
Mục đích: Tái lập Bảng 5 trong bài báo Paper 3:
So sánh mức độ nhạy cảm giữa các kênh giám sát trên N=300 mẫu thực nghiệm độc lập:
1. AQSM: ESM Dropoff vs Top1Mean
2. AQSM: ESM Margin vs Top1Mean
3. AQSM: ESM Dropoff vs ActiveQ Split
4. Query Vanishing: ActiveQ Split vs Top1Mean
5. Query Vanishing: ActiveQ Split vs ESM Dropoff
Xuất log chi tiết, bảng 2x2 contingency table và file JSON metrics.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
from scipy import stats

# Đảm bảo import đúng workspace
EXPERIMENTS_DIR = Path(__file__).resolve().parent
PAPER3_DIR = EXPERIMENTS_DIR.parent
WORKSPACE_DIR = PAPER3_DIR.parent

if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))
if str(WORKSPACE_DIR / "edge_sec_pro") not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR / "edge_sec_pro"))

from edge_sec_pro.experiments.common_exp import CHECKPOINTS_DIR
from paper_3_defense_guardrail.experiments.exp_logger import (
    setup_experiment_logger,
    save_audit_json,
    PAPER3_DIR,
    RESULTS_DIR,
)


def mcnemar_test(y_true_a: np.ndarray, y_true_b: np.ndarray) -> Dict[str, Any]:
    """
    Tính kiểm định McNemar chính xác giữa 2 phân loại nhị phân trên cùng tập mẫu N.
    y_true_a: mảng boolean [N] (1 = phát hiện đúng, 0 = lọt)
    y_true_b: mảng boolean [N] (1 = phát hiện đúng, 0 = lọt)
    """
    assert len(y_true_a) == len(y_true_b), "Kích thước tập mẫu phải bằng nhau"
    n = len(y_true_a)

    # Bảng 2x2:
    #             B = 1       B = 0
    # A = 1         a           b
    # A = 0         c           d
    a = int(np.sum((y_true_a == 1) & (y_true_b == 1)))
    b = int(np.sum((y_true_a == 1) & (y_true_b == 0)))  # A bắt được, B lọt
    c = int(np.sum((y_true_a == 0) & (y_true_b == 1)))  # A lọt, B bắt được
    d = int(np.sum((y_true_a == 0) & (y_true_b == 0)))

    # McNemar statistic với hiệu chỉnh tính liên tục Edwards: (|b - c| - 1)^2 / (b + c)
    discordant = b + c
    if discordant == 0:
        chi2 = 0.0
        p_val_chi2 = 1.0
        p_val_exact = 1.0
    else:
        chi2 = float((abs(b - c) - 1.0) ** 2 / discordant)
        p_val_chi2 = float(1.0 - stats.chi2.cdf(chi2, df=1))
        p_val_exact = float(stats.binomtest(b, n=discordant, p=0.5, alternative="two-sided").pvalue)

    odds_ratio = float(b / c) if c > 0 else float("inf")

    return {
        "N": n,
        "contingency_table": {"a_both_detect": a, "b_only_A": b, "c_only_B": c, "d_neither": d},
        "tpr_A": round(float(np.mean(y_true_a) * 100.0), 2),
        "tpr_B": round(float(np.mean(y_true_b) * 100.0), 2),
        "tpr_diff": round(float((np.mean(y_true_a) - np.mean(y_true_b)) * 100.0), 2),
        "discordant_pairs": discordant,
        "mcnemar_chi2": round(chi2, 4),
        "p_value_chi2": round(p_val_chi2, 6),
        "p_value_exact_binomial": float(p_val_exact),
        "odds_ratio": round(odds_ratio, 3),
        "statistically_significant_05": p_val_exact < 0.05,
    }


def run_mcnemar():
    logger, log_file = setup_experiment_logger("step11_mcnemar_tests")
    logger.info("BƯỚC 11: KIỂM ĐỊNH THỐNG KÊ GHÉP CẶP MCNEMAR (PAIRED HYPOTHESIS TESTING)")

    npz_file = WORKSPACE_DIR / "edge_sec_pro" / "results" / "rtdetr_advanced_defense_raw_scores.npz"
    json_file = WORKSPACE_DIR / "edge_sec_pro" / "results" / "rtdetr_advanced_defense_results.json"

    if not npz_file.exists() or not json_file.exists():
        err_msg = f"Không tìm thấy file kết quả thô: {npz_file} hoặc {json_file}"
        logger.error(err_msg)
        raise FileNotFoundError(err_msg)

    data = np.load(npz_file, allow_pickle=True)
    with open(json_file, "r", encoding="utf-8") as f:
        meta = json.load(f)

    thresholds = meta["cal_thresholds"]
    logger.info(f"Ngưỡng hiệu chuẩn phân vị danh định 5.0%: {thresholds}")

    comparisons = [
        ("Attack__Query_Selection_Manipulation_AQSM", "ESM_Dropoff", "Top1Mean_Split", "AQSM: ESM Drop vs. Top1Mean"),
        ("Attack__Query_Selection_Manipulation_AQSM", "ESM_Margin", "Top1Mean_Split", "AQSM: ESM Margin vs. Top1Mean"),
        ("Attack__Query_Selection_Manipulation_AQSM", "ESM_Dropoff", "ActiveQ_Split", "AQSM: ESM Drop vs. ActiveQ"),
        ("Attack__PGD_Query_Vanishing_15_steps", "ActiveQ_Split", "Top1Mean_Split", "Vanishing: ActiveQ vs. Top1Mean"),
        ("Attack__PGD_Query_Vanishing_15_steps", "ActiveQ_Split", "ESM_Dropoff", "Vanishing: ActiveQ vs. ESM Drop"),
    ]

    results = {}
    logger.info("\n" + "=" * 90)
    logger.info("KẾT QUẢ KIỂM ĐỊNH GHÉP CẶP MCNEMAR TRÊN N=300 MẪU (BẢNG 5 TRONG BÀI BÁO):")
    logger.info("-" * 90)
    logger.info(f"{'So Sánh Kênh Giám Sát':<35} | {'TPR A':<8} | {'TPR B':<8} | {'Chi2':<8} | {'Exact p-value':<14} | {'Ý Nghĩa'}")
    logger.info("-" * 90)

    for prefix, ch_a, ch_b, label in comparisons:
        key_a = f"{prefix}__{ch_a}"
        key_b = f"{prefix}__{ch_b}"
        if key_a not in data or key_b not in data:
            logger.warning(f"Warning: {key_a} hoặc {key_b} không có trong file NPZ!")
            continue

        scores_a = data[key_a]
        scores_b = data[key_b]
        th_a = thresholds[ch_a]
        th_b = thresholds[ch_b]

        bin_a = (scores_a >= th_a).astype(int)
        bin_b = (scores_b >= th_b).astype(int)

        res = mcnemar_test(bin_a, bin_b)
        results[label] = res

        sig_text = "p < 0.01 ***" if res["p_value_exact_binomial"] < 0.01 else ("p < 0.05 *" if res["statistically_significant_05"] else "n.s.")
        logger.info(f"{label:<35} | {res['tpr_A']:5.1f}%  | {res['tpr_B']:5.1f}%  | {res['mcnemar_chi2']:6.3f} | {res['p_value_exact_binomial']:<14.4e} | {sig_text}")

    logger.info("=" * 90)

    # Xuất file kết quả JSON
    res_json_paper3 = RESULTS_DIR / "step11_mcnemar_test_results.json"
    res_json_global = WORKSPACE_DIR / "edge_sec_pro" / "results" / "rtdetr_mcnemar_test_results.json"

    export_payload = {
        "experiment": "step11_mcnemar_tests",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "comparisons": results,
        "artifacts": {
            "log_file": str(log_file),
        }
    }

    save_audit_json(res_json_paper3, export_payload)
    save_audit_json(res_json_global, export_payload)

    logger.info(f"\n-> Đã lưu Audit JSON (Paper 3) : {res_json_paper3}")
    logger.info(f"-> Đã đồng bộ JSON toàn dự án  : {res_json_global}")
    logger.info("\n>>> BƯỚC 11 ĐÃ HOÀN TẤT THÀNH CÔNG VỚI ĐẦY ĐỦ DỮ LIỆU KIỂM TOÁN! <<<")


if __name__ == "__main__":
    run_mcnemar()
