# -*- coding: utf-8 -*-
"""
Module: paper_2_phantom_fabrication.utils.stats
Mục đích: Cung cấp các phương pháp thống kê khoa học và kiểm định giả thuyết:
  - Bootstrap Paired Difference & Confidence Interval 95%.
  - Wilson Score Interval cho tỷ lệ False Positive / Detection Rate.
  - Paired difference statistical tests (Wilcoxon / t-test).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import stats


def bootstrap_ci_95(
    data: Sequence[float],
    n_boot: int = 10000,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Tính khoảng tin cậy 95% bằng phương pháp Bootstrap Resampling (10,000 lần lặp).
    Đảm bảo tính chính xác khoa học và không phụ thuộc giả định chuẩn.
    """
    arr = np.asarray(data, dtype=np.float64)
    if len(arr) == 0:
        return (0.0, 0.0)
    if len(arr) == 1:
        return (float(arr[0]), float(arr[0]))

    rng = np.random.RandomState(seed)
    boot_means = np.empty(n_boot, dtype=np.float64)
    n = len(arr)
    for i in range(n_boot):
        indices = rng.randint(0, n, size=n)
        boot_means[i] = np.mean(arr[indices])

    lower = float(np.percentile(boot_means, 2.5))
    upper = float(np.percentile(boot_means, 97.5))
    return (round(lower, 4), round(upper, 4))


def save_per_image_telemetry(
    records: List[Dict[str, Any]],
    output_path: Union[str, Path],
) -> Path:
    """
    Lưu trữ kết quả chi tiết từng ảnh ra file JSONL phục vụ thẩm định khoa học (Data Provenance).
    """
    import json
    p = Path(output_path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in records:
            # Chuyển đổi an toàn các kiểu dữ liệu NumPy
            clean_r = {}
            for k, v in r.items():
                if isinstance(v, (np.integer, int)):
                    clean_r[k] = int(v)
                elif isinstance(v, (np.floating, float)):
                    clean_r[k] = float(v)
                elif isinstance(v, np.ndarray):
                    clean_r[k] = v.tolist()
                else:
                    clean_r[k] = v
            f.write(json.dumps(clean_r) + "\n")
    return p


def wilson_score_interval(
    successes: int,
    total: int,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """
    Tính khoảng tin cậy Wilson Score cho tỷ lệ nhị thức p = successes / total.
    """
    if total <= 0:
        return (0.0, 0.0)

    p_hat = successes / total
    z = stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0)
    denom = 1.0 + (z ** 2) / total
    center = (p_hat + (z ** 2) / (2.0 * total)) / denom
    half_width = (z * np.sqrt((p_hat * (1.0 - p_hat) + (z ** 2) / (4.0 * total)) / total)) / denom

    low = max(0.0, center - half_width)
    high = min(1.0, center + half_width)
    return (round(float(low), 4), round(float(high), 4))


def paired_diff_test(
    a: Sequence[float],
    b: Sequence[float],
    n_boot: int = 1000,
) -> Dict[str, Any]:
    """
    Kiểm định thống kê sự khác biệt ghép cặp giữa hai phân phối mẫu:
    Trả về mean diff, Bootstrap 95% CI và p-value (Wilcoxon hoặc Paired t-test).
    """
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    diff = arr_a - arr_b

    mean_diff = float(np.mean(diff))
    ci_low, ci_high = bootstrap_ci_95(diff, n_boot=n_boot)

    p_val = float("nan")
    try:
        if not np.all(diff == 0):
            res = stats.wilcoxon(arr_a, arr_b)
            p_val = float(res.pvalue)
    except Exception:
        try:
            res = stats.ttest_rel(arr_a, arr_b)
            p_val = float(res.pvalue)
        except Exception:
            pass

    return {
        "mean_diff": round(mean_diff, 4),
        "ci_95": [ci_low, ci_high],
        "p_value": p_val,
        "is_significant_05": bool(p_val < 0.05) if not np.isnan(p_val) else False,
    }
