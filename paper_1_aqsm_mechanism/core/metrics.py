# -*- coding: utf-8 -*-
"""
Module: paper_1_aqsm_mechanism.core.metrics
Mục đích: Cung cấp bộ công cụ tính toán các chỉ số thống kê chuẩn mực:
1. 95% Clopper-Pearson Exact Binomial Confidence Intervals.
2. AUROC kèm Bootstrap CI 95%.
3. Task-level Attack Success Rate (ASR).
4. TPR @ Fixed FPR.
5. Tiện ích tuần tự hóa JSON cho dữ liệu numpy / PyTorch.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.stats import beta
from sklearn.metrics import roc_auc_score, roc_curve
import torch


def clopper_pearson_interval(
    k: int,
    n: int,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """
    Tính toán khoảng tin cậy nhị thức chính xác Clopper-Pearson (Exact Binomial CI)
    dựa trên hàm phân phối tích lũy nghịch đảo Beta (Beta PPF).

    Args:
        k: Số lượng mẫu thành công (hoặc số báo động dương tính).
        n: Tổng số lượng mẫu thử nghiệm.
        confidence: Độ tin cậy thống kê (mặc định 0.95 cho mức ý nghĩa alpha = 0.05).

    Returns:
        Tuple (lower_bound, upper_bound) trong khoảng [0.0, 1.0].
    """
    if n <= 0:
        return 0.0, 0.0
    k = max(0, min(k, n))
    alpha = 1.0 - confidence
    lower = 0.0 if k == 0 else float(beta.ppf(alpha / 2.0, k, n - k + 1))
    upper = 1.0 if k == n else float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return lower, upper


clopper_pearson_ci = clopper_pearson_interval


def compute_auroc_and_bootstrap_ci(
    labels: np.ndarray,
    scores: np.ndarray,
    n_boot: int = 500,
    seed: int = 42,
) -> Tuple[float, Tuple[float, float]]:
    """
    Tính toán chỉ số AUROC và khoảng tin cậy 95% Bootstrap không phụ thuộc ngưỡng.
    """
    labels = np.asarray(labels, dtype=int)
    scores = np.nan_to_num(np.asarray(scores, dtype=float), nan=0.0, posinf=1e6, neginf=-1e6)

    pos_mask = (labels == 1)
    neg_mask = (labels == 0)
    n_pos = int(np.sum(pos_mask))
    n_neg = int(np.sum(neg_mask))

    if n_pos == 0 or n_neg == 0:
        return 0.5, (0.5, 0.5)

    pos_scores = scores[pos_mask]
    neg_scores = scores[neg_mask]

    diff = pos_scores[:, None] - neg_scores[None, :]
    base_auroc = float(np.mean(diff > 0) + 0.5 * np.mean(diff == 0))

    rng = np.random.RandomState(seed)
    n = len(labels)
    aurocs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        b_labels = labels[idx]
        b_scores = scores[idx]
        b_pos = (b_labels == 1)
        b_neg = (b_labels == 0)

        if np.sum(b_pos) == 0 or np.sum(b_neg) == 0:
            continue
        b_diff = b_scores[b_pos][:, None] - b_scores[b_neg][None, :]
        aurocs.append(float(np.mean(b_diff > 0) + 0.5 * np.mean(b_diff == 0)))

    if len(aurocs) > 10:
        ci_lower = max(0.0, float(np.percentile(aurocs, 2.5)))
        ci_upper = min(1.0, float(np.percentile(aurocs, 97.5)))
        ci = (ci_lower, ci_upper)
    else:
        ci = (base_auroc, base_auroc)

    return base_auroc, ci


def calculate_task_level_asr(
    attack_type: str,
    clean_logits: torch.Tensor,
    adv_logits: torch.Tensor,
    target_class_idx: int = 1,
) -> Tuple[float, int, int, List[bool]]:
    """
    Tính Attack Success Rate (ASR) dựa trên sự suy giảm thực chất của tác vụ nhận diện:
    - Vanishing: Độ tự tin cao nhất của queries tụt xuống < 0.20.
    - Fabrication / Patch / Adaptive: Có ít nhất 1 query bị kích hoạt ở target_class_idx >= 0.35.
    - AQSM / Selection Starvation: Độ tự tin trung bình của các queries foreground sạch rơi xuống < 0.35
      hoặc suy giảm >= 0.15.
    """
    probs_c = torch.sigmoid(clean_logits).cpu()
    probs_a = torch.sigmoid(adv_logits).cpu()
    batch_size = probs_c.shape[0]

    successes = 0
    total_valid = 0
    is_success_list = []

    for b in range(batch_size):
        p_c = probs_c[b]
        p_a = probs_a[b]

        if attack_type in ["AQSM", "Selection_Starvation"]:
            c_max_per_query = p_c.max(dim=-1).values
            c_active_mask = (c_max_per_query >= 0.35)
            if c_active_mask.sum() > 0:
                total_valid += 1
                a_on_c_active = p_a[c_active_mask].max(dim=-1).values
                mean_c = float(c_max_per_query[c_active_mask].mean())
                mean_a = float(a_on_c_active.mean())
                succ = (mean_a < 0.35) or ((mean_c - mean_a) >= 0.15)
            else:
                succ = False
        elif attack_type in ["Vanishing", "Query_Vanishing"]:
            total_valid += 1
            max_conf = float(p_a.max())
            succ = (max_conf < 0.20)
        else:
            total_valid += 1
            t_confs = p_a[..., target_class_idx]
            succ = (float(t_confs.max()) >= 0.35)

        is_success_list.append(succ)
        if succ:
            successes += 1

    if total_valid == 0:
        total_valid = batch_size
        successes = sum(is_success_list)

    asr = float(successes) / float(total_valid)
    return asr, successes, total_valid, is_success_list


def compute_tpr_at_fixed_fpr(
    scores_pos: np.ndarray,
    scores_neg: np.ndarray,
    target_fpr: float = 0.05,
) -> Tuple[float, float]:
    """
    Xác định ngưỡng cảnh báo trên tập âm tính (clean calibration) để khống chế
    tỷ lệ báo động sai đúng bằng target_fpr, sau đó đánh giá độ nhạy (TPR) trên tập dương tính.
    """
    scores_neg = np.nan_to_num(np.asarray(scores_neg, dtype=float), nan=0.0)
    scores_pos = np.nan_to_num(np.asarray(scores_pos, dtype=float), nan=0.0)

    if len(scores_neg) == 0:
        return 0.0, 0.0

    threshold = float(np.quantile(scores_neg, 1.0 - target_fpr))
    if len(scores_pos) == 0:
        return 0.0, threshold

    tpr = float(np.mean(scores_pos > threshold))
    return tpr, threshold


def make_serializable(obj: Any) -> Any:
    """
    Chuyển đổi đệ quy các cấu trúc dữ liệu phức tạp (numpy, torch, dataclass)
    về định dạng chuẩn nguyên thủy (native types) có thể xuất sang JSON.
    """
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(x) for x in obj]
    elif isinstance(obj, dict):
        return {str(k): make_serializable(v) for k, v in obj.items()}
    return obj


to_json_serializable = make_serializable
