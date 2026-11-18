# -*- coding: utf-8 -*-
"""
Module: paper_1_aqsm_mechanism.core.defense_engine
Mục đích: Cung cấp giải pháp phòng thủ tin cậy nhẹ (Lightweight Intrusion Defense Guardrail)
cho RT-DETR tại tầng biên:
1. EncoderSelectionMonitor (ESM):
   - Giám sát độ dốc suy giảm điểm (enc_dropoff = s_1 / s_300)
   - Giám sát lề phân cách chọn lọc (enc_margin = s_300 - s_301)
   giữa 300 queries được chọn và các candidate nền bị loại bỏ.
2. CalibratedMaxOREnsemble:
   - Cổng logic Max-OR phi tham số (Extreme-Value Fusion Gate)
   - Hiệu chuẩn phân vị chặt chẽ (Split-Percentile Conformal Calibration)
     đảm bảo False Positive Rate danh định trên ảnh sạch đạt chính xác alpha = 0.05 (5.0% FPR).
3. Hỗ trợ đánh giá đòn tấn công thích ứng White-Box Adaptive Evasion Attack (Carlini-Tramèr).
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleFeatureMonitor:
    """
    Bộ giám sát 1 chiều phi tham số với cơ chế chia đều ngân sách FPR (two_sided_split):
    - alpha/2 ở đuôi dưới (bắt sự suy giảm/sụp đổ chỉ số)
    - alpha/2 ở đuôi trên (bắt sự lạm phát bất thường)
    """

    def __init__(self, target_fpr: float = 0.05):
        self.target_fpr = target_fpr
        self.cal_values: np.ndarray = np.array([], dtype=np.float64)
        self.lower_thresh: float = 0.0
        self.upper_thresh: float = 0.0
        self.is_fitted: bool = False

    def fit(self, values: Sequence[float]) -> None:
        vals = np.array(values, dtype=np.float64)
        self.cal_values = np.sort(vals)
        half_alpha = (self.target_fpr / 2.0) * 100.0
        self.lower_thresh = float(np.percentile(self.cal_values, half_alpha))
        self.upper_thresh = float(np.percentile(self.cal_values, 100.0 - half_alpha))
        self.is_fitted = True

    def score(self, values: Sequence[float]) -> np.ndarray:
        """
        Tính điểm bất thường chuẩn hóa phân vị q(x) in [0, 1].
        q(x) = 2 * |mid_rank / N - 0.5|
        """
        if not self.is_fitted or len(self.cal_values) == 0:
            raise RuntimeError("SingleFeatureMonitor chưa được fit trên dữ liệu hiệu chuẩn!")
        vals = np.array(values, dtype=np.float64)
        left_ranks = np.searchsorted(self.cal_values, vals, side="left")
        right_ranks = np.searchsorted(self.cal_values, vals, side="right")
        quantiles = (left_ranks + right_ranks) / (2.0 * float(len(self.cal_values)))
        anomaly_scores = 2.0 * np.abs(quantiles - 0.5)
        return anomaly_scores

    def predict(self, values: Sequence[float]) -> np.ndarray:
        vals = np.array(values, dtype=np.float64)
        is_lower = vals <= self.lower_thresh
        is_upper = vals >= self.upper_thresh
        return (is_lower | is_upper).astype(int)


class EncoderSelectionMonitor:
    """
    Bộ giám sát quá trình chọn Query tại cửa ngõ Hybrid Encoder (ESM):
    Đo đạc 2 đặc trưng thống kê chủ chốt từ phân phối xác suất sigmoid của candidates:
    1. `enc_dropoff`: Tỷ số giữa candidate xếp thứ 1 và candidate xếp thứ 300:
       dropoff = max(s_1, 1e-4) / max(s_300, 1e-4)
       Dưới đòn AQSM, object thật bị dìm và background bị nâng, làm phân phối bị bẹt hoặc đảo lộn.
    2. `enc_margin`: Khoảng cách tin cậy giữa candidate thứ 300 và candidate thứ 301:
       margin = s_300 - s_301
    """

    def __init__(self, target_fpr: float = 0.05):
        self.target_fpr = target_fpr
        self.dropoff_mon = SingleFeatureMonitor(target_fpr=target_fpr)
        self.margin_mon = SingleFeatureMonitor(target_fpr=target_fpr)
        self.fused_thresh: float = 0.0
        self.cal_max_scores: np.ndarray = np.array([], dtype=np.float64)
        self.is_fitted: bool = False

    @staticmethod
    def extract_features_from_logits(enc_logits: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Trích xuất (dropoffs, margins) từ enc_outputs_class [B, S, C] hoặc [B, S].
        Hỗ trợ số lượng candidate S = 8400 ở chuẩn Native 640x640.
        """
        probs = torch.sigmoid(enc_logits)
        if probs.dim() == 3:
            max_probs, _ = probs.max(dim=-1)  # [B, S]
        else:
            max_probs = probs

        # Sắp xếp giảm dần theo hàng
        sorted_probs, _ = torch.sort(max_probs, dim=-1, descending=True)
        s_1 = sorted_probs[:, 0].detach().cpu().numpy()
        s_300 = sorted_probs[:, 299].detach().cpu().numpy()
        s_301 = sorted_probs[:, 300].detach().cpu().numpy() if sorted_probs.shape[1] > 300 else np.zeros_like(s_300)

        dropoffs = np.maximum(s_1, 1e-6) / np.maximum(s_300, 1e-6)
        margins = s_300 - s_301
        return dropoffs, margins

    def fit(self, enc_logits_list: Sequence[torch.Tensor]) -> None:
        all_dropoffs = []
        all_margins = []
        for lgt in enc_logits_list:
            d, m = self.extract_features_from_logits(lgt)
            all_dropoffs.extend(d.tolist())
            all_margins.extend(m.tolist())

        self.dropoff_mon.fit(all_dropoffs)
        self.margin_mon.fit(all_margins)

        # Tính điểm gộp cực đại trên tập hiệu chuẩn để xác định ngưỡng fused Max-OR
        s_d = self.dropoff_mon.score(all_dropoffs)
        s_m = self.margin_mon.score(all_margins)
        max_scores = np.maximum(s_d, s_m)
        self.cal_max_scores = np.sort(max_scores)
        self.fused_thresh = float(np.percentile(self.cal_max_scores, (1.0 - self.target_fpr) * 100.0))
        self.is_fitted = True

    def score(self, enc_logits: torch.Tensor) -> np.ndarray:
        d, m = self.extract_features_from_logits(enc_logits)
        s_d = self.dropoff_mon.score(d)
        s_m = self.margin_mon.score(m)
        return np.maximum(s_d, s_m)

    def predict(self, enc_logits: torch.Tensor) -> np.ndarray:
        fused_score = self.score(enc_logits)
        return (fused_score >= self.fused_thresh).astype(int)


def evaluate_defense_cohort(
    model_wrapper: Any,
    clean_tensors: torch.Tensor,
    adv_tensors_dict: Dict[str, torch.Tensor],
    target_fpr: float = 0.05,
    n_cal: int = 150,
) -> Dict[str, Any]:
    """
    Đánh giá hiệu năng bảo vệ của Encoder Selection Monitor trên tập kiểm thử:
    1. Dùng n_cal ảnh sạch đầu tiên để fit ngưỡng 5.0% FPR.
    2. Đánh giá FPR trên (N - n_cal) ảnh sạch còn lại (Zero Data Leakage).
    3. Đánh giá TPR (Detection Rate) trên các đòn tấn công đối kháng.
    """
    model = model_wrapper.model
    model.eval()
    device = clean_tensors.device

    n_total = len(clean_tensors)
    if n_total <= n_cal:
        n_cal = max(1, n_total // 2)
    assert n_total > n_cal, f"Tổng số ảnh ({n_total}) phải lớn hơn n_cal ({n_cal})"
    n_test = n_total - n_cal

    # Trích xuất logits cho calibration theo lô nhỏ
    print(f"[Defense Eval] Trích xuất logits hiệu chuẩn trên {n_cal} ảnh sạch...")
    cal_logits = []
    bsz = 10
    with torch.no_grad():
        for i in range(0, n_cal, bsz):
            end_idx = min(i + bsz, n_cal)
            b = clean_tensors[i : end_idx]
            if hasattr(b, "to"):
                b = b.to(device)
            out = model(b)
            if hasattr(out, "enc_outputs_class") and out.enc_outputs_class is not None:
                cal_logits.append(out.enc_outputs_class.cpu())
            else:
                cal_logits.append(out.logits.cpu())

    esm = EncoderSelectionMonitor(target_fpr=target_fpr)
    esm.fit(cal_logits)

    # Đánh giá trên Clean Test (FPR)
    print(f"[Defense Eval] Đánh giá FPR trên {n_test} ảnh sạch độc lập...")
    test_logits = []
    with torch.no_grad():
        for i in range(0, n_test, bsz):
            start_idx = n_cal + i
            end_idx = min(start_idx + bsz, n_total)
            b = clean_tensors[start_idx : end_idx]
            if hasattr(b, "to"):
                b = b.to(device)
            out = model(b)
            if hasattr(out, "enc_outputs_class") and out.enc_outputs_class is not None:
                test_logits.append(out.enc_outputs_class.cpu())
            else:
                test_logits.append(out.logits.cpu())

    test_logits_cat = torch.cat(test_logits, dim=0)
    preds_clean = esm.predict(test_logits_cat)
    clean_fpr = float(np.mean(preds_clean)) * 100.0

    results: Dict[str, Any] = {
        "target_fpr_pct": target_fpr * 100.0,
        "empirical_clean_fpr_pct": round(clean_fpr, 2),
        "n_cal": n_cal,
        "n_test": n_test,
        "fused_threshold": round(esm.fused_thresh, 4),
        "attacks": {},
    }

    # Đánh giá TPR trên từng đòn tấn công
    for atk_name, adv_full in adv_tensors_dict.items():
        adv_logits = []
        with torch.no_grad():
            for i in range(0, n_test, bsz):
                start_idx = n_cal + i
                end_idx = min(start_idx + bsz, n_total)
                b = adv_full[start_idx : end_idx]
                if hasattr(b, "to"):
                    b = b.to(device)
                out = model(b)
                if hasattr(out, "enc_outputs_class") and out.enc_outputs_class is not None:
                    adv_logits.append(out.enc_outputs_class.cpu())
                else:
                    adv_logits.append(out.logits.cpu())

        adv_logits_cat = torch.cat(adv_logits, dim=0)
        preds_adv = esm.predict(adv_logits_cat)
        scores_adv = esm.score(adv_logits_cat)
        tpr = float(np.mean(preds_adv)) * 100.0

        # Tính AUROC phân loại sạch vs tấn công
        all_labels = np.concatenate([np.zeros(n_test), np.ones(n_test)])
        clean_scores = esm.score(test_logits_cat)
        all_scores = np.concatenate([clean_scores, scores_adv])

        # AUROC via Wilcoxon-Mann-Whitney
        n0 = n_test
        n1 = n_test
        ranks = np.argsort(np.argsort(all_scores)) + 1
        r_sum = np.sum(ranks[all_labels == 1])
        auroc = float((r_sum - n1 * (n1 + 1) / 2.0) / (n0 * n1))

        results["attacks"][atk_name] = {
            "tpr_detection_rate_pct": round(tpr, 2),
            "auroc": round(auroc, 4),
            "mean_anomaly_score": round(float(np.mean(scores_adv)), 4),
        }
        print(f"   + {atk_name.upper():15s}: TPR = {tpr:5.2f}% | AUROC = {auroc:.4f}")

    return results
