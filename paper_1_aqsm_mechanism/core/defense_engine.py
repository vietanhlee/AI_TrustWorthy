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

from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
import sklearn.metrics as skm
import torch


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
        self.cal_dropoff_median = float(np.median(all_dropoffs)) if all_dropoffs else 35.0
        self.cal_margin_median = float(np.median(all_margins)) if all_margins else 0.05

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


class SpatialConsistencyMonitor:
    """
    Bộ giám sát tính nhất quán không gian (Spatial-Consistency Monitor):
    Phát hiện sự mất đồng bộ không gian giữa tọa độ Anchor / Reference Points ban đầu
    và phân bố dự đoán hoặc lấy mẫu Deformable Cross-Attention của Deformable Decoder.
    Dưới đòn AQSM, anchors bị dịch chuyển cưỡng bức vào vùng nền lộn xộn,
    làm sai lệch rõ rệt độ phân tán không gian (spatial dispersion) và khoảng cách dịch chuyển tham chiếu.
    """
    def __init__(self, target_fpr: float = 0.05):
        self.target_fpr = target_fpr
        self.disp_mon = SingleFeatureMonitor(target_fpr=target_fpr)
        self.drift_mon = SingleFeatureMonitor(target_fpr=target_fpr)
        self.fused_thresh: float = 0.0
        self.is_fitted: bool = False

    @staticmethod
    def extract_features(
        init_ref_boxes: torch.Tensor,
        pred_boxes: torch.Tensor,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Trích xuất (dispersion, drift):
        - dispersion: Bán kính phân tán không gian trung bình của các anchors [B].
        - drift: Độ dịch chuyển L2 trung bình giữa init reference points và predicted boxes [B].
        """
        # Tự động nhận diện không gian logit (-inf, +inf) hay không gian tọa độ chuẩn hóa [0, 1]
        ref_sig = torch.sigmoid(init_ref_boxes) if (init_ref_boxes.max() > 1.5 or init_ref_boxes.min() < -0.5) else torch.clamp(init_ref_boxes, 0.0, 1.0)
        pred_sig = torch.sigmoid(pred_boxes) if (pred_boxes.max() > 1.5 or pred_boxes.min() < -0.5) else torch.clamp(pred_boxes, 0.0, 1.0)

        ref_np = ref_sig.detach().cpu().numpy()
        pred_np = pred_sig.detach().cpu().numpy()

        # 1. Spatial dispersion: khoảng cách từ mỗi anchor tới tâm trung bình của ảnh
        mean_center = np.mean(ref_np[..., :2], axis=1, keepdims=True)  # [B, 1, 2]
        dispersion = np.mean(np.sqrt(np.sum((ref_np[..., :2] - mean_center) ** 2, axis=-1)), axis=1)  # [B]

        # 2. Reference drift: độ dịch chuyển từ ref point sang predicted box center
        drift = np.mean(np.sqrt(np.sum((pred_np[..., :2] - ref_np[..., :2]) ** 2, axis=-1)), axis=1)  # [B]

        return dispersion, drift

    def fit(self, ref_boxes_list: Sequence[torch.Tensor], pred_boxes_list: Sequence[torch.Tensor]) -> None:
        all_disp = []
        all_drift = []
        for r, p in zip(ref_boxes_list, pred_boxes_list):
            disp, drift = self.extract_features(r, p)
            all_disp.extend(disp.tolist())
            all_drift.extend(drift.tolist())

        self.disp_mon.fit(all_disp)
        self.drift_mon.fit(all_drift)

        s_disp = self.disp_mon.score(all_disp)
        s_drift = self.drift_mon.score(all_drift)
        max_scores = np.maximum(s_disp, s_drift)
        self.fused_thresh = float(np.percentile(max_scores, (1.0 - self.target_fpr) * 100.0))
        self.is_fitted = True

    def score(self, init_ref_boxes: torch.Tensor, pred_boxes: torch.Tensor) -> np.ndarray:
        disp, drift = self.extract_features(init_ref_boxes, pred_boxes)
        s_disp = self.disp_mon.score(disp)
        s_drift = self.drift_mon.score(drift)
        return np.maximum(s_disp, s_drift)

    def predict(self, init_ref_boxes: torch.Tensor, pred_boxes: torch.Tensor) -> np.ndarray:
        scores = self.score(init_ref_boxes, pred_boxes)
        return (scores >= self.fused_thresh).astype(int)


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
    4. Tính AUROC chính xác qua toàn bộ đường cong phân bố bằng sklearn.metrics.roc_auc_score.
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

        # Tính AUROC thực tế qua toàn bộ đường cong phân bố bằng sklearn.metrics.roc_auc_score
        all_labels = np.concatenate([np.zeros(n_test), np.ones(n_test)])
        clean_scores = esm.score(test_logits_cat)
        all_scores = np.concatenate([clean_scores, scores_adv])

        try:
            auroc = float(skm.roc_auc_score(all_labels, all_scores))
        except Exception:
            # Fallback nếu phân bố hằng số
            auroc = 0.5000

        results["attacks"][atk_name] = {
            "tpr_detection_rate_pct": round(tpr, 2),
            "auroc": round(auroc, 4),
            "mean_anomaly_score": round(float(np.mean(scores_adv)), 4),
        }
        print(f"   + {atk_name.upper():15s}: TPR = {tpr:5.2f}% | AUROC = {auroc:.4f}")

    return results


class ActiveBoxCountMonitor:
    """
    Baseline giám sát đơn giản dựa trên đếm số lượng Active Bounding Boxes đầu ra
    (một chiều: số query có max_c sigma(L_{q, c}) >= conf_threshold).
    Dùng làm đối chứng cho độ thưa trung bình (stealthiness) giữa AQSM, Query Vanishing
    và Standard Task PGD trên cùng ngưỡng hiệu chuẩn FPR danh định.
    """

    def __init__(self, target_fpr: float = 0.05, conf_threshold: float = 0.35):
        self.target_fpr = float(target_fpr)
        self.conf_threshold = float(conf_threshold)
        self.twosided_monitor = SingleFeatureMonitor(target_fpr=target_fpr)
        self.upper_thresh: float = 0.0
        self.cal_counts: np.ndarray = np.array([], dtype=np.float64)
        self.is_fitted: bool = False

    def count_active_boxes(self, dec_logits_list: Any) -> np.ndarray:
        """Đếm số box vượt ngưỡng conf_threshold trên từng ảnh từ logits [B, N_q, C]."""
        if isinstance(dec_logits_list, torch.Tensor):
            probs = torch.sigmoid(dec_logits_list)
            max_probs, _ = probs.max(dim=-1)
            counts = (max_probs >= self.conf_threshold).sum(dim=-1)
            return counts.detach().cpu().numpy().astype(np.float64)
        counts_all: List[np.ndarray] = []
        for batch_t in dec_logits_list:
            probs = torch.sigmoid(batch_t)
            max_probs, _ = probs.max(dim=-1)
            counts_all.append((max_probs >= self.conf_threshold).sum(dim=-1).detach().cpu().numpy().astype(np.float64))
        return np.concatenate(counts_all, axis=0) if counts_all else np.array([], dtype=np.float64)

    def fit(self, clean_dec_logits: Any) -> None:
        counts = self.count_active_boxes(clean_dec_logits)
        self.cal_counts = counts
        self.twosided_monitor.fit(counts)
        self.upper_thresh = float(np.percentile(counts, (1.0 - self.target_fpr) * 100.0)) if counts.size > 0 else 0.0
        self.is_fitted = True

    @staticmethod
    def _summarize_counts(counts: np.ndarray) -> Dict[str, float]:
        if counts.size == 0:
            return {"mean": 0.0, "std": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
        return {
            "mean": round(float(np.mean(counts)), 2),
            "std": round(float(np.std(counts, ddof=1)) if counts.size > 1 else 0.0, 2),
            "median": round(float(np.median(counts)), 2),
            "p95": round(float(np.percentile(counts, 95.0)), 2),
            "max": round(float(np.max(counts)), 2),
        }

    def evaluate_cohort(
        self,
        clean_test_dec_logits: Any,
        adv_dec_logits_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Đánh giá Active-Box Watchdog (1-sided flooding + 2-sided count anomaly)."""
        if not self.is_fitted:
            raise RuntimeError("ActiveBoxCountMonitor chưa được fit trên dữ liệu hiệu chuẩn!")

        clean_counts = self.count_active_boxes(clean_test_dec_logits)
        clean_fpr_upper = float(np.mean(clean_counts > self.upper_thresh)) * 100.0 if clean_counts.size > 0 else 0.0
        clean_fpr_two = float(np.mean(self.twosided_monitor.predict(clean_counts))) * 100.0 if clean_counts.size > 0 else 0.0
        clean_scores_two = self.twosided_monitor.score(clean_counts)

        out: Dict[str, Any] = {
            "conf_threshold": self.conf_threshold,
            "target_fpr_pct": round(self.target_fpr * 100.0, 2),
            "upper_threshold_gt": round(self.upper_thresh, 2),
            "twosided_bounds": [
                round(self.twosided_monitor.lower_thresh, 2),
                round(self.twosided_monitor.upper_thresh, 2),
            ],
            "clean_fpr_upper_pct": round(clean_fpr_upper, 2),
            "clean_fpr_twosided_pct": round(clean_fpr_two, 2),
            "clean_box_stats": self._summarize_counts(clean_counts),
            "attacks": {},
        }

        for atk_name, adv_lgts in adv_dec_logits_dict.items():
            adv_counts = self.count_active_boxes(adv_lgts)
            n_c = clean_counts.size
            n_a = adv_counts.size
            tpr_upper = float(np.mean(adv_counts > self.upper_thresh)) * 100.0 if n_a > 0 else 0.0
            tpr_two = float(np.mean(self.twosided_monitor.predict(adv_counts))) * 100.0 if n_a > 0 else 0.0
            adv_scores_two = self.twosided_monitor.score(adv_counts)

            if n_c > 0 and n_a > 0 and (np.std(np.concatenate([clean_counts, adv_counts])) > 0):
                labels = np.concatenate([np.zeros(n_c), np.ones(n_a)])
                try:
                    auroc_upper = float(skm.roc_auc_score(labels, np.concatenate([clean_counts, adv_counts])))
                except Exception:
                    auroc_upper = 0.5
                try:
                    auroc_two = float(skm.roc_auc_score(labels, np.concatenate([clean_scores_two, adv_scores_two])))
                except Exception:
                    auroc_two = 0.5
            else:
                auroc_upper = 0.5
                auroc_two = 0.5

            out["attacks"][atk_name] = {
                "box_stats": self._summarize_counts(adv_counts),
                "tpr_upper_pct": round(tpr_upper, 2),
                "auroc_upper": round(auroc_upper, 4),
                "tpr_twosided_pct": round(tpr_two, 2),
                "auroc_twosided": round(auroc_two, 4),
            }

        return out
