"""
Module: rtdetr_advanced_monitor.py
Mục đích: Cài đặt các cơ chế phòng thủ chuyên sâu cho RT-DETR theo chuẩn học thuật nghiêm ngặt:
1. SingleFeatureBaseline:
   - 'two_sided': Z-score hai phía đối xứng truyền thống (|z| = |x - mu| / std).
   - 'two_sided_split': Giao thức phi tham số chia đều ngân sách FPR (2.5% đuôi dưới, 2.5% đuôi trên),
     giải quyết triệt để lỗi phân phối lệch của z-score đối xứng mà không cần Oracle.
   - 'upper': Bắt dị thường đuôi trên ((x - mu) / std) - dùng làm đối chuẩn Oracle cho Fabrication.
   - 'lower': Bắt dị thường đuôi dưới ((mu - x) / std) - dùng làm đối chuẩn Oracle cho Vanishing.
2. CalibratedMaxOREnsemble:
   - Cổng logic Cực Đại Chuẩn Hóa (Calibrated Extreme-Value / Max-OR Gate).
   - Tinh gọn trên 4 kênh thống kê có đóng góp biên thực chất (Pruned High-Signal Channels):
     + act_lower: Bắt Vanishing & Selection Starvation (AQSM)
     + act_upper: Bắt Fabrication (tạo vật thể giả)
     + top1_lower: Bắt suy giảm độ tự tin toàn cục
     + top1_upper: Bắt lạm phát độ tự tin
   - Hiệu chuẩn ngưỡng tau trực tiếp trên phân phối điểm max của tập Calibration để đảm bảo
     FPR thực nghiệm trên dữ liệu sạch đạt chuẩn chính xác 5.0%.
3. LayerwiseMahalanobisDetector (Lee et al., NeurIPS 2018):
   - Tuyến phòng thủ sâu thứ hai dựa trên manifold đặc trưng ẩn tại các tầng Decoder,
     cung cấp khả năng phát hiện khi kẻ tấn công thích ứng vượt qua các bộ đếm 1D.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from edge_sec_pro.transformer_monitors.rtdetr_advanced_features import RTDETRDeepFeatures


class SingleFeatureBaseline:
    """
    Bộ phát hiện bất thường 1 chiều hỗ trợ phân tích hướng đuôi phân phối:
    - 'upper': Bắt dị thường ở đuôi trên (z = (x - mean) / std).
    - 'lower': Bắt dị thường ở đuôi dưới (z = (mean - x) / std).
    - 'two_sided': Bắt dị thường hai phía đối xứng (|z| = |x - mean| / std).
    - 'two_sided_split': Phân phối phi tham số chia ngân sách FPR:
      (target_fpr / 2) ở đuôi dưới và (target_fpr / 2) ở đuôi trên.
    """

    def __init__(self, attr_name: str, tail: str = "two_sided", target_fpr: float = 0.05):
        assert tail in ["upper", "lower", "two_sided", "two_sided_split"], f"tail không hợp lệ: {tail}"
        self.attr_name = attr_name
        self.tail = tail
        self.target_fpr = target_fpr
        self.mean = 0.0
        self.std = 1.0
        self.threshold = 0.0
        self.lower_threshold = 0.0
        self.upper_threshold = 0.0
        self.sorted_cal_vals: np.ndarray = np.array([])

    def fit(self, features: List[RTDETRDeepFeatures]) -> None:
        vals = np.array([getattr(f, self.attr_name) for f in features], dtype=np.float64)
        self.sorted_cal_vals = np.sort(vals)
        self.mean = float(np.mean(vals))
        self.std = float(np.std(vals))
        if self.std < 1e-8:
            self.std = 1e-8

        if self.tail == "two_sided_split":
            scores = self._compute_scores(vals)
            self.threshold = float(np.percentile(scores, (1.0 - self.target_fpr) * 100.0))
            half_fpr = (self.target_fpr / 2.0) * 100.0
            self.lower_threshold = float(np.percentile(vals, half_fpr))
            self.upper_threshold = float(np.percentile(vals, 100.0 - half_fpr))
        else:
            scores = self._compute_scores(vals)
            self.threshold = float(np.percentile(scores, (1.0 - self.target_fpr) * 100.0))

    def _compute_scores(self, vals: np.ndarray) -> np.ndarray:
        if self.tail == "upper":
            return (vals - self.mean) / self.std
        elif self.tail == "lower":
            return (self.mean - vals) / self.std
        elif self.tail == "two_sided":
            return np.abs(vals - self.mean) / self.std
        else:  # two_sided_split
            if len(self.sorted_cal_vals) == 0:
                return np.abs(vals - self.mean) / self.std
            # Tính mid-rank phân vị: (rank_left + rank_right) / (2 * N)
            # để xử lý phân phối liên tục và triệt tiêu sai lệch discrete ties
            left_ranks = np.searchsorted(self.sorted_cal_vals, vals, side="left")
            right_ranks = np.searchsorted(self.sorted_cal_vals, vals, side="right")
            quantiles = (left_ranks + right_ranks) / (2.0 * float(len(self.sorted_cal_vals)))
            return 2.0 * np.abs(quantiles - 0.5)

    def score(self, features: List[RTDETRDeepFeatures]) -> np.ndarray:
        vals = np.array([getattr(f, self.attr_name) for f in features], dtype=np.float64)
        return self._compute_scores(vals)

    def predict(self, features: List[RTDETRDeepFeatures]) -> np.ndarray:
        scores = self.score(features)
        return (scores >= self.threshold).astype(int)


class LayerwiseMahalanobisDetector:
    """
    Phát hiện bất thường đa tầng (Layer-wise Mahalanobis) theo Lee et al. (NeurIPS 2018).
    Fit phân phối Gaussian độc lập trên từng tầng Decoder với Shrinkage Regularization.
    """

    def __init__(self, shrinkage: float = 0.05, target_fpr: float = 0.05):
        self.shrinkage = shrinkage
        self.target_fpr = target_fpr
        self.num_layers = 0
        self.layer_means: List[np.ndarray] = []
        self.layer_precisions: List[np.ndarray] = []
        self.cal_layer_maha_means: List[float] = []
        self.cal_layer_maha_stds: List[float] = []
        self.threshold = 0.0
        self.is_fitted = False

    def fit(self, features: List[RTDETRDeepFeatures]) -> None:
        self.num_layers = len(features[0].layer_vectors)
        self.layer_means = []
        self.layer_precisions = []

        all_layer_data = [
            np.array([f.layer_vectors[l] for f in features], dtype=np.float64)
            for l in range(self.num_layers)
        ]

        for l in range(self.num_layers):
            X_l = all_layer_data[l]
            N, D = X_l.shape
            mu = np.mean(X_l, axis=0)
            diff = X_l - mu
            cov = np.cov(diff, rowvar=False)

            trace_cov = np.trace(cov) / max(1, D)
            cov_reg = (1.0 - self.shrinkage) * cov + self.shrinkage * trace_cov * np.eye(D)

            try:
                prec = np.linalg.pinv(cov_reg)
            except np.linalg.LinAlgError:
                prec = np.eye(D)

            self.layer_means.append(mu)
            self.layer_precisions.append(prec)

        self.cal_layer_maha_means = []
        self.cal_layer_maha_stds = []
        layer_maha_scores = []

        for l in range(self.num_layers):
            X_l = all_layer_data[l]
            mu = self.layer_means[l]
            prec = self.layer_precisions[l]
            diff = X_l - mu
            d_sq = np.sum(np.dot(diff, prec) * diff, axis=1)
            m_l = float(np.mean(d_sq))
            s_l = float(np.std(d_sq))
            if s_l < 1e-8:
                s_l = 1e-8
            self.cal_layer_maha_means.append(m_l)
            self.cal_layer_maha_stds.append(s_l)
            layer_maha_scores.append((d_sq - m_l) / s_l)

        combined_cal_scores = np.max(np.column_stack(layer_maha_scores), axis=1)
        self.threshold = float(np.percentile(combined_cal_scores, (1.0 - self.target_fpr) * 100.0))
        self.is_fitted = True

    def score(self, features: List[RTDETRDeepFeatures]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("LayerwiseMahalanobisDetector chưa được fit!")

        all_layer_data = [
            np.array([f.layer_vectors[l] for f in features], dtype=np.float64)
            for l in range(self.num_layers)
        ]

        layer_scores = []
        for l in range(self.num_layers):
            X_l = all_layer_data[l]
            mu = self.layer_means[l]
            prec = self.layer_precisions[l]
            diff = X_l - mu
            d_sq = np.sum(np.dot(diff, prec) * diff, axis=1)
            z_l = (d_sq - self.cal_layer_maha_means[l]) / self.cal_layer_maha_stds[l]
            layer_scores.append(z_l)

        return np.max(np.column_stack(layer_scores), axis=1)

    def predict(self, features: List[RTDETRDeepFeatures]) -> np.ndarray:
        scores = self.score(features)
        return (scores >= self.threshold).astype(int)


class CalibratedMaxOREnsemble:
    """
    Bộ kết hợp Cực Trị Chuẩn Hóa Khớp FPR (Reliability-Aware 4-Tier Ensemble):
    Kết hợp 4 kênh giám sát độc lập đại diện cho các tầng phòng thủ:
    - Kênh 1 (Tier 0 - Query Population Activity): ActiveQ_Split (Phát hiện Vanishing & Fabrication)
    - Kênh 2 (Tier 0 - Global Confidence Drift)  : Top1Mean_Split (Phát hiện Confidence Inflation/Deflation)
    - Kênh 3 (Tier 1 - Encoder Selection Dynamics): ESM_Dropoff (Phát hiện AQSM / Information Starvation)
    - Kênh 4 (Tier 2 - Deep Latent Representation): Layer_Maha (Phát hiện Gradient Patch & Local Perturbations)

    Toàn bộ 4 kênh được chuẩn hóa phân vị thực nghiệm trên tập Calibration:
        q_k(x) = Rank_{clean}(s_k(x)) / N_cal in [0, 1]
    Điểm gộp cực trị:
        S_fused(x) = max_{k=1..4} q_k(x)
    Ngưỡng tau được hiệu chuẩn tại phân vị (1 - target_fpr) của {S_fused(x_i)}_{i=1}^{N_cal},
    đảm bảo FPR trên tập Calibration đạt chính xác target_fpr (5.0%).
    """

    def __init__(
        self,
        target_fpr: float = 0.05,
        shrinkage: float = 0.05,
    ):
        self.target_fpr = target_fpr
        self.shrinkage = shrinkage
        self.detectors: Dict[str, Any] = {}
        self.cal_sorted_scores: Dict[str, np.ndarray] = {}
        self.threshold = 0.0
        self.is_fitted = False

    def fit(self, features: List[RTDETRDeepFeatures]) -> None:
        # 1. Khởi tạo 4 kênh giám sát đại diện 4 tầng
        # Kênh 1 sử dụng active_query_soft (Continuous Soft Query Activation) để triệt tiêu hoàn toàn Discrete Tie
        self.detectors = {
            "ActiveQ_Split": SingleFeatureBaseline("active_query_soft", tail="two_sided_split", target_fpr=self.target_fpr),
            "Top1Mean_Split": SingleFeatureBaseline("query_top1_mean", tail="two_sided_split", target_fpr=self.target_fpr),
            "ESM_Dropoff": SingleFeatureBaseline("enc_dropoff", tail="two_sided_split", target_fpr=self.target_fpr),
            "Layer_Maha": LayerwiseMahalanobisDetector(shrinkage=self.shrinkage, target_fpr=self.target_fpr),
        }

        # 2. Huấn luyện từng detector trên tập Calibration và lưu phân phối điểm sạch
        self.cal_sorted_scores = {}
        raw_cal_scores: Dict[str, np.ndarray] = {}
        for name, det in self.detectors.items():
            det.fit(features)
            s = det.score(features)
            raw_cal_scores[name] = s
            self.cal_sorted_scores[name] = np.sort(s)

        # 3. Tính điểm Max-Score cho từng mẫu trong tập Calibration
        N_cal = len(features)
        cal_max_scores = []
        for i in range(N_cal):
            quantiles = []
            for name in self.detectors:
                val = raw_cal_scores[name][i]
                rank_l = np.searchsorted(self.cal_sorted_scores[name], val, side="left")
                rank_r = np.searchsorted(self.cal_sorted_scores[name], val, side="right")
                q = (rank_l + rank_r) / (2.0 * float(N_cal))
                quantiles.append(q)
            cal_max_scores.append(max(quantiles))

        cal_max_arr = np.array(cal_max_scores, dtype=np.float64)
        # 4. Hiệu chuẩn ngưỡng tau_max trên chính phân phối điểm max của tập Calibration
        self.threshold = float(np.percentile(cal_max_arr, (1.0 - self.target_fpr) * 100.0))
        self.is_fitted = True

    def score(self, features: List[RTDETRDeepFeatures]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("CalibratedMaxOREnsemble chưa được fit!")

        all_quantiles = []
        for name, det in self.detectors.items():
            s = det.score(features)
            rank_l = np.searchsorted(self.cal_sorted_scores[name], s, side="left")
            rank_r = np.searchsorted(self.cal_sorted_scores[name], s, side="right")
            q = (rank_l + rank_r) / (2.0 * float(len(self.cal_sorted_scores[name])))
            all_quantiles.append(q)

        # S_max = max qua tất cả các detectors đã chuẩn hóa
        matrix = np.column_stack(all_quantiles)
        return np.max(matrix, axis=1)

    def predict(self, features: List[RTDETRDeepFeatures]) -> np.ndarray:
        scores = self.score(features)
        return (scores >= self.threshold).astype(int)


class EncoderSelectionMonitor:
    """
    Bộ giám sát Quá trình Tuyển chọn Query tại Cửa ngõ Encoder (Encoder Selection Monitor - ESM):
    Chuyên biệt phát hiện đòn tấn công Adversarial Query Selection Manipulation (AQSM).
    Theo dõi tỷ số suy giảm điểm (enc_dropoff) và margin giữa Top-300 và 1800 candidates nền (enc_margin).
    Sử dụng hiệu chuẩn phân vị phi tham số hai phía (two_sided_split) với ngân sách target_fpr = 0.05.
    """

    def __init__(self, target_fpr: float = 0.05):
        self.target_fpr = target_fpr
        self.dropoff_detector = SingleFeatureBaseline("enc_dropoff", tail="two_sided_split", target_fpr=target_fpr)
        self.margin_detector = SingleFeatureBaseline("enc_margin", tail="two_sided_split", target_fpr=target_fpr)
        self.is_fitted = False

    def fit(self, features: List[RTDETRDeepFeatures]) -> None:
        self.dropoff_detector.fit(features)
        self.margin_detector.fit(features)
        self.is_fitted = True

    def score(self, features: List[RTDETRDeepFeatures]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("EncoderSelectionMonitor chưa được fit!")
        s_drop = self.dropoff_detector.score(features)
        s_marg = self.margin_detector.score(features)
        return np.maximum(s_drop, s_marg)

    def predict(self, features: List[RTDETRDeepFeatures]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("EncoderSelectionMonitor chưa được fit!")
        p_drop = self.dropoff_detector.predict(features)
        p_marg = self.margin_detector.predict(features)
        return np.bitwise_or(p_drop, p_marg)

