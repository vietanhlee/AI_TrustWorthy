"""
Module: detr_query_monitor.py
Mục đích: Bộ phát hiện bất thường tầng mô hình (Model-Layer Anomaly Detector)
cho Transformer Object Detectors bằng thuật toán Mahalanobis++ chuẩn hóa đa tỉ lệ.

Đặc tính kỹ thuật:
- Chuẩn hóa Z-Score theo từng chiều đặc trưng với ngưỡng bảo vệ phương sai tối thiểu (Regularized Std).
- Tính toán khoảng cách Mahalanobis chính quy hóa (Ledoit-Wolf / Ridge Shrinkage).
- Tích hợp điểm số Radial Log-Magnitude linh hoạt để bắt cả sự lệch hướng lẫn đột biến độ lớn.
- Hoạt động 100% dựa trên nội tại của mô hình (Query & Cross-Attention features).
- Tuyệt đối KHÔNG phụ thuộc vào thời gian thực thi (Latency/Timers) hay telemetry phần cứng.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np


class DETRQueryMonitor:
    """
    Bộ giám sát an toàn một lớp (One-Class Safety Monitor) trên đa tạp đặc trưng Transformer.
    """

    def __init__(
        self,
        shrinkage: float = 0.20,
        radial_weight: float = 0.0,
        target_fpr: float = 0.05,
        min_std: float = 0.02,
        epsilon: float = 1e-6,
    ) -> None:
        """
        Khởi tạo DETRQueryMonitor.

        Tham số:
            shrinkage: Hệ số co cụm hiệp phương sai về ma trận đơn vị (đảm bảo ma trận khả nghịch).
            radial_weight: Trọng số của điểm số bất thường theo độ lớn vector (Radial Magnitude).
            target_fpr: Tỷ lệ báo động sai mục tiêu (False Positive Rate) khi hiệu chuẩn ngưỡng (mặc định 5%).
            min_std: Ngưỡng dưới cho độ lệch chuẩn từng chiều (chống nổ số học khi phương sai gần 0).
            epsilon: Giá trị nhỏ để tránh chia cho 0.
        """
        self.shrinkage = shrinkage
        self.radial_weight = radial_weight
        self.target_fpr = target_fpr
        self.min_std = min_std
        self.epsilon = epsilon

        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.inv_cov: Optional[np.ndarray] = None
        self.rad_mean: float = 0.0
        self.rad_std: float = 1.0
        self.threshold: float = 0.0
        self.is_fitted: bool = False

    def fit(self, clean_features: np.ndarray) -> "DETRQueryMonitor":
        """
        Huấn luyện và ước lượng phân phối đặc trưng chuẩn mực từ dữ liệu ảnh sạch.

        Tham số:
            clean_features: Ma trận đặc trưng [M, d] từ M mẫu ảnh sạch (d=16).

        Trả về:
            Self (instance đã được fit).
        """
        M, d = clean_features.shape
        if M < 2:
            raise ValueError(f"Cần ít nhất 2 mẫu để tính toán hiệp phương sai, nhận được M={M}")

        # 1. Chuẩn hóa quy mô từng đặc trưng (Feature-wise Standardization với Regularized Std)
        self.mean = np.mean(clean_features, axis=0)  # [d]
        raw_std = np.std(clean_features, axis=0)  # [d]
        self.std = np.maximum(raw_std, self.min_std)  # [d]
        z_clean = (clean_features - self.mean) / self.std  # [M, d]

        # 2. Phân tích thành phần Radial (Log-Magnitude trong không gian chuẩn hóa Z)
        z_norms = np.linalg.norm(z_clean, axis=1) + self.epsilon
        log_norms = np.log(z_norms)
        self.rad_mean = float(np.mean(log_norms))
        self.rad_std = float(np.std(log_norms)) + self.epsilon

        # 3. Ước lượng ma trận hiệp phương sai co cụm (Shrinkage Covariance)
        sample_cov = np.cov(z_clean, rowvar=False)  # [d, d]
        if sample_cov.ndim == 0:
            sample_cov = np.array([[float(sample_cov)]])

        prior_cov = np.trace(sample_cov) / d * np.eye(d)
        shrunk_cov = (1.0 - self.shrinkage) * sample_cov + self.shrinkage * prior_cov

        # Nghịch đảo ma trận ổn định
        self.inv_cov = np.linalg.pinv(shrunk_cov)
        self.is_fitted = True

        # 4. Hiệu chuẩn ngưỡng (Threshold Calibration) dựa trên target_fpr
        clean_scores = self.compute_anomaly_scores(clean_features)
        self.threshold = float(np.percentile(clean_scores, (1.0 - self.target_fpr) * 100))

        return self

    def compute_anomaly_scores(self, features: np.ndarray) -> np.ndarray:
        """
        Tính điểm số bất thường (Anomaly Score) cho tập vector đặc trưng.

        Tham số:
            features: Ma trận đặc trưng [K, d].

        Trả về:
            numpy.ndarray: Mảng 1 chiều [K] chứa điểm số bất thường.
        """
        if not self.is_fitted or self.mean is None or self.std is None or self.inv_cov is None:
            raise RuntimeError("Mô hình DETRQueryMonitor chưa được fit. Hãy gọi fit() trước.")

        z = (features - self.mean) / self.std  # [K, d]

        # Khoảng cách Mahalanobis bình phương: z * inv_cov * z^T
        mahal_dist_sq = np.sum(z @ self.inv_cov * z, axis=1)  # [K]
        mahal_dist_sq = np.maximum(0.0, mahal_dist_sq)

        if self.radial_weight > 0.0:
            z_norms = np.linalg.norm(z, axis=1) + self.epsilon
            log_norms = np.log(z_norms)
            rad_score = ((log_norms - self.rad_mean) / self.rad_std) ** 2  # [K]
            return mahal_dist_sq + self.radial_weight * rad_score

        return mahal_dist_sq

    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Phát hiện tấn công cho một hoặc nhiều mẫu.
        """
        scores = self.compute_anomaly_scores(features)
        is_attack = scores > self.threshold
        return is_attack, scores

    def explain_anomaly(
        self,
        feature_vector: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Giải thích đóng góp của từng chiều đặc trưng vào độ lệch bất thường.
        """
        if not self.is_fitted or self.mean is None or self.std is None:
            raise RuntimeError("Mô hình chưa được fit.")

        vec = feature_vector.squeeze()
        z_scores = np.abs(vec - self.mean) / self.std

        names = feature_names or [f"feat_{i}" for i in range(len(vec))]
        contributions = {name: float(z_scores[i]) for i, name in enumerate(names)}
        return dict(sorted(contributions.items(), key=lambda item: item[1], reverse=True))
