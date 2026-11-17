# -*- coding: utf-8 -*-
"""
Module: paper_3_defense_guardrail.experiments.exp_logger
Mục đích: Cung cấp khung Logging và Audit dữ liệu chuẩn Production cho các thực nghiệm Paper 3:
1. Ghi log có cấu trúc (Console + File) với timestamp, level, thông tin tài nguyên và tham số.
2. Theo dõi và lưu vết quá trình tối ưu hóa / huấn luyện theo từng epoch / step ra file CSV
   (MetricHistoryTracker), cho phép parse và vẽ đồ thị học tập mà không cần chạy lại.
3. Ghi vết toàn bộ dự đoán, điểm dị thường (scores) và nhãn từng mẫu (SampleAuditTracker) ra CSV.
4. Tính toán và lưu trữ toàn bộ chỉ số định lượng (AUROC, TPR, FPR, CIs, Confusion Matrix,
   tọa độ ROC curve 100 điểm) ra file JSON có cấu trúc phục vụ kiểm toán (Audit).
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch

# Đường dẫn thư mục chuẩn
EXPERIMENTS_DIR = Path(__file__).resolve().parent
PAPER3_DIR = EXPERIMENTS_DIR.parent
WORKSPACE_DIR = PAPER3_DIR.parent
LOGS_DIR = PAPER3_DIR / "logs"
RESULTS_DIR = PAPER3_DIR / "results"

LOGS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def setup_experiment_logger(exp_name: str, log_dir: Optional[Path] = None) -> Tuple[logging.Logger, Path]:
    """
    Khởi tạo Logger chuẩn cho thực nghiệm:
    - Ghi ra StreamHandler (Console UTF-8)
    - Ghi ra FileHandler ({exp_name}_{timestamp}.log và {exp_name}_latest.log)
    """
    target_dir = log_dir if log_dir is not None else LOGS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = target_dir / f"{exp_name}_{timestamp}.log"
    latest_log_file = target_dir / f"{exp_name}_latest.log"

    logger = logging.getLogger(f"Paper3.{exp_name}")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # Định dạng log chuẩn
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)-7s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 1. Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 2. File Handler (Timestamped)
    file_handler = logging.FileHandler(str(log_file), mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 3. File Handler (Latest)
    latest_handler = logging.FileHandler(str(latest_log_file), mode="w", encoding="utf-8")
    latest_handler.setLevel(logging.DEBUG)
    latest_handler.setFormatter(formatter)
    logger.addHandler(latest_handler)

    logger.propagate = False

    # Ghi log môi trường khởi tạo
    logger.info("=" * 85)
    logger.info(f"KHỞI TẠO TIẾN TRÌNH THỰC NGHIỆM: {exp_name}")
    logger.info(f"Thời gian bắt đầu  : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"File log chi tiết  : {log_file}")
    logger.info(f"Python Runtime     : {sys.version.split()[0]} | Platform: {sys.platform}")
    logger.info(f"PyTorch Version    : {torch.__version__} | Threads: {torch.get_num_threads()}")
    logger.info(f"CUDA khả dụng      : {torch.cuda.is_available()}")
    logger.info("=" * 85)

    return logger, log_file


class MetricHistoryTracker:
    """
    Theo dõi và ghi vết lịch sử huấn luyện / tối ưu hóa theo từng Epoch / Step.
    Ví dụ: Tối ưu Adversarial Patch qua các epochs, PGD gradient steps, hoặc sweep tham số.
    Cho phép xuất CSV để vẽ biểu đồ Loss, ASR, Learning Rate mà không cần chạy lại từ đầu.
    """

    def __init__(self, exp_name: str, output_dir: Optional[Path] = None):
        self.exp_name = exp_name
        self.output_dir = output_dir if output_dir is not None else RESULTS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history: List[Dict[str, Any]] = []
        self.start_time = time.time()

    def log_step(
        self,
        epoch: int,
        step: int,
        total_steps: int,
        loss: float,
        task_loss: Optional[float] = None,
        reg_loss: Optional[float] = None,
        asr: Optional[float] = None,
        lr: Optional[float] = None,
        extra_metrics: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> Dict[str, Any]:
        """Ghi nhận một bước tối ưu hóa / epoch."""
        elapsed = time.time() - self.start_time
        record = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": round(elapsed, 2),
            "epoch": epoch,
            "step": step,
            "total_steps": total_steps,
            "loss": float(loss),
            "task_loss": float(task_loss) if task_loss is not None else None,
            "reg_loss": float(reg_loss) if reg_loss is not None else None,
            "asr": float(asr) if asr is not None else None,
            "learning_rate": float(lr) if lr is not None else None,
        }
        if extra_metrics:
            for k, v in extra_metrics.items():
                record[k] = float(v) if isinstance(v, (int, float, np.floating, np.integer)) else str(v)

        self.history.append(record)

        # Định dạng câu log
        msg_parts = [f"[Epoch {epoch:2d} | Step {step:3d}/{total_steps:3d}] Loss: {loss:.5f}"]
        if task_loss is not None:
            msg_parts.append(f"Task_Loss: {task_loss:.5f}")
        if reg_loss is not None:
            msg_parts.append(f"Reg_Loss: {reg_loss:.5f}")
        if asr is not None:
            msg_parts.append(f"ASR: {asr*100:5.1f}%")
        if lr is not None:
            msg_parts.append(f"LR: {lr:.6f}")
        msg_parts.append(f"Elapsed: {elapsed:.1f}s")

        full_msg = " | ".join(msg_parts)
        if logger:
            logger.info(full_msg)

        # Tự động lưu bản cập nhật mới nhất ra CSV
        self.save_csv()
        return record

    def save_csv(self, filename: Optional[str] = None) -> Path:
        """Lưu toàn bộ lịch sử ra file CSV."""
        fname = filename if filename is not None else f"{self.exp_name}_history.csv"
        csv_path = self.output_dir / fname
        if not self.history:
            return csv_path

        import csv
        fieldnames = list(self.history[0].keys())
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.history)
        return csv_path


class SampleAuditTracker:
    """
    Theo dõi và ghi vết toàn bộ dự đoán, điểm dị thường (scores) và nhãn ở mức từng mẫu (Sample-level).
    Cho phép kiểm toán, trích xuất ma trận nhầm lẫn, hoặc vẽ lại ROC curves cho mọi detector mà
    không cần nạp lại mô hình hay chạy lại suy luận.
    """

    def __init__(self, exp_name: str, output_dir: Optional[Path] = None):
        self.exp_name = exp_name
        self.output_dir = output_dir if output_dir is not None else RESULTS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.samples: List[Dict[str, Any]] = []

    def record_sample(
        self,
        sample_idx: int,
        condition: str,
        ground_truth: int,
        scores: Dict[str, float],
        preds: Dict[str, int],
        is_attack_success: Optional[int] = None,
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Ghi nhận một mẫu:
        - ground_truth: 0 (Clean / Neg) hoặc 1 (Adversarial / Pos)
        - scores: {detector_name: score_val}
        - preds: {detector_name: 0 hoặc 1}
        """
        row: Dict[str, Any] = {
            "sample_idx": sample_idx,
            "condition": condition,
            "ground_truth": ground_truth,
            "is_attack_success": is_attack_success if is_attack_success is not None else -1,
        }
        for det_name, val in scores.items():
            row[f"score_{det_name}"] = float(val)
        for det_name, val in preds.items():
            row[f"pred_{det_name}"] = int(val)

        if extra_info:
            for k, v in extra_info.items():
                row[k] = float(v) if isinstance(v, (int, float, np.floating, np.integer)) else str(v)

        self.samples.append(row)

    def save_csv(self, filename: Optional[str] = None) -> Path:
        """Xuất bảng dữ liệu chi tiết từng mẫu ra CSV."""
        fname = filename if filename is not None else f"{self.exp_name}_samples.csv"
        csv_path = self.output_dir / fname
        if not self.samples:
            return csv_path

        import csv
        fieldnames = list(self.samples[0].keys())
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.samples)
        return csv_path


def compute_roc_coordinates(labels: np.ndarray, scores: np.ndarray, num_points: int = 100) -> Dict[str, List[float]]:
    """
    Tính toán tọa độ đường cong ROC (FPR, TPR, Thresholds) được lấy mẫu đều tại num_points
    để lưu vào JSON. Về sau có thể vẽ lại toàn bộ ROC Curve chỉ bằng 1 lệnh đọc JSON.
    """
    from sklearn.metrics import roc_curve
    fprs, tprs, thresholds = roc_curve(labels, scores)

    if len(fprs) <= num_points:
        return {
            "fpr": [float(x) for x in fprs],
            "tpr": [float(x) for x in tprs],
            "thresholds": [float(x) for x in thresholds],
        }

    # Lấy mẫu đều theo trục FPR
    target_fprs = np.linspace(0.0, 1.0, num_points)
    sampled_tprs = np.interp(target_fprs, fprs, tprs)
    # Lấy mẫu thresholds tương ứng
    sampled_thresh = np.interp(target_fprs, fprs, thresholds)

    return {
        "fpr": [round(float(x), 5) for x in target_fprs],
        "tpr": [round(float(x), 5) for x in sampled_tprs],
        "thresholds": [round(float(x), 5) for x in sampled_thresh],
    }


def compute_confusion_stats(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Tính toán ma trận nhầm lẫn và các độ đo dẫn xuất."""
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    total = len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "specificity": round(specificity, 4),
    }


def save_audit_json(json_path: Path, data: Dict[str, Any]) -> None:
    """Lưu dữ liệu kiểm toán ra file JSON với định dạng tương thích và thụt lề chuẩn."""
    try:
        from paper_3_defense_guardrail.core.metrics import make_serializable
    except ImportError:
        try:
            from core.metrics import make_serializable
        except ImportError:
            from edge_sec_pro.core.metrics import make_serializable

    json_path.parent.mkdir(parents=True, exist_ok=True)
    serializable_data = make_serializable(data)
    with open(json_path, mode="w", encoding="utf-8") as f:
        json.dump(serializable_data, f, indent=2, ensure_ascii=False)
