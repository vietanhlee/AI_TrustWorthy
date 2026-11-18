# -*- coding: utf-8 -*-
"""
Module: paper_2_phantom_fabrication.experiments.exp_logger_p2
Mục đích: Khung Logging và Audit dữ liệu thực nghiệm chuẩn Production cho Paper 2:
1. Ghi log có cấu trúc (Console UTF-8 + FileHandler) kèm timestamp, CPU/GPU, tiến độ chi tiết.
2. Theo dõi và ghi vết lịch sử huấn luyện / tối ưu hóa / sweep (MetricHistoryTracker) ra CSV.
   Cho phép parse các metrics qua từng epochs/steps để vẽ lại biểu đồ học tập mà không cần chạy lại.
3. Ghi vết toàn bộ dự đoán, số lượng phantoms, active queries, displacement rates ở mức từng ảnh
   (SampleAuditTracker) ra CSV, phục vụ kiểm toán (audit) độc lập và phân tích hồi cứu.
4. Tính toán và lưu trữ các thống kê phân phối, khoảng tin cậy 95% Bootstrap, kiểm định Wilcoxon
   ra file JSON có cấu trúc đầy đủ.
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

EXPERIMENTS_DIR = Path(__file__).resolve().parent
PAPER2_DIR = EXPERIMENTS_DIR.parent
WORKSPACE_DIR = PAPER2_DIR.parent
LOGS_DIR = PAPER2_DIR / "logs"
RESULTS_DIR = PAPER2_DIR / "results"

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

    logger = logging.getLogger(f"Paper2.{exp_name}")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

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

    logger.info("=" * 85)
    logger.info(f"KHỞI TẠO TIẾN TRÌNH THỰC NGHIỆM PAPER 2: {exp_name}")
    logger.info(f"Thời gian bắt đầu  : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"File log chi tiết  : {log_file}")
    logger.info(f"Python Runtime     : {sys.version.split()[0]} | Platform: {sys.platform}")
    logger.info(f"PyTorch Version    : {torch.__version__} | Threads: {torch.get_num_threads()}")
    logger.info(f"CUDA khả dụng      : {torch.cuda.is_available()}")
    logger.info("=" * 85)

    return logger, log_file


class MetricHistoryTracker:
    """
    Theo dõi và ghi vết lịch sử huấn luyện / tối ưu hóa theo từng Epoch / Step ra CSV.
    Ví dụ: PGD iteration steps, optimization loss, learning rate, margin violations.
    Cho phép parse trực tiếp để vẽ biểu đồ convergence mà không cần chạy lại mô hình.
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
        }
        if extra_metrics:
            for k, v in extra_metrics.items():
                record[k] = float(v) if isinstance(v, (int, float, np.floating, np.integer)) else str(v)

        self.history.append(record)

        msg_parts = [f"[Epoch {epoch:2d} | Step {step:3d}/{total_steps:3d}] Loss: {loss:.5f}"]
        if task_loss is not None:
            msg_parts.append(f"Task_Loss: {task_loss:.5f}")
        if reg_loss is not None:
            msg_parts.append(f"Reg_Loss: {reg_loss:.5f}")
        if extra_metrics:
            for k, v in extra_metrics.items():
                if isinstance(v, (int, float, np.floating, np.integer)):
                    msg_parts.append(f"{k}: {float(v):.4f}")
        msg_parts.append(f"Elapsed: {elapsed:.1f}s")

        if logger:
            logger.info(" | ".join(msg_parts))

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
    Ghi vết toàn bộ chỉ số thực nghiệm ở mức từng ảnh (Sample-level) ra file CSV.
    Cho phép kiểm toán chi tiết, phân tích tương quan, hoặc vẽ lại các biểu đồ phân phối
    (KDE, Boxplot, Histogram) mà không cần chạy lại suy luận mô hình.
    """

    def __init__(self, exp_name: str, output_dir: Optional[Path] = None):
        self.exp_name = exp_name
        self.output_dir = output_dir if output_dir is not None else RESULTS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.samples: List[Dict[str, Any]] = []

    def record_sample(
        self,
        sample_idx: int,
        image_id: Union[int, str],
        condition: str,
        metrics: Dict[str, Any],
    ) -> None:
        """Ghi nhận dữ liệu của 1 ảnh."""
        row: Dict[str, Any] = {
            "sample_idx": int(sample_idx),
            "image_id": str(image_id),
            "condition": str(condition),
        }
        for k, v in metrics.items():
            if isinstance(v, (int, np.integer)):
                row[k] = int(v)
            elif isinstance(v, (float, np.floating)):
                row[k] = round(float(v), 5)
            elif isinstance(v, (bool, np.bool_)):
                row[k] = int(v)
            else:
                row[k] = str(v)

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


def compute_distribution_stats(arr: Union[List[float], np.ndarray]) -> Dict[str, float]:
    """Tính các thống kê phân phối đầy đủ (mean, std, median, IQR, percentiles)."""
    a = np.array(arr, dtype=np.float64)
    if len(a) == 0:
        return {}
    return {
        "mean": float(np.mean(a)),
        "std": float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
        "median": float(np.median(a)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "p2_5": float(np.percentile(a, 2.5)),
        "p25": float(np.percentile(a, 25.0)),
        "p75": float(np.percentile(a, 75.0)),
        "p97_5": float(np.percentile(a, 97.5)),
        "iqr": float(np.percentile(a, 75.0) - np.percentile(a, 25.0)),
    }


def save_audit_json(data: Dict[str, Any], filepath: Path) -> Path:
    """Lưu kết quả kiểm toán ra JSON chuẩn format indent=2."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return filepath
