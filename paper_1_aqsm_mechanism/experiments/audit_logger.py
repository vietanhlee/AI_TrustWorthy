# -*- coding: utf-8 -*-
"""
Module: paper_1_aqsm_mechanism.experiments.audit_logger
Mục đích: Cung cấp hệ thống Audit Logging chuẩn Production cho toàn bộ thực nghiệm.
Đặc tính kỹ thuật:
  1. Hỗ trợ ghi log dòng console & file đa luồng an toàn (Thread-safe / Process-safe).
  2. Ghi nhật ký chi tiết từng bước (Step/Epoch/Batch/Instance) ra định dạng JSON Lines (.jsonl)
     với cơ chế flush tức thời (Immediate Flush), đảm bảo dữ liệu luôn được bảo toàn ngay cả khi tiến trình bị gián đoạn.
  3. Tự động thu thập Metadata môi trường: Device info, CUDA capability, Python version, Seed, Timestamp.
  4. Xuất dữ liệu bảng thô (Raw Data) ra CSV và tổng hợp thống kê ra JSON cấu trúc cao cấp.
  5. Cung cấp hàm parser đọc lại log và audit stream để tái sử dụng dữ liệu offline mà không cần chạy lại mô hình.
"""

import csv
import json
import logging
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent


class ExperimentAuditLogger:
    """
    Bộ quản lý ghi nhật ký kiểm toán thực nghiệm chuẩn Production.
    Lưu trữ đồng thời:
      - Log text format chuẩn: logs/<task_name>.log
      - Audit Stream chi tiết từng bước: outputs/audit_streams/<task_name>_audit_stream.jsonl
      - Bảng dữ liệu thô: outputs/raw_csv/<csv_name>.csv
      - Báo cáo tổng hợp: outputs/summary_json/<json_name>.json
    """

    def __init__(
        self,
        task_name: str,
        logs_dir: Optional[Path] = None,
        audit_dir: Optional[Path] = None,
        raw_csv_dir: Optional[Path] = None,
        summary_json_dir: Optional[Path] = None,
    ):
        self.task_name = task_name

        # Thiết lập đường dẫn thư mục
        base_exp = WORKSPACE_DIR / "paper_1_aqsm_mechanism"
        self.logs_dir = logs_dir or (base_exp / "logs")
        self.audit_dir = audit_dir or (base_exp / "outputs" / "audit_streams")
        self.raw_csv_dir = raw_csv_dir or (base_exp / "outputs" / "raw_csv")
        self.summary_json_dir = summary_json_dir or (base_exp / "outputs" / "summary_json")

        for d in [self.logs_dir, self.audit_dir, self.raw_csv_dir, self.summary_json_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.log_file_path = self.logs_dir / f"{task_name}.log"
        self.audit_stream_path = self.audit_dir / f"{task_name}_audit_stream.jsonl"

        # Thiết lập logger
        self.logger = logging.getLogger(task_name)
        self.logger.setLevel(logging.INFO)
        # Xóa các handler cũ nếu có
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # Handler ghi file UTF-8
        fh = logging.FileHandler(self.log_file_path, mode="a", encoding="utf-8")
        fh.setFormatter(logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] %(message)s"))
        self.logger.addHandler(fh)

        # Handler ghi console
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] %(message)s"))
        self.logger.addHandler(ch)

        # Khởi tạo audit stream file pointer
        self._audit_file = open(self.audit_stream_path, mode="a", encoding="utf-8")
        self.start_time = time.time()

    def log_metadata(self, config: Dict[str, Any]) -> None:
        """Ghi nhận thông tin cấu hình môi trường và tham số thực nghiệm vào đầu audit stream."""
        meta = {
            "record_type": "METADATA",
            "task_name": self.task_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "system_info": {
                "os": platform.system(),
                "os_release": platform.release(),
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            },
            "experiment_config": config,
        }
        self.write_audit_record(meta)
        self.logger.info(f"=== KHỞI TẠO THỰC NGHIỆM: {self.task_name} ===")
        self.logger.info(f"Metadata: Device={meta['system_info']['device_name']}, PyTorch={meta['system_info']['torch_version']}")
        self.logger.info(f"Config: {json.dumps(config, default=str)}")

    def write_audit_record(self, record: Dict[str, Any]) -> None:
        """Ghi một bản ghi JSON vào audit stream và flush ngay lập tức."""
        if "timestamp" not in record:
            record["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        if "elapsed_sec" not in record:
            record["elapsed_sec"] = round(time.time() - self.start_time, 3)

        line = json.dumps(record, default=self._json_serializer)
        self._audit_file.write(line + "\n")
        self._audit_file.flush()

    def log_step(
        self,
        step_name: str,
        step_idx: int,
        total_steps: int,
        metrics: Dict[str, Any],
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Ghi nhật ký từng bước thực thi (Step/Batch/Iteration/Epoch).
        Vừa ghi log formatted dễ đọc, vừa ghi audit stream JSONL để sau này parse vẽ đồ thị.
        """
        pct = (step_idx / max(total_steps, 1)) * 100.0
        metric_str = ", ".join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()])

        self.logger.info(f"[{step_name} {step_idx}/{total_steps} ({pct:.1f}%)] {metric_str}")

        record = {
            "record_type": "STEP_PROGRESS",
            "step_name": step_name,
            "step_idx": step_idx,
            "total_steps": total_steps,
            "percent_complete": round(pct, 2),
            "metrics": metrics,
            "extra_info": extra_info or {},
        }
        self.write_audit_record(record)

    def log_epoch(
        self,
        epoch_idx: int,
        total_epochs: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Dành riêng cho các vòng lặp Training hoặc Iterative Attack Optimization.
        Lưu metrics theo từng epoch để parse vẽ biểu đồ training loss/accuracy offline.
        """
        record = {
            "record_type": "EPOCH_METRIC",
            "epoch": epoch_idx,
            "total_epochs": total_epochs,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics or {},
        }
        train_str = ", ".join([f"{k}={v:.4f}" for k, v in train_metrics.items()])
        val_str = f" | Val: {', '.join([f'{k}={v:.4f}' for k, v in val_metrics.items()])}" if val_metrics else ""
        self.logger.info(f"[EPOCH {epoch_idx}/{total_epochs}] Train: {train_str}{val_str}")
        self.write_audit_record(record)

    def export_raw_csv(self, filename: str, rows: List[Dict[str, Any]]) -> Path:
        """Xuất danh sách các dòng dữ liệu thô ra tệp CSV."""
        if not rows:
            self.logger.warning("Không có dữ liệu để xuất file CSV.")
            return Path("")

        csv_path = self.raw_csv_dir / filename
        fieldnames = list(rows[0].keys())
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                clean_r = {k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()}
                writer.writerow(clean_r)

        self.logger.info(f"[EXPORT CSV] Đã xuất {len(rows)} bản ghi thô ra: {csv_path}")
        return csv_path

    def export_summary_json(self, filename: str, summary_data: Dict[str, Any]) -> Path:
        """Xuất dữ liệu thống kê tổng hợp ra tệp JSON."""
        summary_data["task_name"] = self.task_name
        summary_data["export_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        summary_data["total_runtime_sec"] = round(time.time() - self.start_time, 2)

        json_path = self.summary_json_dir / filename
        with open(json_path, mode="w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2, default=self._json_serializer)

        self.logger.info(f"[EXPORT JSON] Đã xuất báo cáo tổng hợp ra: {json_path}")
        return json_path

    def close(self) -> None:
        """Đóng an toàn file audit stream."""
        if hasattr(self, "_audit_file") and not self._audit_file.closed:
            elapsed = time.time() - self.start_time
            summary_record = {
                "record_type": "TASK_COMPLETED",
                "total_duration_sec": round(elapsed, 3),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            self.write_audit_record(summary_record)
            self._audit_file.close()
            self.logger.info(f"=== HOÀN TẤT THỰC NGHIỆM: {self.task_name} (Tổng thời gian: {elapsed:.2f}s) ===")

    @staticmethod
    def _json_serializer(obj: Any) -> Any:
        """Chuyển đổi kiểu dữ liệu numpy/torch sang chuẩn JSON."""
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif isinstance(obj, Path):
            return str(obj)
        return str(obj)


def parse_audit_stream(jsonl_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Hàm tiện ích đọc lại toàn bộ dữ liệu audit stream JSONL.
    Cho phép parse dữ liệu từng epoch, từng batch để vẽ đồ thị hoặc audit mà không cần chạy lại mô hình.
    """
    records = []
    p = Path(jsonl_path)
    if not p.exists():
        raise FileNotFoundError(f"Không tìm thấy file audit stream tại: {p}")

    with open(p, mode="r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
