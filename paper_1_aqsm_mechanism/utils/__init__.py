# -*- coding: utf-8 -*-
"""
Package: paper_1_aqsm_mechanism.utils
Mục đích: Cung cấp bộ công cụ tiện ích chuẩn hóa (Shared Utilities) phục vụ Paper 1:
  - cli: Khởi tạo parser tham số dòng lệnh chung.
  - device: Dò phần cứng, cấu hình GPU/CPU và quản lý VRAM.
  - io_helpers: Ghi/đọc JSON serializable, ghi CSV per-image, format thời gian.
  - stats: Phân tích phương sai 2x2 RM-ANOVA, Paired Bootstrap 95% CI.
"""

from .cli import create_base_parser
from .device import clear_gpu_memory, get_device_info, setup_cpu_threads
from .io_helpers import format_duration, load_json, save_json, to_serializable, write_csv_rows
from .stats import bootstrap_paired_diff, repeated_measures_2x2_anova

__all__ = [
    "create_base_parser",
    "get_device_info",
    "setup_cpu_threads",
    "clear_gpu_memory",
    "save_json",
    "load_json",
    "write_csv_rows",
    "format_duration",
    "to_serializable",
    "bootstrap_paired_diff",
    "repeated_measures_2x2_anova",
]
