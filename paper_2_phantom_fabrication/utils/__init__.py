# -*- coding: utf-8 -*-
"""
Package: paper_2_phantom_fabrication.utils
Mục đích: Cung cấp hệ thống tiện ích chuẩn hóa dùng chung cho toàn bộ Paper 2:
  - cli: Khởi tạo parser tham số dòng lệnh tiêu chuẩn.
  - device: Nhận diện và quản lý thiết bị phần cứng (GPU/CPU, Multi-GPU, VRAM).
  - io_helpers: Đọc ghi file JSON, CSV, định dạng thời gian và serialization.
  - stats: Các thuật toán thống kê khoa học (Bootstrap CI 95%, Wilson Score Interval).
"""

from paper_2_phantom_fabrication.utils.cli import create_base_parser
from paper_2_phantom_fabrication.utils.device import (
    clear_gpu_memory,
    detect_hardware,
    get_device,
    setup_cpu_threads,
)
from paper_2_phantom_fabrication.utils.io_helpers import (
    format_duration,
    load_json,
    save_json,
    to_serializable,
    write_csv_rows,
)
from paper_2_phantom_fabrication.utils.stats import (
    bootstrap_ci_95,
    paired_diff_test,
    wilson_score_interval,
)

__all__ = [
    # CLI
    "create_base_parser",
    # Device
    "detect_hardware",
    "get_device",
    "clear_gpu_memory",
    "setup_cpu_threads",
    # IO Helpers
    "save_json",
    "load_json",
    "write_csv_rows",
    "format_duration",
    "to_serializable",
    # Stats
    "bootstrap_ci_95",
    "wilson_score_interval",
    "paired_diff_test",
]
