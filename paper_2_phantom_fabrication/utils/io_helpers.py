# -*- coding: utf-8 -*-
"""
Module: paper_2_phantom_fabrication.utils.io_helpers
Mục đích: Cung cấp các tiện ích xử lý vào/ra tệp tin (I/O Helpers) chuẩn hóa:
  - Lưu và nạp JSON an toàn với kiểu dữ liệu NumPy / PyTorch.
  - Ghi CSV buffer tối ưu.
  - Định dạng thời gian thực thi (format_duration).
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

import numpy as np
import torch


def to_serializable(obj: Any) -> Any:
    """Chuyển đổi các đối tượng NumPy, PyTorch hoặc Path sang kiểu dữ liệu Python thuần."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(x) for x in obj]
    if isinstance(obj, set):
        return [to_serializable(x) for x in sorted(list(obj))]
    return obj


def save_json(data: Any, path: Union[str, Path], indent: int = 2) -> None:
    """Lưu dữ liệu ra file JSON, tự động tạo thư mục cha nếu chưa có."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    serializable_data = to_serializable(data)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(serializable_data, f, indent=indent, ensure_ascii=False)


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Nạp dữ liệu từ file JSON."""
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def write_csv_rows(path: Union[str, Path], header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    """Ghi dữ liệu dạng bảng ra file CSV chuẩn UTF-8."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def format_duration(seconds: float) -> str:
    """Định dạng thời gian tính bằng giây thành chuỗi thân thiện với người đọc."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        m = int(seconds // 60)
        s = seconds % 60
        return f"{m}m {s:.1f}s"
    else:
        h = int(seconds // 3600)
        rem = seconds % 3600
        m = int(rem // 60)
        s = rem % 60
        return f"{h}h {m}m {s:.0f}s"
