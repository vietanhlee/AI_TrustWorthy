# -*- coding: utf-8 -*-
"""
Module: paper_1_aqsm_mechanism.utils.io_helpers
Mục đích: Cung cấp các tiện ích I/O an toàn, chuyển đổi kiểu dữ liệu numpy sang JSON serializable,
đọc/ghi file CSV per-image, và định dạng thời gian chuẩn xác phục vụ nghiên cứu khoa học.
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch


def to_serializable(val: Any) -> Any:
    """
    Chuyển đổi đệ quy các đối tượng kiểu numpy, torch hoặc cấu trúc dữ liệu lồng nhau
    sang kiểu dữ liệu chuẩn của Python (int, float, list, dict, str, bool, None).
    Triệt tiêu 100% lỗi 'TypeError: Object of type int64 is not JSON serializable'.
    """
    if isinstance(val, (int, float, str, bool)) or val is None:
        return val
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, np.bool_):
        return bool(val)
    if isinstance(val, np.ndarray):
        return [to_serializable(v) for v in val.tolist()]
    if isinstance(val, torch.Tensor):
        return [to_serializable(v) for v in val.detach().cpu().tolist()]
    if isinstance(val, Path):
        return str(val)
    if isinstance(val, dict):
        return {str(k): to_serializable(v) for k, v in val.items()}
    if isinstance(val, (list, tuple, set)):
        return [to_serializable(v) for v in val]
    return str(val)


def save_json(path: Union[str, Path], data: Any, indent: int = 2) -> Path:
    """
    Ghi dữ liệu ra file JSON một cách an toàn và chuẩn format.
    Tự động tạo các thư mục cha nếu chưa tồn tại.
    """
    out_path = Path(path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    serializable_data = to_serializable(data)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(serializable_data, f, indent=indent, ensure_ascii=False)
    return out_path


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Đọc dữ liệu từ file JSON một cách an toàn.
    """
    in_path = Path(path).resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file JSON tại: {in_path}")
    with open(in_path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_csv_rows(
    path: Union[str, Path],
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
) -> Path:
    """
    Ghi dữ liệu dạng bảng ra file CSV với chuẩn UTF-8 (có hỗ trợ ký tự tiếng Việt)
    và newline='' để tránh dòng trắng thừa trên Windows.
    """
    out_path = Path(path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for r in rows:
            writer.writerow([to_serializable(v) for v in r])
    return out_path


def format_duration(seconds: float) -> str:
    """
    Định dạng số giây thành chuỗi thời gian trực quan dễ đọc (hh:mm:ss).
    """
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}h:{m:02d}m:{s:02d}s"
