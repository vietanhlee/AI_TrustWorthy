# -*- coding: utf-8 -*-
"""
Module: paper_1_aqsm_mechanism.utils.io_helpers
Mục đích: Cung cấp các tiện ích I/O an toàn, chuyển đổi kiểu dữ liệu numpy sang JSON serializable,
đọc/ghi file CSV per-image, và định dạng thời gian chuẩn xác phục vụ nghiên cứu khoa học.
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

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
    if isinstance(val, torch.Tensor):
        if val.ndim == 0:
            return to_serializable(val.item())
        res = val.detach().cpu().tolist()
        return [to_serializable(v) for v in res] if isinstance(res, list) else res
    if isinstance(val, np.ndarray):
        if val.ndim == 0:
            return to_serializable(val.item())
        res = val.tolist()
        return [to_serializable(v) for v in res] if isinstance(res, list) else res
    if isinstance(val, Path):
        return str(val)
    if isinstance(val, dict):
        return {str(k): to_serializable(v) for k, v in val.items()}
    if isinstance(val, (list, tuple, set)):
        return [to_serializable(v) for v in val]
    return str(val)


def save_json(
    arg1: Any,
    arg2: Any = None,
    indent: int = 2,
    *,
    path: Optional[Union[str, Path]] = None,
    data: Any = None,
) -> Path:
    """
    Ghi dữ liệu ra file JSON một cách an toàn và chuẩn format.
    Tự động tạo các thư mục cha nếu chưa tồn tại.
    Hỗ trợ linh hoạt cả 2 thứ tự tham số:
      - save_json(data, path) (chuẩn phổ biến tương tự json.dump)
      - save_json(path, data)
    Tự động chuyển đổi đệ quy các tensor/numpy/path sang kiểu dữ liệu có thể serialize.
    """
    if path is not None and data is not None:
        actual_path, actual_data = path, data
    elif arg2 is None:
        raise ValueError("save_json yêu cầu cả 2 tham số: dữ liệu (data) và đường dẫn file (path).")
    elif isinstance(arg1, (str, Path)) and not isinstance(arg2, (str, Path)):
        # Trường hợp: save_json(path, data)
        actual_path, actual_data = arg1, arg2
    elif isinstance(arg2, (str, Path)) and not isinstance(arg1, (str, Path)):
        # Trường hợp: save_json(data, path)
        actual_path, actual_data = arg2, arg1
    elif isinstance(arg1, (str, Path)) and isinstance(arg2, (str, Path)):
        # Cả 2 đều là str hoặc Path: phân biệt dựa trên phần mở rộng hoặc dấu phân cách
        str1, str2 = str(arg1), str(arg2)
        if str2.endswith(".json") or "/" in str2 or "\\" in str2:
            actual_path, actual_data = arg2, arg1
        else:
            actual_path, actual_data = arg1, arg2
    else:
        # Mặc định fallback: data trước, path sau
        actual_data, actual_path = arg1, arg2

    out_path = Path(actual_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    serializable_data = to_serializable(actual_data)
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
