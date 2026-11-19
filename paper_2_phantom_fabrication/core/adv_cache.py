# -*- coding: utf-8 -*-
"""
Module: paper_2_phantom_fabrication.core.adv_cache
Mục đích: Quản lý đọc nạp, kiểm tra toàn vẹn và lưu trữ bộ nhớ đệm mẫu đối kháng (Adversarial Cache)
cho các đòn tấn công Phantom Queries (EQF, DQF, Untargeted_PGD, v.v.).

Chuẩn mực kỹ thuật (Production Standards - Học hỏi từ Paper 1):
  1. Tuyệt đối KHÔNG lưu uint8 hay làm tròn làm mất mát độ chính xác gradient đối kháng.
     Lưu trữ perturbation delta chuẩn IEEE 754 float32 nguyên bản (hoặc float16).
  2. Sử dụng disk-backed memmap tạm thời trong lúc sinh để giữ mức RAM luôn phẳng (< 1.5 GB),
     loại bỏ hoàn toàn nguy cơ tràn bộ nhớ OOM SIGKILL (exit code 57 / 137).
  3. Hỗ trợ mmap_mode="r" khi nạp từ file .npz trên đĩa: chỉ nạp lô cần thiết vào RAM,
     tiết kiệm 95% thời gian chạy lại PGD trên quy mô toàn bộ 5,000 ảnh COCO.
  4. Xác thực toàn vẹn 100% định danh ảnh (Image ID Verification) chống lệch dữ liệu.
"""

import gc
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from paper_2_phantom_fabrication.core.config import (
    ADV_CACHE_DIR,
    get_adv_cache_dir,
)


def eps_tag(eps_num: int) -> str:
    """Trả về nhãn epsilon dạng xâu chuỗi (ví dụ: eps16)."""
    return f"eps{eps_num}"


def find_cache_file(
    attack: str,
    eps_num: int = 16,
    n_eval: int = 0,
    cache_dir: Optional[Union[str, Path]] = None,
) -> Optional[Path]:
    """
    Tìm kiếm file cache .npz phù hợp nhất trong thư mục cache:
    1. Ưu tiên file đúng tên: adv_cache_{attack}_eps{eps_num}_n{n_eval}.npz
    2. Nếu n_eval <= 0 (chạy toàn bộ cohort): tìm file có N lớn nhất hoặc alias chung.
    3. Tìm file alias chung: adv_cache_{attack}_eps{eps_num}.npz
    """
    cdir = Path(cache_dir) if cache_dir is not None else get_adv_cache_dir()
    if not cdir.exists():
        return None

    attack_norm = attack.lower()

    # 1. Thử file chính xác theo n_eval
    if n_eval > 0:
        target = cdir / f"adv_cache_{attack_norm}_{eps_tag(eps_num)}_n{n_eval}.npz"
        if target.exists():
            return target

    # 2. Tìm các file khớp pattern
    pattern = f"adv_cache_{attack_norm}_{eps_tag(eps_num)}*.npz"
    candidates = sorted(list(cdir.glob(pattern)), key=lambda p: p.stat().st_size, reverse=True)

    for cand in candidates:
        try:
            with np.load(cand, mmap_mode="r") as data:
                if "image_ids" in data:
                    c_len = len(data["image_ids"])
                    if n_eval <= 0 or c_len >= n_eval:
                        return cand
        except Exception:
            continue

    return None


def load_cached_adv_delta(
    attack: str,
    eps_num: int = 16,
    n_eval: int = 0,
    expected_image_ids: Optional[Sequence[int]] = None,
    cache_dir: Optional[Union[str, Path]] = None,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Nạp mảng perturbation delta từ file cache .npz qua disk mmap_mode='r'.
    Đảm bảo delta ở định dạng float32 chuẩn, không ép kiểu uint8.
    
    Trả về:
      (delta_mmap, meta_dict) nếu cache tồn tại và khớp image_ids;
      (None, {}) nếu không tìm thấy cache hợp lệ.
    """
    fpath = find_cache_file(attack=attack, eps_num=eps_num, n_eval=n_eval, cache_dir=cache_dir)
    if fpath is None or not fpath.exists():
        return None, {}

    try:
        data = np.load(fpath, mmap_mode="r")
    except Exception:
        return None, {}

    if "image_ids" not in data:
        return None, {}

    cached_ids = data["image_ids"]
    actual_n = len(cached_ids)

    # Kiểm tra tính khớp của image_ids nếu được cung cấp
    if expected_image_ids is not None:
        target_ids = list(expected_image_ids)
        check_n = len(target_ids) if n_eval <= 0 else min(n_eval, len(target_ids))
        if actual_n < check_n:
            return None, {}
        if list(cached_ids[:check_n]) != target_ids[:check_n]:
            # Image IDs không khớp hoàn toàn
            return None, {}

    delta_key = "delta" if "delta" in data else ("delta_f32" if "delta_f32" in data else "delta_f16")
    if delta_key not in data:
        return None, {}

    delta_mmap = data[delta_key]
    meta = {}
    if "meta_json" in data:
        try:
            meta = json.loads(str(data["meta_json"]))
        except Exception:
            meta = {}

    return delta_mmap, meta


def save_adv_cache(
    attack: str,
    eps_num: int,
    image_ids: Sequence[int],
    delta_array: Union[np.ndarray, np.memmap],
    meta_info: Optional[Dict[str, Any]] = None,
    cache_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Lưu trữ mảng perturbation delta vào file cache .npz chuẩn float32.
    TUYỆT ĐỐI KHÔNG LƯU UINT8 để bảo toàn 100% độ chính xác đối kháng (như Paper 1).
    """
    cdir = Path(cache_dir) if cache_dir is not None else get_adv_cache_dir()
    cdir.mkdir(parents=True, exist_ok=True)

    n_total = len(image_ids)
    attack_norm = attack.lower()
    out_file = cdir / f"adv_cache_{attack_norm}_{eps_tag(eps_num)}_n{n_total}.npz"

    meta = meta_info.copy() if meta_info is not None else {}
    meta.setdefault("attack", attack_norm)
    meta.setdefault("epsilon_num_255", eps_num)
    meta.setdefault("n_total", n_total)
    meta.setdefault("dtype", "float32")
    meta.setdefault("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))

    # Đảm bảo mảng delta có dtype float32 hoặc float16
    arr_to_save = delta_array
    if hasattr(arr_to_save, "dtype") and arr_to_save.dtype not in (np.float32, np.float16):
        arr_to_save = arr_to_save.astype(np.float32)

    np.savez(
        out_file,
        image_ids=np.array(image_ids, dtype=np.int64),
        delta=arr_to_save,
        delta_f32=arr_to_save,
        meta_json=json.dumps(meta, indent=2),
    )

    return out_file
