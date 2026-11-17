# -*- coding: utf-8 -*-
"""
Module: paper_1_aqsm_mechanism.core.adv_cache
Mục đích: Quản lý đọc nạp bộ nhớ đệm mẫu đối kháng (adversarial image cache)
đảm bảo 100% tính nhất quán thực nghiệm qua các task phân tích độc lập.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch

from paper_1_aqsm_mechanism.experiments.config import ADV_CACHE_DIR

DEFAULT_CACHE_DIR: Path = ADV_CACHE_DIR


def eps_tag(eps_num: int) -> str:
    """Trả về nhãn epsilon dạng xâu chuỗi (ví dụ: eps16)."""
    return f"eps{eps_num}"


def cache_path(
    attack: str,
    eps_num: int,
    n_eval: int,
    cache_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Xây dựng đường dẫn file nén npz chứa perturbation delta."""
    cdir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    return cdir / f"adv_cache_{attack}_{eps_tag(eps_num)}_n{n_eval}.npz"


def load_cached_adv_images(
    attack: str,
    eps_num: int,
    n_eval: int,
    clean_imgs: torch.Tensor,
    expected_image_ids: List[int],
    cache_dir: Optional[Union[str, Path]] = None,
) -> Tuple[torch.Tensor, Dict]:
    """
    Nạp ảnh đối kháng từ file cache npz trên đĩa và cộng trực tiếp vào tập ảnh sạch.
    """
    fpath = cache_path(attack, eps_num, n_eval, cache_dir=cache_dir)
    if not fpath.exists():
        raise FileNotFoundError(f"Chưa tìm thấy cache đối kháng tại: {fpath}")

    data = np.load(fpath, allow_pickle=False)
    cached_ids = data["image_ids"].tolist()
    if cached_ids != list(expected_image_ids):
        raise ValueError(
            f"Image IDs không khớp giữa cache {fpath.name} và cohort hiện tại! "
            f"Cache có {len(cached_ids)} ảnh, cohort có {len(expected_image_ids)} ảnh."
        )

    delta = torch.from_numpy(data["delta_f16"].astype(np.float32))
    # Đảm bảo device khớp với clean_imgs
    if clean_imgs.device != delta.device:
        delta = delta.to(clean_imgs.device)

    adv = torch.clamp(clean_imgs + delta, 0.0, 1.0)
    meta = json.loads(str(data["meta_json"]))
    return adv, meta
