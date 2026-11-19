# -*- coding: utf-8 -*-
"""
Module: paper_2_phantom_fabrication.experiments.engines.cache_engine
Mục đích: Động cơ sinh bộ nhớ đệm đối kháng (Adversarial Cache Engine) chuẩn float32,
áp dụng disk-backed np.memmap để kiểm soát RAM luôn < 1.5 GB, triệt tiêu 100% lỗi OOM (-9).
"""

import gc
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from paper_2_phantom_fabrication.attacks.phantom_attacks import generate_batch_adversarial
from paper_2_phantom_fabrication.core.adv_cache import save_adv_cache
from paper_2_phantom_fabrication.core.config import get_adv_cache_dir


def generate_single_attack_cache(
    model: torch.nn.Module,
    dataloader: DataLoader,
    cohort: Dict[str, Any],
    attack_name: str,
    eps_int: int = 16,
    steps: int = 8,
    target_class_idx: int = 1,
    cache_dir: Optional[Union[str, Path]] = None,
    device: torch.device = torch.device("cpu"),
    force_recompute: bool = False,
) -> Path:
    """
    Sinh toàn bộ perturbation delta cho một đòn tấn công trên toàn bộ tập ảnh
    và lưu thành file .npz chuẩn float32 nguyên bản qua memory-mapped streaming.
    """
    cdir = Path(cache_dir) if cache_dir is not None else get_adv_cache_dir()
    cdir.mkdir(parents=True, exist_ok=True)

    n_total = len(cohort["dataset"])
    image_ids = cohort["image_ids"]
    eps = float(eps_int) / 255.0
    alpha = 2.0 / 255.0 if eps_int >= 8 else 1.0 / 255.0

    target_file = cdir / f"adv_cache_{attack_name.lower()}_eps{eps_int}_n{n_total}.npz"

    # Kiểm tra nếu cache đã tồn tại và hợp lệ
    if not force_recompute and target_file.exists():
        try:
            with np.load(target_file, mmap_mode="r") as data:
                if "image_ids" in data and len(data["image_ids"]) == n_total:
                    if list(data["image_ids"]) == list(image_ids):
                        print(f"[CACHE GEN] Đã tìm thấy cache hoàn chỉnh tại: {target_file.name}. Bỏ qua bước sinh.")
                        return target_file
        except Exception:
            print(f"[CACHE GEN] Cache hiện tại bị lỗi đọc, tiến hành sinh lại: {target_file.name}")

    print(f"\n[CACHE GEN] ==================================================================")
    print(f"[CACHE GEN] Đang sinh: {attack_name.upper()} | eps={eps_int}/255 ({eps:.4f}) | N={n_total} ảnh")
    print(f"[CACHE GEN] PGD Steps={steps} | Alpha={alpha:.5f} | Device={device}")
    print(f"[CACHE GEN] ==================================================================")

    # Tạo disk-backed memmap tạm thời trên đĩa cứng để kiểm soát RAM < 1.5 GB
    temp_mmap_path = cdir / f"tmp_{attack_name.lower()}_eps{eps_int}_{os.getpid()}_{int(time.time())}.dat"
    delta_mmap = np.memmap(
        temp_mmap_path,
        dtype=np.float32,
        mode="w+",
        shape=(n_total, 3, 640, 640),
    )

    t_start = time.time()
    pbar = tqdm(dataloader, desc=f"Gen {attack_name.upper()}_eps{eps_int}", dynamic_ncols=True)

    for batch_images, batch_indices in pbar:
        bx = batch_images.to(device)

        adv_b = generate_batch_adversarial(
            model=model,
            bx=bx,
            attack_type=attack_name,
            target_class_idx=target_class_idx,
            epsilon=eps,
            alpha=alpha,
            steps=steps,
        )

        delta_batch = (adv_b - bx).detach().cpu().to(torch.float32).numpy()
        delta_mmap[batch_indices.numpy()] = delta_batch

        del bx, adv_b, delta_batch
        if device.type == "cuda":
            torch.cuda.empty_cache()

    delta_mmap.flush()
    elapsed = time.time() - t_start

    meta_info = {
        "attack": attack_name.lower(),
        "n_total": n_total,
        "epsilon_num_255": eps_int,
        "epsilon": eps,
        "alpha": alpha,
        "steps": steps,
        "resolution": "640x640",
        "dtype": "float32",
        "elapsed_sec": round(elapsed, 2),
        "throughput_fps": round(n_total / max(elapsed, 1e-4), 2),
        "device": str(device),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    print(f"[CACHE GEN] Đang lưu file cache chuẩn float32 vào: {target_file.name}...")
    save_adv_cache(
        attack=attack_name,
        eps_num=eps_int,
        image_ids=image_ids,
        delta_array=delta_mmap,
        meta_info=meta_info,
        cache_dir=cdir,
    )

    # Dọn dẹp memmap tạm
    del delta_mmap
    gc.collect()
    if temp_mmap_path.exists():
        try:
            temp_mmap_path.unlink()
        except Exception:
            pass

    print(f"[CACHE GEN] Hoàn thành sinh {attack_name.upper()} trong {elapsed:.1f}s ({n_total / max(elapsed, 1e-4):.2f} FPS).")
    return target_file
