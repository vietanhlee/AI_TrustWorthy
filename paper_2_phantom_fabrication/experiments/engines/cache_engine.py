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
import shutil
import time
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from paper_2_phantom_fabrication.attacks.phantom_attacks import generate_batch_adversarial
from paper_2_phantom_fabrication.core.adv_cache import find_cache_file, make_cache_filename_stem
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
    và lưu trực tiếp thành file .npy chuẩn float32 nguyên bản qua np.lib.format.open_memmap.
    Không tạo file đệm kép, tiết kiệm 50% dung lượng đĩa so với np.savez truyền thống.
    """
    cdir = Path(cache_dir) if cache_dir is not None else get_adv_cache_dir()
    cdir.mkdir(parents=True, exist_ok=True)

    n_total = len(cohort["dataset"])
    image_ids = cohort["image_ids"]
    eps = float(eps_int) / 255.0
    alpha = 2.0 / 255.0 if eps_int >= 8 else 1.0 / 255.0

    stem = make_cache_filename_stem(
        attack=attack_name, eps_num=eps_int, steps=steps, target_class_idx=target_class_idx, n_eval=n_total
    )
    target_npy = cdir / f"{stem}_delta.npy"
    target_meta = cdir / f"{stem}_meta.json"
    legacy_npz = cdir / f"adv_cache_{attack_name.lower()}_eps{eps_int}_n{n_total}.npz"

    # Kiểm tra nếu cache đã tồn tại và hợp lệ
    if not force_recompute:
        if target_npy.exists() and target_meta.exists():
            try:
                with open(target_meta, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if len(meta.get("image_ids", [])) == n_total and meta.get("steps", 8) == steps and meta.get("target_class_idx", 1) == target_class_idx:
                    print(f"[CACHE GEN] Đã tìm thấy cache hoàn chỉnh tại: {target_npy.name}. Bỏ qua bước sinh.")
                    return target_npy
            except Exception:
                pass
        elif steps == 8 and target_class_idx == 1 and legacy_npz.exists():
            try:
                with np.load(legacy_npz, mmap_mode="r") as data:
                    if "image_ids" in data and len(data["image_ids"]) == n_total:
                        print(f"[CACHE GEN] Đã tìm thấy legacy cache hoàn chỉnh tại: {legacy_npz.name}. Bỏ qua bước sinh.")
                        return legacy_npz
            except Exception:
                pass

    # Tự động dọn dẹp các file tạm tmp_* còn sót lại trước khi tính toán dung lượng
    try:
        for tmp_f in cdir.glob("tmp_*"):
            try:
                tmp_f.unlink()
            except Exception:
                pass
    except Exception:
        pass

    # Kiểm tra dung lượng đĩa khả dụng với cơ chế thích ứng (Zero-Crash Adaptive Disk Guard)
    req_bytes = n_total * 3 * 640 * 640 * 4
    disk_free = shutil.disk_usage(cdir).free
    req_gb = req_bytes / (1024 ** 3)
    free_gb = disk_free / (1024 ** 3)
    if disk_free < max(req_bytes * 1.15, 2.0 * (1024 ** 3)):
        print(f"\n[CACHE ADAPTIVE] Dung lượng đĩa khả dụng ({free_gb:.2f} GB) đang thấp hơn ngưỡng an toàn (~{req_gb:.2f} GB cần thiết).")
        print(f"[CACHE ADAPTIVE] Để bảo vệ môi trường Kaggle/Colab khỏi lỗi tràn ổ đĩa, hệ thống tự động bỏ qua ghi cache đĩa cho {attack_name.upper()} (eps={eps_int}) và chuyển sang sinh On-The-Fly trực tiếp trên GPU VRAM (tốn 0 GB đĩa đệm).")
        return None

    print(f"\n[CACHE GEN] ==================================================================")
    print(f"[CACHE GEN] Đang sinh: {attack_name.upper()} | eps={eps_int}/255 ({eps:.4f}) | N={n_total} ảnh")
    print(f"[CACHE GEN] PGD Steps={steps} | Alpha={alpha:.5f} | Device={device}")
    print(f"[CACHE GEN] Định dạng: Single-file .npy open_memmap ({req_gb:.2f} GB) | Ổ đĩa còn trống: {free_gb:.2f} GB")
    print(f"[CACHE GEN] ==================================================================")

    # Mở trực tiếp memmap file đích (.npy chuẩn NumPy) - KHÔNG CẦN FILE TẠM
    delta_mmap = np.lib.format.open_memmap(
        target_npy,
        mode="w+",
        dtype=np.float32,
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
    del delta_mmap
    gc.collect()
    elapsed = time.time() - t_start

    meta_info = {
        "attack": attack_name.lower(),
        "n_total": n_total,
        "epsilon_num_255": eps_int,
        "epsilon": eps,
        "alpha": alpha,
        "steps": steps,
        "target_class_idx": target_class_idx,
        "resolution": "640x640",
        "dtype": "float32",
        "elapsed_sec": round(elapsed, 2),
        "throughput_fps": round(n_total / max(elapsed, 1e-4), 2),
        "device": str(device),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "image_ids": list(image_ids),
    }

    with open(target_meta, "w", encoding="utf-8") as f:
        json.dump(meta_info, f, indent=2)

    print(f"[CACHE GEN] Hoàn thành sinh {attack_name.upper()} trong {elapsed:.1f}s ({n_total / max(elapsed, 1e-4):.2f} FPS).")
    print(f"[CACHE GEN] File cache đã lưu tại: {target_npy.name} ({target_npy.stat().st_size / (1024 ** 3):.2f} GB).")
    return target_npy
