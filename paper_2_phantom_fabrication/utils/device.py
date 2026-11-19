# -*- coding: utf-8 -*-
"""
Module: paper_2_phantom_fabrication.utils.device
Mục đích: Quản lý thiết bị phần cứng, tự động tối ưu hóa tài nguyên GPU/CPU,
nhận diện Multi-GPU DataParallelism và dọn dẹp bộ nhớ VRAM sạch sẽ.
"""

import gc
import os
import platform
from typing import Any, Dict, List, Optional, Tuple

import torch


def detect_hardware(preferred_device: str = "auto") -> Dict[str, Any]:
    """
    Phát hiện và phân tích chi tiết môi trường phần cứng hiện tại (GPU/CPU).
    """
    info: Dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_threads": os.cpu_count() or 4,
        "device_type": "cpu",
        "primary_device": "cpu",
        "num_gpus": 0,
        "gpu_names": [],
        "vram_gb": [],
        "cuda_version": None,
        "recommended_batch_size": 4,
    }

    use_cuda = False
    if preferred_device.lower() in ("cuda", "gpu"):
        use_cuda = torch.cuda.is_available()
    elif preferred_device.lower() == "auto":
        use_cuda = torch.cuda.is_available()

    if use_cuda:
        info["device_type"] = "cuda"
        info["primary_device"] = "cuda:0"
        info["num_gpus"] = torch.cuda.device_count()
        info["cuda_version"] = torch.version.cuda

        for i in range(info["num_gpus"]):
            props = torch.cuda.get_device_properties(i)
            info["gpu_names"].append(props.name)
            vram = props.total_memory / (1024 ** 3)
            info["vram_gb"].append(round(vram, 2))

        # Đề xuất batch size tối ưu theo VRAM
        min_vram = min(info["vram_gb"]) if info["vram_gb"] else 8.0
        if min_vram >= 24.0:
            base_bs = 24
        elif min_vram >= 15.0:
            base_bs = 16
        elif min_vram >= 7.5:
            base_bs = 8
        else:
            base_bs = 4
        info["recommended_batch_size"] = base_bs * max(1, info["num_gpus"])
    else:
        info["device_type"] = "cpu"
        info["primary_device"] = "cpu"
        info["recommended_batch_size"] = 4

    return info


def get_device(preferred: str = "auto") -> torch.device:
    """Trả về đối tượng torch.device tương ứng."""
    hw = detect_hardware(preferred_device=preferred)
    return torch.device(hw["primary_device"])


def clear_gpu_memory() -> None:
    """Giải phóng toàn bộ VRAM không sử dụng và chạy Garbage Collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()


def setup_cpu_threads(num_threads: int = 8) -> None:
    """Thiết lập số luồng tối đa cho PyTorch trên CPU."""
    if num_threads > 0:
        torch.set_num_threads(num_threads)
        os.environ["OMP_NUM_THREADS"] = str(num_threads)
        os.environ["MKL_NUM_THREADS"] = str(num_threads)
