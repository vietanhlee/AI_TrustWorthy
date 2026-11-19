# -*- coding: utf-8 -*-
"""
Module: paper_1_aqsm_mechanism.utils.device
Mục đích: Cung cấp các tiện ích phát hiện phần cứng, cấu hình thiết bị tính toán (CUDA/CPU),
tối ưu hóa số luồng CPU và quản lý thu hồi bộ nhớ đồ họa (VRAM) chuẩn công nghiệp.
"""

import gc
import os
import sys
from typing import Any, Dict, Optional, Union
import torch


def get_device_info(requested_device: str = "auto") -> Dict[str, Any]:
    """
    Tự động dò tìm và trả về thông tin chi tiết về môi trường phần cứng.
    Hỗ trợ các tùy chọn: 'auto', 'cuda', 'cuda:0', 'cpu'.
    """
    cuda_available = torch.cuda.is_available()

    if requested_device == "auto":
        device_type = "cuda" if cuda_available else "cpu"
    elif requested_device.startswith("cuda"):
        if not cuda_available:
            print("[CẢNH BÁO] Yêu cầu thiết bị CUDA nhưng hệ thống không có GPU! Tự động chuyển về CPU.")
            device_type = "cpu"
        else:
            device_type = requested_device
    else:
        device_type = "cpu"

    device = torch.device(device_type)
    cpu_cores = os.cpu_count() or 4

    info: Dict[str, Any] = {
        "device": str(device),
        "device_type": device.type,
        "cuda_available": cuda_available,
        "cpu_cores": cpu_cores,
    }

    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        total_vram_gb = props.total_memory / (1024 ** 3)
        allocated_gb = torch.cuda.memory_allocated(idx) / (1024 ** 3)
        reserved_gb = torch.cuda.memory_reserved(idx) / (1024 ** 3)
        free_vram_gb = total_vram_gb - reserved_gb

        info.update({
            "gpu_index": idx,
            "gpu_name": props.name,
            "vram_total_gb": round(total_vram_gb, 2),
            "vram_allocated_gb": round(allocated_gb, 2),
            "vram_reserved_gb": round(reserved_gb, 2),
            "vram_free_gb": round(free_vram_gb, 2),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        })
    else:
        info.update({
            "gpu_name": "CPU",
            "vram_total_gb": 0.0,
            "vram_free_gb": 0.0,
        })

    return info


# Alias tương thích
detect_hardware = get_device_info


def setup_cpu_threads(num_threads: int = 4) -> int:
    """
    Cấu hình số luồng tính toán CPU an toàn cho PyTorch và OpenMP/MKL.
    """
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)

    actual_threads = min(num_threads, os.cpu_count() or num_threads)
    try:
        torch.set_num_threads(actual_threads)
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(max(1, actual_threads // 2))
    except RuntimeError:
        pass
    return actual_threads


def clear_gpu_memory() -> None:
    """
    Thu hồi bộ nhớ rác và giải phóng cache VRAM của PyTorch.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
