# -*- coding: utf-8 -*-
"""
Module: paper_2_phantom_fabrication.core.hardware
Mục đích: Tự động nhận diện phần cứng thực thi (CPU, Single-GPU, Multi-GPU)
và đo lường độ trễ suy luận thời gian thực (Real-time Latency Profiling) chuẩn xác.

Đặc tính kỹ thuật:
  - Trên NVIDIA GPU: Sử dụng torch.cuda.Event(enable_timing=True) độc quyền để đo chính xác
    thời gian thực thi của CUDA Kernels, loại bỏ overhead truyền nhận CPU-GPU.
  - Trên Edge CPU: Sử dụng time.perf_counter() độ phân giải cao kết hợp warm-up passes.
  - Tự động nhận diện cấu hình Multi-GPU để kích hoạt DataParallelism.
"""

import datetime
import logging
import os
import time
from typing import Any, Dict

import numpy as np
import torch

logger = logging.getLogger("HardwareProfiler")


def detect_hardware(requested_device: str = "auto") -> Dict[str, Any]:
    """
    Tự động nhận diện phần cứng thực thi (CPU / Single-GPU / Multi-GPU).
    Trả về cấu hình thiết bị cùng metadata phục vụ ghi log telemetry.
    """
    if requested_device == "auto":
        use_cuda = torch.cuda.is_available()
    elif requested_device.lower() in ("cuda", "gpu"):
        use_cuda = torch.cuda.is_available()
        if not use_cuda:
            logger.warning("Yêu cầu GPU nhưng CUDA không khả dụng. Tự động chuyển về CPU.")
    else:
        use_cuda = False

    hw_info = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    if use_cuda:
        num_gpus = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(i) for i in range(num_gpus)]
        vram_gb = [
            round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 2)
            for i in range(num_gpus)
        ]
        hw_info.update({
            "device_type": "cuda",
            "num_gpus": num_gpus,
            "gpu_names": gpu_names,
            "vram_gb": vram_gb,
            "cuda_version": torch.version.cuda,
            "primary_device": "cuda:0",
            "is_multi_gpu": (num_gpus > 1),
        })
        if num_gpus > 1:
            logger.info(
                f"Phát hiện Multi-GPU: {num_gpus} GPUs khả dụng -> {gpu_names} "
                f"(Tổng VRAM: {sum(vram_gb):.1f} GB). Tự động kích hoạt Data Parallelism."
            )
        else:
            logger.info(f"Phát hiện Single-GPU: {gpu_names[0]} (VRAM: {vram_gb[0]} GB).")
    else:
        cpu_count = os.cpu_count() or 4
        hw_info.update({
            "device_type": "cpu",
            "num_gpus": 0,
            "cpu_cores": cpu_count,
            "primary_device": "cpu",
            "is_multi_gpu": False,
        })
        logger.info(f"Chạy trên CPU (Số luồng CPU khả dụng: {cpu_count})")

    return hw_info


def benchmark_model_latency(
    model: torch.nn.Module,
    sample_tensor: torch.Tensor,
    device: torch.device,
    num_warmup: int = 10,
    num_runs: int = 25,
) -> Dict[str, float]:
    """
    Đo độ trễ suy luận chuẩn xác theo đặc thù thiết bị:
    - GPU: sử dụng torch.cuda.Event(enable_timing=True) với warmup và đồng bộ CUDA.
    - CPU: sử dụng time.perf_counter() với warmup.
    """
    model.eval()
    bx = sample_tensor[:1].to(device)

    with torch.no_grad():
        # Vòng lặp khởi động warm-up ổn định xung nhịp phần cứng
        for _ in range(num_warmup):
            _ = model(pixel_values=bx)

        latencies = []
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            for _ in range(num_runs):
                starter.record()
                _ = model(pixel_values=bx)
                ender.record()
                torch.cuda.synchronize(device)
                latencies.append(starter.elapsed_time(ender))  # milliseconds
        else:
            for _ in range(num_runs):
                t0 = time.perf_counter()
                _ = model(pixel_values=bx)
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000.0)  # milliseconds

    return {
        "mean_ms": float(np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
        "median_ms": float(np.median(latencies)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
    }
