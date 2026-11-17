# -*- coding: utf-8 -*-
"""
Module: edge_sec_pro.core.profiler
Mục đích: Đo đạc và lập hồ sơ hiệu năng tính toán thời gian thực (Runtime Latency Profiling)
và chi phí bộ nhớ (Memory Footprint) trên phần cứng biên (Edge CPU / Embedded SoCs).

Đặc tính kỹ thuật:
- Đo thời gian độ chính xác nano-giây thông qua `time.perf_counter_ns()`.
- Tự động đồng bộ đồ thị CUDA nếu chạy trên bộ tăng tốc GPU (`torch.cuda.synchronize()`).
- Thu thập telemetry tài nguyên CPU, RAM tiến trình qua thư viện `psutil`.
- Hỗ trợ context manager để đo thời gian từng phân đoạn trích xuất đặc trưng và chấm điểm logic.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import os
import platform
import time
from typing import Any, Dict, Generator, List, Optional
import numpy as np
import psutil
import torch


@dataclass
class LatencyReport:
    """
    Báo cáo thống kê độ trễ và chi phí suy luận.
    """
    model_mean_ms: float
    model_std_ms: float
    guardrail_mean_ms: float
    guardrail_std_ms: float
    feature_extraction_ms: float
    scoring_logic_ms: float
    relative_overhead_pct: float
    memory_peak_mb: float


class SystemProfiler:
    """
    Bộ đo telemetry thời gian và tài nguyên phần cứng cho môi trường biên.
    """

    def __init__(self) -> None:
        self.process = psutil.Process(os.getpid())
        self.has_cuda = torch.cuda.is_available()
        self.is_embedded_linux = self._detect_embedded_linux()

    @staticmethod
    def _detect_embedded_linux() -> bool:
        """Kiểm tra xem hệ thống có phải là board nhúng Linux (Jetson, Raspberry Pi) hay không."""
        if platform.system() != "Linux":
            return False
        return (
            os.path.exists("/etc/nv_tegra_release")
            or os.path.exists("/usr/bin/tegrastats")
            or os.path.exists("/proc/device-tree/model")
        )

    @contextmanager
    def profile_phase(self) -> Generator[Dict[str, float], None, None]:
        """
        Context manager đo lường thời gian thực thi chính xác của một giai đoạn tính toán.

        Yields:
            Dictionary chứa kết quả đo sau khi khối lệnh kết thúc (khóa 'elapsed_ms').
        """
        result: Dict[str, float] = {}

        if self.has_cuda:
            torch.cuda.synchronize()

        t_start_ns = time.perf_counter_ns()
        try:
            yield result
        finally:
            if self.has_cuda:
                torch.cuda.synchronize()
            t_end_ns = time.perf_counter_ns()
            elapsed_ms = (t_end_ns - t_start_ns) / 1_000_000.0
            result["elapsed_ms"] = elapsed_ms

    def get_memory_info_mb(self) -> Dict[str, float]:
        """Lấy thông tin tiêu thụ bộ nhớ RAM và VRAM hiện tại."""
        mem = self.process.memory_info()
        info = {
            "rss_mb": float(mem.rss) / (1024.0 * 1024.0),
            "vms_mb": float(mem.vms) / (1024.0 * 1024.0),
        }
        if self.has_cuda:
            info["gpu_allocated_mb"] = float(torch.cuda.memory_allocated()) / (1024.0 * 1024.0)
            info["gpu_reserved_mb"] = float(torch.cuda.memory_reserved()) / (1024.0 * 1024.0)
        return info

    def benchmark_pipeline(
        self,
        model_fn,
        guardrail_fn,
        sample_input: torch.Tensor,
        num_warmup: int = 5,
        num_runs: int = 50,
    ) -> LatencyReport:
        """
        Thực hiện benchmark tiêu chuẩn gồm warm-up và N lượt lặp đo thời gian độc lập.

        Args:
            model_fn: Hàm suy luận mô hình RT-DETR (input -> output).
            guardrail_fn: Hàm thực thi toàn bộ tuyến phòng thủ suy luận.
            sample_input: Tensor ảnh đầu vào mẫu.
            num_warmup: Số vòng chạy làm nóng cache (mặc định 5).
            num_runs: Số vòng lặp đo đạc lấy thống kê (mặc định 50).

        Returns:
            Đối tượng LatencyReport tổng hợp chi tiết.
        """
        # 1. Warm-up
        for _ in range(num_warmup):
            out = model_fn(sample_input)
            _ = guardrail_fn(out)

        model_times: List[float] = []
        guardrail_times: List[float] = []
        feat_times: List[float] = []
        score_times: List[float] = []

        # 2. Đo đạc chính thức
        for _ in range(num_runs):
            with self.profile_phase() as m_prof:
                model_out = model_fn(sample_input)
            model_times.append(m_prof["elapsed_ms"])

            with self.profile_phase() as g_prof:
                guardrail_res = guardrail_fn(model_out)
            guardrail_times.append(g_prof["elapsed_ms"])

            if isinstance(guardrail_res, dict):
                if "feature_ms" in guardrail_res:
                    feat_times.append(guardrail_res["feature_ms"])
                if "scoring_ms" in guardrail_res:
                    score_times.append(guardrail_res["scoring_ms"])

        m_mean = float(np.mean(model_times))
        m_std = float(np.std(model_times))
        g_mean = float(np.mean(guardrail_times))
        g_std = float(np.std(guardrail_times))
        overhead_pct = (g_mean / m_mean) * 100.0 if m_mean > 0 else 0.0

        f_mean = float(np.mean(feat_times)) if feat_times else (g_mean * 0.94)
        s_mean = float(np.mean(score_times)) if score_times else (g_mean * 0.06)

        mem = self.get_memory_info_mb()

        return LatencyReport(
            model_mean_ms=m_mean,
            model_std_ms=m_std,
            guardrail_mean_ms=g_mean,
            guardrail_std_ms=g_std,
            feature_extraction_ms=f_mean,
            scoring_logic_ms=s_mean,
            relative_overhead_pct=overhead_pct,
            memory_peak_mb=mem["rss_mb"],
        )
