# -*- coding: utf-8 -*-
"""
Module: paper_2_phantom_fabrication.core.defense_engine
Mục đích: Cung cấp các giải pháp phòng thủ đối kháng chuyên sâu và chuẩn đối chứng:
  1. Spatial-Query Divergence (SQD) Filter: Bộ lọc kiểm định tính nhất quán không gian - truy vấn.
  2. Baseline Input Preprocessing Defenses:
     - JPEG Compression (Quality = 75)
     - Gaussian Blur (Kernel = 5x5, Sigma = 1.0)
     - Median Filter (Kernel = 3x3)
     - Bit-Depth Reduction (4-bit quantization)
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

from paper_2_phantom_fabrication.defenses.baseline_defenses import (
    BitDepthReductionDefense,
    GaussianBlurDefense,
    JPEGCompressionDefense,
    MedianFilterDefense,
)
from paper_2_phantom_fabrication.defenses.sqd_filter import (
    SQDFilter,
    evaluate_sqd_filter_on_detections,
)


def get_all_baseline_defenses() -> Dict[str, Any]:
    """Khởi tạo và trả về từ điển chứa toàn bộ 4 cơ chế phòng thủ đối chứng."""
    return {
        "jpeg": JPEGCompressionDefense(quality=75),
        "gaussian": GaussianBlurDefense(kernel_size=5, sigma=1.0),
        "median": MedianFilterDefense(kernel_size=3),
        "bitdepth": BitDepthReductionDefense(bits=4),
    }
