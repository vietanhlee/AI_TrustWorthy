# -*- coding: utf-8 -*-
"""
Module: edge_sec_pro.core.detector
Mục đích: Facade module cung cấp giao diện truy cập chuẩn mực cho các mô hình
Transformer Object Detectors (RT-DETR).

Tái xuất và đóng gói RTDETRModelWrapper từ transformer_monitors, cho phép
khởi tạo mô hình nhanh với các siêu tham số định sẵn từ config.
"""

from typing import Optional, Union
import torch

from edge_sec_pro.core.config import MODEL_NAME, get_device
from edge_sec_pro.transformer_monitors.detr_wrapper import (
    DETRInternalState,
    DETRModelWrapper,
    RTDETRModelWrapper,
)


def load_edge_detector(
    model_name: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> RTDETRModelWrapper:
    """
    Hàm tiện ích nạp mô hình RT-DETR đã cấu hình sẵn các hook trích xuất trạng thái nội tại.

    Args:
        model_name: Tên hoặc đường dẫn trọng số mô hình (mặc định từ config).
        device: Thiết bị đích để chuyển đổi trọng số (CPU / CUDA).

    Returns:
        Instance RTDETRModelWrapper sẵn sàng suy luận và can thiệp bảo mật.
    """
    target_model = model_name or MODEL_NAME
    target_device = torch.device(device) if device is not None else get_device()
    wrapper = RTDETRModelWrapper(pretrained_model_name=target_model, device=target_device)
    return wrapper


__all__ = [
    "DETRInternalState",
    "DETRModelWrapper",
    "RTDETRModelWrapper",
    "load_edge_detector",
]
