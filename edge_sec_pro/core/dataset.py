# -*- coding: utf-8 -*-
"""
Module: edge_sec_pro.core.dataset
Mục đích: Quản lý nạp và tiền xử lý tập dữ liệu hình ảnh COCO val2017 theo giao thức
phân vùng nghiêm ngặt không rò rỉ dữ liệu (Disjoint Zero-Leakage Partition Protocol).

Hỗ trợ:
1. Nạp và chuẩn hóa kích thước ảnh (640x640) sang Tensor [0.0, 1.0].
2. Phân chia độc lập tập Hiệu chuẩn (N_cal = 1000) và tập Kiểm thử sạch (N_test = 500).
3. Lớp Dataset PyTorch chuẩn phục vụ suy luận hàng loạt (Batch Inference).
4. Các bộ biến đổi tạo nhóm đối chứng âm tự nhiên (Natural Environmental Corruptions).
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from tqdm import tqdm

from edge_sec_pro.core.config import (
    COCO_VAL_DIR,
    IMG_SIZE,
    N_CALIBRATION_SAMPLES,
    N_TEST_CLEAN_SAMPLES,
)


class COCOValSubsetDataset(Dataset):
    """
    Dataset PyTorch quản lý tập ảnh COCO val2017 đã được nạp sẵn vào bộ nhớ.
    """

    def __init__(self, tensors: torch.Tensor, filenames: Optional[List[str]] = None) -> None:
        self.tensors = tensors
        self.filenames = filenames or [f"img_{i:05d}.jpg" for i in range(len(tensors))]

    def __len__(self) -> int:
        return len(self.tensors)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        return self.tensors[idx], self.filenames[idx]


def load_coco_images(
    coco_dir: Optional[Union[str, Path]] = None,
    n_cal: int = N_CALIBRATION_SAMPLES,
    n_test: int = N_TEST_CLEAN_SAMPLES,
    img_size: int = IMG_SIZE,
    seed: int = 42,
    show_progress: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Nạp danh sách ảnh COCO val2017 và phân chia thành 2 tập độc lập:
    - Tập Hiệu chuẩn (Calibration Set): Dùng để fitting các bộ giám sát và ngưỡng Max-OR.
    - Tập Kiểm thử sạch (Clean Test Set): Dùng để đánh giá FPR danh định (Zero Leakage).

    Args:
        coco_dir: Thư mục chứa ảnh .jpg của COCO val2017.
        n_cal: Số lượng ảnh hiệu chuẩn (mặc định 1000).
        n_test: Số lượng ảnh kiểm thử sạch (mặc định 500).
        img_size: Kích thước cạnh vuông chuẩn hóa (mặc định 640).
        seed: Random seed để cố định danh sách chọn mẫu.
        show_progress: Bật thanh tiến trình tqdm nếu cần theo dõi.

    Returns:
        Tuple (cal_tensors, test_tensors) dạng Tensor [N, 3, img_size, img_size].
    """
    data_dir = Path(coco_dir) if coco_dir is not None else COCO_VAL_DIR
    all_files = sorted(list(data_dir.glob("*.jpg")))
    total_needed = n_cal + n_test

    if len(all_files) < total_needed:
        raise FileNotFoundError(
            f"Thư mục {data_dir} chỉ chứa {len(all_files)} ảnh, không đủ yêu cầu {total_needed} ảnh."
        )

    rng = np.random.RandomState(seed)
    chosen_indices = rng.choice(len(all_files), size=total_needed, replace=False)
    chosen_files = [all_files[idx] for idx in chosen_indices]

    iterator = tqdm(chosen_files, desc="[COCO] Nạp tập ảnh COCO val2017", leave=False) if show_progress else chosen_files
    tensor_list: List[torch.Tensor] = []

    for fpath in iterator:
        try:
            with Image.open(fpath) as img:
                rgb = img.convert("RGB").resize((img_size, img_size))
                tensor_list.append(TF.to_tensor(rgb))
        except Exception:
            # Fallback nếu file ảnh bị hỏng: trả về ảnh xám trung tính 0.5
            tensor_list.append(torch.full((3, img_size, img_size), 0.5, dtype=torch.float32))

    all_tensors = torch.stack(tensor_list, dim=0)
    cal_tensors = all_tensors[:n_cal]
    test_tensors = all_tensors[n_cal:total_needed]

    return cal_tensors, test_tensors


def apply_natural_corruption(
    images: torch.Tensor,
    corruption_type: str,
    severity_param: Optional[float] = None,
) -> torch.Tensor:
    """
    Áp dụng các biến dạng tự nhiên và đối chứng âm (Negative Controls) lên batch ảnh:
    - 'gaussian_noise': Nhiễu cảm biến Gauss (sigma = 16/255).
    - 'random_patch': Patch nhiễu ngẫu nhiên kích thước 48x48 dán vào vị trí ngẫu nhiên.
    - 'gaussian_blur': Mờ chuyển động thấu kính (kernel 5x5, sigma = 1.0/1.5).
    - 'brightness_reduction': Giảm độ sáng -30% (ánh sáng chập choạng / sương mù).

    Args:
        images: Tensor ảnh [B, 3, H, W] trong dải [0.0, 1.0].
        corruption_type: Tên loại biến dạng.
        severity_param: Tham số điều chỉnh mức độ nghiêm trọng.

    Returns:
        Tensor ảnh đã áp dụng biến dạng [B, 3, H, W] kẹp trong [0.0, 1.0].
    """
    corrupted = images.clone()
    device = images.device

    if corruption_type in ["gaussian_noise", "noise"]:
        sigma = severity_param if severity_param is not None else (16.0 / 255.0)
        noise = torch.randn_like(corrupted) * sigma
        corrupted = torch.clamp(corrupted + noise, 0.0, 1.0)

    elif corruption_type in ["random_patch", "patch"]:
        patch_sz = int(severity_param) if severity_param is not None else 48
        B, C, H, W = corrupted.shape
        rng = np.random.RandomState(42)
        for b in range(B):
            top = rng.randint(0, max(1, H - patch_sz))
            left = rng.randint(0, max(1, W - patch_sz))
            random_pixels = torch.rand((C, patch_sz, patch_sz), device=device)
            corrupted[b, :, top : top + patch_sz, left : left + patch_sz] = random_pixels

    elif corruption_type in ["gaussian_blur", "blur"]:
        sigma = severity_param if severity_param is not None else 1.5
        # Sử dụng torchvision gaussian blur
        import torchvision.transforms as T
        blurrer = T.GaussianBlur(kernel_size=5, sigma=sigma)
        corrupted = blurrer(corrupted)

    elif corruption_type in ["brightness_reduction", "brightness"]:
        factor = severity_param if severity_param is not None else 0.70  # Giảm 30%
        corrupted = torch.clamp(corrupted * factor, 0.0, 1.0)

    return corrupted
