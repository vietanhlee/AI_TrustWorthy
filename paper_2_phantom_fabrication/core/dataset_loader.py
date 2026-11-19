# -*- coding: utf-8 -*-
"""
Module: paper_2_phantom_fabrication.core.dataset_loader
Mục đích: Bộ nạp dữ liệu ảnh COCO dạng luồng lười (Streaming Lazy DataLoader)
giúp xử lý quy mô lớn (5,000 ảnh COCO val2017) mà không bao giờ bị tràn RAM (Zero-OOM).

Cơ chế hoạt động:
  - Chỉ lưu trữ metadata và đường dẫn tệp tin trong bộ nhớ.
  - Khi slice batch `clean_proxy[start:end]`, ảnh mới được đọc từ đĩa, resize 640x640 và chuyển thành Tensor.
  - Sau khi kết thúc batch, bộ nhớ được thu hồi tức thì qua `del` và `gc.collect()`.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF

from paper_2_phantom_fabrication.core.coco_gt import discover_coco_annotations, load_flexible_coco_cohort


class LazyImageBatchProxy:
    """
    Proxy nạp ảnh lười: Giả lập một Tensor [N, 3, H, W] nhưng chỉ thực sự nạp ảnh khi được slice.
    """

    def __init__(
        self,
        image_paths: Sequence[Union[str, Path]],
        img_size: int = 640,
        device: torch.device = torch.device("cpu"),
    ):
        self.image_paths = [Path(p) for p in image_paths]
        self.img_size = img_size
        self.device = device
        self._len = len(self.image_paths)

    def __len__(self) -> int:
        return self._len

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        return (self._len, 3, self.img_size, self.img_size)

    def __getitem__(self, idx: Union[int, slice]) -> torch.Tensor:
        if isinstance(idx, int):
            if idx < 0:
                idx += self._len
            if idx < 0 or idx >= self._len:
                raise IndexError(f"Chỉ số ảnh vượt quá giới hạn: {idx} (độ dài: {self._len})")
            paths = [self.image_paths[idx]]
        elif isinstance(idx, slice):
            paths = self.image_paths[idx]
        else:
            raise TypeError(f"Kiểu chỉ mục không hợp lệ: {type(idx)}")

        tensors = []
        for p in paths:
            with Image.open(p) as img:
                rgb = img.convert("RGB")
                if rgb.size != (self.img_size, self.img_size):
                    rgb = rgb.resize((self.img_size, self.img_size), Image.BILINEAR)
                t = TF.to_tensor(rgb)
                tensors.append(t)

        batch_t = torch.stack(tensors, dim=0)
        return batch_t.to(self.device)


class LazyAdvImageBatchProxy:
    """
    Proxy nạp ảnh đối kháng lười:
    Cộng perturbation delta (float32 từ mmap trên đĩa) vào ảnh sạch tương ứng on-the-fly.
    """

    def __init__(
        self,
        clean_proxy: LazyImageBatchProxy,
        delta_mmap: Union[np.ndarray, np.memmap],
        device: torch.device = torch.device("cpu"),
    ):
        self.clean_proxy = clean_proxy
        self.delta_mmap = delta_mmap
        self.device = device
        self._len = len(clean_proxy)

    def __len__(self) -> int:
        return self._len

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        return self.clean_proxy.shape

    def __getitem__(self, idx: Union[int, slice]) -> torch.Tensor:
        clean_b = self.clean_proxy[idx]
        delta_b_np = self.delta_mmap[idx]
        delta_t = torch.from_numpy(delta_b_np if delta_b_np.dtype == np.float32 else delta_b_np.astype(np.float32)).to(clean_b.device)
        adv_b = torch.clamp(clean_b + delta_t, 0.0, 1.0)
        return adv_b.to(self.device)


class FlexibleCOCODataset:
    """
    Dataset COCO linh hoạt hỗ trợ Lazy Loading toàn diện:
    - Nạp danh sách annotations và file ảnh.
    - Cung cấp `tensors` dạng LazyImageBatchProxy.
    - Cung cấp `gt_list` và `image_ids` chuẩn xác.
    """

    def __init__(
        self,
        folder_image: Union[str, Path],
        annotations_file: Optional[Union[str, Path]] = None,
        img_size: int = 640,
        n_samples: int = 0,
        seed: int = 42,
        device: torch.device = torch.device("cpu"),
    ):
        self.img_folder = Path(folder_image).resolve()
        self.ann_file = discover_coco_annotations(self.img_folder, annotations_file)
        self.img_size = img_size
        self.device = device

        cohort = load_flexible_coco_cohort(
            img_folder=self.img_folder,
            ann_file=self.ann_file,
            n_samples=n_samples,
            seed=seed,
            img_size=img_size,
            preload_images=False,
        )

        self.cohort = cohort
        self.dataset = cohort["dataset"]
        self.image_ids = cohort["image_ids"]
        self.gt_list = cohort["gt_list"]
        self.image_paths = [item["img_path"] for item in self.dataset]

        self.tensors = LazyImageBatchProxy(
            image_paths=self.image_paths,
            img_size=img_size,
            device=device,
        )

    def __len__(self) -> int:
        return len(self.image_ids)
