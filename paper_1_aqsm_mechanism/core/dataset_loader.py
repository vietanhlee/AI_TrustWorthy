# -*- coding: utf-8 -*-
"""
Module: paper_1_aqsm_mechanism.core.dataset_loader
Mục đích: Bộ nạp dataset COCO val2017 chuẩn công nghiệp (Production-Grade Dataset Loader).
Hỗ trợ:
  1. Nạp linh hoạt toàn bộ tập ảnh trong folder chỉ định (không giới hạn cứng, không hardcode N=30 hay N=300).
  2. Cơ chế nạp theo lô lười (Lazy Batch Loading / Streaming Image Proxy) nhằm ngăn chặn cạn kiệt RAM
     khi thực thi trên quy mô toàn bộ 5,000 ảnh COCO val2017 (24.5 GB raw tensors).
  3. Ánh xạ danh mục nhãn chuẩn xác 100% giữa 80 lớp COCO gốc và RT-DETR head.
  4. Hoàn toàn tương thích ngược với cú pháp tensor slicing `imgs[start:end]` của codebase hiện tại.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF

from paper_1_aqsm_mechanism.core.coco_gt import (
    _COCO_NAME_ALIASES,
    build_coco_category_mapping,
)


class LazyImageBatchProxy:
    """
    Lớp bao bọc lười (Lazy Image Batch Proxy) cung cấp giao diện slicing `proxy[start:end]`
    như một Tensor PyTorch chuẩn, nhưng chỉ nạp và chuẩn hóa ảnh từ đĩa khi có yêu cầu.
    Giúp tiết kiệm 95% dung lượng RAM, cho phép chạy mượt mà trên 5,000 ảnh mà không lo OOM.
    """

    def __init__(
        self,
        image_paths: Sequence[Path],
        img_size: int = 640,
        device: Optional[torch.device] = None,
    ) -> None:
        self.paths = list(image_paths)
        self.img_size = img_size
        self.device = device or torch.device("cpu")

    def __len__(self) -> int:
        return len(self.paths)

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        return (len(self.paths), 3, self.img_size, self.img_size)

    def to(self, device: torch.device) -> "LazyImageBatchProxy":
        """Thiết lập thiết bị đích khi nạp tensor."""
        return LazyImageBatchProxy(self.paths, self.img_size, device=device)

    def __getitem__(self, item: Union[int, slice, Sequence[int]]) -> torch.Tensor:
        if isinstance(item, slice):
            indices = range(*item.indices(len(self.paths)))
            selected_paths = [self.paths[i] for i in indices]
        elif isinstance(item, (int, np.integer)):
            selected_paths = [self.paths[int(item)]]
        elif isinstance(item, (list, tuple, np.ndarray)):
            selected_paths = [self.paths[int(i)] for i in item]
        else:
            raise TypeError(f"Kiểu chỉ số không hợp lệ cho LazyImageBatchProxy: {type(item)}")

        tensors = []
        for p in selected_paths:
            with Image.open(p) as img:
                rgb = img.convert("RGB").resize((self.img_size, self.img_size))
                tensors.append(TF.to_tensor(rgb))

        if not tensors:
            return torch.empty((0, 3, self.img_size, self.img_size), device=self.device)

        batch_tensor = torch.stack(tensors, dim=0)
        if self.device != torch.device("cpu"):
            batch_tensor = batch_tensor.to(self.device)
        return batch_tensor


class LazyAdvImageBatchProxy:
    """
    Lớp bao bọc lười cho ảnh đối kháng (Adversarial Lazy Proxy).
    Tự động áp dụng perturbation delta từ mảng nạp sẵn lên lô ảnh sạch tương ứng.
    """

    def __init__(
        self,
        clean_proxy: LazyImageBatchProxy,
        deltas: np.ndarray,
        device: Optional[torch.device] = None,
    ) -> None:
        self.clean_proxy = clean_proxy
        self.deltas = deltas  # Mảng float32 chuẩn [N, 3, H, W]
        self.device = device or clean_proxy.device

    def __len__(self) -> int:
        return len(self.clean_proxy)

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        return self.clean_proxy.shape

    def to(self, device: torch.device) -> "LazyAdvImageBatchProxy":
        return LazyAdvImageBatchProxy(self.clean_proxy.to(device), self.deltas, device=device)

    def __getitem__(self, item: Union[int, slice, Sequence[int]]) -> torch.Tensor:
        clean_batch = self.clean_proxy[item]
        if isinstance(item, slice):
            delta_sub = self.deltas[item]
        elif isinstance(item, (int, np.integer)):
            delta_sub = self.deltas[int(item) : int(item) + 1]
        elif isinstance(item, (list, tuple, np.ndarray)):
            delta_sub = self.deltas[list(item)]
        else:
            raise TypeError(f"Kiểu chỉ số không hợp lệ: {type(item)}")

        delta_tensor = torch.from_numpy(
            delta_sub if delta_sub.dtype == np.float32 else delta_sub.astype(np.float32)
        ).to(clean_batch.device)
        return torch.clamp(clean_batch + delta_tensor, 0.0, 1.0)


class OnTheFlyAdvImageBatchProxy:
    """
    Lớp bao bọc đối kháng On-The-Fly (Zero-Disk Overhead):
    Khi gọi slice lô ảnh [start:end], sinh trực tiếp mẫu đối kháng trên GPU VRAM,
    đáp ứng ngay lập tức cho các tác vụ phân tích mà KHÔNG CẦN LƯU VÀO ĐĨA (0 GB đĩa).
    Sau khi xử lý xong lô, bộ nhớ VRAM tự động được giải phóng.
    """

    def __init__(
        self,
        clean_proxy: Any,
        attack_name: str,
        eps_num: int = 16,
        wrapper: Any = None,
        steps: int = 15,
        device: Optional[torch.device] = None,
    ) -> None:
        self.clean_proxy = clean_proxy
        self.attack_name = attack_name.lower()
        self.eps_num = eps_num
        self.eps = float(eps_num) / 255.0
        self.alpha = (2.0 / 255.0) if eps_num >= 8 else (1.0 / 255.0)
        self.steps = steps
        self.wrapper = wrapper
        self.device = device or getattr(clean_proxy, "device", torch.device("cpu"))
        self._attacker = None

    def __len__(self) -> int:
        return len(self.clean_proxy)

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        return getattr(self.clean_proxy, "shape", (len(self.clean_proxy), 3, 640, 640))

    def to(self, device: torch.device) -> "OnTheFlyAdvImageBatchProxy":
        return OnTheFlyAdvImageBatchProxy(
            clean_proxy=self.clean_proxy.to(device) if hasattr(self.clean_proxy, "to") else self.clean_proxy,
            attack_name=self.attack_name,
            eps_num=self.eps_num,
            wrapper=self.wrapper,
            steps=self.steps,
            device=device,
        )

    def _get_attacker(self):
        if self._attacker is None:
            if self.wrapper is None:
                from paper_1_aqsm_mechanism.core.detr_wrapper import RTDETRModelWrapper
                from paper_1_aqsm_mechanism.experiments.config import MODEL_NAME
                self.wrapper = RTDETRModelWrapper(pretrained_model_name=MODEL_NAME, device=self.device)
                self.wrapper.eval()
            from paper_1_aqsm_mechanism.core.attack_generator import DETRAttackGenerator
            self._attacker = DETRAttackGenerator(self.wrapper, epsilon=self.eps, alpha=self.alpha, num_steps=self.steps)
        return self._attacker

    def __getitem__(self, item: Union[int, slice, Sequence[int]]) -> torch.Tensor:
        clean_batch = self.clean_proxy[item]
        single_item = False
        if isinstance(item, (int, np.integer)):
            clean_batch = clean_batch.unsqueeze(0)
            single_item = True

        if hasattr(clean_batch, "to") and clean_batch.device != self.device:
            clean_batch = clean_batch.to(self.device)

        with torch.enable_grad():
            attacker = self._get_attacker()
            if self.attack_name == "aqsm":
                adv_batch = attacker.generate_query_selection_attack(clean_batch, steps=self.steps, alpha=self.alpha)
            elif self.attack_name in ["vanishing", "query_vanishing"]:
                adv_batch = attacker.generate_vanishing_attack(clean_batch, steps=self.steps, alpha=self.alpha)
            elif self.attack_name in ["taskpgd", "task_pgd"]:
                adv_batch = attacker.generate_standard_task_pgd(clean_batch, steps=self.steps, alpha=self.alpha)
            elif self.attack_name in ["fabrication", "query_fabrication"]:
                adv_batch = attacker.generate_fabrication_attack(clean_batch, steps=self.steps, alpha=self.alpha)
            elif self.attack_name in ["encoder_fabrication", "eqf"]:
                adv_batch = attacker.generate_encoder_fabrication_attack(clean_batch, steps=self.steps, alpha=self.alpha)
            elif self.attack_name in ["adaptive", "adaptive_evasion"]:
                adv_batch = attacker.generate_adaptive_attack(clean_batch, steps=self.steps, alpha=self.alpha)
            elif self.attack_name in ["adaptive_esm", "esm_evasion"]:
                adv_batch = attacker.generate_adaptive_esm_attack(clean_batch, steps=self.steps, alpha=self.alpha)
            elif self.attack_name in ["adaptive_fused", "fused_evasion"]:
                adv_batch = attacker.generate_adaptive_fused_attack(clean_batch, steps=self.steps, alpha=self.alpha)
            elif self.attack_name in ["noise", "random_noise"]:
                adv_batch = attacker.generate_random_noise(clean_batch)
            elif self.attack_name in ["blur", "brightness"]:
                adv_batch = attacker.generate_clean_shifts(clean_batch, shift_type=self.attack_name)
            else:
                raise ValueError(f"Đòn tấn công không hỗ trợ trong On-The-Fly: {self.attack_name}")

            adv_batch = adv_batch.detach()

        if single_item:
            adv_batch = adv_batch.squeeze(0)
        return adv_batch


class FlexibleCOCODataset:
    """
    Dataset COCO linh hoạt:
    - Quét toàn bộ ảnh trong thư mục chỉ định.
    - Đọc metadata và annotations từ file instances_val2017.json.
    - Cung cấp dữ liệu Ground Truth và tensor ảnh (hoặc proxy lười) đầy đủ cho pipeline thực nghiệm.
    """

    def __init__(
        self,
        folder_image: Union[str, Path],
        annotations_file: Union[str, Path],
        img_size: int = 640,
        id2label: Optional[Dict[Union[int, str], str]] = None,
        lazy_tensors: bool = True,
        device: Optional[torch.device] = None,
        max_samples: Optional[int] = None,
    ) -> None:
        self.folder_image = Path(folder_image).resolve()
        self.annotations_file = Path(annotations_file).resolve()
        self.img_size = img_size
        self.device = device or torch.device("cpu")
        self.lazy_tensors = lazy_tensors
        self.max_samples = max_samples

        if not self.folder_image.is_dir():
            raise FileNotFoundError(f"Thư mục ảnh không tồn tại: {self.folder_image}")
        if not self.annotations_file.is_file():
            raise FileNotFoundError(f"File annotations không tồn tại: {self.annotations_file}")

        self._load_dataset(id2label=id2label)

    def _load_dataset(self, id2label: Optional[Dict[Union[int, str], str]] = None) -> None:
        """Đọc file annotations và đối chiếu ảnh trong thư mục."""
        print(f"[Dataset Loader] Đang nạp annotations từ: {self.annotations_file.name}...")
        with open(self.annotations_file, "r", encoding="utf-8") as f:
            coco_data = json.load(f)

        self.cat_map = build_coco_category_mapping(coco_data["categories"], id2label=id2label)
        img_meta: Dict[int, Dict[str, Any]] = {int(im["id"]): im for im in coco_data["images"]}
        anns_by_img: Dict[int, List[Dict[str, Any]]] = {}
        for ann in coco_data["annotations"]:
            img_id = int(ann["image_id"])
            anns_by_img.setdefault(img_id, []).append(ann)

        # Quét các file ảnh hợp lệ trong folder
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        all_image_paths = sorted([
            p for p in self.folder_image.iterdir()
            if p.suffix.lower() in valid_extensions
        ])

        if not all_image_paths:
            raise FileNotFoundError(f"Không tìm thấy file ảnh hợp lệ nào trong: {self.folder_image}")

        if self.max_samples is not None and self.max_samples > 0:
            all_image_paths = all_image_paths[:self.max_samples]
            print(f"[Dataset Loader] Giới hạn xử lý {len(all_image_paths)} ảnh (theo --max_samples={self.max_samples}).")
        else:
            print(f"[Dataset Loader] Tìm thấy {len(all_image_paths)} ảnh trong {self.folder_image.name}.")

        self.paths: List[Path] = []
        self.image_ids: List[int] = []
        self.gt_list: List[Dict[str, Any]] = []
        self.hf_labels: List[Dict[str, torch.Tensor]] = []

        skipped_no_meta = 0
        for p in all_image_paths:
            # Xác định image_id từ tên file (chuẩn COCO: 000000123456.jpg -> 123456)
            stem = p.stem
            try:
                img_id = int(stem)
            except ValueError:
                # Nếu tên file không phải số nguyên thuần túy, thử tìm trong img_meta theo file_name
                matched = [iid for iid, im in img_meta.items() if im.get("file_name") == p.name]
                if matched:
                    img_id = matched[0]
                else:
                    skipped_no_meta += 1
                    continue

            if img_id not in img_meta:
                skipped_no_meta += 1
                continue

            meta = img_meta[img_id]
            orig_w = float(meta["width"])
            orig_h = float(meta["height"])

            boxes_xyxy: List[List[float]] = []
            boxes_cxcywh: List[List[float]] = []
            labels: List[int] = []
            iscrowd: List[bool] = []

            for ann in anns_by_img.get(img_id, []):
                cid = int(ann["category_id"])
                if cid not in self.cat_map:
                    continue
                x, y, w, h = [float(v) for v in ann["bbox"]]
                if w <= 1e-3 or h <= 1e-3:
                    continue

                x1 = max(0.0, min(1.0, x / orig_w))
                y1 = max(0.0, min(1.0, y / orig_h))
                x2 = max(x1 + 1e-4, min(1.0, (x + w) / orig_w))
                y2 = max(y1 + 1e-4, min(1.0, (y + h) / orig_h))
                bw = max(1e-4, x2 - x1)
                bh = max(1e-4, y2 - y1)
                cx = x1 + 0.5 * bw
                cy = y1 + 0.5 * bh

                boxes_xyxy.append([x1, y1, x2, y2])
                boxes_cxcywh.append([cx, cy, bw, bh])
                labels.append(self.cat_map[cid])
                iscrowd.append(bool(ann.get("iscrowd", 0) == 1))

            arr_xyxy = np.array(boxes_xyxy, dtype=np.float32) if boxes_xyxy else np.zeros((0, 4), dtype=np.float32)
            arr_cxcywh = np.array(boxes_cxcywh, dtype=np.float32) if boxes_cxcywh else np.zeros((0, 4), dtype=np.float32)
            arr_lbl = np.array(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64)
            arr_crowd = np.array(iscrowd, dtype=bool) if iscrowd else np.zeros((0,), dtype=bool)

            self.paths.append(p)
            self.image_ids.append(img_id)
            self.gt_list.append({
                "image_id": img_id,
                "file_name": p.name,
                "orig_size": (int(orig_w), int(orig_h)),
                "boxes_xyxy": arr_xyxy,
                "boxes_cxcywh": arr_cxcywh,
                "labels": arr_lbl,
                "iscrowd": arr_crowd,
            })

            non_crowd_mask = ~arr_crowd
            self.hf_labels.append({
                "class_labels": torch.from_numpy(arr_lbl[non_crowd_mask].copy()),
                "boxes": torch.from_numpy(arr_cxcywh[non_crowd_mask].copy()),
            })

        if skipped_no_meta > 0:
            print(f"[Dataset Loader] Cảnh báo: Bỏ qua {skipped_no_meta} ảnh không tìm thấy metadata trong annotations.")

        print(f"[Dataset Loader] Đã tải thành công {len(self.paths)} ảnh hợp lệ kèm Ground Truth.")

        # Khởi tạo proxy lười hoặc tensor
        if self.lazy_tensors:
            self.tensors = LazyImageBatchProxy(self.paths, self.img_size, device=self.device)
        else:
            print(f"[Dataset Loader] Đang nạp toàn bộ {len(self.paths)} ảnh vào RAM (Eager mode)...")
            tensor_list = []
            for p in self.paths:
                with Image.open(p) as img:
                    rgb = img.convert("RGB").resize((self.img_size, self.img_size))
                    tensor_list.append(TF.to_tensor(rgb))
            self.tensors = torch.stack(tensor_list, dim=0).to(self.device)

    def __len__(self) -> int:
        return len(self.paths)

    def to_cohort_dict(self) -> Dict[str, Any]:
        """Trả về cấu trúc dict tương thích 100% với hàm `load_coco_val2017_gt_cohort`."""
        return {
            "tensors": self.tensors,
            "paths": self.paths,
            "image_ids": self.image_ids,
            "gt_list": self.gt_list,
            "hf_labels": self.hf_labels,
            "cat_id_to_idx": self.cat_map,
        }
