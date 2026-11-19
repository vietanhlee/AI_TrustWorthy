# -*- coding: utf-8 -*-
"""
Module: paper_2_phantom_fabrication.experiments.generate_adv_cache
Mục đích: Tiền xử lý và sinh toàn bộ bộ nhớ đệm đối kháng (Adversarial Cache Pre-generation)
cho toàn bộ tập dữ liệu (ví dụ toàn bộ 5,000 ảnh COCO val2017) theo đúng chuẩn Paper 1.

Đặc tính kỹ thuật (Production Standards):
  1. Chạy 1 lượt duy nhất qua 5,000 ảnh để tính và khóa cứng toàn bộ perturbation deltas.
     Không bao giờ bị "miss" ảnh vì chỉ mục image_ids được khớp chính xác 1-1 với COCO.
  2. Tuyệt đối KHÔNG lưu uint8 hay nén quantization làm mất mát độ chính xác gradient đối kháng.
     Lưu giữ chuẩn xác 100% theo định dạng IEEE 754 float32 nguyên bản.
  3. Sử dụng disk-backed np.memmap tạm thời trên đĩa trong quá trình sinh, đảm bảo mức RAM
     luôn phẳng dưới 1.5 GB ngay cả khi xử lý toàn bộ 5,000 ảnh kích thước 640x640.
  4. Hỗ trợ tự động nhận diện Multi-GPU DataParallelism để tăng tốc độ sinh lên tối đa.
  5. Sau khi sinh xong, tất cả các bài thử nghiệm Multi-Seed (Seed 42, 123, 456) và Ablations
     chỉ cần đọc trực tiếp từ cache qua mmap_mode="r", hoàn toàn không phải chạy lại PGD.
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import RTDetrForObjectDetection

# Đảm bảo import được các module từ project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from paper_2_phantom_fabrication.core.adv_cache import save_adv_cache
from paper_2_phantom_fabrication.core.coco_gt import (
    CocoCohortDataset,
    discover_coco_annotations,
    load_flexible_coco_cohort,
)
from paper_2_phantom_fabrication.core.config import (
    ADV_CACHE_DIR,
    get_adv_cache_dir,
    set_adv_cache_dir,
)
from paper_2_phantom_fabrication.core.hardware import detect_hardware
from paper_2_phantom_fabrication.attacks.phantom_attacks import generate_batch_adversarial
from paper_2_phantom_fabrication.experiments.streaming_evaluator import get_cohort_dataloader


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
    và lưu thành file .npz chuẩn float32 nguyên bản.
    """
    cdir = Path(cache_dir) if cache_dir is not None else get_adv_cache_dir()
    cdir.mkdir(parents=True, exist_ok=True)

    n_total = len(cohort["dataset"])
    image_ids = cohort["image_ids"]
    eps = float(eps_int) / 255.0
    alpha = 2.0 / 255.0 if eps_int >= 8 else 1.0 / 255.0

    target_file = cdir / f"adv_cache_{attack_name.lower()}_eps{eps_int}_n{n_total}.npz"

    # Kiểm tra xem cache đã tồn tại và đủ số lượng ảnh chưa
    if not force_recompute and target_file.exists():
        try:
            with np.load(target_file, mmap_mode="r") as data:
                if "image_ids" in data and len(data["image_ids"]) == n_total:
                    if list(data["image_ids"]) == list(image_ids):
                        print(f"[CACHE GEN] Đã tìm thấy cache hoàn chỉnh tại: {target_file.name}. Bỏ qua bước sinh.")
                        return target_file
        except Exception:
            print(f"[CACHE GEN] Cache hiện tại bị lỗi đọc, tiến hành sinh lại: {target_file.name}")

    print(f"\n[CACHE GEN] ==================================================================")
    print(f"[CACHE GEN] Đang sinh: {attack_name.upper()} | eps={eps_int}/255 ({eps:.4f}) | N={n_total} ảnh")
    print(f"[CACHE GEN] PGD Steps={steps} | Alpha={alpha:.5f} | Device={device}")
    print(f"[CACHE GEN] ==================================================================")

    # Tạo disk-backed memmap tạm thời để giữ mức RAM luôn < 1.5 GB
    temp_mmap_path = cdir / f"tmp_{attack_name.lower()}_eps{eps_int}_{os.getpid()}_{int(time.time())}.dat"
    delta_mmap = np.memmap(
        temp_mmap_path,
        dtype=np.float32,
        mode="w+",
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
    elapsed = time.time() - t_start

    meta_info = {
        "attack": attack_name.lower(),
        "n_total": n_total,
        "epsilon_num_255": eps_int,
        "epsilon": eps,
        "alpha": alpha,
        "steps": steps,
        "resolution": "640x640",
        "dtype": "float32",
        "elapsed_sec": round(elapsed, 2),
        "throughput_fps": round(n_total / max(elapsed, 1e-4), 2),
        "device": str(device),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    print(f"[CACHE GEN] Đang lưu file cache chuẩn float32 vào: {target_file.name}...")
    save_adv_cache(
        attack=attack_name,
        eps_num=eps_int,
        image_ids=image_ids,
        delta_array=delta_mmap,
        meta_info=meta_info,
        cache_dir=cdir,
    )

    # Dọn dẹp memmap tạm
    del delta_mmap
    gc.collect()
    if temp_mmap_path.exists():
        try:
            temp_mmap_path.unlink()
        except Exception:
            pass

    print(f"[CACHE GEN] Hoàn thành sinh {attack_name.upper()} trong {elapsed:.1f}s ({n_total / max(elapsed, 1e-4):.2f} FPS).")
    return target_file


def main():
    parser = argparse.ArgumentParser(description="Adversarial Cache Generator for Paper 2 (Native 640x640, float32)")
    parser.add_argument("--folder_image", type=str, required=True, help="Đường dẫn thư mục ảnh COCO (e.g. val2017)")
    parser.add_argument("--annotations", type=str, default=None, help="Đường dẫn file annotations instances_val2017.json")
    parser.add_argument("--num_images", type=int, default=0, help="Số ảnh cần sinh cache (0 = toàn bộ ảnh trong thư mục)")
    parser.add_argument("--cache_dir", type=str, default=str(ADV_CACHE_DIR), help="Thư mục lưu trữ cache .npz")
    parser.add_argument("--device", type=str, default="auto", help="Thiết bị: auto, cuda, cpu")
    parser.add_argument("--batch_size", type=int, default=0, help="Batch size (0 = tự động tính theo VRAM)")
    parser.add_argument("--force_recompute", action="store_true", default=False, help="Bắt buộc sinh lại kể cả khi đã có cache")
    parser.add_argument("--include_sweep", action="store_true", default=False, help="Sinh thêm cache eps 4, 8, 12 cho ablation sweep")

    args = parser.parse_args()
    set_adv_cache_dir(args.cache_dir)

    print("======================================================================")
    print("  KHỞI ĐỘNG TIỀN XỬ LÝ VÀ SINH BỘ NHỚ ĐỆM ĐỐI KHÁNG (PAPER 2)")
    print("======================================================================")

    # 1. Hardware Detection
    hw_info = detect_hardware(args.device)
    device = torch.device(hw_info["primary_device"])

    num_gpus = max(1, hw_info.get("num_gpus", 1))
    if args.batch_size <= 0:
        if device.type == "cuda":
            single_gpu_vram = hw_info["vram_gb"][0] if hw_info["vram_gb"] else 8.0
            base_batch = 16 if single_gpu_vram >= 15.0 else (8 if single_gpu_vram >= 7.5 else 4)
            batch_size = base_batch * num_gpus
        else:
            batch_size = 4
    else:
        batch_size = args.batch_size

    print(f" Thiết bị thực thi:  {device} ({num_gpus} GPUs)")
    print(f" Cấu hình Batch Size:{batch_size}")
    print(f" Thư mục Cache:      {Path(args.cache_dir).resolve()}")

    img_folder = Path(args.folder_image).resolve()
    ann_file = discover_coco_annotations(img_folder, args.annotations)

    # 2. Nạp Dataset
    cohort = load_flexible_coco_cohort(
        img_folder, ann_file, n_samples=args.num_images, seed=42, img_size=640, preload_images=False
    )
    dataloader = get_cohort_dataloader(cohort, batch_size=batch_size, device=device)
    print(f" Đã sẵn sàng nạp:    {len(cohort['dataset'])} ảnh COCO chuẩn 640x640")

    # 3. Nạp Model RT-DETR
    raw_model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r18vd")
    raw_model.eval()
    raw_model.to(device)

    if hw_info.get("num_gpus", 0) > 1 and device.type == "cuda":
        model = torch.nn.DataParallel(raw_model)
        print(f" Kích hoạt Multi-GPU DataParallel trên {hw_info['num_gpus']} GPUs: {hw_info['gpu_names']}")
    else:
        model = raw_model

    # 4. Sinh cache cho các đòn tấn công chính (eps = 16/255)
    attacks_to_generate = ["Untargeted_PGD", "EQF", "DQF"]
    for attack in attacks_to_generate:
        generate_single_attack_cache(
            model=model,
            dataloader=dataloader,
            cohort=cohort,
            attack_name=attack,
            eps_int=16,
            steps=8,
            target_class_idx=1,
            cache_dir=args.cache_dir,
            device=device,
            force_recompute=args.force_recompute,
        )

    # 5. Nếu có cờ include_sweep: sinh thêm eps 4, 8, 12 cho ablation
    if args.include_sweep:
        for eps_val in [4, 8, 12]:
            for attack in ["EQF", "DQF"]:
                generate_single_attack_cache(
                    model=model,
                    dataloader=dataloader,
                    cohort=cohort,
                    attack_name=attack,
                    eps_int=eps_val,
                    steps=8,
                    target_class_idx=1,
                    cache_dir=args.cache_dir,
                    device=device,
                    force_recompute=args.force_recompute,
                )

    print("\n======================================================================")
    print(" TOÀN BỘ BỘ NHỚ ĐỆM ĐỐI KHÁNG ĐÃ ĐƯỢC KHÓA CỨNG TRÊN ĐĨA!")
    print(f" Thư mục: {Path(args.cache_dir).resolve()}")
    print(" Giờ đây bạn có thể chạy `run_all.sh` với tốc độ tức thì (100% Cache Hit).")
    print("======================================================================")


if __name__ == "__main__":
    main()
