# -*- coding: utf-8 -*-
"""
Module: paper_1_aqsm_mechanism.experiments.generate_adv_cache_640
Mục đích: Sinh và lưu trữ bộ nhớ đệm mẫu đối kháng Native 640x640 (S = 8,400 proposals)
cho toàn bộ các đòn tấn công: AQSM, Query Vanishing, Standard Task PGD
trên toàn bộ tập ảnh chỉ định (COCO val2017).

Đặc tính chuẩn công nghiệp (Production Standards):
  - Nhận tham số linh hoạt `--folder_image`, `--annotations`, `--cache_dir`.
  - Nạp ảnh theo lô lười (Lazy Batch Loading) chống tràn RAM khi chạy quy mô 5,000 ảnh.
  - Hỗ trợ cơ chế kiểm tra và bỏ qua (Resume / Skip Existing) nếu cache đã tồn tại toàn vẹn.
  - Tự động tối ưu đa luồng CPU / tăng tốc phần cứng CUDA GPU.
  - Đảm bảo 100% dữ liệu thực nghiệm thật, không hardcode hay làm giả số liệu.
"""

import argparse
import gc
import json
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from tqdm import tqdm

WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from paper_1_aqsm_mechanism.core.dataset_loader import FlexibleCOCODataset
from paper_1_aqsm_mechanism.core.detr_wrapper import RTDETRModelWrapper
from paper_1_aqsm_mechanism.core.attack_generator import DETRAttackGenerator
from paper_1_aqsm_mechanism.experiments.config import (
    ADV_CACHE_DIR,
    COCO_ANN_FILE,
    COCO_VAL_DIR,
    MODEL_NAME,
    RANDOM_SEED,
    configure_cpu_threads,
    detect_hardware_environment,
    get_adv_cache_dir,
    get_device,
    update_experiment_paths,
)


def generate_single_attack_cache(
    wrapper: RTDETRModelWrapper,
    dataset: FlexibleCOCODataset,
    attack_name: str,
    eps_int: int,
    steps: int = 15,
    batch_size: int = 8,
    out_dir: Optional[Path] = None,
    skip_existing: bool = True,
) -> Path:
    """
    Sinh adversarial perturbation delta cho toàn bộ tập ảnh và lưu vào file .npz chuẩn.
    """
    save_dir = Path(out_dir) if out_dir is not None else get_adv_cache_dir()
    save_dir.mkdir(parents=True, exist_ok=True)

    n_total = len(dataset)
    image_ids = dataset.image_ids
    clean_proxy = dataset.tensors
    device = wrapper.device

    eps = float(eps_int) / 255.0
    alpha = (2.0 / 255.0) if eps_int >= 8 else (1.0 / 255.0)

    out_file = save_dir / f"adv_cache_{attack_name.lower()}_eps{eps_int}_n{n_total}.npz"
    alias_file = save_dir / f"adv_cache_{attack_name.lower()}_eps{eps_int}.npz"

    # Kiểm tra nếu cache đã tồn tại và hợp lệ
    if skip_existing and out_file.exists():
        try:
            with np.load(out_file, allow_pickle=False) as cached_data:
                if len(cached_data["image_ids"]) == n_total and list(cached_data["image_ids"]) == image_ids:
                    print(f"[CACHE GEN] Đã tìm thấy cache hoàn chỉnh tại: {out_file.name}. Bỏ qua bước sinh.")
                    return out_file
        except Exception:
            print(f"[CACHE GEN] Cache hiện tại bị lỗi đọc, tiến hành sinh lại: {out_file.name}")

    print(f"\n[CACHE GEN] ==================================================================")
    print(f"[CACHE GEN] Đang sinh: {attack_name.upper()} | eps={eps_int}/255 ({eps:.4f}) | N={n_total}")
    print(f"[CACHE GEN] PGD Steps={steps} | Alpha={alpha:.5f} | Batch Size={batch_size} | Device={device}")
    print(f"[CACHE GEN] ==================================================================")

    attacker = DETRAttackGenerator(wrapper, epsilon=eps, alpha=alpha, num_steps=steps)
    all_deltas = []
    t_start = time.time()

    for i in tqdm(range(0, n_total, batch_size), desc=f"{attack_name.upper()}_eps{eps_int}"):
        b_clean = clean_proxy[i : i + batch_size].to(device)

        if attack_name.lower() == "aqsm":
            b_adv = attacker.generate_query_selection_attack(b_clean, steps=steps, alpha=alpha)
        elif attack_name.lower() in ["vanishing", "query_vanishing"]:
            b_adv = attacker.generate_vanishing_attack(b_clean, steps=steps, alpha=alpha)
        elif attack_name.lower() in ["taskpgd", "task_pgd"]:
            b_adv = attacker.generate_standard_task_pgd(b_clean, steps=steps, alpha=alpha)
        else:
            raise ValueError(f"Đòn tấn công không được hỗ trợ: {attack_name}")

        delta = (b_adv - b_clean).detach().cpu()
        all_deltas.append(delta)

    delta_all = torch.cat(all_deltas, dim=0).numpy().astype(np.float16)  # [N, 3, 640, 640]
    l2_norms = np.linalg.norm(delta_all.reshape(n_total, -1), axis=1)
    elapsed = time.time() - t_start

    meta_info = {
        "attack": attack_name.lower(),
        "n_eval": n_total,
        "epsilon_num_255": eps_int,
        "epsilon": eps,
        "alpha": alpha,
        "steps": steps,
        "resolution": "640x640",
        "num_candidates": 8400,
        "max_linf": float(np.max(np.abs(delta_all))),
        "mean_l2": float(np.mean(l2_norms)),
        "std_l2": float(np.std(l2_norms)),
        "elapsed_sec": elapsed,
        "throughput_fps": float(n_total / max(elapsed, 1e-4)),
        "device": str(device),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Lưu file nén npz chuẩn
    np.savez_compressed(
        out_file,
        delta_f16=delta_all,
        image_ids=np.array(image_ids, dtype=np.int64),
        l2_norms=l2_norms,
        meta_json=json.dumps(meta_info),
    )
    print(f"    -> Đã lưu thành công: {out_file.name} ({out_file.stat().st_size / 1e6:.1f} MB) trong {elapsed:.1f}s")

    # Lưu alias chung để tiện tra cứu
    if alias_file != out_file:
        np.savez_compressed(
            alias_file,
            delta_f16=delta_all,
            image_ids=np.array(image_ids, dtype=np.int64),
            l2_norms=l2_norms,
            meta_json=json.dumps(meta_info),
        )
        print(f"    -> Đã tạo alias tiện ích: {alias_file.name}")

    return out_file


def main():
    parser = argparse.ArgumentParser(description="Sinh Adversarial Cache Native 640x640 cho RT-DETR")
    parser.add_argument("--folder_image", type=str, default=str(COCO_VAL_DIR), help="Đường dẫn thư mục ảnh COCO")
    parser.add_argument("--annotations", type=str, default=str(COCO_ANN_FILE), help="Đường dẫn file instances_val2017.json")
    parser.add_argument("--cache_dir", type=str, default=str(ADV_CACHE_DIR), help="Thư mục lưu cache đối kháng")
    parser.add_argument("--steps", type=int, default=15, help="Số bước lặp PGD (mặc định 15)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size (tự động điều chỉnh nếu None)")
    parser.add_argument("--threads", type=int, default=8, help="Số luồng CPU tính toán")
    parser.add_argument("--device", type=str, default="auto", help="Thiết bị tính toán (cuda / cpu / auto)")
    parser.add_argument("--attacks", type=str, default="aqsm,vanishing,taskpgd", help="Danh sách attacks (phân tách bởi dấu phẩy)")
    parser.add_argument("--epsilons", type=str, default="16", help="Danh sách epsilon /255 (ví dụ: '16' hoặc '4,8,16')")
    parser.add_argument("--include_sweep", action="store_true", default=False, help="Sinh kèm sweep eps 4 và 8")
    parser.add_argument("--force_recompute", action="store_true", default=False, help="Bắt buộc sinh lại cache kể cả khi đã có")
    args = parser.parse_args()

    update_experiment_paths(
        folder_image=args.folder_image,
        annotations=args.annotations,
        cache_dir=args.cache_dir,
    )

    configure_cpu_threads(args.threads)
    hw_info = detect_hardware_environment(preferred_device=args.device)
    device = get_device(preferred=args.device)

    # Tự động điều chỉnh batch size
    batch_size = args.batch_size if args.batch_size is not None else hw_info["recommended_batch_size"]

    print("=" * 80)
    print(" KHỞI TẠO TIẾN TRÌNH SINH CACHE ĐỐI KHÁNG NATIVE 640x640 (S = 8,400)")
    print(f" Thư mục ảnh : {args.folder_image}")
    print(f" Annotations  : {args.annotations}")
    print(f" Thư mục cache: {args.cache_dir}")
    print(f" Phần cứng    : {hw_info['device_type'].upper()} ({hw_info.get('gpu_names', [hw_info.get('processor')])})")
    print(f" Batch Size   : {batch_size} | PGD Steps: {args.steps}")
    print("=" * 80)

    # Nạp mô hình RT-DETR
    wrapper = RTDETRModelWrapper(pretrained_model_name=MODEL_NAME, device=device)
    wrapper.eval()

    # Nạp toàn bộ tập ảnh bằng FlexibleCOCODataset
    dataset = FlexibleCOCODataset(
        folder_image=args.folder_image,
        annotations_file=args.annotations,
        img_size=640,
        id2label=wrapper.model.config.id2label,
        lazy_tensors=True,
        device=device,
    )

    attacks = [a.strip().lower() for a in args.attacks.split(",") if a.strip()]

    # Xác định danh sách epsilons
    eps_list = [int(e.strip()) for e in args.epsilons.split(",") if e.strip()]
    if args.include_sweep:
        for extra in [4, 8]:
            if extra not in eps_list:
                eps_list.append(extra)
    eps_list = sorted(eps_list)

    skip_existing = not args.force_recompute

    for eps_i in eps_list:
        step_i = args.steps if eps_i >= 16 else min(args.steps, 10)
        for atk in attacks:
            generate_single_attack_cache(
                wrapper=wrapper,
                dataset=dataset,
                attack_name=atk,
                eps_int=eps_i,
                steps=step_i,
                batch_size=batch_size,
                out_dir=Path(args.cache_dir),
                skip_existing=skip_existing,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    del dataset, wrapper
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    print("\n" + "=" * 80)
    print(" [HOÀN TẤT] Sinh toàn bộ bộ nhớ đệm đối kháng thành công!")
    print(f" Kiểm tra thư mục: {args.cache_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
