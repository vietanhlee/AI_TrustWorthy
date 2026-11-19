# -*- coding: utf-8 -*-
"""
Module: paper_2_phantom_fabrication.experiments.tasks.task1_multiseed_benchmark
Mục đích: CLI Task 1 - Đo đạc độ lệch chuẩn đa hạt giống (Multi-Seed Benchmark) trên 3 seeds (42, 123, 456).
Đánh giá RT-DETR vs YOLOv8n trên Clean, Untargeted PGD, EQF, DQF phục vụ Table 1 & Table 4 của bài báo.
"""

import gc
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
from pathlib import Path

import torch

WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from paper_2_phantom_fabrication.core.config import set_adv_cache_dir
from paper_2_phantom_fabrication.experiments.engines.benchmark_engine import run_stage_1_benchmark
from paper_2_phantom_fabrication.utils.cli import create_base_parser
from paper_2_phantom_fabrication.utils.device import (
    clear_gpu_memory,
    detect_hardware,
    setup_cpu_threads,
)
from paper_2_phantom_fabrication.utils.io_helpers import save_json


def main():
    parser = create_base_parser("Multi-Seed Benchmark (Stage 1)", task_name="Task 1")
    args = parser.parse_args()

    set_adv_cache_dir(args.cache_dir)
    setup_cpu_threads(args.threads)
    hw_info = detect_hardware(args.device)
    device = torch.device(hw_info["primary_device"])
    batch_size = args.batch_size if args.batch_size is not None and args.batch_size > 0 else hw_info["recommended_batch_size"]

    seed_list = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    print("=" * 80)
    print(" [TASK 1] KHỞI CHẠY MULTI-SEED BENCHMARK (RT-DETR vs YOLOv8n)")
    print(f" Thư mục ảnh : {args.folder_image}")
    print(f" Hạt giống   : {seed_list} (N={args.num_images}/seed)")
    print(f" Thiết bị    : {device} | Batch Size: {batch_size}")
    print("=" * 80)

    results = run_stage_1_benchmark(
        image_folder=args.folder_image,
        annotations_path=args.annotations,
        seeds=seed_list,
        n_samples_per_seed=args.num_images,
        device=device,
        hw_info=hw_info,
        batch_size=batch_size,
        cache_dir=args.cache_dir,
        skip_cache_gen=args.skip_cache_gen,
        force_recompute=args.force_recompute,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "stage1_benchmark_telemetry.json"
    save_json(results, out_json)
    print(f"\n[HOÀN TẤT TASK 1] Đã lưu kết quả đo đạc tại: {out_json}")

    clear_gpu_memory()


if __name__ == "__main__":
    main()
