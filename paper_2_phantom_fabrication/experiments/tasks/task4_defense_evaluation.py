# -*- coding: utf-8 -*-
"""
Module: paper_2_phantom_fabrication.experiments.tasks.task4_defense_evaluation
Mục đích: CLI Task 4 - Đánh giá chuyên sâu giải pháp phòng thủ SQD-Filter và đối chứng baseline (Table 4):
  1. SQD-Filter (Ours)
  2. Baseline Defenses (JPEG Compression, Gaussian Blur, Median Filter, Bit-Depth Reduction)
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
from paper_2_phantom_fabrication.utils.cli import create_base_parser
from paper_2_phantom_fabrication.utils.device import (
    clear_gpu_memory,
    detect_hardware,
    setup_cpu_threads,
)
from paper_2_phantom_fabrication.utils.io_helpers import save_json


def main():
    parser = create_base_parser("Defense Evaluation (Table 4)", task_name="Task 4")
    args = parser.parse_args()

    from paper_2_phantom_fabrication.experiments.engines.defense_engine import run_comprehensive_defense_evaluation

    set_adv_cache_dir(args.cache_dir)
    setup_cpu_threads(args.threads)
    hw_info = detect_hardware(args.device)
    device = torch.device(hw_info["primary_device"])
    batch_size = args.batch_size if args.batch_size is not None and args.batch_size > 0 else hw_info["recommended_batch_size"]

    print("=" * 80)
    print(" [TASK 4] KHỞI CHẠY ĐÁNH GIÁ CÁC GIẢI PHÁP PHÒNG THỦ (SQD-FILTER vs BASELINES)")
    print(f" Thư mục ảnh : {args.folder_image}")
    print(f" Số mẫu      : {args.num_images if args.num_images > 0 else 'Toàn bộ tập'}")
    print(f" Thiết bị    : {device} | Batch Size: {batch_size}")
    print("=" * 80)

    results = run_comprehensive_defense_evaluation(
        image_folder=args.folder_image,
        annotations_path=args.annotations,
        n_samples=args.num_images,
        device=device,
        hw_info=hw_info,
        batch_size=batch_size,
        cache_dir=args.cache_dir,
        skip_cache_gen=args.skip_cache_gen,
        force_recompute=args.force_recompute,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "stage1_defense_telemetry.json"
    save_json(results, out_json)
    print(f"\n[HOÀN TẤT TASK 4] Đã lưu kết quả đo đạc tại: {out_json}")

    clear_gpu_memory()


if __name__ == "__main__":
    main()
