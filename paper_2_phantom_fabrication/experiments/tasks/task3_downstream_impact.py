# -*- coding: utf-8 -*-
"""
Module: paper_2_phantom_fabrication.experiments.tasks.task3_downstream_impact
Mục đích: CLI Task 3 - Đánh giá chuyển giao liên lớp và tác động Cyber-Physical Downstream (Table 3):
  1. Cross-Class Generalization (person, bicycle, car, motorcycle, bus).
  2. Size-Stratified Breakdown (Small, Medium, Large object eviction).
  3. Real ByteTrack Hungarian Assignment Simulation (Bipartite Matching latency).
  4. AEB 3D Pinhole Hazard Projection (False braking risk at 28m TTC).
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
from paper_2_phantom_fabrication.experiments.engines.downstream_engine import run_stage_2_downstream_eval
from paper_2_phantom_fabrication.utils.cli import create_base_parser
from paper_2_phantom_fabrication.utils.device import (
    clear_gpu_memory,
    detect_hardware,
    setup_cpu_threads,
)
from paper_2_phantom_fabrication.utils.io_helpers import save_json


def main():
    parser = create_base_parser("Downstream & Cross-Class Impact (Table 3)", task_name="Task 3")
    args = parser.parse_args()

    set_adv_cache_dir(args.cache_dir)
    setup_cpu_threads(args.threads)
    hw_info = detect_hardware(args.device)
    device = torch.device(hw_info["primary_device"])
    batch_size = args.batch_size if args.batch_size is not None and args.batch_size > 0 else hw_info["recommended_batch_size"]

    print("=" * 80)
    print(" [TASK 3] KHỞI CHẠY ĐÁNH GIÁ DOWNSTREAM VÀ CROSS-CLASS GENERALIZATION")
    print(f" Thư mục ảnh : {args.folder_image}")
    print(f" Số mẫu      : {args.num_images if args.num_images > 0 else 'Toàn bộ tập'}")
    print(f" Thiết bị    : {device} | Batch Size: {batch_size}")
    print("=" * 80)

    results = run_stage_2_downstream_eval(
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
    out_json = out_dir / "stage2_downstream_telemetry.json"
    save_json(results, out_json)
    print(f"\n[HOÀN TẤT TASK 3] Đã lưu kết quả đo đạc tại: {out_json}")

    clear_gpu_memory()


if __name__ == "__main__":
    main()
