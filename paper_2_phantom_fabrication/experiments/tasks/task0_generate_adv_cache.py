# -*- coding: utf-8 -*-
"""
Module: paper_2_phantom_fabrication.experiments.tasks.task0_generate_adv_cache
Mục đích: CLI Task 0 - Tiền sinh toàn bộ bộ nhớ đệm đối kháng (Adversarial Cache Pre-generation)
chuẩn IEEE 754 float32 nguyên bản cho các đòn tấn công: Untargeted_PGD, EQF, DQF.
"""

import gc
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
from pathlib import Path

WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from paper_2_phantom_fabrication.core.coco_gt import (
    discover_coco_annotations,
    load_flexible_coco_cohort,
)
from paper_2_phantom_fabrication.core.config import set_adv_cache_dir
from paper_2_phantom_fabrication.experiments.engines.cache_engine import generate_single_attack_cache
from paper_2_phantom_fabrication.experiments.streaming_evaluator import get_cohort_dataloader
from paper_2_phantom_fabrication.utils.cli import create_base_parser
from paper_2_phantom_fabrication.utils.device import (
    clear_gpu_memory,
    detect_hardware,
    setup_cpu_threads,
)


def main():
    parser = create_base_parser("Pre-generate Adversarial Cache (float32)", task_name="Task 0")
    parser.add_argument("--steps", type=int, default=8, help="Số bước lặp PGD sinh mẫu")
    parser.add_argument("--attacks", type=str, default="Untargeted_PGD,EQF,DQF", help="Danh sách attacks (phân tách bởi dấu phẩy)")
    parser.add_argument("--include_sweep", action="store_true", default=False, help="Sinh thêm cache eps 4, 8, 12 cho ablation sweep")
    args = parser.parse_args()

    import torch
    from transformers import RTDetrForObjectDetection

    set_adv_cache_dir(args.cache_dir)
    setup_cpu_threads(args.threads)
    hw_info = detect_hardware(args.device)
    device = torch.device(hw_info["primary_device"])
    batch_size = args.batch_size if args.batch_size is not None and args.batch_size > 0 else hw_info["recommended_batch_size"]

    print("=" * 80)
    print(" [TASK 0] KHỞI TẠO TIẾN TRÌNH SINH ADVERSARIAL CACHE (ZERO-OOM MEMMAP)")
    print(f" Thư mục ảnh : {args.folder_image}")
    print(f" Thiết bị    : {device} ({hw_info.get('num_gpus', 0)} GPUs)")
    print(f" Batch Size  : {batch_size} | PGD Steps: {args.steps}")
    print("=" * 80)

    img_folder = Path(args.folder_image).resolve()
    ann_file = discover_coco_annotations(img_folder, args.annotations)

    cohort = load_flexible_coco_cohort(
        img_folder, ann_file, n_samples=args.num_images, seed=42, img_size=640, preload_images=False
    )
    dataloader = get_cohort_dataloader(cohort, batch_size=batch_size, device=device)
    print(f" Đã sẵn sàng nạp: {len(cohort['dataset'])} ảnh COCO")

    raw_model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r18vd")
    raw_model.eval()
    raw_model.to(device)

    if hw_info.get("num_gpus", 0) > 1 and device.type == "cuda":
        model = torch.nn.DataParallel(raw_model)
    else:
        model = raw_model

    attacks_to_gen = [a.strip() for a in args.attacks.split(",") if a.strip()]
    for atk in attacks_to_gen:
        generate_single_attack_cache(
            model=model,
            dataloader=dataloader,
            cohort=cohort,
            attack_name=atk,
            eps_int=16,
            steps=args.steps,
            target_class_idx=1,
            cache_dir=args.cache_dir,
            device=device,
            force_recompute=args.force_recompute,
        )

    if args.include_sweep:
        for eps_val in [4, 8, 12]:
            for atk in ["EQF", "DQF"]:
                generate_single_attack_cache(
                    model=model,
                    dataloader=dataloader,
                    cohort=cohort,
                    attack_name=atk,
                    eps_int=eps_val,
                    steps=args.steps,
                    target_class_idx=1,
                    cache_dir=args.cache_dir,
                    device=device,
                    force_recompute=args.force_recompute,
                )

    del raw_model, model, dataloader, cohort
    clear_gpu_memory()
    print("\n[HOÀN THÀNH TOÀN BỘ] Task 0: Pre-generate Adversarial Cache thành công!")


if __name__ == "__main__":
    main()
