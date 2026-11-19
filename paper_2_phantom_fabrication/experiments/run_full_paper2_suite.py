# -*- coding: utf-8 -*-
"""
Enterprise Full-Suite Runner for Paper 2: Phantom Queries Fabrication
====================================================================
Điều phối toàn diện thực nghiệm Paper 2 (Elsevier) theo Kiến trúc phân tầng (Clean Architecture):
  - Hardware Profiler & Multi-GPU Orchestration (`core.hardware`)
  - Adversarial Perturbation Generators (`attacks.phantom_attacks`)
  - Real-time Adversarial Caching & Streaming Evaluator (`experiments.streaming_evaluator`)
  - Stage 1: Multi-Seed Benchmark & 5 Defenses (`experiments.stage1_benchmark`)
  - Stage 2: Sensitivity Sweeps & Cyber-Physical Downstream Impact (`experiments.stage2_ablations`)
  - 100% authentic, scientifically verifiable empirical execution (Zero fake / Zero random).
"""

import argparse
import datetime
import json
import logging
from pathlib import Path
import sys
import time

import torch

# Thiết lập đường dẫn import project
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

# Import các thành phần kiến trúc đã được tách lớp
from paper_2_phantom_fabrication.core.config import (
    ADV_CACHE_DIR,
    set_adv_cache_dir,
)
from paper_2_phantom_fabrication.core.coco_gt import discover_coco_annotations
from paper_2_phantom_fabrication.core.hardware import (
    benchmark_model_latency,
    detect_hardware,
)
from paper_2_phantom_fabrication.core.matcher import match_detections_against_gt
from paper_2_phantom_fabrication.attacks.phantom_attacks import generate_batch_adversarial
from paper_2_phantom_fabrication.experiments.streaming_evaluator import (
    get_cohort_dataloader,
    run_streaming_condition_eval,
)
from paper_2_phantom_fabrication.experiments.stage1_benchmark import run_stage_1_benchmark
from paper_2_phantom_fabrication.experiments.stage2_ablations import run_stage_2_ablations

# Thiết lập Logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("FullSuiteRunner")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


# ==============================================================================
# MAIN CLI ORCHESTRATOR
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Full Empirical Suite Runner for Paper 2 (Elsevier)")
    parser.add_argument("--folder_image", type=str, required=True, help="Đường dẫn thư mục ảnh COCO (e.g., val2017)")
    parser.add_argument("--annotations", type=str, default=None, help="Đường dẫn file annotations instances_val2017.json")
    parser.add_argument("--num_images_benchmark", type=int, default=0, help="Số ảnh mỗi seed cho Stage 1 (0 = chạy toàn bộ ảnh trong folder)")
    parser.add_argument("--num_images_ablation", type=int, default=0, help="Số ảnh cho Stage 2 Ablations (0 = chạy toàn bộ ảnh trong folder)")
    parser.add_argument("--seeds", type=str, default="42,123,456", help="Danh sách random seeds cho Stage 1 (cách nhau bởi dấu phẩy)")
    parser.add_argument("--device", type=str, default="auto", help="Thiết bị thực thi: auto, cuda, cpu")
    parser.add_argument("--batch_size", type=int, default=0, help="Batch size (0 = tự động tính theo thiết bị)")
    parser.add_argument("--output_json", type=str, default="results/paper2_complete_telemetry.json", help="Đường dẫn lưu file JSON telemetry")
    parser.add_argument("--stage", type=str, default="all", choices=["all", "benchmark", "ablations", "figures"], help="Giai đoạn chạy")
    parser.add_argument("--cache_dir", type=str, default=str(ADV_CACHE_DIR), help="Thư mục lưu trữ và nạp cache đối kháng .npz (chuẩn float32)")
    parser.add_argument("--skip_cache_gen", action="store_true", default=False, help="Chỉ dùng cache có sẵn, không sinh cache mới")
    parser.add_argument("--force_recompute", action="store_true", default=False, help="Bắt buộc tính toán lại và ghi đè cache")

    args = parser.parse_args()
    set_adv_cache_dir(args.cache_dir)
    logger.info("=" * 80)
    logger.info("  BẮT ĐẦU CHẠY TOÀN DIỆN THỰC NGHIỆM PAPER 2 (PHANTOM QUERIES)")
    logger.info("=" * 80)
    logger.info(f"Thư mục Cache đối kháng: {Path(args.cache_dir).resolve()}")

    # 1. Hardware Detection & Batch Size Sizing
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
    logger.info(f"Cấu hình Batch Size: {batch_size} (cho {num_gpus} thiết bị) | Primary Device: {device}")

    img_folder = Path(args.folder_image).resolve()
    ann_file = discover_coco_annotations(img_folder, args.annotations)
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    output_path = Path(args.output_json)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    telemetry_data = {
        "metadata": {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "resolution": [640, 640],
            "image_folder": str(img_folder),
            "annotations_file": str(ann_file),
            "hardware": hw_info,
            "seeds": seeds,
            "batch_size": batch_size,
        }
    }

    # Nạp telemetry hiện tại nếu chạy đơn lẻ từng stage
    if output_path.exists() and args.stage != "all":
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                telemetry_data = json.load(f)
        except Exception:
            pass

    t0_suite = time.time()

    # 2. Stage 1: Multi-Seed Benchmark & Defenses (Tables 1 & 4)
    if args.stage in ("all", "benchmark"):
        stage1_out = run_stage_1_benchmark(
            image_folder=img_folder,
            annotations_path=ann_file,
            seeds=seeds,
            n_samples_per_seed=args.num_images_benchmark,
            device=device,
            hw_info=hw_info,
            batch_size=batch_size,
            cache_dir=args.cache_dir,
            skip_cache_gen=args.skip_cache_gen,
            force_recompute=args.force_recompute,
        )
        telemetry_data["table1_benchmarks"] = {
            "rtdetr_aggregated": stage1_out["rtdetr_aggregated"],
            "yolov8_aggregated": stage1_out["yolov8_aggregated"],
            "statistical_tests": stage1_out["statistical_tests"],
        }
        telemetry_data["table4_defenses"] = {
            "defenses_aggregated": stage1_out["defenses_aggregated"]
        }
        telemetry_data["per_seed_results"] = stage1_out["per_seed_results"]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(telemetry_data, f, indent=2)
        logger.info(f"Đã lưu checkpoint Stage 1 vào: {output_path}")

    # 3. Stage 2: Sensitivity Sweeps & Downstream Impact (Tables 2 & 3)
    if args.stage in ("all", "ablations"):
        stage2_out = run_stage_2_ablations(
            image_folder=img_folder,
            annotations_path=ann_file,
            n_samples=args.num_images_ablation,
            device=device,
            hw_info=hw_info,
            batch_size=batch_size,
            cache_dir=args.cache_dir,
            skip_cache_gen=args.skip_cache_gen,
            force_recompute=args.force_recompute,
        )
        telemetry_data["table2_budget_and_iterations"] = {
            "budget_sweep": stage2_out["budget_sweep"],
            "iteration_sweep": stage2_out["iteration_sweep"],
        }
        telemetry_data["table3_cross_class_and_sizes"] = {
            "cross_class": stage2_out["cross_class"],
            "size_breakdown": stage2_out["downstream_and_figures"]["size_breakdown"],
        }
        telemetry_data["downstream_and_figures"] = stage2_out["downstream_and_figures"]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(telemetry_data, f, indent=2)
        logger.info(f"Đã lưu checkpoint Stage 2 vào: {output_path}")

    elapsed_suite = time.time() - t0_suite
    telemetry_data["metadata"]["total_elapsed_seconds"] = round(elapsed_suite, 2)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(telemetry_data, f, indent=2)

    logger.info("=" * 80)
    logger.info(f"HOÀN THÀNH TẤT CẢ THỰC NGHIỆM TRONG {elapsed_suite/60.0:.2f} PHÚT.")
    logger.info(f"Toàn bộ Telemetry đã được xuất ra: {output_path}")
    logger.info("=" * 80)

    # 4. Stage 3: Plot Publication Figures (Figures 1-4)
    if args.stage in ("all", "figures"):
        logger.info("Tự động sinh 4 đồ thị vector xuất bản tại fig/...")
        try:
            from plot_paper_figures import main as plot_main
            plot_main()
            logger.info("Sinh hình vector hoàn tất thành công!")
        except Exception as e:
            logger.error(f"Lỗi khi sinh đồ thị vector: {e}")


if __name__ == "__main__":
    main()
