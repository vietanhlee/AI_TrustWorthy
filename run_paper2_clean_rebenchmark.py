# -*- coding: utf-8 -*-
"""
Production Clean Re-Benchmark Orchestrator for Paper 2 (Elsevier EAAI)
======================================================================
Script thực thi và tái tạo toàn bộ thực nghiệm Paper 2 (Adversarial Object Fabrication in RT-DETR).

Mục tiêu thiết kế:
  1. Hỗ trợ đa môi trường: Kaggle GPU (T4/P100), Google Colab (A100/L4), Workstation Local (RTX 3090/4090) hoặc CPU fallback.
  2. Tách bạch rõ ràng 3 giai đoạn (Stages):
     - Stage 1: Multi-Seed Benchmark (Bảng 1: RT-DETR vs YOLOv8n trên Clean, Untargeted PGD, EQF, DQF, RAB-EQF)
       và 5 cơ chế phòng thủ (Bảng 4: No Defense, Standard NMS, Threshold Elevation, Spatial DBSCAN, SQD-Filter).
     - Stage 2: Sensitivity Sweeps & Phân rã độ phân giải (Bảng 2: Budget epsilon in {4,8,12,16}/255, T in {2,4,8,15})
       và Phân rã theo lớp & kích thước vật thể (Bảng 3: person, bicycle, car; small, medium, large).
     - Stage 3: Phân tích Downstream Cyber-Physical (ByteTrack MOT & AEB corridor proxy) và sinh toàn bộ Figure vector (Figs 1-4).
  3. Chuẩn hóa Cache Đối Kháng (Adversarial Cache Keying): Khóa cache chặt chẽ theo `model_condition_eps_steps_alpha_seed_index`,
     tránh triệt để lỗi va chạm khóa (cache collision) và lỗi số liệu cố định (hardcode/sponge leakage).
  4. Đồng bộ tự động Telemetry JSON -> Figures (PDF/PNG) -> LaTeX Manuscript (`main_phantom.tex`) -> Biên dịch PDF nộp bài.

Sử dụng:
  # Chạy toàn bộ suite với 300 ảnh/seed trên GPU:
  python run_paper2_clean_rebenchmark.py --folder_image /path/to/coco/val2017 --annotations /path/to/instances_val2017.json

  # Chạy nhanh kiểm tra pipeline (smoke test 10 ảnh):
  python run_paper2_clean_rebenchmark.py --folder_image /path/to/coco/val2017 --quick_test

  # Chỉ vẽ lại đồ thị và biên dịch paper từ telemetry JSON đã có:
  python run_paper2_clean_rebenchmark.py --stage figures --sync_paper
"""

import argparse
import os
import sys

# Khắc phục lỗi OpenMP duplicate runtime trên môi trường Windows Anaconda
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import time
import shutil
import logging
import subprocess
from pathlib import Path

# Thiết lập đường dẫn root
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PAPER2_DIR = PROJECT_ROOT / "paper_2_phantom_fabrication"
EXPERIMENTS_DIR = PAPER2_DIR / "experiments"
OUTPUTS_DIR = PAPER2_DIR / "outputs"
FIG_DIR = PAPER2_DIR / "fig"
SUBMISSION_DIR = PROJECT_ROOT / "SUBMISSION_PDFS"

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("CleanRebenchmark")


def check_environment():
    """Kiểm tra và ghi log thông số phần cứng & thư viện."""
    logger.info("=" * 80)
    logger.info(" KIỂM TRA MÔI TRƯỜNG THỰC THI (ENVIRONMENT AUDIT)")
    logger.info("=" * 80)
    
    try:
        import torch
        logger.info(f"PyTorch Version: {torch.__version__}")
        has_cuda = torch.cuda.is_available()
        logger.info(f"CUDA Available: {has_cuda}")
        if has_cuda:
            device_count = torch.cuda.device_count()
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                logger.info(f"  [GPU {i}] {props.name} | VRAM: {props.total_memory / (1024**3):.2f} GB")
        else:
            logger.info("  Chạy trên CPU (Lưu ý: Đo latency inference CPU chuẩn xác, không dùng mock).")
    except ImportError:
        logger.error("Không tìm thấy PyTorch! Vui lòng cài đặt PyTorch trước khi chạy.")
        sys.exit(1)


def run_benchmark_suite(args):
    """Gọi Runner chính của Paper 2."""
    runner_script = EXPERIMENTS_DIR / "run_full_paper2_suite.py"
    if not runner_script.exists():
        logger.error(f"Không tìm thấy script: {runner_script}")
        sys.exit(1)

    cmd = [
        sys.executable,
        str(runner_script),
        "--folder_image", str(args.folder_image),
        "--stage", str(args.stage),
        "--device", str(args.device),
        "--output_json", str(args.output_json),
    ]

    if args.annotations:
        cmd.extend(["--annotations", str(args.annotations)])

    if args.quick_test:
        cmd.extend(["--num_images_benchmark", "10", "--num_images_ablation", "10", "--seeds", "42"])
    else:
        if args.num_images_benchmark > 0:
            cmd.extend(["--num_images_benchmark", str(args.num_images_benchmark)])
        if args.num_images_ablation > 0:
            cmd.extend(["--num_images_ablation", str(args.num_images_ablation)])
        cmd.extend(["--seeds", str(args.seeds)])

    if args.batch_size > 0:
        cmd.extend(["--batch_size", str(args.batch_size)])

    if args.cache_dir:
        cmd.extend(["--cache_dir", str(args.cache_dir)])

    if args.skip_cache_gen:
        cmd.append("--skip_cache_gen")

    if args.force_recompute:
        cmd.append("--force_recompute")

    logger.info("Khởi chạy lệnh thực nghiệm:")
    logger.info(" ".join(cmd))
    
    t0 = time.time()
    res = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - t0
    
    if res.returncode != 0:
        logger.error(f"Thực nghiệm dừng với mã lỗi: {res.returncode}")
        sys.exit(res.returncode)
        
    logger.info(f"Thực nghiệm hoàn tất thành công trong {elapsed/60.0:.2f} phút.")


def sync_figures_and_paper(telemetry_file: Path):
    """Vẽ lại figures và đồng bộ vào LaTeX, sau đó biên dịch PDF."""
    logger.info("=" * 80)
    logger.info(" ĐỒNG BỘ FIGURE VECTOR & MANUSCRIPT LATEX")
    logger.info("=" * 80)

    # 1. Chạy figures engine
    try:
        from paper_2_phantom_fabrication.experiments.engines.figures_engine import generate_all_paper2_figures
        logger.info(f"Sinh 4 figure vector (PDF/PNG) từ telemetry: {telemetry_file}")
        generate_all_paper2_figures(str(telemetry_file), str(FIG_DIR))
        logger.info(f"  -> Sinh 4 đồ thị vector thành công tại: {FIG_DIR}")
    except Exception as e:
        logger.warning(f"Lỗi khi gọi figures_engine trực tiếp: {e}. Thử fallback qua script độc lập.")
        plot_script = EXPERIMENTS_DIR / "plot_paper_figures.py"
        if plot_script.exists():
            subprocess.run([sys.executable, str(plot_script)], cwd=str(EXPERIMENTS_DIR))

    # 2. Biên dịch LaTeX paper
    tex_path = PAPER2_DIR / "main_phantom.tex"
    if tex_path.exists():
        logger.info(f"Biên dịch LaTeX: {tex_path}")
        cmd_compile = [
            ["pdflatex", "-interaction=nonstopmode", "main_phantom.tex"],
            ["bibtex", "main_phantom"],
            ["pdflatex", "-interaction=nonstopmode", "main_phantom.tex"],
            ["pdflatex", "-interaction=nonstopmode", "main_phantom.tex"],
        ]
        for cmd in cmd_compile:
            subprocess.run(cmd, cwd=str(PAPER2_DIR), capture_output=True)

        pdf_src = PAPER2_DIR / "main_phantom.pdf"
        if pdf_src.exists():
            SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
            eaai_dst = SUBMISSION_DIR / "2_Paper_Phantom_Fabrication_Elsevier_EAAI.pdf"
            shutil.copy2(pdf_src, eaai_dst)
            # Đồng bộ cả tên phụ
            alt_dst = SUBMISSION_DIR / "3_Paper_Phantom_Queries_Fabrication_Elsevier.pdf"
            shutil.copy2(pdf_src, alt_dst)
            logger.info(f"Đã sao chép PDF hoàn thiện tới: {eaai_dst} ({eaai_dst.stat().st_size:,} bytes)")


def main():
    parser = argparse.ArgumentParser(description="Clean Re-benchmark Orchestrator for Paper 2 (Elsevier EAAI)")
    parser.add_argument("--folder_image", type=str, default="", help="Đường dẫn thư mục ảnh COCO (e.g. val2017)")
    parser.add_argument("--annotations", type=str, default=None, help="Đường dẫn annotations instances_val2017.json")
    parser.add_argument("--num_images_benchmark", type=int, default=300, help="Số ảnh mỗi seed cho Stage 1 (mặc định 300)")
    parser.add_argument("--num_images_ablation", type=int, default=300, help="Số ảnh cho Stage 2 Ablations (mặc định 300)")
    parser.add_argument("--seeds", type=str, default="42,123,456", help="Seeds thực nghiệm (mặc định 42,123,456)")
    parser.add_argument("--device", type=str, default="auto", help="Thiết bị (auto, cuda, cpu)")
    parser.add_argument("--batch_size", type=int, default=0, help="Batch size (0 = tự động)")
    parser.add_argument("--output_json", type=str, default=str(OUTPUTS_DIR / "paper2_complete_telemetry.json"), help="File lưu telemetry")
    parser.add_argument("--stage", type=str, default="all", choices=["all", "benchmark", "ablations", "figures"], help="Giai đoạn thực thi")
    parser.add_argument("--cache_dir", type=str, default="", help="Thư mục adversarial cache")
    parser.add_argument("--skip_cache_gen", action="store_true", help="Chỉ dùng cache đối kháng có sẵn")
    parser.add_argument("--force_recompute", action="store_true", help="Tính toán lại toàn bộ, ghi đè cache")
    parser.add_argument("--quick_test", action="store_true", help="Chạy kiểm tra nhanh 10 ảnh (smoke test)")
    parser.add_argument("--sync_paper", action="store_true", default=True, help="Tự động đồng bộ đồ thị và biên dịch paper sau khi chạy")

    args = parser.parse_args()

    check_environment()

    # Nếu chỉ chạy phase figures hoặc nếu không truyền folder_image mà file telemetry đã tồn tại:
    telemetry_path = Path(args.output_json)
    if args.stage == "figures" or (not args.folder_image and telemetry_path.exists()):
        logger.info(f"Đã phát hiện telemetry file: {telemetry_path}")
        sync_figures_and_paper(telemetry_path)
        logger.info("Hoàn tất quy trình đồng bộ và biên dịch.")
        return

    if not args.folder_image:
        logger.error("Vui lòng cung cấp --folder_image (hoặc chạy với --stage figures nếu đã có telemetry JSON).")
        sys.exit(1)

    run_benchmark_suite(args)

    if args.sync_paper and telemetry_path.exists():
        sync_figures_and_paper(telemetry_path)

    logger.info("=" * 80)
    logger.info(" HOÀN TẤT TOÀN DIỆN QUY TRÌNH RE-BENCHMARK CHO PAPER 2!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
