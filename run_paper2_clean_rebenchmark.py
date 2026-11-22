# -*- coding: utf-8 -*-
"""
Production Clean Re-Benchmark Orchestrator for Paper 2 (Elsevier EAAI)
======================================================================
Script thực thi và tái tạo toàn bộ thực nghiệm Paper 2 (Adversarial Object Fabrication in RT-DETR).

Mục tiêu thiết kế:
  1. Hỗ trợ đa môi trường: Kaggle GPU (T4/P100), Google Colab (A100/L4), Workstation Local (RTX 3090/4090) hoặc CPU fallback.
  2. Quy trình khép kín một lệnh duy nhất (Single-Command End-to-End Pipeline):
     - Task 0: Tiền sinh cache đối kháng float32 memmap (hoặc bypass on-the-fly qua GPU VRAM).
     - Task 1: Multi-Seed Benchmark (Bảng 1: RT-DETR vs YOLOv8n trên Clean, Untargeted PGD, EQF, DQF, RAB-EQF).
     - Task 2: Sensitivity Sweeps & Phân rã độ phân giải (Bảng 2: Budget epsilon, số bước lặp PGD T).
     - Task 3: Phân tích Downstream Cyber-Physical (Bảng 3: Cross-class, size breakdown, ByteTrack MOT & AEB corridor proxy).
     - Task 4: Comprehensive Defense Evaluation (Bảng 4: No Defense, Standard NMS, Threshold Elevation, DBSCAN, SQD-Filter).
     - Task 5: Xuất toàn bộ 4 Figure vector xuất bản (Figs 1-4) và hình định tính 6-panel.
     - Task 6: Chuỗi hậu xử lý khoa học:
         + Tính toán chuẩn xác kiểm định Student's t hai phía (df=2).
         + Tự động sinh báo cáo kết quả Markdown (EXPERIMENT_RESULTS_REPORT.md).
         + Tự động sinh macro tham số LaTeX (paper2_metrics.tex).
         + Kiểm toán tính nhất quán 100% giữa số liệu telemetry và TeX manuscript.
         + Tự động biên dịch LaTeX sang PDF qua pdflatex + bibtex, kiểm định đạt CHÍNH XÁC 24 TRANG.
         + Tự động xuất PDF nghiệm thu tới SUBMISSION_PDFS/.
         + Tự động chạy pytest unit test suite để đảm bảo không có lỗi kỹ thuật.

Sử dụng:
  # 1. Chạy toàn bộ thực nghiệm khép kín từ A-Z (trên GPU):
  python run_paper2_clean_rebenchmark.py --folder_image /path/to/coco/val2017

  # 2. Chế độ hậu xử lý tức thì (sau khi tải outputs/ từ Kaggle về máy):
  python run_paper2_clean_rebenchmark.py --sync_only

  # 3. Chạy kiểm tra nhanh pipeline (smoke test 10 ảnh):
  python run_paper2_clean_rebenchmark.py --folder_image /path/to/coco/val2017 --quick_test
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Khắc phục lỗi OpenMP duplicate runtime trên môi trường Windows Anaconda
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("MasterBenchmark")

PROJECT_ROOT = Path(__file__).resolve().parent
PAPER2_DIR = PROJECT_ROOT / "paper_2_phantom_fabrication"
OUTPUTS_DIR = PAPER2_DIR / "outputs"
RESULTS_DIR = PAPER2_DIR / "results"
FIG_DIR = PAPER2_DIR / "fig"
SUBMISSION_DIR = PROJECT_ROOT / "SUBMISSION_PDFS"


def check_environment() -> None:
    """Kiểm tra môi trường runtime PyTorch và GPU."""
    logger.info("=" * 80)
    logger.info(" KIỂM TRA MÔI TRƯỜNG THỰC THI (ENVIRONMENT CHECK)")
    logger.info("=" * 80)
    logger.info(f"Project Root : {PROJECT_ROOT}")
    logger.info(f"Python Exec  : {sys.executable}")

    try:
        import torch
        logger.info(f"PyTorch Ver  : {torch.__version__}")
        if torch.cuda.is_available():
            n_gpu = torch.cuda.device_count()
            gpu_names = [torch.cuda.get_device_name(i) for i in range(n_gpu)]
            logger.info(f"Thiết bị     : CUDA ({n_gpu} GPUs: {', '.join(gpu_names)})")
        else:
            logger.info("Thiết bị     : CPU (Không tìm thấy CUDA)")
    except ImportError:
        logger.error("Không tìm thấy PyTorch! Vui lòng cài đặt PyTorch trước khi chạy.")
        sys.exit(1)


def run_master_orchestrator(args: argparse.Namespace) -> None:
    """Chuyển tiếp thực thi tới Master Orchestrator run_all_experiments.py."""
    orchestrator_script = PAPER2_DIR / "run_all_experiments.py"
    if not orchestrator_script.exists():
        logger.error(f"Không tìm thấy master orchestrator: {orchestrator_script}")
        sys.exit(1)

    cmd = [
        sys.executable,
        str(orchestrator_script),
        "--output_dir", str(OUTPUTS_DIR),
        "--fig_dir", str(FIG_DIR),
        "--device", str(args.device),
    ]

    if args.sync_only or args.stage == "figures":
        cmd.append("--sync_only")
    else:
        if args.folder_image:
            cmd.extend(["--folder_image", str(args.folder_image)])
        if args.annotations:
            cmd.extend(["--annotations", str(args.annotations)])

        if args.quick_test:
            cmd.extend(["--num_images", "10", "--seeds", "42"])
        else:
            if args.num_images > 0:
                cmd.extend(["--num_images", str(args.num_images)])
            cmd.extend(["--seeds", str(args.seeds)])

        if args.tasks:
            cmd.extend(["--tasks", str(args.tasks)])
        elif args.stage == "benchmark":
            cmd.extend(["--tasks", "0,1,4,5,6"])
        elif args.stage == "ablations":
            cmd.extend(["--tasks", "2,3,5,6"])

        if args.batch_size > 0:
            cmd.extend(["--batch_size", str(args.batch_size)])

        if args.cache_dir:
            cmd.extend(["--cache_dir", str(args.cache_dir)])

        if args.skip_cache_gen:
            cmd.append("--skip_cache_gen")

        if args.force_recompute:
            cmd.append("--force_recompute")

        if args.skip_paper_compile:
            cmd.append("--skip_paper_compile")

    logger.info("Khởi chạy lệnh điều phối:")
    logger.info(" ".join(cmd))

    t0 = time.time()
    res = subprocess.run(cmd, cwd=str(PAPER2_DIR))
    elapsed = time.time() - t0

    if res.returncode != 0:
        logger.error(f"Thực nghiệm dừng lại với mã lỗi: {res.returncode}")
        sys.exit(res.returncode)

    logger.info(f"Toàn bộ chuỗi thực nghiệm hoàn tất thành công trong {elapsed/60.0:.2f} phút.")


def main():
    parser = argparse.ArgumentParser(
        description="Clean Re-benchmark Orchestrator for Paper 2 (Elsevier EAAI)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--folder_image", type=str, default="", help="Đường dẫn thư mục ảnh COCO (e.g. val2017)")
    parser.add_argument("--annotations", type=str, default=None, help="Đường dẫn annotations instances_val2017.json")
    parser.add_argument("--num_images", type=int, default=300, help="Số ảnh giới hạn (0 = toàn bộ)")
    parser.add_argument("--num_images_benchmark", type=int, default=300, help="Tương thích ngược: số ảnh cho Stage 1")
    parser.add_argument("--num_images_ablation", type=int, default=300, help="Tương thích ngược: số ảnh cho Stage 2")
    parser.add_argument("--seeds", type=str, default="42,123,456", help="Seeds thực nghiệm (mặc định 42,123,456)")
    parser.add_argument("--device", type=str, default="auto", help="Thiết bị: auto, cuda, cpu")
    parser.add_argument("--batch_size", type=int, default=0, help="Batch size (0 = tự động)")
    parser.add_argument("--output_json", type=str, default=str(OUTPUTS_DIR / "paper2_complete_telemetry.json"), help="File lưu telemetry")
    parser.add_argument("--stage", type=str, default="all", choices=["all", "benchmark", "ablations", "figures", "sync"], help="Giai đoạn thực thi")
    parser.add_argument("--tasks", type=str, default="", help="Danh sách task ID cụ thể (ví dụ: '0,1,2,3,4,5,6')")
    parser.add_argument("--cache_dir", type=str, default="", help="Thư mục adversarial cache")
    parser.add_argument("--skip_cache_gen", action="store_true", default=False, help="Chỉ dùng cache đối kháng có sẵn")
    parser.add_argument("--force_recompute", action="store_true", default=False, help="Tính toán lại toàn bộ, ghi đè cache")
    parser.add_argument("--quick_test", action="store_true", default=False, help="Chạy kiểm tra nhanh 10 ảnh (smoke test)")
    parser.add_argument("--sync_only", action="store_true", default=False, help="Chế độ hậu xử lý: Chỉ cập nhật đồ thị, báo cáo Markdown, đồng bộ TeX, build PDF 24 trang")
    parser.add_argument("--skip_paper_compile", action="store_true", default=False, help="Bỏ qua bước pdflatex (ví dụ trên máy chủ cloud không có pdflatex)")

    args = parser.parse_args()

    check_environment()

    # Xử lý tương thích ngược num_images
    if args.num_images <= 0:
        if args.num_images_benchmark > 0:
            args.num_images = args.num_images_benchmark
        elif args.num_images_ablation > 0:
            args.num_images = args.num_images_ablation

    telemetry_path = Path(args.output_json)

    # Nếu chỉ chạy phase figures / sync, hoặc không truyền folder_image mà telemetry đã có sẵn:
    if args.sync_only or args.stage in ("figures", "sync") or (not args.folder_image and telemetry_path.exists()):
        args.sync_only = True
        logger.info(f"Phát hiện chế độ hậu xử lý (Sync Only) với file telemetry: {telemetry_path}")
        run_master_orchestrator(args)
        logger.info("Hoàn tất quy trình hậu xử lý tự động.")
        return

    if not args.folder_image:
        # Tự động dò tìm thư mục data/coco5k/val2017 mặc định
        default_coco = PROJECT_ROOT / "data" / "coco5k" / "val2017"
        if default_coco.exists():
            args.folder_image = str(default_coco)
            logger.info(f"Tự động chọn thư mục ảnh COCO mặc định: {default_coco}")
        else:
            logger.error("Vui lòng cung cấp --folder_image (hoặc sử dụng --sync_only nếu đã có kết quả telemetry JSON).")
            sys.exit(1)

    run_master_orchestrator(args)

    logger.info("=" * 80)
    logger.info(" HOÀN TẤT TOÀN DIỆN QUY TRÌNH RE-BENCHMARK CHO PAPER 2!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
