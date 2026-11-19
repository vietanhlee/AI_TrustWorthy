# -*- coding: utf-8 -*-
"""
Script: kaggle_run_experiments.py
Mục đích: Chạy toàn bộ chuỗi thực nghiệm Paper 1 trên môi trường GPU (Kaggle T4 x 2, P100 hoặc Google Colab).
Tự động kích hoạt CUDA, gọi Master Orchestrator (run_all_experiments.py),
và tự động nén thư mục outputs thành `paper1_experiment_results.zip` để người dùng tải về chỉ với 1 click.
"""

import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

# Thêm thư mục gốc vào PYTHONPATH
WORKSPACE_DIR = Path(__file__).resolve().parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

import torch


def main():
    parser = argparse.ArgumentParser(description="Kaggle/Colab GPU Master Experiment Runner")
    parser.add_argument("--folder_image", type=str, default=None, help="Đường dẫn thư mục ảnh COCO val2017")
    parser.add_argument("--annotations", type=str, default=None, help="Đường dẫn file instances_val2017.json")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size tối ưu cho GPU (mặc định 16)")
    parser.add_argument("--max_samples", type=int, default=300, help="Số lượng ảnh thực nghiệm (mặc định 300 ảnh theo chuẩn Paper 1)")
    parser.add_argument("--num_images", type=int, default=None, help="Alias tương đương --max_samples")
    parser.add_argument("--tasks", type=str, default="0,1,2,3,4,5,6,8,7", help="Danh sách tasks cần chạy")
    parser.add_argument("--skip_cache_gen", action="store_true", default=False, help="Bỏ qua Task 0 nếu cache đã có")
    parser.add_argument("--include_sweep", action="store_true", default=False, help="Sinh kèm sweep eps 4 và 8 cho Task 5")
    args = parser.parse_args()

    if args.num_images is not None:
        args.max_samples = args.num_images

    print("================================================================================")
    print(" [KAGGLE/COLAB GPU RUNNER] KHỞI CHẠY MASTER ORCHESTRATOR TRÊN GPU")
    print("================================================================================")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f">>> Thiết bị phát hiện: {device.upper()}")
    if device == "cuda":
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f">>> GPU Hardware: {gpu_count}x {gpu_name} ({vram_gb:.2f} GB VRAM)")
        print(">>> Tốc độ dự kiến: Tăng tốc ~15x so với CPU!")
    else:
        print(">>> [CẢNH BÁO] Không tìm thấy GPU, đang chạy trên CPU.")

    # Kiểm tra dung lượng đĩa khả dụng và tự động tính toán giới hạn an toàn
    disk_stat = shutil.disk_usage(".")
    free_gb = disk_stat.free / (1024 ** 3)
    total_gb = disk_stat.total / (1024 ** 3)
    print(f">>> Dung lượng ổ cứng khả dụng: {free_gb:.2f} GB / {total_gb:.2f} GB")

    if args.skip_cache_gen:
        print(">>> Cờ --skip_cache_gen ĐƯỢC BẬT: Bỏ qua Task 0 (tiền sinh cache đối kháng ra đĩa).")
        print(">>> KÍCH HOẠT CHẾ ĐỘ ON-THE-FLY VRAM STREAMING (0 GB đĩa, giải phóng VRAM theo từng batch)!")
        effective_max_samples = args.max_samples
    else:
        bytes_per_sample = 3 * 640 * 640 * 4  # 4.9152 MB/ảnh ở chuẩn float32
        num_attacks = 3  # aqsm, vanishing, taskpgd
        needed_disk_gb = (args.max_samples * bytes_per_sample * num_attacks) / (1024 ** 3)

        # Để an toàn, chỉ dùng tối đa 75% đĩa trống hiện có (dành 25% cho outputs, zip và OS)
        safe_disk_capacity_gb = free_gb * 0.75
        max_safe_samples = int((safe_disk_capacity_gb * (1024 ** 3)) / (num_attacks * bytes_per_sample))

        if needed_disk_gb > safe_disk_capacity_gb:
            effective_max_samples = min(max_safe_samples, 300)
            print("=" * 80)
            print(f"⚠️  [CẢNH BÁO DUNG LƯỢNG Ổ CỨNG KAGGLE]")
            print(f" Yêu cầu tiền sinh cache cho {args.max_samples} ảnh -> Cần tối thiểu {needed_disk_gb:.2f} GB đĩa cho {num_attacks} attacks,")
            print(f" nhưng ổ cứng khả dụng chỉ còn {free_gb:.2f} GB!")
            print(f" Để ngăn ngừa lỗi cạn đĩa ([Errno 28] No space left on device), hệ thống tự động bảo vệ:")
            print(f" --> Tự động điều chỉnh: max_samples = {effective_max_samples} ảnh")
            print(f" (Mẹo: Thêm cờ --skip_cache_gen để chạy On-The-Fly số ảnh tùy thích với 0 GB đĩa)")
            print("=" * 80)
        else:
            effective_max_samples = args.max_samples

    print(f">>> Cấu hình Max Samples thực thi: {effective_max_samples} ảnh")

    paper_dir = WORKSPACE_DIR / "paper_1_aqsm_mechanism"
    output_dir = paper_dir / "outputs"
    fig_dir = paper_dir / "fig"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Chuẩn bị lệnh gọi run_all_experiments.py
    cmd = [
        sys.executable,
        str(paper_dir / "run_all_experiments.py"),
        "--device", device,
        "--batch_size", str(args.batch_size),
        "--output_dir", str(output_dir),
        "--fig_dir", str(fig_dir),
    ]

    if effective_max_samples is not None:
        cmd.extend(["--max_samples", str(effective_max_samples)])
    if args.folder_image:
        cmd.extend(["--folder_image", str(args.folder_image)])
    if args.annotations:
        cmd.extend(["--annotations", str(args.annotations)])
    if args.tasks:
        cmd.extend(["--tasks", str(args.tasks)])
    if args.skip_cache_gen:
        cmd.append("--skip_cache_gen")
    if args.include_sweep:
        cmd.append("--include_sweep")

    start_time = time.time()
    print(f"\n>>> Đang thực thi lệnh: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=str(WORKSPACE_DIR))

    if result.returncode != 0:
        print(f"\n[LỖI] Thực nghiệm thất bại với mã thoát: {result.returncode}")
        sys.exit(result.returncode)

    elapsed = time.time() - start_time
    print(f"\n>>> Toàn bộ thực nghiệm hoàn thành xuất sắc trong: {elapsed:.2f}s ({elapsed/60:.2f} phút)!")

    # Đóng gói thư mục outputs và fig thành file ZIP
    zip_path = WORKSPACE_DIR / "paper1_experiment_results.zip"
    print(f"\n>>> Đang đóng gói kết quả vào file ZIP: {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Thêm outputs
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                file_full = Path(root) / file
                arcname = file_full.relative_to(paper_dir)
                zipf.write(file_full, arcname)
        # Thêm figures
        for root, dirs, files in os.walk(fig_dir):
            for file in files:
                file_full = Path(root) / file
                arcname = file_full.relative_to(paper_dir)
                zipf.write(file_full, arcname)

    print(f">>> [HOÀN TẤT] File kết quả đã sẵn sàng tải về: {zip_path.resolve()} ({zip_path.stat().st_size / 1e6:.1f} MB)")
    print(">>> Bạn chỉ cần tải file `paper1_experiment_results.zip` về máy và đưa vào thư mục dự án.")


if __name__ == "__main__":
    main()
