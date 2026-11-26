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
from paper_1_aqsm_mechanism.experiments.config import DEFAULT_MULTI_SEEDS, get_img_size


def main():
    default_seeds_str = ",".join(map(str, DEFAULT_MULTI_SEEDS))
    parser = argparse.ArgumentParser(description="Kaggle/Colab GPU Master Experiment Runner")
    parser.add_argument("--folder_image", type=str, default=None, help="Đường dẫn thư mục ảnh COCO val2017")
    parser.add_argument("--annotations", type=str, default=None, help="Đường dẫn file instances_val2017.json")
    parser.add_argument("--model", type=str, default="r18", help="Kiến trúc hoặc checkpoint RT-DETR ('r18', 'r50', 'PekingU/rtdetr_r18vd', 'PekingU/rtdetr_r50vd')")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size tối ưu cho GPU (mặc định 16)")
    parser.add_argument("--max_samples", type=int, default=1000, help="Số lượng ảnh thực nghiệm (mặc định 1000 ảnh chuẩn)")
    parser.add_argument("--num_images", type=int, default=None, help="Alias tương đương --max_samples")
    parser.add_argument("--tasks", type=str, default="0,1,2,3,4,5,6,8,10,7,9", help="Danh sách tasks cần chạy (mặc định gồm cả Task 10 đa seed)")
    parser.add_argument("--skip_cache_gen", action="store_true", default=False, help="Bỏ qua Task 0 nếu cache đã có")
    parser.add_argument("--force_recompute", action="store_true", default=False, help="Bắt buộc sinh lại cache từ đầu kể cả khi đã có")
    parser.add_argument("--include_sweep", action="store_true", default=False, help="Sinh kèm sweep eps 4 và 8 cho Task 5")
    parser.add_argument("--run_sqa_ablation", action="store_true", default=False, help="Chạy Live GPU Forward Pass cho SQA và controls trong Task 8")
    parser.add_argument("--run_multiseed", action="store_true", default=False, help="Bật Task 10 chạy thực nghiệm AQSM đa seed khi truyền tập --tasks tùy chỉnh")
    parser.add_argument("--seeds", type=str, default=default_seeds_str, help=f"Danh sách random seeds cho Task 4 & Task 10 (mặc định: {default_seeds_str})")
    parser.add_argument("--pgd_steps", type=int, default=15, help="Số bước lặp PGD sinh mẫu đối kháng (mặc định 15)")
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
        eff_img = int(get_img_size())
        bytes_per_sample = 3 * eff_img * eff_img * 4  # Float32 delta per image
        num_attacks = 3  # aqsm, vanishing, taskpgd
        actual_dir_count = 5000
        if args.folder_image and Path(args.folder_image).is_dir():
            valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
            found_cnt = len([p for p in Path(args.folder_image).iterdir() if p.suffix.lower() in valid_exts])
            if found_cnt > 0:
                actual_dir_count = found_cnt
        est_samples = args.max_samples if (args.max_samples is not None and args.max_samples > 0) else actual_dir_count
        needed_disk_gb = (est_samples * bytes_per_sample * num_attacks) / (1024 ** 3)

        # Để an toàn, chỉ dùng tối đa 75% đĩa trống hiện có (dành 25% cho outputs, zip và OS)
        safe_disk_capacity_gb = free_gb * 0.75
        max_safe_samples = int((safe_disk_capacity_gb * (1024 ** 3)) / (num_attacks * bytes_per_sample))

        if needed_disk_gb > safe_disk_capacity_gb:
            effective_max_samples = min(est_samples, max(max_safe_samples, 1))
            print("=" * 80)
            print(f"⚠️  [CẢNH BÁO DUNG LƯỢNG Ổ CỨNG KAGGLE]")
            print(f" Yêu cầu tiền sinh cache cho {args.max_samples} ảnh -> Cần tối thiểu {needed_disk_gb:.2f} GB đĩa cho {num_attacks} attacks,")
            print(f" nhưng ổ cứng khả dụng chỉ còn {free_gb:.2f} GB!")
            print(f" Để ngăn ngừa lỗi cạn đĩa ([Errno 28] No space left on device), hệ thống tự động bảo vệ:")
            print(f" --> Điều chỉnh an toàn cho cache trên đĩa: max_samples = {effective_max_samples} ảnh")
            print(f" (Mẹo: Thêm cờ --skip_cache_gen để chạy On-The-Fly số ảnh đầy đủ {args.max_samples} với 0 GB đĩa)")
            print("=" * 80)
        else:
            effective_max_samples = args.max_samples

    print(f">>> Cấu hình Max Samples thực thi: {effective_max_samples} ảnh")
    print(f">>> Mô hình chỉ định            : {args.model}")

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
        "--model", str(args.model),
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
    if args.force_recompute:
        cmd.append("--force_recompute")
    if args.include_sweep:
        cmd.append("--include_sweep")
    if args.run_sqa_ablation:
        cmd.append("--run_sqa_ablation")
    if args.run_multiseed:
        cmd.append("--run_multiseed")
    if args.seeds:
        cmd.extend(["--seeds", str(args.seeds)])
    if args.pgd_steps is not None:
        cmd.extend(["--pgd_steps", str(args.pgd_steps)])

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
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zipf:
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
