# -*- coding: utf-8 -*-
"""
Module: paper_2_phantom_fabrication.kaggle_run_experiments
Mục đích: Script chuyên biệt khởi chạy chuỗi thực nghiệm Paper 2 trên nền tảng Kaggle GPU.
  - Tự động tìm kiếm tập dữ liệu COCO val2017 trong các thư mục `/kaggle/input`.
  - Tối ưu hóa cấu hình bộ nhớ VRAM, CPU threads, và batch size tự động.
  - Chạy Orchestrator `run_all_experiments.py` với cơ chế chống OOM triệt để.
  - Tự động đóng gói kết quả thực nghiệm và các biểu đồ thành `paper2_experiment_results.zip`.
"""

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import List, Optional

PACKAGE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = PACKAGE_DIR.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from paper_2_phantom_fabrication.utils.device import detect_hardware


def discover_kaggle_coco_dir() -> Optional[Path]:
    """
    Tự động dò tìm đường dẫn thư mục ảnh val2017 trên môi trường Kaggle.
    """
    candidates = [
        Path("/kaggle/input/coco-2017-dataset/coco2017/val2017"),
        Path("/kaggle/input/coco2017/val2017"),
        Path("/kaggle/input/coco-val2017/val2017"),
        Path("/kaggle/input/coco-2017/val2017"),
        Path("/kaggle/input/val2017"),
        PACKAGE_DIR / "data" / "coco5k" / "val2017",
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            return c

    # Dò tìm đệ quy nếu có thư mục /kaggle/input
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for p in kaggle_input.glob("**/val2017"):
            if p.is_dir():
                return p
    return None


def zip_results(outputs_dir: Path, fig_dir: Path, zip_dest: Path) -> None:
    """
    Đóng gói toàn bộ outputs và biểu đồ figures thành file ZIP tiện tải về.
    """
    zip_dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_dest, "w", zipfile.ZIP_DEFLATED) as zipf:
        if outputs_dir.exists():
            for root, _, files in os.walk(outputs_dir):
                for f in files:
                    full_p = Path(root) / f
                    rel_p = full_p.relative_to(outputs_dir.parent)
                    zipf.write(full_p, arcname=str(rel_p))
        if fig_dir.exists():
            for root, _, files in os.walk(fig_dir):
                for f in files:
                    full_p = Path(root) / f
                    rel_p = full_p.relative_to(fig_dir.parent)
                    zipf.write(full_p, arcname=str(rel_p))
    print(f"\n[KAGGLE PACKAGE] Đã đóng gói thành công file ZIP: {zip_dest} ({zip_dest.stat().st_size / (1024 * 1024):.2f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Kaggle GPU Runner for Paper 2: Phantom Queries Fabrication",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--folder_image", type=str, default="", help="Đường dẫn thư mục ảnh val2017 (Tự động phát hiện nếu bỏ trống)")
    parser.add_argument("--annotations", type=str, default="", help="Đường dẫn file annotations JSON (Tự động phát hiện nếu bỏ trống)")
    parser.add_argument("--tasks", type=str, default="0,1,2,3,4,5", help="Danh sách task ID thực thi (e.g. '0,1,2,3,4,5' hoặc 'all')")
    parser.add_argument("--num_images", type=int, default=0, help="Số ảnh giới hạn (0 = chạy toàn bộ)")
    parser.add_argument("--batch_size", type=int, default=0, help="Batch size suy luận (0 = tự động)")
    parser.add_argument("--skip_cache_gen", action="store_true", default=False, help="Bỏ qua sinh cache nếu đã có")
    parser.add_argument("--force_recompute", action="store_true", default=False, help="Bắt buộc tính toán lại từ đầu")
    args = parser.parse_args()

    # Dò tìm thư mục ảnh nếu chưa cung cấp
    img_folder = Path(args.folder_image) if args.folder_image else discover_kaggle_coco_dir()
    if img_folder is None or not img_folder.exists():
        print("[LỖI] Không tìm thấy thư mục ảnh COCO val2017 trên Kaggle!")
        print("Vui lòng truyền đối số --folder_image /kaggle/input/.../val2017")
        sys.exit(1)

    # Thư mục làm việc trên Kaggle
    working_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else PACKAGE_DIR
    outputs_dir = working_dir / "outputs"
    fig_dir = working_dir / "fig"
    cache_dir = working_dir / "cache" / "adv_cache_640"
    zip_dest = working_dir / "paper2_experiment_results.zip"

    hw_info = detect_hardware("auto")
    print("=" * 80)
    print(" KHỞI ĐỘNG KAGGLE EXPERIMENTS RUNNER - PAPER 2")
    print(f" Thư mục ảnh      : {img_folder}")
    print(f" Thiết bị phần cứng: {hw_info['primary_device']} ({hw_info.get('num_gpus', 0)} GPUs)")
    print(f" GPU Model        : {hw_info.get('gpu_name', 'N/A')}")
    print(f" VRAM khả dụng    : {hw_info.get('gpu_memory_gb', 0.0):.2f} GB")
    print(f" Thư mục đầu ra   : {working_dir}")
    print("=" * 80)

    cmd = [
        sys.executable,
        str(PACKAGE_DIR / "run_all_experiments.py"),
        "--folder_image", str(img_folder),
        "--tasks", str(args.tasks),
        "--num_images", str(args.num_images),
        "--batch_size", str(args.batch_size),
        "--cache_dir", str(cache_dir),
        "--output_dir", str(outputs_dir),
        "--fig_dir", str(fig_dir),
    ]
    if args.annotations:
        cmd.extend(["--annotations", str(Path(args.annotations).resolve())])
    if args.skip_cache_gen:
        cmd.append("--skip_cache_gen")
    if args.force_recompute:
        cmd.append("--force_recompute")

    print(f"\n[KAGGLE ORCHESTRATOR] Bắt đầu gọi run_all_experiments.py...")
    ret = subprocess.call(cmd, cwd=str(PACKAGE_DIR))

    if ret == 0:
        print("\n[THÀNH CÔNG] Toàn bộ chuỗi thực nghiệm Paper 2 đã hoàn tất mỹ mãn!")
        zip_results(outputs_dir, fig_dir, zip_dest)
    else:
        print(f"\n[LỖI] Chuỗi thực nghiệm kết thúc với mã lỗi: {ret}")

    sys.exit(ret)


if __name__ == "__main__":
    main()
