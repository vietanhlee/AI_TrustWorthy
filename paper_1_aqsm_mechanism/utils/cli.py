# -*- coding: utf-8 -*-
"""
Module: paper_1_aqsm_mechanism.utils.cli
Mục đích: Cung cấp bộ cấu hình tham số dòng lệnh (CLI Argument Parser) chuẩn mực,
giúp các Task 1 -> Task 8 và Orchestrator kế thừa đồng nhất, loại bỏ hơn 150 dòng lặp code (DRY).
"""

import argparse
from typing import Optional

from paper_1_aqsm_mechanism.experiments.config import (
    COHORT_SIZE,
    get_adv_cache_dir,
    get_coco_ann_file,
    get_coco_val_dir,
    get_model_name,
    get_outputs_dir,
)


def create_base_parser(
    description: str,
    task_name: Optional[str] = None,
) -> argparse.ArgumentParser:
    """
    Tạo ArgumentParser chứa sẵn các tham số cấu hình chung chuẩn hóa:
      --folder_image: Thư mục chứa ảnh COCO (hỗ trợ tập con hoặc toàn bộ 5,000 ảnh).
      --annotations: Đường dẫn file instances_val2017.json.
      --output_dir: Thư mục xuất kết quả (summary JSON & raw CSV).
      --cache_dir: Thư mục lưu trữ và đọc bộ nhớ đệm mẫu đối kháng .npz.
      --batch_size: Kích thước batch suy luận (None = tự động tính theo VRAM/CPU).
      --threads: Số luồng CPU tính toán.
      --device: Thiết bị tính toán (auto / cuda / cpu).
    """
    desc = f"[{task_name}] {description}" if task_name else description
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        "--folder_image",
        type=str,
        default=str(get_coco_val_dir()),
        help="Đường dẫn thư mục ảnh COCO (mặc định lấy từ cấu hình hệ thống)",
    )
    parser.add_argument(
        "--annotations",
        type=str,
        default=str(get_coco_ann_file()),
        help="Đường dẫn file instances_val2017.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(get_outputs_dir()),
        help="Thư mục xuất kết quả thực nghiệm (mặc định: outputs/)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=str(get_adv_cache_dir()),
        help="Thư mục chứa bộ nhớ đệm mẫu đối kháng .npz (mặc định: cache/adv_cache_640/)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Kích thước batch suy luận (mặc định tự động tính tối ưu theo VRAM phần cứng)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="Số luồng CPU tính toán song song",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Thiết bị thực thi (auto = tự nhận diện GPU nếu có)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(get_model_name()),
        help="Kiến trúc hoặc checkpoint RT-DETR ('r18', 'r50', 'PekingU/rtdetr_r18vd', 'PekingU/rtdetr_r50vd')",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=COHORT_SIZE,
        help=f"Số lượng mẫu ảnh tối đa cần đánh giá (mặc định: {COHORT_SIZE} ảnh, <= 0 để chạy toàn bộ dataset)",
    )

    return parser
