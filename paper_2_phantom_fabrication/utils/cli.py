# -*- coding: utf-8 -*-
"""
Module: paper_2_phantom_fabrication.utils.cli
Mục đích: Khởi tạo ArgumentParser chuẩn hóa cho tất cả các Task của Paper 2,
đảm bảo đồng nhất 100% về giao diện tham số dòng lệnh giữa các môi trường chạy.
"""

import argparse
from pathlib import Path
from typing import Optional

from paper_2_phantom_fabrication.core.config import (
    ADV_CACHE_DIR,
    COCO_ANN_FILE,
    COCO_VAL_DIR,
    OUTPUTS_DIR,
)


def create_base_parser(
    description: str,
    task_name: Optional[str] = None,
) -> argparse.ArgumentParser:
    """
    Tạo parser chuẩn hóa chứa các tham số dòng lệnh cốt lõi của Paper 2.
    """
    prefix = f"[{task_name}] " if task_name else ""
    parser = argparse.ArgumentParser(
        description=f"{prefix}{description}",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 1. Tham số dữ liệu đầu vào
    parser.add_argument(
        "--folder_image",
        type=str,
        default=str(COCO_VAL_DIR),
        help="Đường dẫn thư mục ảnh COCO (ví dụ: val2017)",
    )
    parser.add_argument(
        "--annotations",
        type=str,
        default=str(COCO_ANN_FILE) if Path(COCO_ANN_FILE).exists() else None,
        help="Đường dẫn file instances_val2017.json (tự động dò tìm nếu None)",
    )

    # 2. Tham số lưu trữ đầu ra
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(OUTPUTS_DIR),
        help="Thư mục xuất kết quả (summary JSON, raw CSV, audit streams)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=str(ADV_CACHE_DIR),
        help="Thư mục lưu trữ bộ nhớ đệm đối kháng .npz chuẩn float32",
    )

    # 3. Tham số thiết bị phần cứng & phân bổ tài nguyên
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Thiết bị tính toán (auto / cuda / cpu)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size suy luận (0 hoặc None = tự động tối ưu theo VRAM)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="Số luồng CPU tối đa cho PyTorch",
    )

    # 4. Tham số mẫu và hạt giống ngẫu nhiên
    parser.add_argument(
        "--num_images",
        type=int,
        default=0,
        help="Số ảnh xử lý (0 = toàn bộ ảnh trong thư mục)",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,123,456",
        help="Danh sách random seeds (phân tách bởi dấu phẩy)",
    )

    # 5. Cờ điều khiển Cache
    parser.add_argument(
        "--skip_cache_gen",
        action="store_true",
        default=False,
        help="Chỉ sử dụng cache sẵn có trên đĩa, không sinh mới",
    )
    parser.add_argument(
        "--force_recompute",
        action="store_true",
        default=False,
        help="Bắt buộc tính toán lại và ghi đè cache hiện có",
    )

    return parser
