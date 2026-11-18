# -*- coding: utf-8 -*-
"""
Module: paper_1_aqsm_mechanism.experiments.tasks.task7_generate_figures
Mục đích: Task tổng hợp đọc các file JSON thực nghiệm và tự động xuất toàn bộ biểu đồ vector PDF/PNG chuẩn xuất bản.

Đặc tính chuẩn công nghiệp (Production Standards):
  - Nhận tham số linh hoạt `--output_dir` và `--fig_dir`.
  - Đọc trực tiếp từ kết quả JSON thật của các tasks trước, không hardcode số liệu.
"""

import argparse
import logging
import sys
from pathlib import Path

WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from paper_1_aqsm_mechanism.experiments.config import (
    FIG_DIR,
    LOGS_DIR,
    OUTPUTS_DIR,
    SUMMARY_JSON_DIR,
    update_experiment_paths,
)
from paper_1_aqsm_mechanism.experiments.generate_figures_and_tables import generate_all_figures_and_tables


def main():
    parser = argparse.ArgumentParser(description="Task 7: Generate Publication Figures (PDF/PNG)")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUTS_DIR), help="Thư mục kết quả chứa summary_json")
    parser.add_argument("--fig_dir", type=str, default=str(FIG_DIR), help="Thư mục xuất file ảnh PDF/PNG")
    args = parser.parse_args()

    update_experiment_paths(
        output_dir=args.output_dir,
        fig_dir=args.fig_dir,
    )

    print("\n=======================================================")
    print("--> [Task 7] Đang sinh toàn bộ biểu đồ vector PDF/PNG xuất bản...")
    print(f" Thư mục dữ liệu: {args.output_dir}")
    print(f" Thư mục xuất ảnh: {args.fig_dir}")
    print("=======================================================")

    s_dir = Path(args.output_dir) / "summary_json"
    generate_all_figures_and_tables(summary_dir=s_dir, fig_dir=Path(args.fig_dir))

    print("\n[HOÀN THÀNH TOÀN BỘ] Task 7: Generate Publication Figures thành công!")


if __name__ == "__main__":
    main()
