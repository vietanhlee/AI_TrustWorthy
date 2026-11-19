# -*- coding: utf-8 -*-
"""
Module: paper_1_aqsm_mechanism.experiments.tasks.task7_generate_figures
Mục đích: CLI điều phối Task 7 - Đọc các file JSON thực nghiệm và tự động xuất toàn bộ biểu đồ vector PDF/PNG chuẩn xuất bản Elsevier EAAI.
"""

import sys
from pathlib import Path

WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from paper_1_aqsm_mechanism.experiments.config import (
    FIG_DIR,
    OUTPUTS_DIR,
    update_experiment_paths,
)
from paper_1_aqsm_mechanism.experiments.engines.generate_figures_and_tables import (
    generate_all_figures_and_tables,
)
from paper_1_aqsm_mechanism.utils.cli import create_base_parser


def main():
    parser = create_base_parser("Generate Publication Figures (PDF/PNG)", task_name="Task 7")
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
