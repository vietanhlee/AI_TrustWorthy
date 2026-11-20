# -*- coding: utf-8 -*-
"""
Module: paper_2_phantom_fabrication.experiments.tasks.task5_generate_figures
Mục đích: CLI Task 5 - Tự động hóa xuất toàn bộ biểu đồ khoa học chuẩn IEEE/Elsevier
cho Paper 2 (Figure 1, 2, 3, 4) định dạng PDF vector và PNG 300 DPI từ dữ liệu thực nghiệm thực tế.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
from pathlib import Path

WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from paper_2_phantom_fabrication.core.config import FIG_DIR, OUTPUTS_DIR, RESULTS_DIR
from paper_2_phantom_fabrication.experiments.engines.figures_engine import generate_all_paper2_figures
from paper_2_phantom_fabrication.utils.cli import create_base_parser


def main():
    parser = create_base_parser("Generate Scientific Figures (Fig 1-4)", task_name="Task 5")
    parser.add_argument(
        "--telemetry_path",
        type=str,
        default="",
        help="Đường dẫn file JSON telemetry đầy đủ. Nếu trống, tự động tìm kiếm.",
    )
    parser.add_argument(
        "--fig_dir",
        type=str,
        default=str(FIG_DIR),
        help="Thư mục xuất các biểu đồ PDF và PNG",
    )
    parser.add_argument(
        "--generate_qualitative",
        action="store_true",
        default=False,
        help="Có sinh lại hình định tính 6-panel không (mặc định False nếu đã tồn tại).",
    )
    args = parser.parse_args()

    # Tìm kiếm file telemetry
    telemetry_file: Path
    if args.telemetry_path and Path(args.telemetry_path).exists():
        telemetry_file = Path(args.telemetry_path)
    else:
        candidates = [
            Path(args.output_dir) / "paper2_complete_telemetry.json",
            RESULTS_DIR / "paper2_complete_telemetry.json",
            Path(args.output_dir) / "stage1_benchmark_telemetry.json",
        ]
        found = None
        for c in candidates:
            if c.exists():
                found = c
                break
        telemetry_file = found if found is not None else (RESULTS_DIR / "paper2_complete_telemetry.json")

    out_fig_dir = Path(args.fig_dir)
    out_fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(" [TASK 5] XUẤT BIỂU ĐỒ KHOA HỌC PAPER 2 (FIGURE 1, 2, 3, 4)")
    print(f" Nguồn dữ liệu Telemetry: {telemetry_file}")
    print(f" Thư mục xuất biểu đồ   : {out_fig_dir}")
    print("=" * 80)

    if not telemetry_file.exists():
        print(f"[CẢNH BÁO] Không tìm thấy file telemetry tại: {telemetry_file}")
        print("Vui lòng chạy Task 1/2/3 hoặc cung cấp đường dẫn --telemetry_path hợp lệ.")
        return

    generate_all_paper2_figures(telemetry_path=telemetry_file, fig_dir=out_fig_dir)

    if args.generate_qualitative or not (out_fig_dir / "fig_qualitative_detections.pdf").exists():
        print("\n [TASK 5] Xuất Biểu đồ Định tính Thực nghiệm (Qualitative Visualizations 6-panel)...")
        try:
            from paper_2_phantom_fabrication.experiments.generate_qualitative_figure import (
                generate_paper2_qualitative_figure,
            )
            generate_paper2_qualitative_figure(
                image_path=None,
                output_dir=out_fig_dir,
                device_str=str(args.device) if hasattr(args, "device") and args.device else None,
            )
        except Exception as e:
            print(f"[CẢNH BÁO] Không thể sinh hình định tính tự động: {e}")
    else:
        print("\n [TASK 5] Hình định tính fig_qualitative_detections.pdf đã tồn tại chuẩn. Bỏ qua tái tạo.")

    print(f"\n[HOÀN TẤT TASK 5] Đã xuất toàn bộ biểu đồ vector tại: {out_fig_dir}")


if __name__ == "__main__":
    main()
