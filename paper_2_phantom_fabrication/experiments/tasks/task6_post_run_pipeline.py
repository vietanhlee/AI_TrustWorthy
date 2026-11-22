# -*- coding: utf-8 -*-
"""
Module: paper_2_phantom_fabrication.experiments.tasks.task6_post_run_pipeline
Mục đích: CLI Task 6 - Tự động hóa toàn diện chuỗi hậu xử lý thực nghiệm khoa học Paper 2:
  1. Chuẩn hóa & lưu trữ các kiểm định ý nghĩa thống kê (df=2, Student's t hai phía).
  2. Xuất báo cáo tổng kết thực nghiệm toàn diện chuẩn Markdown (EXPERIMENT_RESULTS_REPORT.md).
  3. Xuất bảng tham số macro LaTeX (paper2_metrics.tex).
  4. Kiểm toán tự động tính nhất quán 100% giữa số liệu telemetry và mã nguồn bản thảo TeX.
  5. Tự động biên dịch bản thảo sang PDF qua pdflatex & bibtex, kiểm tra đạt CHÍNH XÁC 24 TRANG.
  6. Tự động sao chép tệp PDF nghiệm thu sang SUBMISSION_PDFS/.
  7. Khởi chạy bộ kiểm thử tự động pytest để đảm bảo chất lượng kỹ thuật chuẩn production.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
from pathlib import Path

WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from paper_2_phantom_fabrication.core.config import OUTPUTS_DIR, RESULTS_DIR
from paper_2_phantom_fabrication.experiments.engines.post_run_engine import run_post_experiment_pipeline
from paper_2_phantom_fabrication.utils.cli import create_base_parser


def main():
    parser = create_base_parser("Post-Run Scientific Pipeline (Report, Sync, Paper, Verify)", task_name="Task 6")
    parser.add_argument(
        "--telemetry_path",
        type=str,
        default="",
        help="Đường dẫn file JSON telemetry đầy đủ. Nếu trống, tự động tìm kiếm.",
    )
    parser.add_argument(
        "--skip_paper_compile",
        action="store_true",
        default=False,
        help="Bỏ qua biên dịch pdflatex (ví dụ khi chạy trên môi trường cloud không cài pdflatex).",
    )
    parser.add_argument(
        "--expected_pages",
        type=int,
        default=24,
        help="Số trang kỳ vọng của bản thảo nộp bài Elsevier EAAI (mặc định: 24 trang).",
    )
    args = parser.parse_args()

    # Tìm kiếm file telemetry
    telemetry_file: Path
    if args.telemetry_path and Path(args.telemetry_path).exists():
        telemetry_file = Path(args.telemetry_path).resolve()
    else:
        candidates = [
            Path(args.output_dir).resolve() / "paper2_complete_telemetry.json",
            RESULTS_DIR / "paper2_complete_telemetry.json",
            Path(args.output_dir).resolve() / "stage1_benchmark_telemetry.json",
        ]
        found = None
        for c in candidates:
            if c.exists():
                found = c
                break
        telemetry_file = found if found is not None else (RESULTS_DIR / "paper2_complete_telemetry.json")

    paper2_dir = Path(__file__).resolve().parent.parent.parent
    project_root = paper2_dir.parent
    submission_dir = project_root / "SUBMISSION_PDFS"

    success = run_post_experiment_pipeline(
        telemetry_path=telemetry_file,
        paper2_dir=paper2_dir,
        submission_dir=submission_dir,
        project_root=project_root,
        skip_compile=args.skip_paper_compile,
        expected_pages=args.expected_pages,
    )

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
