# -*- coding: utf-8 -*-
"""
Module: paper_1_aqsm_mechanism.experiments.tasks.task9_generate_tables
Mục đích: Tự động kết xuất toàn bộ 13 Bảng biểu thực nghiệm chuẩn xuất bản Elsevier EAAI
từ 100% dữ liệu output đầu ra (JSON trong summary_json/), không fix cứng số liệu,
và tự động đồng bộ lại Data Provenance Manifest (data_provenance_manifest.json).
"""

import argparse
import sys
from pathlib import Path

WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from paper_1_aqsm_mechanism.tools.generate_publication_tables import main as gen_tables_main
from paper_1_aqsm_mechanism.tools.generate_data_provenance_manifest import generate_provenance_manifest
from paper_1_aqsm_mechanism.experiments.config import (
    get_outputs_dir,
    get_summary_json_dir,
    update_experiment_paths,
)


def main():
    parser = argparse.ArgumentParser(description="[Task 9] Kết xuất toàn bộ 13 bảng biểu và cập nhật provenance manifest.")
    parser.add_argument("--output_dir", type=str, default=None, help="Thư mục output chứa summary_json/")
    args, _ = parser.parse_known_args()

    if args.output_dir is not None:
        update_experiment_paths(output_dir=args.output_dir)

    # 1. Sinh toàn bộ 13 bảng biểu LaTeX và CSV
    gen_tables_main()

    # 2. Tự động cập nhật Data Provenance Manifest từ dữ liệu thực nghiệm mới
    summary_dir = get_summary_json_dir()
    outputs_dir = get_outputs_dir()
    print(f"\n>>> [TASK 9] Tự động cập nhật Data Provenance Manifest từ {summary_dir}...")
    try:
        generate_provenance_manifest(summary_dir, outputs_dir / "data_provenance_manifest.json")
    except Exception as e:
        print(f"[CẢNH BÁO] Không thể cập nhật manifest trong Task 9: {e}")


if __name__ == "__main__":
    main()
