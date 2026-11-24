# -*- coding: utf-8 -*-
"""
Module: paper_1_aqsm_mechanism.experiments.tasks.task9_generate_tables
Mục đích: Tự động kết xuất toàn bộ 13 Bảng biểu thực nghiệm chuẩn xuất bản Elsevier EAAI
từ 100% dữ liệu output đầu ra (JSON trong summary_json/), không fix cứng số liệu,
và tự động đồng bộ lại Data Provenance Manifest (data_provenance_manifest.json).
"""

import sys
from pathlib import Path

WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from paper_1_aqsm_mechanism.tools.generate_publication_tables import main as gen_tables_main
from paper_1_aqsm_mechanism.tools.generate_data_provenance_manifest import generate_provenance_manifest
from paper_1_aqsm_mechanism.experiments.config import OUTPUTS_DIR, SUMMARY_JSON_DIR


def main():
    # 1. Sinh toàn bộ 13 bảng biểu LaTeX và CSV
    gen_tables_main()

    # 2. Tự động cập nhật Data Provenance Manifest từ dữ liệu thực nghiệm mới
    print("\n>>> [TASK 9] Tự động cập nhật Data Provenance Manifest từ summary_json...")
    try:
        generate_provenance_manifest(SUMMARY_JSON_DIR, OUTPUTS_DIR / "data_provenance_manifest.json")
    except Exception as e:
        print(f"[CẢNH BÁO] Không thể cập nhật manifest trong Task 9: {e}")


if __name__ == "__main__":
    main()
