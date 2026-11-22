# -*- coding: utf-8 -*-
"""
Module: paper_1_aqsm_mechanism.experiments.tasks.task9_generate_tables
Mục đích: Tự động kết xuất toàn bộ 12 Bảng biểu thực nghiệm chuẩn xuất bản Elsevier EAAI
từ 100% dữ liệu output đầu ra (JSON trong summary_json/), không fix cứng số liệu.
"""

import sys
from pathlib import Path

WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from paper_1_aqsm_mechanism.tools.generate_publication_tables import main

if __name__ == "__main__":
    main()
