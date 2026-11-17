# -*- coding: utf-8 -*-
"""
Module: paper_1_aqsm_mechanism.experiments.tasks.task5_generate_figures
Mục đích: Task tổng hợp đọc các file JSON thực nghiệm và tự động xuất các biểu đồ vector PDF/PNG (Figure 3, 4, 5).
"""

import logging
import sys
from pathlib import Path

WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from paper_1_aqsm_mechanism.experiments.config import LOGS_DIR
from paper_1_aqsm_mechanism.experiments.generate_figures_and_tables import generate_all_figures_and_tables


def main():
    log_path = LOGS_DIR / "task5_figures.log"
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][Task 5][%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
    )
    logging.info("Khởi chạy Task 5: Sinh toàn bộ Figures vector PDF/PNG...")
    generate_all_figures_and_tables()
    logging.info("Hoàn thành Task 5 thành công!")


if __name__ == "__main__":
    main()
