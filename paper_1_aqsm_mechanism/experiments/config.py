# -*- coding: utf-8 -*-
"""
Module: paper_1_aqsm_mechanism.experiments.config
Mục đích: Quản lý cấu hình tập trung, siêu tham số thực nghiệm, đường dẫn thư mục chuẩn
phục vụ bài báo số 1 (Mechanistic Breakdown of Dynamic Query Selection).

Tất cả các tham số về cohort thực nghiệm N=300 ảnh chuẩn COCO val2017,
ngân sách tấn công chuẩn hóa L_inf (eps=16/255, T=15 steps), giao thức can thiệp nhân quả 2x2,
và cấu trúc lưu trữ dữ liệu thô (raw CSV/JSON/logs) được định nghĩa tại đây.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
import torch

# Ngăn ngừa xung đột OpenMP trên hệ điều hành Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# -----------------------------------------------------------------------------
# 1. ĐƯỜNG DẪN HỆ THỐNG VÀ THƯ MỤC LÀM VIỆC
# -----------------------------------------------------------------------------
EXPERIMENTS_DIR: Path = Path(__file__).resolve().parent
PAPER_DIR: Path = EXPERIMENTS_DIR.parent
WORKSPACE_DIR: Path = PAPER_DIR.parent

# Thư mục chứa dataset COCO gốc
DATA_DIR: Path = Path(os.environ.get("EDGE_SEC_DATA_DIR", str(WORKSPACE_DIR / "edge_sec_pro" / "data")))
COCO_VAL_DIR: Path = DATA_DIR / "coco5k" / "val2017"
COCO_ANN_FILE: Path = DATA_DIR / "coco5k" / "annotations" / "instances_val2017.json"
SPLIT_FILE: Path = DATA_DIR / "split.json"

# Thư mục lưu trữ bộ nhớ đệm mẫu đối kháng (Adversarial Cache)
ADV_CACHE_DIR: Path = Path(os.environ.get("EDGE_SEC_ADV_CACHE_DIR", str(WORKSPACE_DIR / "edge_sec_pro" / "results" / "checkpoints" / "adv_cache")))

# Thư mục lưu trữ hình ảnh xuất bản và dữ liệu thực nghiệm phục vụ audit
FIG_DIR: Path = PAPER_DIR / "fig"
FIG_DIR.mkdir(parents=True, exist_ok=True)

OUTPUTS_DIR: Path = PAPER_DIR / "outputs"
LOGS_DIR: Path = OUTPUTS_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

RAW_CSV_DIR: Path = OUTPUTS_DIR / "raw_csv"
RAW_CSV_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_JSON_DIR: Path = OUTPUTS_DIR / "summary_json"
SUMMARY_JSON_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# 2. CẤU HÌNH MÔ HÌNH RT-DETR (EDGE EMBEDDED PROFILE)
# -----------------------------------------------------------------------------
MODEL_NAME: str = os.environ.get("EDGE_SEC_MODEL", "PekingU/rtdetr_r18vd")
IMG_SIZE: int = int(os.environ.get("EDGE_SEC_IMG_SIZE", "320"))
NUM_CANDIDATES: int = 2100   # 40x40 (S3, 1/8) + 20x20 (S4, 1/16) + 10x10 (S5, 1/32)
NUM_QUERIES: int = 300       # N_q = 300 top-k encoder proposals truyền sang Decoder
HIDDEN_DIM: int = 256        # Vector nhúng ẩn d_model = 256
NUM_DECODER_LAYERS: int = 3  # Số tầng Deformable Cross-Attention trong RT-DETR-R18
NUM_CLASSES: int = 80        # Số lớp COCO object classes

# -----------------------------------------------------------------------------
# 3. THAM SỐ GIAO THỨC TẤN CÔNG ĐỐI KHÁNG CHẨN ĐOÁN (DIAGNOSTIC THREAT MODEL)
# -----------------------------------------------------------------------------
# Ngân sách nhiễu chuẩn hóa L_inf
EPSILON_L_INF: float = float(os.environ.get("EDGE_SEC_EPS", str(16.0 / 255.0)))
PGD_STEP_SIZE: float = float(os.environ.get("EDGE_SEC_ALPHA", str(2.0 / 255.0)))
PGD_NUM_STEPS: int = int(os.environ.get("EDGE_SEC_STEPS", "15"))

# Tham số hàm mất mát phân hạng bù trừ AQSM (Adversarial Query Selection Manipulation)
AQSM_RANKING_MARGIN: float = 0.25   # gamma: lề phân hạng tối thiểu ép foreground tụt hạng
AQSM_LAMBDA_RANK: float = 1.20      # lambda_rank: trọng số phạt phân hạng
AQSM_LAMBDA_SUPP: float = 0.50      # lambda_supp: trọng số triệt tiêu độ tự tin tuyệt đối

# Ngân sách quét so sánh Matched-Budget
SWEEP_EPSILONS: List[float] = [4.0 / 255.0, 8.0 / 255.0, 16.0 / 255.0]

# -----------------------------------------------------------------------------
# 4. THAM SỐ ĐO ĐẠC THỰC NGHIỆM VÀ PHÂN TÍCH NHÂN QUẢ (CAUSAL & TELEMETRY PROTOCOL)
# -----------------------------------------------------------------------------
COHORT_SIZE: int = 300       # N = 300 ảnh COCO val2017 tiêu chuẩn
RANDOM_SEED: int = 42        # Seed cố định bảo đảm khả năng tái lập 100%
N_BOOTSTRAP: int = 1000      # Số lần lấy mẫu Bootstrap tính khoảng tin cậy 95% CI

# Ngưỡng phát hiện danh định
DETECTION_CONF_THRESHOLD: float = 0.50
IOU_THRESHOLD_EVAL: float = 0.50

# -----------------------------------------------------------------------------
# 5. TIỆN ÍCH THIẾT BỊ VÀ CẤU HÌNH ĐA LUỒNG CPU
# -----------------------------------------------------------------------------
def get_device(preferred: Optional[str] = None) -> torch.device:
    """
    Xác định thiết bị tính toán tối ưu (Ưu tiên GPU nếu có, mặc định CPU).
    """
    if preferred is not None:
        return torch.device(preferred)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def configure_cpu_threads(num_threads: int = 8) -> None:
    """
    Thiết lập số luồng tính toán song song tối đa cho PyTorch trên CPU đa lõi.
    """
    torch.set_num_threads(num_threads)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(num_threads)
        except RuntimeError:
            pass
