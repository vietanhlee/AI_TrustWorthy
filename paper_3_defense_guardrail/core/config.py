# -*- coding: utf-8 -*-
"""
Module: edge_sec_pro.core.config
Mục đích: Quản lý cấu hình tập trung, siêu tham số và đường dẫn thư mục chuẩn production
cho toàn bộ hệ sinh thái nghiên cứu và thực thi bảo mật trên Real-Time DETR (RT-DETR).

Tất cả các hằng số về kiến trúc mô hình, giao thức tấn công đối kháng 4 tiers,
tham số hiệu chuẩn phân vị không rò rỉ dữ liệu (Zero Data Leakage), và ngưỡng
giám sát suy luận thời gian thực trên phần cứng biên đều được định nghĩa tại đây.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
import torch

# Ngăn ngừa lỗi xung đột OpenMP giữa PyTorch và các thư viện C++ trên Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# -----------------------------------------------------------------------------
# 1. ĐƯỜNG DẪN HỆ THỐNG VÀ THƯ MỤC LÀM VIỆC
# -----------------------------------------------------------------------------
CORE_DIR: Path = Path(__file__).resolve().parent
PACKAGE_DIR: Path = CORE_DIR.parent
WORKSPACE_DIR: Path = PACKAGE_DIR.parent

DATA_DIR: Path = PACKAGE_DIR / "data" if (PACKAGE_DIR / "data").exists() else (WORKSPACE_DIR / "edge_sec_pro" / "data")
COCO_VAL_DIR: Path = DATA_DIR / "coco5k" / "val2017"
SPLIT_FILE: Path = DATA_DIR / "split.json"

RESULTS_DIR: Path = PACKAGE_DIR / "results"
CHECKPOINTS_DIR: Path = RESULTS_DIR / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# 2. CẤU HÌNH KIẾN TRÚC MÔ HÌNH RT-DETR (EDGE PROFILE)
# -----------------------------------------------------------------------------
MODEL_NAME: str = os.environ.get("EDGE_SEC_MODEL", "PekingU/rtdetr_r18vd")
IMG_SIZE: int = int(os.environ.get("EDGE_SEC_IMG_SIZE", "640"))
NUM_CANDIDATES: int = 8400   # 80x80 (S3) + 40x40 (S4) + 20x20 (S5)
NUM_QUERIES: int = 300       # Số queries top-k được chọn đưa vào Deformable Decoder
HIDDEN_DIM: int = 256        # Kích thước vector ẩn của Deformable Decoder
PROJECTION_DIM: int = 64     # Chiều chiếu trực giao cho Layer Mahalanobis
NUM_DECODER_LAYERS: int = 3  # Số tầng Deformable Cross-Attention của RT-DETR-R18

# -----------------------------------------------------------------------------
# 3. THAM SỐ GIAO THỨC TẤN CÔNG ĐỐI KHÁNG (THREAT TAXONOMY - 4 TIERS)
# -----------------------------------------------------------------------------
# Ngân sách nhiễu chuẩn L_inf cho tấn công số hóa
EPSILON_L_INF: float = float(os.environ.get("EDGE_SEC_EPS", str(16.0 / 255.0)))
PGD_STEP_SIZE: float = float(os.environ.get("EDGE_SEC_ALPHA", str(2.0 / 255.0)))
PGD_NUM_STEPS: int = int(os.environ.get("EDGE_SEC_STEPS", "15"))

# Kích thước Adversarial Patch cục bộ (Tier 0)
PATCH_SIZE: int = int(os.environ.get("EDGE_SEC_PATCH_SIZE", "48"))

# Tham số hàm mất mát phân hạng bù trừ AQSM (Tier 1) - đồng bộ worker_b1_aqsm
AQSM_RANKING_MARGIN: float = 0.25
AQSM_LAMBDA_RANK: float = 1.20
AQSM_LAMBDA_SUPP: float = 0.50

# Tham số tấn công thích ứng né tránh (Tier 3)
ADAPTIVE_LAMBDA_1D: float = 0.50
ADAPTIVE_LAMBDA_JOINT: float = 0.50
ADAPTIVE_LAMBDA_ESM: float = 2.00
ADAPTIVE_LAMBDA_ALL: float = 1.00

# -----------------------------------------------------------------------------
# 4. THAM SỐ GIÁM SÁT SUY LUẬN & HIỆU CHUẨN PHÂN VỊ (GUARDRAILS & CALIBRATION)
# -----------------------------------------------------------------------------
# Ngân sách False Positive Rate (FPR) danh định
TARGET_FPR: float = 0.05

# Kích thước tập hiệu chuẩn và kiểm thử độc lập (Zero Leakage)
N_CALIBRATION_SAMPLES: int = 1200
N_TEST_CLEAN_SAMPLES: int = 600

# Tham số hàm kích hoạt mềm giải quyết Discrete Tie Artifact (Tier 2 ActiveQ)
SOFT_ACTIVATION_TEMP: float = 20.0
SOFT_ACTIVATION_TAU: float = 0.35

# Tham số điều chuẩn co ngót ma trận hiệp phương sai Ledoit-Wolf (Tier 3 Mahalanobis)
SHRINKAGE_BETA: float = 0.05

# Ngưỡng phát hiện vật thể danh định
DETECTION_CONF_THRESHOLD: float = 0.50

# -----------------------------------------------------------------------------
# 5. TIỆN ÍCH THIẾT BỊ VÀ ĐA LUỒNG
# -----------------------------------------------------------------------------
def get_device(preferred: Optional[str] = None) -> torch.device:
    """
    Xác định thiết bị tính toán tối ưu (CUDA GPU hoặc Edge CPU).
    Mặc định ưu tiên CUDA nếu có sẵn, hoặc cho phép ép kiểu chạy CPU để đo latency chuẩn.
    """
    if preferred is not None:
        return torch.device(preferred)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def configure_runtime_threads(num_threads: int = 4) -> None:
    """
    Cấu hình số luồng tính toán song song cho PyTorch trên CPU biên.
    """
    torch.set_num_threads(num_threads)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(num_threads)
        except RuntimeError:
            pass
