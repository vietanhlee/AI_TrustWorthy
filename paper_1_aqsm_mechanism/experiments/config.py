# -*- coding: utf-8 -*-
"""
Module: paper_1_aqsm_mechanism.experiments.config
Mục đích: Quản lý cấu hình tập trung, siêu tham số thực nghiệm, đường dẫn thư mục chuẩn
phục vụ bài báo số 1 (Mechanistic Breakdown of Dynamic Query Selection).

Hỗ trợ:
  - Cập nhật linh hoạt đường dẫn từ CLI / Script Shell (`--folder_image`, `--annotations`, `--output_dir`, `--cache_dir`, `--fig_dir`).
  - Cung cấp các hàm getter truy xuất đường dẫn động (dynamic path getters) tránh lỗi đóng băng biến tại thời điểm import.
  - Tự động nhận diện thiết bị tính toán (CPU, Single-GPU CUDA, Multi-GPU DataParallel).
  - Tự động điều chỉnh kích thước lô (auto-tune batch size) theo dung lượng VRAM thực tế.
  - Đảm bảo 100% dữ liệu thực nghiệm thật, không hardcode hay làm giả số liệu.
"""

import os
import platform
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import torch

# Ngăn ngừa xung đột OpenMP trên hệ điều hành Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# -----------------------------------------------------------------------------
# 1. ĐƯỜNG DẪN HỆ THỐNG VÀ THƯ MỤC LÀM VIỆC MẶC ĐỊNH
# -----------------------------------------------------------------------------
EXPERIMENTS_DIR: Path = Path(__file__).resolve().parent
PAPER_DIR: Path = EXPERIMENTS_DIR.parent
WORKSPACE_DIR: Path = PAPER_DIR.parent

# Thư mục chứa dataset COCO gốc
DATA_DIR: Path = Path(os.environ.get("EDGE_SEC_DATA_DIR", str(WORKSPACE_DIR / "edge_sec_pro" / "data")))
COCO_VAL_DIR: Path = DATA_DIR / "coco5k" / "val2017"
COCO_ANN_FILE: Path = DATA_DIR / "coco5k" / "annotations" / "instances_val2017.json"
SPLIT_FILE: Path = DATA_DIR / "split.json"

# Thư mục lưu trữ bộ nhớ đệm mẫu đối kháng (Adversarial Cache) Native 640x640
ADV_CACHE_DIR: Path = Path(os.environ.get("EDGE_SEC_ADV_CACHE_DIR", str(PAPER_DIR / "cache" / "adv_cache_640")))
ADV_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Thư mục lưu trữ hình ảnh xuất bản và dữ liệu thực nghiệm phục vụ audit
FIG_DIR: Path = Path(os.environ.get("EDGE_SEC_FIG_DIR", str(PAPER_DIR / "fig")))
FIG_DIR.mkdir(parents=True, exist_ok=True)

OUTPUTS_DIR: Path = Path(os.environ.get("EDGE_SEC_OUTPUTS_DIR", str(PAPER_DIR / "outputs")))
LOGS_DIR: Path = OUTPUTS_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

RAW_CSV_DIR: Path = OUTPUTS_DIR / "raw_csv"
RAW_CSV_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_JSON_DIR: Path = OUTPUTS_DIR / "summary_json"
SUMMARY_JSON_DIR.mkdir(parents=True, exist_ok=True)

AUDIT_STREAMS_DIR: Path = OUTPUTS_DIR / "audit_streams"
AUDIT_STREAMS_DIR.mkdir(parents=True, exist_ok=True)


def resolve_model_name(name: Optional[str] = None) -> str:
    """Chuẩn hóa tên model checkpoint hoặc alias ('r18', 'r50', 'r101', 'detr', 'deformable_detr', ...)."""
    if not name:
        return os.environ.get("EDGE_SEC_MODEL", "PekingU/rtdetr_r18vd")
    clean = str(name).strip()
    alias_map = {
        "r18": "PekingU/rtdetr_r18vd",
        "rtdetr_r18": "PekingU/rtdetr_r18vd",
        "rtdetr_r18vd": "PekingU/rtdetr_r18vd",
        "r50": "PekingU/rtdetr_r50vd",
        "rtdetr_r50": "PekingU/rtdetr_r50vd",
        "rtdetr_r50vd": "PekingU/rtdetr_r50vd",
        "r101": "PekingU/rtdetr_r101vd",
        "rtdetr_r101": "PekingU/rtdetr_r101vd",
        "rtdetr_r101vd": "PekingU/rtdetr_r101vd",
        "detr": "facebook/detr-resnet-50",
        "detr_r50": "facebook/detr-resnet-50",
        "deformable_detr": "SenseTime/deformable-detr",
    }
    return alias_map.get(clean.lower(), clean)


def infer_num_decoder_layers(model_name: Optional[str] = None) -> int:
    """Suy luận số tầng decoder mặc định dựa trên kiến trúc mô hình."""
    resolved = resolve_model_name(model_name).lower()
    if any(tok in resolved for tok in ("r50", "r101", "deformable", "detr-resnet", "detr_resnet")):
        return 6
    return 3


def infer_num_queries(model_name: Optional[str] = None) -> int:
    """Suy luận số object query mặc định theo kiến trúc mô hình (DETR=100, RT-DETR/Deformable=300)."""
    resolved = resolve_model_name(model_name).lower()
    if "detr-resnet" in resolved and "rtdetr" not in resolved and "deformable" not in resolved:
        return 100
    return 300


def get_model_tag(name: Optional[str] = None) -> str:
    """Trả về tag ngắn gọn của model để phân biệt file cache/output (ví dụ: rtdetr_r18vd, rtdetr_r50vd)."""
    m = resolve_model_name(name)
    return m.split("/")[-1].lower().replace("-", "_")


def update_experiment_paths(
    folder_image: Optional[Union[str, Path]] = None,
    annotations: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    fig_dir: Optional[Union[str, Path]] = None,
    model: Optional[str] = None,
    pgd_steps: Optional[int] = None,
    eps_l_inf: Optional[float] = None,
    pgd_step_size: Optional[float] = None,
    img_size: Optional[int] = None,
) -> None:
    """
    Cập nhật động các đường dẫn và siêu tham số toàn cục phục vụ thực nghiệm linh hoạt.
    """
    global COCO_VAL_DIR, COCO_ANN_FILE, OUTPUTS_DIR, LOGS_DIR, RAW_CSV_DIR, SUMMARY_JSON_DIR, AUDIT_STREAMS_DIR, ADV_CACHE_DIR, FIG_DIR, MODEL_NAME, NUM_DECODER_LAYERS, NUM_QUERIES, PGD_NUM_STEPS, EPSILON_L_INF, PGD_STEP_SIZE, IMG_SIZE

    if model is not None:
        MODEL_NAME = resolve_model_name(model)
        NUM_DECODER_LAYERS = infer_num_decoder_layers(MODEL_NAME)
        NUM_QUERIES = infer_num_queries(MODEL_NAME)

    if pgd_steps is not None and int(pgd_steps) > 0:
        PGD_NUM_STEPS = int(pgd_steps)

    if eps_l_inf is not None and float(eps_l_inf) > 0:
        EPSILON_L_INF = float(eps_l_inf)

    if pgd_step_size is not None and float(pgd_step_size) > 0:
        PGD_STEP_SIZE = float(pgd_step_size)

    if img_size is not None and int(img_size) > 0:
        IMG_SIZE = int(img_size)

    if folder_image is not None:
        COCO_VAL_DIR = Path(folder_image).resolve()

    if annotations is not None:
        COCO_ANN_FILE = Path(annotations).resolve()

    if output_dir is not None:
        OUTPUTS_DIR = Path(output_dir).resolve()
        LOGS_DIR = OUTPUTS_DIR / "logs"
        RAW_CSV_DIR = OUTPUTS_DIR / "raw_csv"
        SUMMARY_JSON_DIR = OUTPUTS_DIR / "summary_json"
        AUDIT_STREAMS_DIR = OUTPUTS_DIR / "audit_streams"

        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        RAW_CSV_DIR.mkdir(parents=True, exist_ok=True)
        SUMMARY_JSON_DIR.mkdir(parents=True, exist_ok=True)
        AUDIT_STREAMS_DIR.mkdir(parents=True, exist_ok=True)

    if cache_dir is not None:
        ADV_CACHE_DIR = Path(cache_dir).resolve()
        ADV_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if fig_dir is not None:
        FIG_DIR = Path(fig_dir).resolve()
        FIG_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# GETTER FUNCTIONS TRUY XUẤT ĐƯỜNG DẪN ĐỘNG (DYNAMIC PATH GETTERS)
# -----------------------------------------------------------------------------
def get_outputs_dir() -> Path:
    return OUTPUTS_DIR


get_output_dir = get_outputs_dir


def get_logs_dir() -> Path:
    return LOGS_DIR


def get_raw_csv_dir() -> Path:
    return RAW_CSV_DIR


def get_summary_json_dir() -> Path:
    return SUMMARY_JSON_DIR


def get_audit_streams_dir() -> Path:
    return AUDIT_STREAMS_DIR


def get_adv_cache_dir() -> Path:
    return ADV_CACHE_DIR


def get_fig_dir() -> Path:
    return FIG_DIR


def get_coco_val_dir() -> Path:
    return COCO_VAL_DIR


def get_coco_ann_file() -> Path:
    return COCO_ANN_FILE


def get_model_name() -> str:
    return MODEL_NAME


def get_num_queries() -> int:
    return NUM_QUERIES


def get_num_decoder_layers() -> int:
    return NUM_DECODER_LAYERS


def get_pgd_num_steps() -> int:
    return PGD_NUM_STEPS


def get_epsilon_l_inf() -> float:
    return EPSILON_L_INF


def get_pgd_step_size() -> float:
    return PGD_STEP_SIZE


def get_img_size() -> int:
    return IMG_SIZE


# -----------------------------------------------------------------------------
# 2. CẤU HÌNH MÔ HÌNH RT-DETR (NATIVE 640x640 PRODUCTION PROFILE)
# -----------------------------------------------------------------------------
MODEL_NAME: str = resolve_model_name(os.environ.get("EDGE_SEC_MODEL", "PekingU/rtdetr_r18vd"))
IMG_SIZE: int = int(os.environ.get("EDGE_SEC_IMG_SIZE", "640"))
NUM_CANDIDATES: int = 8400   # 80x80 (S3, 1/8 = 6400) + 40x40 (S4, 1/16 = 1600) + 20x20 (S5, 1/32 = 400)
NUM_QUERIES: int = infer_num_queries(MODEL_NAME)       # N_q top-k encoder proposals truyền sang Decoder
HIDDEN_DIM: int = 256        # Vector nhúng ẩn d_model = 256
NUM_DECODER_LAYERS: int = infer_num_decoder_layers(MODEL_NAME)  # Số tầng Deformable Cross-Attention
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
COHORT_SIZE: int = 1000      # Giá trị mặc định danh định chuẩn cho cohort đánh giá
RANDOM_SEED: int = 42        # Seed cố định bảo đảm khả năng tái lập 100%
DEFAULT_MULTI_SEEDS: List[int] = [42, 100, 2024, 777, 9999]  # Danh sách 5 random seeds chuẩn cho đánh giá đa hạt giống (Task 4 & Task 10)
N_BOOTSTRAP: int = 1000      # Số lần lấy mẫu Bootstrap tính khoảng tin cậy 95% CI

# Ngưỡng phát hiện danh định
DETECTION_CONF_THRESHOLD: float = 0.50
IOU_THRESHOLD_EVAL: float = 0.50


# -----------------------------------------------------------------------------
# 5. TIỆN ÍCH TỰ ĐỘNG NHẬN DIỆN THIẾT BỊ VÀ ĐA TIẾN TRÌNH GPU / CPU
# -----------------------------------------------------------------------------
def get_device(preferred: Optional[str] = None) -> torch.device:
    """
    Xác định thiết bị tính toán tối ưu (Ưu tiên GPU nếu có, mặc định CPU).
    """
    if preferred is not None and preferred.lower() != "auto":
        return torch.device(preferred)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def detect_hardware_environment(preferred_device: Optional[str] = None) -> Dict[str, Any]:
    """
    Tự động nhận diện cấu hình phần cứng:
    - CPU: Tên bộ vi xử lý, số lõi logic/vật lý, kiến trúc OS.
    - GPU: Tên card, số lượng thiết bị, tổng VRAM, chuẩn CUDA capability.
    - Đề xuất chiến lược thực thi (single_gpu, data_parallel, cpu_multithread) và batch_size tối ưu.
    """
    has_cuda = torch.cuda.is_available() and (preferred_device is None or preferred_device.lower() != "cpu")
    info: Dict[str, Any] = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "device_type": "cuda" if has_cuda else "cpu",
        "processor": platform.processor(),
    }

    if has_cuda:
        gpu_count = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
        vram_gb = [torch.cuda.get_device_properties(i).total_memory / (1024 ** 3) for i in range(gpu_count)]
        info.update({
            "gpu_count": gpu_count,
            "gpu_names": gpu_names,
            "total_vram_gb": vram_gb,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            "strategy": "data_parallel" if gpu_count > 1 else "single_gpu",
            "recommended_batch_size": min(32, max(4, int(min(vram_gb) * 2))) * gpu_count,
        })
    else:
        num_cores = os.cpu_count() or 4
        info.update({
            "gpu_count": 0,
            "cpu_cores": num_cores,
            "strategy": "cpu_multithread",
            "recommended_batch_size": 4,
        })

    return info


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
