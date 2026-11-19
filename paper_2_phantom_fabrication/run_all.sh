#!/usr/bin/env bash
# ==============================================================================
# Master Execution Script for Paper 2: Phantom Queries Fabrication
# Supports: Linux / Mac / Google Colab / Kaggle (CPU, Single-GPU, Multi-GPU)
# 100% Backward Compatible with Legacy Flags & Zero-OOM Subtask Architecture
# ==============================================================================

set -e

# Default settings
FOLDER_IMAGE=""
ANNOTATIONS=""
NUM_BENCHMARK=0
NUM_ABLATION=0
NUM_IMAGES=0
SEEDS="42,123,456"
DEVICE="auto"
BATCH_SIZE=0
STAGE="all"
TASKS="0,1,2,3,4,5"
OUTPUT_DIR="outputs"
OUTPUT_JSON="outputs/paper2_complete_telemetry.json"
CACHE_DIR="cache/adv_cache_640"
FIG_DIR="fig"
SKIP_CACHE_GEN=""
FORCE_RECOMPUTE=""

print_usage() {
    echo "======================================================================"
    echo "  PAPER 2: PHANTOM QUERIES FABRICATION - EMPIRICAL SUITE RUNNER"
    echo "======================================================================"
    echo "Usage:"
    echo "  bash run_all.sh --folder_image <path_to_coco_images> [OPTIONS]"
    echo ""
    echo "Required:"
    echo "  --folder_image PATH        Đường dẫn thư mục ảnh COCO (e.g. val2017)"
    echo ""
    echo "Optional:"
    echo "  --annotations PATH         Đường dẫn file instances_val2017.json"
    echo "                             (Tự động tìm kiếm nếu bỏ trống)"
    echo "  --tasks LIST               Danh sách tasks (0,1,2,3,4,5 hoặc all)"
    echo "  --stage STAGE              Giai đoạn: all (mặc định), benchmark, ablations, figures"
    echo "  --num_images INT           Số ảnh giới hạn (0 = chạy toàn bộ ảnh)"
    echo "  --num_benchmark INT        Số ảnh mỗi seed cho Stage 1 (Legacy)"
    echo "  --num_ablation INT         Số ảnh cho Stage 2 Ablations (Legacy)"
    echo "  --seeds LIST               Danh sách seed cách nhau bởi dấu phẩy ('42,123,456')"
    echo "  --device DEV               Thiết bị: auto (mặc định), cuda, cpu"
    echo "  --batch_size INT           Batch size suy luận (0 = tự động tính theo VRAM)"
    echo "  --output_dir PATH          Thư mục lưu trữ kết quả và log (mặc định: outputs)"
    echo "  --cache_dir PATH           Thư mục lưu trữ bộ nhớ đệm đối kháng (mặc định: cache/adv_cache_640)"
    echo "  --fig_dir PATH             Thư mục lưu biểu đồ xuất bản (mặc định: fig)"
    echo "  --skip_cache_gen           Chỉ dùng cache có sẵn, không sinh cache mới"
    echo "  --force_recompute          Bắt buộc tính toán lại và ghi đè cache"
    echo "  -h, --help                 Hiển thị hướng dẫn này"
    echo "======================================================================"
}

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --folder_image) FOLDER_IMAGE="$2"; shift ;;
        --annotations) ANNOTATIONS="$2"; shift ;;
        --tasks) TASKS="$2"; shift ;;
        --stage) STAGE="$2"; shift ;;
        --num_images) NUM_IMAGES="$2"; shift ;;
        --num_benchmark) NUM_BENCHMARK="$2"; shift ;;
        --num_ablation) NUM_ABLATION="$2"; shift ;;
        --seeds) SEEDS="$2"; shift ;;
        --device) DEVICE="$2"; shift ;;
        --batch_size) BATCH_SIZE="$2"; shift ;;
        --output_dir) OUTPUT_DIR="$2"; shift ;;
        --output_json) OUTPUT_JSON="$2"; shift ;;
        --cache_dir) CACHE_DIR="$2"; shift ;;
        --fig_dir) FIG_DIR="$2"; shift ;;
        --skip_cache_gen) SKIP_CACHE_GEN="--skip_cache_gen" ;;
        --force_recompute) FORCE_RECOMPUTE="--force_recompute" ;;
        -h|--help) print_usage; exit 0 ;;
        *) echo "Tham số không nhận dạng: $1"; print_usage; exit 1 ;;
    esac
    shift
done

if [ -z "$FOLDER_IMAGE" ]; then
    echo "LỖI: Thiếu tham số bắt buộc --folder_image!"
    print_usage
    exit 1
fi

# Chuyển đổi stage sang tasks nếu stage được cung cấp
if [ "$STAGE" == "benchmark" ]; then
    TASKS="0,1,4"
elif [ "$STAGE" == "ablations" ]; then
    TASKS="2,3"
elif [ "$STAGE" == "figures" ]; then
    TASKS="5"
fi

if [ "$NUM_IMAGES" -le 0 ]; then
    if [ "$NUM_BENCHMARK" -gt 0 ]; then
        NUM_IMAGES=$NUM_BENCHMARK
    elif [ "$NUM_ABLATION" -gt 0 ]; then
        NUM_IMAGES=$NUM_ABLATION
    fi
fi

# Ensure working directory is script directory or project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check Python environment
PYTHON_CMD="python"
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
fi

echo "======================================================================"
echo " KHỞI ĐỘNG HỆ THỐNG THỰC NGHIỆM PAPER 2 (ZERO-OOM ARCHITECTURE)"
echo "----------------------------------------------------------------------"
echo " Thư mục ảnh COCO : $FOLDER_IMAGE"
if [ -n "$ANNOTATIONS" ]; then
    echo " File Annotations : $ANNOTATIONS"
else
    echo " File Annotations : [Tự động tìm kiếm trong thư mục COCO]"
fi
echo " Thiết bị         : $DEVICE"
echo " Danh sách Tasks  : $TASKS (Stage: $STAGE)"
echo " Số mẫu ảnh       : $NUM_IMAGES (0 = Toàn bộ ảnh)"
echo " Thư mục Cache    : $CACHE_DIR (Disk-backed memmap)"
echo " Thư mục Output   : $OUTPUT_DIR"
echo " Thư mục Figures  : $FIG_DIR"
echo "======================================================================"

# Khởi chạy Master Orchestrator run_all_experiments.py
$PYTHON_CMD run_all_experiments.py \
    --folder_image "$FOLDER_IMAGE" \
    ${ANNOTATIONS:+--annotations "$ANNOTATIONS"} \
    --tasks "$TASKS" \
    --num_images "$NUM_IMAGES" \
    --seeds "$SEEDS" \
    --device "$DEVICE" \
    --batch_size "$BATCH_SIZE" \
    --cache_dir "$CACHE_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --fig_dir "$FIG_DIR" \
    $SKIP_CACHE_GEN \
    $FORCE_RECOMPUTE

echo ""
echo "======================================================================"
echo " TẤT CẢ THỰC NGHIỆM ĐÃ HOÀN TẤT THÀNH CÔNG!"
echo "----------------------------------------------------------------------"
echo " 1. Kết quả Telemetry hợp nhất:"
echo "    -> $OUTPUT_DIR/paper2_complete_telemetry.json"
echo ""
echo " 2. Toàn bộ 4 đồ thị xuất bản vector PDF đã được lưu tại:"
echo "    -> $FIG_DIR/fig1_nms_free_paradox.pdf"
echo "    -> $FIG_DIR/fig2_displacement_tradeoff.pdf"
echo "    -> $FIG_DIR/fig3_downstream_cyberphysical_impact.pdf"
echo "    -> $FIG_DIR/fig4_sqd_filter_defense.pdf"
echo "======================================================================"
