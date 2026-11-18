#!/usr/bin/env bash
# ==============================================================================
# Master Execution Script for Paper 2: Phantom Queries Fabrication
# Supports: Linux / Mac / Google Colab / Kaggle (CPU, Single-GPU, Multi-GPU)
# ==============================================================================

set -e

# Default settings
FOLDER_IMAGE=""
ANNOTATIONS=""
NUM_BENCHMARK=15
NUM_ABLATION=300
SEEDS="42,123,456"
DEVICE="auto"
BATCH_SIZE=0
STAGE="all"
OUTPUT_JSON="results/paper2_complete_telemetry.json"

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
    echo "  --num_benchmark INT        Số ảnh mỗi seed cho Stage 1 (mặc định: 15)"
    echo "  --num_ablation INT         Số ảnh cho Stage 2 Ablations (mặc định: 300)"
    echo "  --seeds LIST               Danh sách seed cách nhau bởi dấu phẩy (mặc định: '42,123,456')"
    echo "  --device DEV               Thiết bị: auto (mặc định), cuda, cpu"
    echo "  --batch_size INT           Batch size suy luận (0 = tự động tính theo VRAM)"
    echo "  --stage STAGE              Giai đoạn: all (mặc định), benchmark, ablations, figures"
    echo "  --output_json PATH         Đường dẫn file JSON đầu ra (mặc định: results/paper2_complete_telemetry.json)"
    echo "  -h, --help                 Hiển thị hướng dẫn này"
    echo "======================================================================"
}

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --folder_image) FOLDER_IMAGE="$2"; shift ;;
        --annotations) ANNOTATIONS="$2"; shift ;;
        --num_benchmark) NUM_BENCHMARK="$2"; shift ;;
        --num_ablation) NUM_ABLATION="$2"; shift ;;
        --seeds) SEEDS="$2"; shift ;;
        --device) DEVICE="$2"; shift ;;
        --batch_size) BATCH_SIZE="$2"; shift ;;
        --stage) STAGE="$2"; shift ;;
        --output_json) OUTPUT_JSON="$2"; shift ;;
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

# Ensure working directory is script directory or project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================================================"
echo " KHỞI ĐỘNG HỆ THỐNG THỰC NGHIỆM PAPER 2"
echo "----------------------------------------------------------------------"
echo " Thư mục ảnh COCO:   $FOLDER_IMAGE"
if [ -n "$ANNOTATIONS" ]; then
    echo " File Annotations:   $ANNOTATIONS"
else
    echo " File Annotations:   [Tự động tìm kiếm trong thư mục COCO]"
fi
echo " Chế độ thiết bị:    $DEVICE"
echo " Số mẫu Benchmark:   $NUM_BENCHMARK ảnh/seed (x 3 seeds)"
echo " Số mẫu Ablation:    $NUM_ABLATION ảnh"
echo " Giai đoạn thực thi: $STAGE"
echo " File Telemetry:     $OUTPUT_JSON"
echo "======================================================================"

# Check Python environment
PYTHON_CMD="python"
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
fi

# Run the master python suite
$PYTHON_CMD experiments/run_full_paper2_suite.py \
    --folder_image "$FOLDER_IMAGE" \
    ${ANNOTATIONS:+--annotations "$ANNOTATIONS"} \
    --num_images_benchmark "$NUM_BENCHMARK" \
    --num_images_ablation "$NUM_ABLATION" \
    --seeds "$SEEDS" \
    --device "$DEVICE" \
    --batch_size "$BATCH_SIZE" \
    --stage "$STAGE" \
    --output_json "$OUTPUT_JSON"

echo ""
echo "======================================================================"
echo " TẤT CẢ THỰC NGHIỆM ĐÃ HOÀN TẤT THÀNH CÔNG!"
echo "----------------------------------------------------------------------"
echo " 1. File Telemetry chứa 100% dữ liệu thực nghiệm đã sẵn sàng tại:"
echo "    -> $OUTPUT_JSON"
echo ""
echo " 2. Toàn bộ 4 đồ thị xuất bản vector PDF đã được lưu tại:"
echo "    -> fig/fig1_nms_free_paradox.pdf"
echo "    -> fig/fig2_displacement_tradeoff.pdf"
echo "    -> fig/fig3_downstream_cyberphysical_impact.pdf"
echo "    -> fig/fig4_sqd_filter_defense.pdf"
echo ""
echo " 3. Bước tiếp theo:"
echo "    Chỉ cần gửi file $OUTPUT_JSON cho Antigravity hoặc chạy:"
echo "    python experiments/sync_telemetry_to_paper.py --telemetry $OUTPUT_JSON"
echo "    để tự động cập nhật toàn bộ Bảng 1-4 và biên dịch bản thảo PDF!"
echo "======================================================================"
