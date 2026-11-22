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
TASKS="0,1,2,3,4,5,6"
OUTPUT_DIR="outputs"
OUTPUT_JSON="outputs/paper2_complete_telemetry.json"
CACHE_DIR="cache/adv_cache_640"
FIG_DIR="fig"
SKIP_CACHE_GEN=""
FORCE_RECOMPUTE=""
SYNC_ONLY=""
SKIP_PAPER_COMPILE=""

print_usage() {
    echo "======================================================================"
    echo "  PAPER 2: PHANTOM QUERIES FABRICATION - EMPIRICAL SUITE RUNNER"
    echo "======================================================================"
    echo "Usage:"
    echo "  bash run_all.sh --folder_image <path_to_coco_images> [OPTIONS]"
    echo "  bash run_all.sh --sync_only"
    echo ""
    echo "Required (trừ khi có --sync_only):"
    echo "  --folder_image PATH        Đường dẫn thư mục ảnh COCO (e.g. val2017)"
    echo ""
    echo "Optional:"
    echo "  --annotations PATH         Đường dẫn file instances_val2017.json"
    echo "                             (Tự động tìm kiếm nếu bỏ trống)"
    echo "  --tasks LIST               Danh sách tasks (0,1,2,3,4,5,6 hoặc all)"
    echo "  --stage STAGE              Giai đoạn: all (mặc định), benchmark, ablations, figures, sync"
    echo "  --sync_only                Chỉ chạy hậu xử lý: Figures, Markdown Report, LaTeX, PDF 24 trang"
    echo "  --skip_paper_compile       Bỏ qua biên dịch PDF qua pdflatex (môi trường cloud)"
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
        --sync_only|--sync-only) SYNC_ONLY="--sync_only" ;;
        --skip_paper_compile|--skip-paper-compile) SKIP_PAPER_COMPILE="--skip_paper_compile" ;;
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

# Chuyển đổi stage sang tasks nếu stage được cung cấp
if [ -n "$SYNC_ONLY" ] || [ "$STAGE" == "figures" ] || [ "$STAGE" == "sync" ]; then
    TASKS="5,6"
    SYNC_ONLY="--sync_only"
elif [ "$STAGE" == "benchmark" ]; then
    TASKS="0,1,4,5,6"
elif [ "$STAGE" == "ablations" ]; then
    TASKS="2,3,5,6"
fi

if [ -z "$FOLDER_IMAGE" ] && [ -z "$SYNC_ONLY" ]; then
    if [ -d "../data/coco5k/val2017" ]; then
        FOLDER_IMAGE="../data/coco5k/val2017"
    elif [ -d "data/coco5k/val2017" ]; then
        FOLDER_IMAGE="data/coco5k/val2017"
    else
        echo "LỖI: Thiếu tham số bắt buộc --folder_image (trừ khi dùng --sync_only)!"
        print_usage
        exit 1
    fi
fi

if [ "$NUM_IMAGES" -le 0 ]; then
    if [ "$NUM_BENCHMARK" -gt 0 ]; then
        NUM_IMAGES=$NUM_BENCHMARK
    elif [ "$NUM_ABLATION" -gt 0 ]; then
        NUM_IMAGES=$NUM_ABLATION
    fi
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_CMD="python"
if [ -f "../.venv/Scripts/python.exe" ]; then
    PYTHON_CMD="../.venv/Scripts/python.exe"
elif [ -f ".venv/Scripts/python.exe" ]; then
    PYTHON_CMD=".venv/Scripts/python.exe"
elif [ -f "../.venv/bin/python" ]; then
    PYTHON_CMD="../.venv/bin/python"
elif [ -f ".venv/bin/python" ]; then
    PYTHON_CMD=".venv/bin/python"
elif command -v python3 &>/dev/null && python3 --version &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null && python --version &>/dev/null; then
    PYTHON_CMD="python"
fi

echo "======================================================================"
echo " KHỞI ĐỘNG HỆ THỐNG THỰC NGHIỆM PAPER 2 (ZERO-OOM ARCHITECTURE)"
echo "----------------------------------------------------------------------"
if [ -n "$FOLDER_IMAGE" ]; then
    echo " Thư mục ảnh COCO : $FOLDER_IMAGE"
else
    echo " Chế độ thực thi  : Hậu xử lý / Đồng bộ (Sync Only)"
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
    ${FOLDER_IMAGE:+--folder_image "$FOLDER_IMAGE"} \
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
    $FORCE_RECOMPUTE \
    $SYNC_ONLY \
    $SKIP_PAPER_COMPILE

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "[LỖI] Chuỗi thực nghiệm thất bại với mã lỗi: $EXIT_CODE"
    exit $EXIT_CODE
fi

echo ""
echo "======================================================================"
echo " TẤT CẢ THỰC NGHIỆM VÀ HẬU XỬ LÝ ĐÃ HOÀN TẤT THÀNH CÔNG 100%!"
echo "----------------------------------------------------------------------"
echo " 1. Kết quả Telemetry hợp nhất:"
echo "    -> $OUTPUT_DIR/paper2_complete_telemetry.json"
echo ""
echo " 2. Báo cáo tổng kết thực nghiệm toàn diện:"
echo "    -> $OUTPUT_DIR/EXPERIMENT_RESULTS_REPORT.md"
echo ""
echo " 3. Toàn bộ 4 đồ thị xuất bản vector PDF đã được lưu tại:"
echo "    -> $FIG_DIR/fig1_nms_free_paradox.pdf"
echo "    -> $FIG_DIR/fig2_displacement_tradeoff.pdf"
echo "    -> $FIG_DIR/fig3_downstream_cyberphysical_impact.pdf"
echo "    -> $FIG_DIR/fig4_sqd_filter_defense.pdf"
echo ""
echo " 4. Tệp PDF nộp bài hoàn thiện (chuẩn 24 trang Elsevier EAAI):"
echo "    -> ../SUBMISSION_PDFS/2_Paper_Phantom_Fabrication_Elsevier_EAAI.pdf"
echo "======================================================================"
