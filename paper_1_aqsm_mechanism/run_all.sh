#!/usr/bin/env bash
# ==============================================================================
# Master Execution Script for Paper 1: Mechanistic Breakdown of Dynamic Query Selection
# Supports: Linux / Mac / Google Colab / Kaggle (CPU, Single-GPU, Multi-GPU)
# 100% Compatible with Paper 2 CLI Flags & On-The-Fly VRAM Streaming (0 GB Disk)
# ==============================================================================

set -e

# Default settings
FOLDER_IMAGE=""
ANNOTATIONS=""
NUM_IMAGES=0
MAX_SAMPLES=0
MODEL="r18"
DEVICE="auto"
BATCH_SIZE=0
TASKS="0,1,2,3,4,5,6,8,10,7,9"
OUTPUT_DIR="outputs"
CACHE_DIR="cache/adv_cache_640"
FIG_DIR="fig"
SKIP_CACHE_GEN=""
FORCE_RECOMPUTE=""
INCLUDE_SWEEP=""
RUN_SQA_ABLATION=""
RUN_MULTISEED=""
SEEDS=""
PGD_STEPS=""

print_usage() {
    echo "======================================================================"
    echo "  PAPER 1: DYNAMIC QUERY SELECTION MECHANISM - EMPIRICAL SUITE RUNNER"
    echo "======================================================================"
    echo "Usage:"
    echo "  bash run_all.sh --folder_image <path_to_coco_images> [OPTIONS]"
    echo ""
    echo "Required:"
    echo "  --folder_image PATH        Đường dẫn thư mục ảnh COCO (e.g. val2017)"
    echo ""
    echo "Optional:"
    echo "  --model ARCH               Kiến trúc mô hình: r18 (mặc định), r50, hoặc checkpoint"
    echo "  --annotations PATH         Đường dẫn file instances_val2017.json"
    echo "  --tasks LIST               Danh sách tasks (mặc định: 0,1,2,3,4,5,6,8,10,7,9)"
    echo "  --num_images INT           Số ảnh giới hạn (0 = chạy toàn bộ ảnh)"
    echo "  --max_samples INT          Alias tương đương --num_images"
    echo "  --device DEV               Thiết bị: auto (mặc định), cuda, cpu"
    echo "  --batch_size INT           Batch size suy luận (0 = tự động tính theo VRAM)"
    echo "  --output_dir PATH          Thư mục lưu trữ kết quả và log (mặc định: outputs)"
    echo "  --cache_dir PATH           Thư mục lưu trữ bộ nhớ đệm đối kháng (mặc định: cache/adv_cache_640)"
    echo "  --fig_dir PATH             Thư mục lưu biểu đồ xuất bản (mặc định: fig)"
    echo "  --skip_cache_gen           Bỏ qua tiền sinh cache Task 0 -> Chạy On-The-Fly VRAM Streaming (0 GB đĩa)"
    echo "  --force_recompute          Bắt buộc tính toán lại và ghi đè cache"
    echo "  --include_sweep            Sinh thêm cache eps 4 và 8 cho Task 5"
    echo "  --run_sqa_ablation         Chạy Live GPU Forward Pass cho SQA và các controls trong Task 8"
    echo "  --run_multiseed            Bật Task 10 chạy thực nghiệm AQSM 5-seed tự động khi chạy tập tasks tùy chỉnh"
    echo "  --seeds LIST               Danh sách seed cho Task 4 & Task 10 (mặc định: 42,100,2024,777,9999)"
    echo "  --pgd_steps INT            Số bước lặp PGD sinh mẫu đối kháng (mặc định: 15)"
    echo "  -h, --help                 Hiển thị hướng dẫn này"
    echo "======================================================================"
}

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --folder_image) FOLDER_IMAGE="$2"; shift ;;
        --annotations) ANNOTATIONS="$2"; shift ;;
        --model) MODEL="$2"; shift ;;
        --tasks) TASKS="$2"; shift ;;
        --num_images) NUM_IMAGES="$2"; shift ;;
        --max_samples) MAX_SAMPLES="$2"; shift ;;
        --device) DEVICE="$2"; shift ;;
        --batch_size) BATCH_SIZE="$2"; shift ;;
        --output_dir) OUTPUT_DIR="$2"; shift ;;
        --cache_dir) CACHE_DIR="$2"; shift ;;
        --fig_dir) FIG_DIR="$2"; shift ;;
        --skip_cache_gen) SKIP_CACHE_GEN="--skip_cache_gen" ;;
        --force_recompute) FORCE_RECOMPUTE="--force_recompute" ;;
        --include_sweep) INCLUDE_SWEEP="--include_sweep" ;;
        --run_sqa_ablation) RUN_SQA_ABLATION="--run_sqa_ablation" ;;
        --run_multiseed) RUN_MULTISEED="--run_multiseed" ;;
        --seeds) SEEDS="$2"; shift ;;
        --pgd_steps) PGD_STEPS="$2"; shift ;;
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

if [ "$NUM_IMAGES" -le 0 ] && [ "$MAX_SAMPLES" -gt 0 ]; then
    NUM_IMAGES=$MAX_SAMPLES
fi

# Ensure working directory is script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check Python environment
PYTHON_CMD="python"
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
fi

echo "======================================================================"
echo " 🚀 KHỞI ĐỘNG HỆ THỐNG THỰC NGHIỆM PAPER 1: DYNAMIC QUERY SELECTION"
echo "----------------------------------------------------------------------"
echo " Thư mục ảnh COCO : $FOLDER_IMAGE"
echo " Kiến trúc mô hình: $MODEL"
if [ -n "$ANNOTATIONS" ]; then
    echo " File Annotations : $ANNOTATIONS"
else
    echo " File Annotations : [Mặc định]"
fi
echo " Thiết bị         : $DEVICE"
echo " Danh sách Tasks  : $TASKS"
echo " Số mẫu ảnh       : $NUM_IMAGES (0 = Toàn bộ ảnh)"
echo " Thư mục Output   : $OUTPUT_DIR"
echo " Thư mục Figures  : $FIG_DIR"
if [ -n "$SKIP_CACHE_GEN" ]; then
    echo " Chế độ Cache     : ON-THE-FLY VRAM STREAMING (0 GB ĐĨA, KHÔNG SỢ HẾT ĐĨA)"
else
    echo " Chế độ Cache     : $CACHE_DIR (Streaming Direct .npz)"
fi
echo "======================================================================"

EXTRA_ARGS=()
if [ -n "$ANNOTATIONS" ]; then
    EXTRA_ARGS+=(--annotations "$ANNOTATIONS")
fi
if [ "$BATCH_SIZE" -gt 0 ]; then
    EXTRA_ARGS+=(--batch_size "$BATCH_SIZE")
fi
EXTRA_ARGS+=(--max_samples "$NUM_IMAGES")
if [ -n "$SKIP_CACHE_GEN" ]; then
    EXTRA_ARGS+=(--skip_cache_gen)
fi
if [ -n "$FORCE_RECOMPUTE" ]; then
    EXTRA_ARGS+=(--force_recompute)
fi
if [ -n "$INCLUDE_SWEEP" ]; then
    EXTRA_ARGS+=(--include_sweep)
fi
if [ -n "$RUN_SQA_ABLATION" ]; then
    EXTRA_ARGS+=(--run_sqa_ablation)
fi
if [ -n "$RUN_MULTISEED" ]; then
    EXTRA_ARGS+=(--run_multiseed)
fi
if [ -n "$SEEDS" ]; then
    EXTRA_ARGS+=(--seeds "$SEEDS")
fi
if [ -n "$PGD_STEPS" ]; then
    EXTRA_ARGS+=(--pgd_steps "$PGD_STEPS")
fi

$PYTHON_CMD run_all_experiments.py \
    --folder_image "$FOLDER_IMAGE" \
    --device "$DEVICE" \
    --model "$MODEL" \
    --tasks "$TASKS" \
    --cache_dir "$CACHE_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --fig_dir "$FIG_DIR" \
    "${EXTRA_ARGS[@]}"

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "[LỖI] Chuỗi thực nghiệm thất bại với mã lỗi: $EXIT_CODE"
    exit $EXIT_CODE
fi

echo ""
echo "======================================================================"
echo " ✅ TẤT CẢ THỰC NGHIỆM PAPER 1 ĐÃ HOÀN TẤT THÀNH CÔNG RỰC RỠ!"
echo "----------------------------------------------------------------------"
echo " 1. Kết quả tóm tắt & kiểm toán:"
echo "    -> $OUTPUT_DIR/summary_json/"
echo "    -> $OUTPUT_DIR/raw_csv/"
echo "    -> $OUTPUT_DIR/final_experiment_manifest.json"
echo ""
echo " 2. Toàn bộ đồ thị xuất bản vector PDF đã được lưu tại:"
echo "    -> $FIG_DIR/fig3_score_spectrum_n300.pdf"
echo "    -> $FIG_DIR/fig4_causal_swap_n300.pdf"
echo "    -> $FIG_DIR/fig5_multiscale_cosine_and_overlap.pdf"
echo "    -> $FIG_DIR/fig6_attention_scattering_comparison.pdf"
echo "    -> $FIG_DIR/fig7_matched_budget_sweep.pdf"
echo "    -> $FIG_DIR/fig8_defense_evaluation.pdf"

echo ""
echo " 3. Đang tự động đóng gói kết quả vào file ZIP..."
if command -v zip &>/dev/null; then
    zip -q -r "paper1_experiment_results.zip" "$OUTPUT_DIR" "$FIG_DIR" 2>/dev/null || true
else
    $PYTHON_CMD - "$OUTPUT_DIR" "$FIG_DIR" "paper1_experiment_results.zip" <<'PYEOF'
import os, sys, zipfile
out_dir, fig_dir, zip_name = sys.argv[1], sys.argv[2], sys.argv[3]
with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
    for root_dir in (out_dir, fig_dir):
        if os.path.isdir(root_dir):
            for r, _, files in os.walk(root_dir):
                for fn in files:
                    fp = os.path.join(r, fn)
                    zf.write(fp, fp)
PYEOF
fi
if [ -f "paper1_experiment_results.zip" ]; then
    echo "    -> File tải về tiện lợi: $(pwd)/paper1_experiment_results.zip"
else
    echo "    -> [CẢNH BÁO] Không tạo được file paper1_experiment_results.zip!"
fi
echo "======================================================================"
