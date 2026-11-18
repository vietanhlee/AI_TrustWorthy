#!/usr/bin/env bash
# ==============================================================================
# SCRIPT ĐIỀU PHỐI CHẠY TOÀN BỘ THỰC NGHIỆM PAPER 1 (AQSM MECHANISM BREAKDOWN)
# ==============================================================================
# Hướng dẫn sử dụng:
#   bash run_experiments.sh --folder_image /path/to/coco/val2017 [OPTIONS]
#
# Các tham số hỗ trợ:
#   --folder_image PATH    : Đường dẫn tới thư mục chứa toàn bộ ảnh COCO val2017 (5,000 ảnh).
#   --annotations PATH     : Đường dẫn file annotations instances_val2017.json (tự tìm nếu bỏ qua).
#   --output_dir PATH      : Thư mục lưu trữ kết quả (summary JSON, raw CSV, logs). Mặc định: ./outputs
#   --cache_dir PATH       : Thư mục lưu trữ bộ nhớ đệm đối kháng .npz. Mặc định: ./cache/adv_cache_640
#   --fig_dir PATH         : Thư mục xuất biểu đồ vector PDF & PNG. Mặc định: ./fig
#   --device DEV           : Thiết bị tính toán: 'cuda', 'cpu', hoặc 'auto' (tự động nhận diện).
#   --batch_size INT       : Kích thước lô (tự động tối ưu theo VRAM card đồ họa nếu bỏ qua).
#   --threads INT          : Số luồng CPU tính toán song song (mặc định: 8).
#   --tasks LIST           : Danh sách task chạy (mặc định: "0,1,2,3,4,5,6,7,8").
#   --skip_cache_gen       : Bỏ qua Task 0 nếu cache đối kháng đã được sinh trước đó.
#   --force_recompute      : Ép sinh lại toàn bộ cache đối kháng kể cả khi đã tồn tại.
# ==============================================================================

set -e

# Định vị thư mục gốc của bài báo
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================================================="
echo " 🚀 KHỞI ĐỘNG HỆ THỐNG THỰC NGHIỆM PAPER 1: DYNAMIC QUERY SELECTION (RT-DETR)"
echo " Thư mục làm việc: $SCRIPT_DIR"
echo "=============================================================================="

# 1. Tự động nhận diện trình thông dịch Python phù hợp
PYTHON_BIN=""
if [ -n "$VIRTUAL_ENV" ] && [ -f "$VIRTUAL_ENV/bin/python" ]; then
    PYTHON_BIN="$VIRTUAL_ENV/bin/python"
elif [ -n "$VIRTUAL_ENV" ] && [ -f "$VIRTUAL_ENV/Scripts/python.exe" ]; then
    PYTHON_BIN="$VIRTUAL_ENV/Scripts/python.exe"
elif [ -f "$SCRIPT_DIR/../.venv/Scripts/python.exe" ]; then
    PYTHON_BIN="$SCRIPT_DIR/../.venv/Scripts/python.exe"
elif [ -f "$SCRIPT_DIR/../.venv/bin/python" ]; then
    PYTHON_BIN="$SCRIPT_DIR/../.venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON_BIN="python3"
elif command -v python &>/dev/null; then
    PYTHON_BIN="python"
else
    echo "❌ LỖI: Không tìm thấy trình thông dịch Python nào trên hệ thống!"
    exit 1
fi

echo " Đang sử dụng Python: $PYTHON_BIN"
"$PYTHON_BIN" -c "import torch; print(f' PyTorch: {torch.__version__} | CUDA Available: {torch.cuda.is_available()} | GPU Count: {torch.cuda.device_count()}')"

# 2. Chạy Master Experiment Orchestrator
echo ""
echo " Đang thực thi Master Orchestrator (run_all_experiments.py)..."
"$PYTHON_BIN" "$SCRIPT_DIR/run_all_experiments.py" "$@"

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=============================================================================="
    echo " ✅ TOÀN BỘ CHUỖI THỰC NGHIỆM ĐÃ HOÀN TẤT THÀNH CÔNG RỰC RỠ!"
    echo " Dữ liệu số liệu Table và Figures đã sẵn sàng trong thư mục outputs/ và fig/."
    echo "=============================================================================="
else
    echo ""
    echo "=============================================================================="
    echo " ❌ THỰC NGHIỆM BỊ GIÁN ĐOẠN HOẶC GẶP LỖI (Exit code: $EXIT_CODE)"
    echo " Kiểm tra chi tiết tại thư mục outputs/logs/."
    echo "=============================================================================="
fi

exit $EXIT_CODE
