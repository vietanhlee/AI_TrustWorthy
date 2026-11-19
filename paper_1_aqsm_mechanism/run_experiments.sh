#!/usr/bin/env bash
# ==============================================================================
# SCRIPT ĐIỀU PHỐI CHẠY TOÀN BỘ THỰC NGHIỆM PAPER 1 (AQSM MECHANISM BREAKDOWN)
# ==============================================================================
# Hướng dẫn sử dụng:
#   bash run_experiments.sh --folder_image /path/to/coco/val2017 [OPTIONS]
#
# Các tham số hỗ trợ:
#   --folder_image PATH    : Đường dẫn tới thư mục chứa toàn bộ ảnh COCO val2017.
#   --annotations PATH     : Đường dẫn file annotations instances_val2017.json.
#   --output_dir PATH      : Thư mục lưu trữ kết quả. Mặc định: ./outputs
#   --cache_dir PATH       : Thư mục lưu trữ bộ nhớ đệm đối kháng .npz. Mặc định: ./cache/adv_cache_640
#   --fig_dir PATH         : Thư mục xuất biểu đồ vector PDF & PNG. Mặc định: ./fig
#   --device DEV           : Thiết bị tính toán: 'cuda', 'cpu', hoặc 'auto'.
#   --batch_size INT       : Kích thước lô.
#   --threads INT          : Số luồng CPU tính toán song song (mặc định: 8).
#   --tasks LIST           : Danh sách task chạy (mặc định: "0,1,2,3,4,5,6,8,7").
#   --num_images INT       : Số lượng ảnh tối đa (mặc định: toàn bộ ảnh).
#   --max_samples INT      : Alias của --num_images.
#   --skip_cache_gen       : Bỏ qua Task 0 -> Tự động chạy On-The-Fly VRAM Streaming (0 GB đĩa).
#   --force_recompute      : Ép sinh lại toàn bộ cache đối kháng kể cả khi đã tồn tại.
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

exec bash "$SCRIPT_DIR/run_all.sh" "$@"
