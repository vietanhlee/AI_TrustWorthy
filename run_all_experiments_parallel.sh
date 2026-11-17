#!/usr/bin/env bash
# ==============================================================================
# Master Parallel Benchmark Runner Script (Bash / POSIX / Git-Bash / WSL)
# Bài báo 1: AQSM Mechanism & Query Selection Vulnerability (Elsevier EAAI)
# Khả năng: Chạy song song tối đa 4 task đồng thời, ghi log kiểm toán chi tiết (.jsonl, .csv, .json).
# ==============================================================================

set -euo pipefail

# Xác định thư mục gốc dự án
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"

# Thiết lập các tham số mặc định
MAX_WORKERS=4
COHORT_SIZE=300
THREADS_PER_TASK=2
TASKS="1,2,3,4,5,6,7"
DRY_RUN=false

# Xử lý cờ dòng lệnh
while [[ $# -gt 0 ]]; do
  case "$1" in
    --max_workers)
      MAX_WORKERS="$2"
      shift 2
      ;;
    --cohort_size)
      COHORT_SIZE="$2"
      shift 2
      ;;
    --threads)
      THREADS_PER_TASK="$2"
      shift 2
      ;;
    --tasks)
      TASKS="$2"
      shift 2
      ;;
    --dry_run)
      DRY_RUN=true
      shift 1
      ;;
    -h|--help)
      echo "Sử dụng: $0 [--max_workers 4] [--cohort_size 300] [--threads 2] [--tasks 1,2,3,4,5,6,7] [--dry_run]"
      exit 0
      ;;
    *)
      echo "Cờ không xác định: $1"
      exit 1
      ;;
  esac
done

# Xác định Python interpreter
PYTHON_EXE="python"
if [[ -f "$ROOT_DIR/.venv/Scripts/python.exe" ]]; then
  PYTHON_EXE="$ROOT_DIR/.venv/Scripts/python.exe"
elif [[ -f "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_EXE="$ROOT_DIR/.venv/bin/python"
fi

echo "================================================================="
echo " HỆ THỐNG ĐIỀU PHỐI THỰC NGHIỆM SONG SONG (MAX ${MAX_WORKERS} WORKERS)"
echo " Tác phẩm: Towards Reliable Real-Time DETR (Elsevier EAAI)"
echo " Thư mục gốc: ${ROOT_DIR}"
echo " Python: ${PYTHON_EXE}"
echo " Cấu hình: MaxWorkers=${MAX_WORKERS} | CohortSize=${COHORT_SIZE} | Tasks=${TASKS}"
echo "================================================================="

MASTER_SCRIPT="$ROOT_DIR/paper_1_aqsm_mechanism/experiments/run_master_benchmark_suite.py"

CMD=("$PYTHON_EXE" "$MASTER_SCRIPT" \
  --max_workers "$MAX_WORKERS" \
  --cohort_size "$COHORT_SIZE" \
  --threads "$THREADS_PER_TASK" \
  --tasks "$TASKS")

if [[ "$DRY_RUN" == true ]]; then
  CMD+=(--dry_run)
fi

# Thực thi bộ điều phối thực nghiệm master
"${CMD[@]}"

EXIT_CODE=$?
if [[ $EXIT_CODE -eq 0 ]]; then
  echo ""
  echo ">>> [THÀNH CÔNG] Toàn bộ thực nghiệm song song đã hoàn tất chuẩn chỉ!"
else
  echo ""
  echo ">>> [CẢNH BÁO] Quá trình chạy thực nghiệm kết thúc với mã lỗi ${EXIT_CODE}."
fi

exit $EXIT_CODE
