#!/usr/bin/env bash
# ==============================================================================
# Master Experiment Runner Script (Bash / POSIX / Git-Bash / WSL)
# Bài báo 1: Mechanistic Breakdown of Dynamic Query Selection in RT-DETR (Elsevier EAAI)
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"

# Xác định Python interpreter
PYTHON_EXE="python"
if [[ -f "$ROOT_DIR/.venv/Scripts/python.exe" ]]; then
  PYTHON_EXE="$ROOT_DIR/.venv/Scripts/python.exe"
elif [[ -f "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_EXE="$ROOT_DIR/.venv/bin/python"
fi

echo "================================================================="
echo " HỆ THỐNG ĐIỀU PHỐI THỰC NGHIỆM MASTER (PAPER 1)"
echo " Tác phẩm: Towards Reliable Real-Time DETR (Elsevier EAAI)"
echo " Thư mục gốc: ${ROOT_DIR}"
echo " Python: ${PYTHON_EXE}"
echo "================================================================="

MASTER_SCRIPT="$ROOT_DIR/paper_1_aqsm_mechanism/run_all_experiments.py"

"$PYTHON_EXE" "$MASTER_SCRIPT" "$@"
