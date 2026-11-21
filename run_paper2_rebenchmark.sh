#!/usr/bin/env bash
# ==============================================================================
# Master Re-Benchmark Runner for Paper 2 (Elsevier EAAI)
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"

PYTHON_EXE="python"
if [[ -f "$ROOT_DIR/.venv/Scripts/python.exe" ]]; then
  PYTHON_EXE="$ROOT_DIR/.venv/Scripts/python.exe"
elif [[ -f "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_EXE="$ROOT_DIR/.venv/bin/python"
elif command -v python3 &>/dev/null; then
  PYTHON_EXE="python3"
fi

echo "================================================================="
echo " HỆ THỐNG THỰC NGHIỆM PAPER 2 (PHANTOM FABRICATION - EAAI)"
echo " Thư mục: ${ROOT_DIR}"
echo " Python : ${PYTHON_EXE}"
echo "================================================================="

# Chuyển tiếp tới run_paper2_clean_rebenchmark.py
"$PYTHON_EXE" "$ROOT_DIR/run_paper2_clean_rebenchmark.py" "$@"
