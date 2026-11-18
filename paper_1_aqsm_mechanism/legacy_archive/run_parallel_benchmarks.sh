#!/usr/bin/env bash
# ==============================================================================
# Script: run_parallel_benchmarks.sh
# Mục đích: Chạy các bài thực nghiệm theo từng mẻ (Batch), song song tối đa 4 task
# trên môi trường Linux / macOS / Git Bash.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MAX_CONCURRENT=4
COHORT_SIZE=300
THREADS_PER_TASK=2

if [[ -f "${WORKSPACE_DIR}/.venv/Scripts/python.exe" ]]; then
    PYTHON_BIN="${WORKSPACE_DIR}/.venv/Scripts/python.exe"
elif [[ -f "${WORKSPACE_DIR}/.venv/bin/python" ]]; then
    PYTHON_BIN="${WORKSPACE_DIR}/.venv/bin/python"
else
    PYTHON_BIN="python"
fi

LOGS_DIR="${SCRIPT_DIR}/../outputs/logs"
mkdir -p "${LOGS_DIR}"

echo "================================================================================"
echo "   BASH BATCH PARALLEL BENCHMARK RUNNER"
echo "   Tối đa tác vụ song song: ${MAX_CONCURRENT}"
echo "   Quy mô cohort: N=${COHORT_SIZE} ảnh COCO val2017"
echo "   Python: ${PYTHON_BIN}"
echo "================================================================================"

echo ""
echo ">>> [MẺ 1] Khởi chạy song song tối đa 4 tasks..."

# Chạy Task 1
"${PYTHON_BIN}" "${SCRIPT_DIR}/tasks/task1_causal_surgery.py" --cohort_size "${COHORT_SIZE}" --threads "${THREADS_PER_TASK}" > "${LOGS_DIR}/bash_task1_causal.log" 2>&1 &
PID1=$!
echo "    [Start] Task 1: Causal Surgery (PID: ${PID1})"

# Chạy Task 2
"${PYTHON_BIN}" "${SCRIPT_DIR}/tasks/task2_telemetry_analyzer.py" --cohort_size "${COHORT_SIZE}" --threads "${THREADS_PER_TASK}" > "${LOGS_DIR}/bash_task2_telemetry.log" 2>&1 &
PID2=$!
echo "    [Start] Task 2: Attention Telemetry (PID: ${PID2})"

# Chạy Task 3
"${PYTHON_BIN}" "${SCRIPT_DIR}/tasks/task3_latency_profiler.py" --n_runs 300 --threads "${THREADS_PER_TASK}" > "${LOGS_DIR}/bash_task3_latency.log" 2>&1 &
PID3=$!
echo "    [Start] Task 3: Latency Profiler (PID: ${PID3})"

# Chạy Task 4
SWEEP_SIZE=$(( COHORT_SIZE < 100 ? COHORT_SIZE : 100 ))
"${PYTHON_BIN}" "${SCRIPT_DIR}/tasks/task4_matched_sweep.py" --cohort_size "${SWEEP_SIZE}" --threads "${THREADS_PER_TASK}" > "${LOGS_DIR}/bash_task4_sweep.log" 2>&1 &
PID4=$!
echo "    [Start] Task 4: Matched-Budget Sweep (PID: ${PID4})"

echo ""
echo ">>> Đang chờ toàn bộ 4 tasks trong Mẻ 1 hoàn tất..."

wait "${PID1}" && echo "    [Hoàn thành] Task 1 -> THÀNH CÔNG" || echo "    [Thất bại] Task 1 -> LỖI"
wait "${PID2}" && echo "    [Hoàn thành] Task 2 -> THÀNH CÔNG" || echo "    [Thất bại] Task 2 -> LỖI"
wait "${PID3}" && echo "    [Hoàn thành] Task 3 -> THÀNH CÔNG" || echo "    [Thất bại] Task 3 -> LỖI"
wait "${PID4}" && echo "    [Hoàn thành] Task 4 -> THÀNH CÔNG" || echo "    [Thất bại] Task 4 -> LỖI"

echo ""
echo ">>> [MẺ 2] Khởi chạy Task 5: Sinh toàn bộ Figures vector PDF/PNG..."
"${PYTHON_BIN}" "${SCRIPT_DIR}/tasks/task5_generate_figures.py" > "${LOGS_DIR}/bash_task5_figures.log" 2>&1
echo "    [Hoàn thành] Task 5 -> Đã sinh đầy đủ Figures PDF/PNG chuẩn xuất bản!"

echo ""
echo "================================================================================"
echo "   QUY TRÌNH CHẠY SONG SONG HOÀN TẤT!"
echo "================================================================================"
