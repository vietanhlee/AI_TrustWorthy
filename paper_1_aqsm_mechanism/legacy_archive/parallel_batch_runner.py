# -*- coding: utf-8 -*-
"""
Module: paper_1_aqsm_mechanism.experiments.parallel_batch_runner
Mục đích: Bộ điều phối chạy thực nghiệm song song theo từng mẻ (Batch-Parallel Pipeline Runner).
Kiểm soát chặt chẽ số lượng tiến trình đồng thời (tối đa 4 workers song song),
tối ưu hóa tài nguyên phần cứng CPU/RAM, ngăn ngừa quá tải bộ nhớ và race condition.

Cấu trúc Mẻ (Batch Pipeline):
  - Mẻ 1 (Parallel, tối đa 4 tiến trình độc lập):
      + Task 1: Causal Surgery 2x2 Factorial Swap (N=300)
      + Task 2: Encoder Memory & Query Selection Dynamics (N=300)
      + Task 3: Deformable Attention Telemetry (N=300)
      + Task 4: Controlled Spatial Misrouting & Tautology Refutation (N=300)
  - Mẻ 2 (Parallel, tối đa 2 tiến trình độc lập):
      + Task 5: Matched-Budget Perturbation Sweep (eps in {4, 8, 16})
      + Task 6: Edge Hardware Latency Profiler (300 runs)
  - Mẻ 3 (Tổng hợp dữ liệu & Sinh đồ thị xuất bản):
      + Task 7: Generate Publication Figures (PDF vector & 300 DPI PNG)
"""

import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent
TASKS_DIR = Path(__file__).resolve().parent / "tasks"
PYTHON_EXE = sys.executable

MAX_PARALLEL_WORKERS = 4


def _run_single_task(task_script: Path, extra_args: List[str] = None) -> Tuple[str, int, float, str]:
    """Chạy một task Python độc lập và đo đạc thời gian thực thi."""
    t_name = task_script.stem
    args = [PYTHON_EXE, str(task_script)] + (extra_args or [])
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}][START] Đang chạy {t_name}...")

    # Chạy process với môi trường kế thừa
    env = os.environ.copy()
    env["PYTHONPATH"] = str(WORKSPACE_DIR)
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    proc = subprocess.run(
        args,
        cwd=str(WORKSPACE_DIR),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    elapsed = time.time() - t_start
    status_str = "SUCCESS" if proc.returncode == 0 else "FAILED"
    print(f"[{time.strftime('%H:%M:%S')}][{status_str}] {t_name} hoàn thành trong {elapsed:.2f}s (Exit code: {proc.returncode})")

    if proc.returncode != 0:
        print(f"[ERROR in {t_name}]:\n{proc.stderr[-1000:]}")

    return t_name, proc.returncode, elapsed, proc.stdout


def run_batch_parallel_pipeline(cohort_size: int = 300, threads_per_task: int = 2):
    """Thực thi toàn bộ pipeline theo mẻ kiểm soát tối đa 4 tiến trình song song."""
    t_total_start = time.time()
    print("=" * 75)
    print("  HỆ THỐNG ĐIỀU PHỐI THỰC NGHIỆM CHẠY SONG SONG THEO MẺ (BATCH PARALLEL)")
    print(f"  Tối đa {MAX_PARALLEL_WORKERS} workers song song | Cohort N={cohort_size} | Luồng/task={threads_per_task}")
    print("=" * 75)

    # ----------------------------------------------------
    # MẺ 1: 4 Task nặng phân tích cơ chế nội tại (N=300)
    # ----------------------------------------------------
    batch1_tasks = [
        (TASKS_DIR / "task1_causal_surgery.py", ["--cohort_size", str(cohort_size), "--threads", str(threads_per_task)]),
        (TASKS_DIR / "task2_encoder_query_dynamics.py", ["--cohort_size", str(cohort_size), "--threads", str(threads_per_task)]),
        (TASKS_DIR / "task3_deformable_telemetry.py", ["--cohort_size", str(cohort_size), "--threads", str(threads_per_task)]),
        (TASKS_DIR / "task4_controlled_misrouting.py", ["--cohort_size", str(cohort_size), "--threads", str(threads_per_task)]),
    ]

    print(f"\n>>> [MẺ 1] Khởi chạy song song {len(batch1_tasks)} Tasks cơ chế nội tại (Worker pool = {MAX_PARALLEL_WORKERS})...")
    t_b1_start = time.time()

    with ProcessPoolExecutor(max_workers=MAX_PARALLEL_WORKERS) as executor:
        futures = {executor.submit(_run_single_task, script, args): script.stem for script, args in batch1_tasks}
        for future in as_completed(futures):
            t_name, code, dur, _ = future.result()
            if code != 0:
                print(f"[FATAL] Task {t_name} gặp lỗi. Dừng pipeline.")
                return False

    print(f">>> [MẺ 1 HOÀN TẤT] Thời gian thực thi Mẻ 1: {time.time() - t_b1_start:.2f}s")

    # ----------------------------------------------------
    # MẺ 2: Quét ngân sách Sweep & Đo Latency
    # ----------------------------------------------------
    batch2_tasks = [
        (TASKS_DIR / "task5_matched_sweep.py", ["--cohort_size", "100", "--threads", str(threads_per_task)]),
        (TASKS_DIR / "task6_latency_profiler.py", ["--runs", "100", "--warmup", "10", "--threads", str(threads_per_task)]),
    ]

    print(f"\n>>> [MẺ 2] Khởi chạy song song {len(batch2_tasks)} Tasks quét ngân sách & đo trễ...")
    t_b2_start = time.time()

    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(_run_single_task, script, args): script.stem for script, args in batch2_tasks}
        for future in as_completed(futures):
            t_name, code, dur, _ = future.result()
            if code != 0:
                print(f"[FATAL] Task {t_name} gặp lỗi.")
                return False

    print(f">>> [MẺ 2 HOÀN TẤT] Thời gian thực thi Mẻ 2: {time.time() - t_b2_start:.2f}s")

    # ----------------------------------------------------
    # MẺ 3: Tổng hợp Figures & Xuất bản đồ thị
    # ----------------------------------------------------
    print("\n>>> [MẺ 3] Đang đọc JSON tổng hợp số liệu và sinh toàn bộ Figures vector...")
    t_b3_start = time.time()
    t_name, code, dur, _ = _run_single_task(TASKS_DIR / "task7_generate_figures.py")
    if code != 0:
        print(f"[FATAL] Task {t_name} thất bại.")
        return False
    print(f">>> [MẺ 3 HOÀN TẤT] Sinh Figures thành công trong {dur:.2f}s!")

    total_time = time.time() - t_total_start
    print("\n" + "=" * 75)
    print(f"  TẤT CẢ 3 MẺ THỰC NGHIỆM ĐÃ HOÀN THÀNH XUẤT SẮC TRONG {total_time:.2f}s ({total_time/60:.2f} phút)!")
    print("=" * 75)
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Master Batch-Parallel Experiments Pipeline")
    parser.add_argument("--cohort_size", type=int, default=300, help="Số lượng ảnh COCO val2017")
    parser.add_argument("--threads_per_task", type=int, default=2, help="Số luồng CPU trên mỗi task")
    args = parser.parse_args()

    success = run_batch_parallel_pipeline(cohort_size=args.cohort_size, threads_per_task=args.threads_per_task)
    sys.exit(0 if success else 1)
