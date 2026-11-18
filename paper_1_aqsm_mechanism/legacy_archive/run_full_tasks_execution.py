# -*- coding: utf-8 -*-
"""
Module: paper_1_aqsm_mechanism.experiments.run_full_tasks_execution
Mục đích: Thực thi toàn bộ các task thực nghiệm khoa học Native 640x640:
  - Task 1: 2x2 Factorial Causal Tensor Surgery
  - Task 2: Dynamic Query Selection & Degradation Dynamics
  - Task 3: Deformable Attention & Spatial Feature Telemetry
  - Task 4: Controlled Misrouting & Geometric Perturbations
  - Task 5: Matched-Budget Perturbation Sweep
  - Task 6: Edge Hardware Latency & Sponge Immunity Profiler
  - Task 8: Defense Mitigation & White-box Adaptive Evasion
Sau khi hoàn tất, tự động vẽ lại toàn bộ Figure vector PDF và đồng bộ số liệu vào main.tex.
"""

import os
import sys
import time
import subprocess
from pathlib import Path

WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent
PYTHON_EXE = sys.executable
TASKS_DIR = WORKSPACE_DIR / "paper_1_aqsm_mechanism" / "experiments" / "tasks"
FIG_SCRIPT = WORKSPACE_DIR / "paper_1_aqsm_mechanism" / "experiments" / "generate_figures_and_tables.py"
UPDATE_SCRIPT = WORKSPACE_DIR / "paper_1_aqsm_mechanism" / "experiments" / "update_main_tex_numbers.py"
LATEX_DIR = WORKSPACE_DIR / "paper_1_aqsm_mechanism"


def run_command_live(cmd: list, desc: str) -> None:
    print("\n" + "=" * 75)
    print(f"[*] BẮT ĐẦU: {desc}")
    print(f"    Lệnh: {' '.join(cmd)}")
    print("=" * 75)
    t0 = time.time()
    res = subprocess.run(cmd)
    elapsed = time.time() - t0
    if res.returncode != 0:
        print(f"[-] LỖI ({elapsed:.1f}s): {desc} thất bại với mã lỗi {res.returncode}")
        sys.exit(res.returncode)
    else:
        print(f"[+] HOÀN THÀNH ({elapsed:.1f}s): {desc} thành công 100%!")


def main():
    print("=" * 80)
    print(" HỆ THỐNG THỰC NGHIỆM ĐỒNG BỘ NATIVE 640x640 TOÀN DIỆN CHO PAPER 1")
    print(f" Python: {PYTHON_EXE}")
    print(f" Thư mục tasks: {TASKS_DIR}")
    print("=" * 80)

    # 1. Chạy Task 1: Causal Surgery
    run_command_live(
        [PYTHON_EXE, str(TASKS_DIR / "task1_causal_surgery.py"), "--cohort_size", "30", "--threads", "4"],
        "Task 1: Causal Surgery 2x2 Factorial Swap (N=30)"
    )

    # 2. Chạy Task 2: Encoder Query Dynamics
    run_command_live(
        [PYTHON_EXE, str(TASKS_DIR / "task2_encoder_query_dynamics.py"), "--cohort_size", "30", "--threads", "4"],
        "Task 2: Encoder Query Dynamics (N=30)"
    )

    # 3. Chạy Task 3: Deformable Attention Telemetry
    run_command_live(
        [PYTHON_EXE, str(TASKS_DIR / "task3_deformable_telemetry.py"), "--cohort_size", "30", "--threads", "4"],
        "Task 3: Deformable Attention Telemetry (N=30)"
    )

    # 4. Chạy Task 4: Controlled Misrouting
    run_command_live(
        [PYTHON_EXE, str(TASKS_DIR / "task4_controlled_misrouting.py"), "--cohort_size", "30", "--threads", "4"],
        "Task 4: Controlled Misrouting (N=30)"
    )

    # 5. Chạy Task 5: Matched Budget Sweep
    run_command_live(
        [PYTHON_EXE, str(TASKS_DIR / "task5_matched_sweep.py"), "--cohort_size", "30", "--threads", "4"],
        "Task 5: Matched Budget Sweep (N=30)"
    )

    # 6. Chạy Task 6: Latency Profiler
    run_command_live(
        [PYTHON_EXE, str(TASKS_DIR / "task6_latency_profiler.py"), "--n_runs", "100"],
        "Task 6: Latency Profiler (100 runs)"
    )

    # 7. Chạy Task 8: Defense Evaluation
    run_command_live(
        [PYTHON_EXE, str(TASKS_DIR / "task8_defense_evaluation.py"), "--cohort_size", "30", "--threads", "4"],
        "Task 8: Defense Evaluation (N=30)"
    )

    # 8. Sinh lại toàn bộ Figure vector PDF từ các file JSON thật
    run_command_live(
        [PYTHON_EXE, str(FIG_SCRIPT)],
        "Tạo toàn bộ Figure vector PDF xuất bản Elsevier EAAI"
    )

    # 9. Cập nhật số liệu chuẩn xác 100% vào main.tex
    if UPDATE_SCRIPT.exists():
        run_command_live(
            [PYTHON_EXE, str(UPDATE_SCRIPT)],
            "Cập nhật số liệu đồng bộ JSON <-> Table vào main.tex"
        )

    # 10. Biên dịch LaTeX pdflatex 2 lần
    print("\n" + "=" * 75)
    print("[*] BIÊN DỊCH BÀI BÁO LATEX MAIN.TEX...")
    print("=" * 75)
    for run_idx in [1, 2]:
        cmd_latex = ["pdflatex", "-interaction=nonstopmode", "main.tex"]
        res = subprocess.run(cmd_latex, cwd=str(LATEX_DIR), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"    Lần {run_idx}: Exit code = {res.returncode}")

    print("\n" + "=" * 80)
    print(" HOÀN THÀNH TOÀN DIỆN QUY TRÌNH THỰC NGHIỆM VÀ XUẤT BẢN PAPER 1!")
    print("=" * 80)


if __name__ == "__main__":
    main()
