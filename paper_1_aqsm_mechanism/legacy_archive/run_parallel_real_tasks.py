# -*- coding: utf-8 -*-
"""
Script: run_parallel_real_tasks.py
Mục đích: Khởi chạy song song 5 task thực nghiệm độc lập (Task 1, Task 2, Task 3, Task 4, Task 8)
trên dữ liệu thật Native 640x640 bằng Subprocess Pool, tận dụng đa lõi CPU.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent
PYTHON_EXE = sys.executable

tasks = [
    {
        "name": "Task 1 (Causal Surgery: Vanishing & TaskPGD)",
        "cmd": [
            PYTHON_EXE,
            "paper_1_aqsm_mechanism/experiments/tasks/task1_causal_surgery.py",
            "--cohort_size", "30",
            "--attacks", "vanishing,taskpgd",
            "--threads", "3",
            "--batch_size", "10"
        ],
        "log": "paper_1_aqsm_mechanism/logs/task1_vanishing_taskpgd.log"
    },
    {
        "name": "Task 2 (Encoder Query Dynamics)",
        "cmd": [
            PYTHON_EXE,
            "paper_1_aqsm_mechanism/experiments/tasks/task2_encoder_query_dynamics.py",
            "--cohort_size", "30",
            "--threads", "3",
            "--batch_size", "10"
        ],
        "log": "paper_1_aqsm_mechanism/logs/task2_encoder_dynamics.log"
    },
    {
        "name": "Task 3 (Deformable Attention Telemetry)",
        "cmd": [
            PYTHON_EXE,
            "paper_1_aqsm_mechanism/experiments/tasks/task3_deformable_telemetry.py",
            "--cohort_size", "30",
            "--threads", "3",
            "--batch_size", "10"
        ],
        "log": "paper_1_aqsm_mechanism/logs/task3_deformable.log"
    },
    {
        "name": "Task 4 (Controlled Misrouting)",
        "cmd": [
            PYTHON_EXE,
            "paper_1_aqsm_mechanism/experiments/tasks/task4_controlled_misrouting.py",
            "--cohort_size", "30",
            "--threads", "3",
            "--batch_size", "10"
        ],
        "log": "paper_1_aqsm_mechanism/logs/task4_misrouting.log"
    },
    {
        "name": "Task 8 (Defense Evaluation ESM)",
        "cmd": [
            PYTHON_EXE,
            "paper_1_aqsm_mechanism/experiments/tasks/task8_defense_evaluation.py",
            "--cohort_size", "30",
            "--threads", "3"
        ],
        "log": "paper_1_aqsm_mechanism/logs/task8_defense.log"
    }
]

def main():
    print("======================================================================")
    print("  KHỞI CHẠY HỒ BƠI 5 TIẾN TRÌNH THỰC NGHIỆM ĐỘC LẬP (NATIVE 640x640)")
    print("======================================================================")
    
    os.makedirs("paper_1_aqsm_mechanism/logs", exist_ok=True)
    
    procs = []
    log_files = []
    start_time = time.time()
    
    for t in tasks:
        lf = open(t["log"], "w", encoding="utf-8")
        log_files.append(lf)
        print(f"[LAUNCH] {t['name']} -> Log: {t['log']}")
        p = subprocess.Popen(
            t["cmd"],
            stdout=lf,
            stderr=subprocess.STDOUT,
            cwd=str(WORKSPACE_DIR),
            text=True
        )
        procs.append((t, p, lf))
        time.sleep(2)  # Delay nhẹ để tránh tranh chấp I/O nạp model
        
    print("\n>>> Tất cả 5 tiến trình đã được kích hoạt. Đang theo dõi...")
    
    while True:
        all_done = True
        statuses = []
        for t, p, _ in procs:
            poll = p.poll()
            if poll is None:
                all_done = False
                statuses.append(f"{t['name'][:18]}: RUNNING")
            else:
                statuses.append(f"{t['name'][:18]}: DONE(code={poll})")
        
        elapsed = time.time() - start_time
        print(f"[{elapsed:6.1f}s] " + " | ".join(statuses), end="\r", flush=True)
        
        if all_done:
            break
        time.sleep(5)
        
    for _, _, lf in procs:
        lf.close()
        
    print(f"\n\n======================================================================")
    print(f" HOÀN THÀNH TẤT CẢ 5 TIẾN TRÌNH TRONG: {time.time() - start_time:.2f}s")
    print(f"======================================================================")
    
    # Kiểm tra mã thoát (exit codes)
    failed = False
    for t, p, _ in procs:
        if p.returncode != 0:
            print(f"[ERROR] {t['name']} thất bại với code {p.returncode}! Vui lòng kiểm tra log: {t['log']}")
            failed = True
        else:
            print(f"[SUCCESS] {t['name']} thành công (code 0)!")
            
    if failed:
        sys.exit(1)

if __name__ == "__main__":
    main()
