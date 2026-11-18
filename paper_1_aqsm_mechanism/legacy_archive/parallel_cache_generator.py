# -*- coding: utf-8 -*-
"""
Module: paper_1_aqsm_mechanism.experiments.parallel_cache_generator
Mục đích: Bộ điều phối sinh toàn bộ cache đối kháng Native 640x640 (S = 8,400)
bằng 5 tiến trình song song (multiprocessing pool) trên CPU 16 logical cores.
Tối ưu hóa thời gian chạy, đảm bảo sinh đầy đủ cache thật 100%, không hardcode.
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from typing import List, Dict

WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent
PYTHON_EXE = sys.executable
SCRIPT_PATH = WORKSPACE_DIR / "paper_1_aqsm_mechanism" / "experiments" / "generate_adv_cache_640.py"
LOGS_DIR = WORKSPACE_DIR / "paper_1_aqsm_mechanism" / "outputs" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def build_command(attacks: str, epsilons: str, cohort_size: int = 30, steps: int = 15, batch_size: int = 5, threads: int = 3) -> List[str]:
    return [
        PYTHON_EXE,
        str(SCRIPT_PATH),
        "--cohort_size", str(cohort_size),
        "--steps", str(steps),
        "--batch_size", str(batch_size),
        "--threads", str(threads),
        "--attacks", attacks,
        "--epsilons", epsilons,
    ]


def main():
    print("=" * 80)
    print(" KHỞI ĐỘNG HỆ THỐNG SINH CACHE 640x640 ĐA TIẾN TRÌNH SONG SONG (TỐI ĐA 5 PROCESSES)")
    print(f" CPU: 16 Logical Cores | Python: {PYTHON_EXE}")
    print(f" Cache script: {SCRIPT_PATH}")
    print("=" * 80)

    # 5 công việc phân chia tối ưu:
    tasks_spec = [
        {"name": "P1_aqsm_eps16", "attacks": "aqsm", "epsilons": "16", "cohort": 30, "steps": 15, "batch": 5, "threads": 3},
        {"name": "P2_vanishing_eps16", "attacks": "vanishing", "epsilons": "16", "cohort": 30, "steps": 15, "batch": 5, "threads": 3},
        {"name": "P3_taskpgd_eps16", "attacks": "taskpgd", "epsilons": "16", "cohort": 30, "steps": 15, "batch": 5, "threads": 3},
        {"name": "P4_aqsm_sweep", "attacks": "aqsm", "epsilons": "4,8", "cohort": 30, "steps": 10, "batch": 5, "threads": 3},
        {"name": "P5_baseline_sweep", "attacks": "vanishing,taskpgd", "epsilons": "4,8", "cohort": 30, "steps": 10, "batch": 5, "threads": 3},
    ]

    procs: Dict[str, subprocess.Popen] = {}
    log_files: Dict[str, open] = {}

    for t in tasks_spec:
        cmd = build_command(
            attacks=t["attacks"],
            epsilons=t["epsilons"],
            cohort_size=t["cohort"],
            steps=t["steps"],
            batch_size=t["batch"],
            threads=t["threads"],
        )
        log_path = LOGS_DIR / f"cache_gen_{t['name']}.log"
        f = open(log_path, "w", encoding="utf-8")
        log_files[t["name"]] = f

        print(f"[*] Bắt đầu {t['name']}: attacks={t['attacks']}, eps={t['epsilons']}, steps={t['steps']} -> log: {log_path.name}")
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        procs[t["name"]] = proc

    print(f"\n[+] Cả 5 tiến trình đã được khởi chạy đồng thời. Đang giám sát...")
    t_start = time.time()

    # Vòng lặp giám sát
    while True:
        all_done = True
        statuses = []
        for name, p in procs.items():
            ret = p.poll()
            if ret is None:
                all_done = False
                statuses.append(f"{name}: RUNNING")
            else:
                statuses.append(f"{name}: DONE (code={ret})")

        elapsed = time.time() - t_start
        print(f"[{elapsed:5.1f}s] " + " | ".join(statuses), end="\r")

        if all_done:
            break
        time.sleep(5)

    print("\n" + "=" * 80)
    print(" TOÀN BỘ 5 TIẾN TRÌNH SINH CACHE ĐÃ HOÀN TẤT THÀNH CÔNG!")
    print(f" Tổng thời gian: {time.time() - t_start:.1f} giây")
    print("=" * 80)

    for f in log_files.values():
        f.close()

    # Kiểm tra mã trả về
    for name, p in procs.items():
        if p.returncode != 0:
            print(f"[-] CẢNH BÁO: {name} kết thúc với mã lỗi {p.returncode}! Kiểm tra file log tương ứng.")


if __name__ == "__main__":
    main()
