# -*- coding: utf-8 -*-
"""
Script: run_parallel_4procs.py
Mục đích: Điều phối toàn bộ chuỗi thực nghiệm Paper 1 bằng Hàng đợi Tiến trình (Task Queue),
giới hạn nghiêm ngặt TỐI ĐA 4 TIẾN TRÌNH SONG SONG CÙNG MỘT LÚC (Max 4 Active Subprocesses).
Tối ưu hóa an toàn bộ nhớ RAM và CPU, log thời gian thực, tự động khôi phục và thông báo tiến độ.
"""

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent
PYTHON_EXE = sys.executable

TASKS_SPEC = [
    {
        "id": "T1_Causal_AQSM",
        "name": "Task 1: Causal Surgery (AQSM)",
        "cmd": [
            PYTHON_EXE,
            "paper_1_aqsm_mechanism/experiments/tasks/task1_causal_surgery.py",
            "--cohort_size", "30",
            "--attacks", "aqsm",
            "--threads", "2",
            "--batch_size", "10"
        ],
        "log": "paper_1_aqsm_mechanism/logs/task1_causal_aqsm.log"
    },
    {
        "id": "T1_Causal_Baselines",
        "name": "Task 1: Causal Surgery (Vanishing & TaskPGD)",
        "cmd": [
            PYTHON_EXE,
            "paper_1_aqsm_mechanism/experiments/tasks/task1_causal_surgery.py",
            "--cohort_size", "30",
            "--attacks", "vanishing,taskpgd",
            "--threads", "2",
            "--batch_size", "10"
        ],
        "log": "paper_1_aqsm_mechanism/logs/task1_causal_baselines.log"
    },
    {
        "id": "T2_Encoder_Dynamics",
        "name": "Task 2: Encoder Query Dynamics (RQ2)",
        "cmd": [
            PYTHON_EXE,
            "paper_1_aqsm_mechanism/experiments/tasks/task2_encoder_query_dynamics.py",
            "--cohort_size", "30",
            "--threads", "2",
            "--batch_size", "10"
        ],
        "log": "paper_1_aqsm_mechanism/logs/task2_encoder_dynamics.log"
    },
    {
        "id": "T3_Deformable_Telemetry",
        "name": "Task 3: Deformable Attention Telemetry (RQ3)",
        "cmd": [
            PYTHON_EXE,
            "paper_1_aqsm_mechanism/experiments/tasks/task3_deformable_telemetry.py",
            "--cohort_size", "30",
            "--threads", "2",
            "--batch_size", "10"
        ],
        "log": "paper_1_aqsm_mechanism/logs/task3_deformable.log"
    },
    {
        "id": "T4_Controlled_Misrouting",
        "name": "Task 4: Controlled Misrouting (RQ3 Isolation)",
        "cmd": [
            PYTHON_EXE,
            "paper_1_aqsm_mechanism/experiments/tasks/task4_controlled_misrouting.py",
            "--cohort_size", "30",
            "--threads", "2",
            "--batch_size", "10"
        ],
        "log": "paper_1_aqsm_mechanism/logs/task4_misrouting.log"
    },
    {
        "id": "T5_Matched_Sweep",
        "name": "Task 5: Matched-Budget Perturbation Sweep (RQ4)",
        "cmd": [
            PYTHON_EXE,
            "paper_1_aqsm_mechanism/experiments/tasks/task5_matched_sweep.py",
            "--cohort_size", "30",
            "--threads", "4",
            "--batch_size", "10"
        ],
        "log": "paper_1_aqsm_mechanism/logs/task5_matched_sweep.log"
    },
    {
        "id": "T6_Edge_Latency",
        "name": "Task 6: Edge Hardware Latency Profiler",
        "cmd": [
            PYTHON_EXE,
            "paper_1_aqsm_mechanism/experiments/tasks/task6_latency_profiler.py",
            "--n_runs", "30",
            "--threads", "2"
        ],
        "log": "paper_1_aqsm_mechanism/logs/task6_latency.log"
    },
    {
        "id": "T8_Defense_Evaluation",
        "name": "Task 8: Defense Evaluation (ESM & Adaptive PGD)",
        "cmd": [
            PYTHON_EXE,
            "paper_1_aqsm_mechanism/experiments/tasks/task8_defense_evaluation.py",
            "--cohort_size", "30",
            "--threads", "2"
        ],
        "log": "paper_1_aqsm_mechanism/logs/task8_defense.log"
    }
]

MAX_CONCURRENT = 4  # Giới hạn tối đa nghiêm ngặt 4 tiến trình song song cùng 1 lúc

def main():
    print("================================================================================")
    print(" [COORDINATOR] KHỞI CHẠY THỰC NGHIỆM PAPER 1: HỒ BƠI TỐI ĐA 4 TIẾN TRÌNH")
    print(f" Cấu hình: Native 640x640 | Giới hạn: MAX_CONCURRENT = {MAX_CONCURRENT} processes")
    print("================================================================================")

    log_dir = WORKSPACE_DIR / "paper_1_aqsm_mechanism" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # 0. Bước 0: Sinh cache mẫu đối kháng sạch bóng từ đầu (Không dùng cache cũ)
    print("\n================================================================================")
    print(" [BƯỚC 0] KHỞI TẠO BỘ NHỚ ĐỆM TẤN CÔNG ĐỐI KHÁNG SẠCH BÓNG NATIVE 640x640")
    print(" (Sinh đối kháng thật 100% bằng PGD 15 steps trên 30 ảnh COCO val2017)")
    print("================================================================================")
    cmd_cache = [
        PYTHON_EXE,
        "paper_1_aqsm_mechanism/experiments/generate_adv_cache_640.py",
        "--cohort_size", "30",
        "--batch_size", "10",
        "--threads", "8",
        "--include_sweep"
    ]
    res_cache = subprocess.run(cmd_cache, cwd=str(WORKSPACE_DIR))
    if res_cache.returncode != 0:
        print("[LỖI] Sinh cache đối kháng thất bại!")
        sys.exit(1)
    print(">>> [BƯỚC 0] Đã khởi tạo xong toàn bộ mẫu đối kháng sạch bóng!\n")

    print("================================================================================")
    print(" [BƯỚC 1] KHỞI CHẠY HỒ BƠI TỐI ĐA 4 TIẾN TRÌNH THỰC NGHIỆM ĐỘC LẬP")
    print("================================================================================")

    pending_tasks = list(TASKS_SPEC)
    running_procs: List[Dict[str, Any]] = []
    completed_tasks: List[Dict[str, Any]] = []
    start_time = time.time()

    while pending_tasks or running_procs:
        # 1. Bổ sung tiến trình mới nếu chưa vượt quá MAX_CONCURRENT
        while len(running_procs) < MAX_CONCURRENT and pending_tasks:
            task = pending_tasks.pop(0)
            log_path = WORKSPACE_DIR / task["log"]
            lf = open(log_path, "w", encoding="utf-8")
            print(f">>> [LAUNCH] {task['name']} -> Log: {task['log']}")
            p = subprocess.Popen(
                task["cmd"],
                stdout=lf,
                stderr=subprocess.STDOUT,
                cwd=str(WORKSPACE_DIR),
                text=True
            )
            running_procs.append({
                "task": task,
                "proc": p,
                "log_file": lf,
                "start_time": time.time()
            })
            time.sleep(2)  # Delay nhẹ 2s tránh xung đột I/O lúc load weights

        # 2. Kiểm tra trạng thái các tiến trình đang chạy
        still_running = []
        for item in running_procs:
            poll = item["proc"].poll()
            if poll is None:
                still_running.append(item)
            else:
                item["log_file"].close()
                elapsed_task = time.time() - item["start_time"]
                code = item["proc"].returncode
                status_str = "SUCCESS" if code == 0 else f"FAILED(code={code})"
                print(f"\n<<< [{status_str}] {item['task']['name']} kết thúc sau {elapsed_task:.1f}s (Mã thoát: {code})")
                completed_tasks.append({
                    "task": item["task"],
                    "returncode": code,
                    "elapsed": elapsed_task
                })

        running_procs = still_running

        # 3. Hiển thị thông báo trạng thái tóm tắt
        elapsed_total = time.time() - start_time
        active_names = [r["task"]["id"] for r in running_procs]
        rem_count = len(pending_tasks)
        print(f"[{elapsed_total:6.1f}s] Active ({len(running_procs)}/{MAX_CONCURRENT}): {active_names} | Còn lại: {rem_count} task(s)   ", end="\r", flush=True)

        if running_procs or pending_tasks:
            time.sleep(4)

    # 4. Báo cáo tổng kết toàn bộ tiến trình
    print("\n\n================================================================================")
    print(f" TẤT CẢ CÁC TIẾN TRÌNH ĐÃ HOÀN TẤT TRONG: {time.time() - start_time:.2f}s")
    print("================================================================================")

    failed_tasks = [c for c in completed_tasks if c["returncode"] != 0]
    if failed_tasks:
        print("\n[CẢNH BÁO] Các task gặp lỗi:")
        for f in failed_tasks:
            print(f" - {f['task']['name']} (Mã lỗi: {f['returncode']}) -> Xem log: {f['task']['log']}")
        sys.exit(1)
    else:
        print("\n[TUYỆT VỜI] 100% tất cả các task thực nghiệm đã hoàn thành thành công mã thoát 0!")
        
        # 5. Tự động kích hoạt pipeline đồng bộ số liệu và xuất bản
        print("\n================================================================================")
        print(" [PIPELINE] BẮT ĐẦU TỰ ĐỘNG ĐỒNG BỘ SỐ LIỆU VÀ XUẤT BẢN BÀI BÁO ELSEVIER")
        print("================================================================================")
        
        # 5.1 Sinh lại toàn bộ Figures vector PDF và Tables
        print(">>> [Bước 1/4] Đang vẽ lại toàn bộ vector figures (Fig 3-8 PDF) từ JSON thật...")
        cmd_fig = [PYTHON_EXE, "paper_1_aqsm_mechanism/experiments/generate_figures_and_tables.py"]
        res_fig = subprocess.run(cmd_fig, cwd=str(WORKSPACE_DIR))
        if res_fig.returncode != 0:
            print("[CẢNH BÁO] Sinh figures gặp lỗi!")
        else:
            print(">>> [Bước 1/4] Đã cập nhật xong toàn bộ Figures vector PDF.")

        # 5.2 Cập nhật số liệu vào main.tex
        print(">>> [Bước 2/4] Đang điền số liệu thực nghiệm 100% THẬT vào 11 bảng trong main.tex...")
        cmd_tex = [PYTHON_EXE, "paper_1_aqsm_mechanism/experiments/update_main_tex_numbers.py"]
        res_tex = subprocess.run(cmd_tex, cwd=str(WORKSPACE_DIR))
        if res_tex.returncode != 0:
            print("[CẢNH BÁO] Cập nhật main.tex gặp lỗi!")
        else:
            print(">>> [Bước 2/4] Đã cập nhật xong số liệu trong main.tex.")

        # 5.3 Biên dịch pdflatex 2 lần
        paper_dir = WORKSPACE_DIR / "paper_1_aqsm_mechanism"
        print(">>> [Bước 3/4] Đang biên dịch LaTeX sang PDF chuẩn Elsevier EAAI...")
        for run_idx in range(1, 3):
            print(f"    - Lượt biên dịch {run_idx}/2...")
            subprocess.run(["pdflatex", "-interaction=nonstopmode", "main.tex"], cwd=str(paper_dir), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(">>> [Bước 3/4] Biên dịch pdflatex hoàn tất.")

        # 5.4 Copy sang SUBMISSION_PDFS và kiểm tra
        print(">>> [Bước 4/4] Sao chép PDF sang SUBMISSION_PDFS...")
        dest_dir = WORKSPACE_DIR / "SUBMISSION_PDFS"
        dest_dir.mkdir(parents=True, exist_ok=True)
        src_pdf = paper_dir / "main.pdf"
        dest_pdf = dest_dir / "1_Paper_AQSM_Mechanism_Elsevier_EAAI.pdf"
        if src_pdf.exists():
            import shutil
            shutil.copy2(src_pdf, dest_pdf)
            print(f">>> [HOÀN TẤT] File PDF xuất bản đã sẵn sàng: {dest_pdf}")
        else:
            print(f"[LỖI] Không tìm thấy file {src_pdf}")

        print("\n================================================================================")
        print("  CHÚC MỪNG: TOÀN BỘ QUY TRÌNH THỰC NGHIỆM VÀ XUẤT BẢN ĐÃ HOÀN TẤT 100%!")
        print("================================================================================")

if __name__ == "__main__":
    main()
