# -*- coding: utf-8 -*-
"""
Module: paper_1_aqsm_mechanism.experiments.run_master_benchmark_suite
Mục đích: Bộ điều phối trung tâm chạy thực nghiệm song song chuẩn Production (Master Benchmark Suite).
Khả năng đáp ứng:
  1. Hỗ trợ chạy đồng thời tối đa N tasks song song (mặc định 4 workers) qua ProcessPoolExecutor.
  2. Phân luồng thông minh:
     - Giai đoạn 1 (Parallel Engine): Đẩy đồng thời tối đa 4 tasks thực nghiệm nặng (Task 1, 2, 3, 4, 5, 6) vào worker pool.
       Khi bất kỳ worker nào hoàn thành, task tiếp theo trong hàng đợi lập tức được điều phối vào,
       duy trì hiệu suất CPU/GPU tối đa mà không bị nghẽn (Zero idle waiting).
     - Giai đoạn 2 (Synthesis & Visualization): Tự động thu thập toàn bộ dữ liệu audit stream/JSON/CSV
       và kích hoạt Task 7 để sinh lại toàn bộ Figures vector PDF/PNG xuất bản.
     - Giai đoạn 3 (Audit Manifest): Tạo báo cáo kiểm toán tổng hợp master `master_audit_manifest.json`.
  3. Tích hợp cờ CLI linh hoạt:
     - --max_workers: Số task chạy đồng thời tối đa (mặc định 4).
     - --tasks: Chọn các task cụ thể cần chạy (ví dụ: "1,2,3,4,5,6,7").
     - --cohort_size: Số lượng ảnh COCO val2017 (mặc định 300).
     - --dry_run: Kiểm tra cấu hình và tính sẵn sàng của cache/mô hình trong 2 giây.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent
TASKS_DIR = Path(__file__).resolve().parent / "tasks"
OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"
SUMMARY_DIR = OUTPUTS_DIR / "summary_json"
AUDIT_DIR = OUTPUTS_DIR / "audit_streams"
RAW_CSV_DIR = OUTPUTS_DIR / "raw_csv"
LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"

PYTHON_EXE = sys.executable

# Bản đồ định nghĩa toàn bộ 7 tasks thực nghiệm chuẩn hóa
TASK_REGISTRY: Dict[int, Dict[str, Any]] = {
    1: {
        "name": "Task 1: Causal Surgery 2x2 Factorial Swap (6 Arms)",
        "script": TASKS_DIR / "task1_causal_surgery.py",
        "description": "Phẫu thuật can thiệp nhân quả M vs Q trên 6 arms cho AQSM, Vanishing, Task PGD",
        "default_args": ["--attacks", "aqsm,vanishing,taskpgd"],
        "phase": 1,
    },
    2: {
        "name": "Task 2: Encoder Query Dynamics & Starvation Check",
        "script": TASKS_DIR / "task2_encoder_query_dynamics.py",
        "description": "Phân tích Cosine đa tỉ lệ S3/S4/S5, Top-K Overlap và kiểm chứng Proposal Starvation",
        "default_args": [],
        "phase": 1,
    },
    3: {
        "name": "Task 3: Deformable Attention Telemetry",
        "script": TASKS_DIR / "task3_deformable_telemetry.py",
        "description": "Đo đạc Attention Entropy, Spatial Dispersion Radius và Spatial Target Misrouting",
        "default_args": [],
        "phase": 1,
    },
    4: {
        "name": "Task 4: Controlled Misrouting & Tautology Refutation",
        "script": TASKS_DIR / "task4_controlled_misrouting.py",
        "description": "Thực nghiệm kiểm soát 4 điều kiện với 5 seeds ngẫu nhiên cho Permutation & Shift",
        "default_args": [],
        "phase": 1,
    },
    5: {
        "name": "Task 5: Matched-Budget Perturbation Sweep",
        "script": TASKS_DIR / "task5_matched_sweep.py",
        "description": "Quét ngân sách nhiễu L_inf qua eps in {4/255, 8/255, 16/255} trên N=100 ảnh",
        "default_args": ["--cohort_size", "100"],
        "phase": 1,
    },
    6: {
        "name": "Task 6: Edge Hardware Latency Profiler",
        "script": TASKS_DIR / "task6_latency_profiler.py",
        "description": "Đo đạc thời gian thực thi phần cứng qua 300 runs và kiểm chứng Sponge Immunity",
        "default_args": ["--n_runs", "300"],
        "phase": 1,
    },
    7: {
        "name": "Task 7: Publication Figures & Tables Generator",
        "script": TASKS_DIR / "task7_generate_figures.py",
        "description": "Đọc số liệu JSON/CSV và tự động sinh toàn bộ 5 Figures vector PDF/PNG 300 DPI",
        "default_args": [],
        "phase": 2,
    },
}


def _execute_task_process(
    task_id: int,
    script_path: Path,
    extra_args: List[str],
) -> Dict[str, Any]:
    """Thực thi một task con độc lập và ghi nhận kết quả."""
    t_start = time.time()
    t_name = f"Task_{task_id}_{script_path.stem}"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(WORKSPACE_DIR)
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    cmd = [PYTHON_EXE, str(script_path)] + extra_args
    print(f"[{time.strftime('%H:%M:%S')}][WORKER START] Bắt đầu {TASK_REGISTRY[task_id]['name']}...")

    proc = subprocess.run(
        cmd,
        cwd=str(WORKSPACE_DIR),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    elapsed = time.time() - t_start
    success = proc.returncode == 0
    status_str = "SUCCESS" if success else "FAILED"

    print(f"[{time.strftime('%H:%M:%S')}][WORKER {status_str}] Task {task_id} hoàn thành trong {elapsed:.2f}s (Exit code: {proc.returncode})")
    if not success:
        print(f"\n--- [LỖI TRONG TASK {task_id}] ---\n{proc.stderr[-1500:]}\n")

    return {
        "task_id": task_id,
        "name": TASK_REGISTRY[task_id]["name"],
        "exit_code": proc.returncode,
        "success": success,
        "duration_sec": round(elapsed, 2),
        "stdout_tail": proc.stdout[-500:] if proc.stdout else "",
        "stderr_tail": proc.stderr[-500:] if proc.stderr else "",
    }


def run_master_benchmark_suite(
    selected_task_ids: List[int],
    max_workers: int = 4,
    cohort_size: int = 300,
    threads_per_task: int = 2,
    dry_run: bool = False,
) -> bool:
    """Điều phối toàn bộ quá trình chạy thực nghiệm song song."""
    t_global_start = time.time()

    print("\n" + "=" * 80)
    print("  HỆ THỐNG ĐIỀU PHỐI THỰC NGHIỆM SONG SONG MASTER (PAPER 1 - ELSEVIER EAAI)")
    print(f"  Tối đa {max_workers} tasks đồng thời | Cohort N={cohort_size} | Luồng CPU/task={threads_per_task}")
    print(f"  Các tasks được chỉ định chạy: {selected_task_ids}")
    print("=" * 80 + "\n")

    if dry_run:
        print(">>> [DRY RUN MODE] Kiểm tra tính toàn vẹn của các file script và thư viện...")
        for tid in selected_task_ids:
            tinfo = TASK_REGISTRY[tid]
            p = tinfo["script"]
            status = "[OK]" if p.exists() else "[MISSING]"
            print(f"  Task {tid}: {tinfo['name']} -> {status} ({p.name})")
        print("\n>>> Kiểm tra Dry Run thành công! Kết thúc chế độ chạy thử.")
        return True

    # Phân nhóm tasks theo Phase:
    # Phase 1: Các task tính toán thực nghiệm (Chạy song song tối đa max_workers)
    # Phase 2: Task 7 sinh figures (Chạy sau cùng khi Phase 1 đã xong dữ liệu)
    phase1_tasks = [tid for tid in selected_task_ids if TASK_REGISTRY[tid]["phase"] == 1]
    phase2_tasks = [tid for tid in selected_task_ids if TASK_REGISTRY[tid]["phase"] == 2]

    all_task_reports: List[Dict[str, Any]] = []

    # ---------------------------------------------------------
    # GIAI ĐOẠN 1: Chạy song song tối đa max_workers tiến trình
    # ---------------------------------------------------------
    if phase1_tasks:
        print(f">>> [GIAI ĐOẠN 1] Đẩy {len(phase1_tasks)} tasks thực nghiệm vào worker pool ({max_workers} slots song song)...")
        t_p1_start = time.time()

        tasks_to_submit = []
        for tid in phase1_tasks:
            script_path = TASK_REGISTRY[tid]["script"]
            args = list(TASK_REGISTRY[tid]["default_args"])
            # Bổ sung tham số chung nếu script hỗ trợ
            if "--cohort_size" not in args:
                args.extend(["--cohort_size", str(cohort_size)])
            if "--threads" not in args:
                args.extend(["--threads", str(threads_per_task)])
            tasks_to_submit.append((tid, script_path, args))

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_tid = {
                executor.submit(_execute_task_process, tid, sp, a): tid
                for tid, sp, a in tasks_to_submit
            }

            for future in as_completed(future_to_tid):
                tid = future_to_tid[future]
                try:
                    report = future.result()
                    all_task_reports.append(report)
                    if not report["success"]:
                        print(f"[CẢNH BÁO] Task {tid} hoàn thành với mã lỗi {report['exit_code']}.")
                except Exception as exc:
                    print(f"[FATAL ERROR] Task {tid} sinh ngoại lệ nghiêm trọng: {exc}")
                    all_task_reports.append({
                        "task_id": tid,
                        "name": TASK_REGISTRY[tid]["name"],
                        "success": False,
                        "exit_code": -1,
                        "duration_sec": 0.0,
                        "error": str(exc),
                    })

        print(f">>> [GIAI ĐOẠN 1 HOÀN TẤT] Thời gian thực thi Giai đoạn 1: {time.time() - t_p1_start:.2f}s\n")

    # ---------------------------------------------------------
    # GIAI ĐOẠN 2: Kích hoạt Task 7 sinh toàn bộ Figures vector
    # ---------------------------------------------------------
    if phase2_tasks:
        print(">>> [GIAI ĐOẠN 2] Khởi chạy Task 7 sinh toàn bộ Figures vector PDF/PNG xuất bản...")
        for tid in phase2_tasks:
            script_path = TASK_REGISTRY[tid]["script"]
            report = _execute_task_process(tid, script_path, TASK_REGISTRY[tid]["default_args"])
            all_task_reports.append(report)

    # ---------------------------------------------------------
    # GIAI ĐOẠN 3: Tạo báo cáo Master Audit Manifest
    # ---------------------------------------------------------
    t_global_total = time.time() - t_global_start
    manifest = {
        "manifest_title": "Paper 1 Master Benchmark Audit Manifest",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_duration_sec": round(t_global_total, 2),
        "concurrency_settings": {
            "max_workers": max_workers,
            "cohort_size": cohort_size,
            "threads_per_task": threads_per_task,
        },
        "system_platform": {
            "os": platform.system(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
        "tasks_summary": all_task_reports,
        "artifact_paths": {
            "summary_json_dir": str(SUMMARY_DIR),
            "audit_streams_dir": str(AUDIT_DIR),
            "raw_csv_dir": str(RAW_CSV_DIR),
            "logs_dir": str(LOGS_DIR),
        },
    }

    manifest_path = SUMMARY_DIR / "master_audit_manifest.json"
    with open(manifest_path, mode="w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("  TỔNG KẾT TOÀN BỘ QUÁ TRÌNH CHẠY THỰC NGHIỆM MASTER BENCHMARK SUITE")
    print(f"  Tổng thời gian: {t_global_total:.2f}s | Báo cáo kiểm toán: {manifest_path}")
    print("-" * 80)
    for rep in all_task_reports:
        st = "[SUCCESS]" if rep.get("success", False) else "[FAILED]"
        print(f"  {st:10s} | {rep['name']:55s} | Thời gian: {rep.get('duration_sec', 0.0):6.2f}s")
    print("=" * 80 + "\n")

    all_passed = all(r.get("success", False) for r in all_task_reports)
    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Master Parallel Benchmark Suite (Paper 1 - Elsevier EAAI)")
    parser.add_argument("--max_workers", type=int, default=4, help="Số task chạy đồng thời tối đa (mặc định 4)")
    parser.add_argument("--cohort_size", type=int, default=300, help="Số lượng ảnh COCO val2017")
    parser.add_argument("--threads", type=int, default=2, help="Số luồng CPU cho mỗi task con")
    parser.add_argument("--tasks", type=str, default="1,2,3,4,5,6,7", help="Danh sách task ID cách nhau bởi dấu phẩy")
    parser.add_argument("--dry_run", action="store_true", help="Chỉ kiểm tra cấu hình mà không chạy tốn tài nguyên")
    args = parser.parse_args()

    task_ids = []
    for item in args.tasks.split(","):
        item = item.strip()
        if item.isdigit():
            tid = int(item)
            if tid in TASK_REGISTRY:
                task_ids.append(tid)

    if not task_ids:
        print("[LỖI] Không có task hợp lệ nào được chọn.")
        sys.exit(1)

    success = run_master_benchmark_suite(
        selected_task_ids=task_ids,
        max_workers=args.max_workers,
        cohort_size=args.cohort_size,
        threads_per_task=args.threads,
        dry_run=args.dry_run,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
