# -*- coding: utf-8 -*-
"""
Script: paper_3_defense_guardrail.run_batch_experiments
Mục đích: Điều phối chạy toàn bộ bộ thực nghiệm Paper 3 một cách tự động và tối ưu hóa thời gian:
1. Giai đoạn 1 (Tuần tự bắt buộc): Chạy Step 1 (run_calibration.py) để sinh checkpoint
   hiệu chuẩn các bộ giám sát (calibration_and_detectors.pkl).
2. Giai đoạn 2 (Song song tối đa 4 workers): Đưa toàn bộ các thực nghiệm độc lập còn lại vào
   hàng đợi (Queue) và chạy đồng thời tối đa 4 tiến trình (concurrency = 4).
   Khi một tiến trình hoàn tất, tiến trình tiếp theo lập tức được kích hoạt ngay.
3. Quản lý log độc lập cho từng tiến trình, hiển thị trạng thái thời gian thực và tổng kết chi tiết.
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Thiết lập đường dẫn workspace
PAPER3_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = PAPER3_DIR.parent
EXPERIMENTS_DIR = PAPER3_DIR / "experiments"
LOGS_DIR = PAPER3_DIR / "logs"
RESULTS_DIR = PAPER3_DIR / "results"

LOGS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Tìm Python executable phù hợp (.venv hoặc system python)
VENV_PYTHON = WORKSPACE_DIR / ".venv" / "Scripts" / "python.exe"
if VENV_PYTHON.exists():
    PYTHON_EXE = str(VENV_PYTHON)
else:
    PYTHON_EXE = sys.executable


# Danh sách định nghĩa các tác vụ thực nghiệm
TASK_REGISTRY: Dict[str, Dict[str, Any]] = {
    "step1_calibration": {
        "name": "Step 1: Calibration & Detectors Fitting (N=1200 / N=600)",
        "script": EXPERIMENTS_DIR / "run_calibration.py",
        "description": "Hiệu chuẩn phân phối sạch và fit 6 detectors",
        "is_dependency": True,
    },
    "step2_controls_fpr": {
        "name": "Step 2: Controls & Natural Shifts FPR Stability (N=300x4)",
        "script": EXPERIMENTS_DIR / "run_controls_and_fpr.py",
        "description": "Đo độ ổn định FPR dưới nhiễu Gaussian, blur, patch, brightness",
        "is_dependency": False,
    },
    "step3_patch_attack": {
        "name": "Step 3: Adversarial Patch 48x48 & Dimension Sweep (16..64)",
        "script": EXPERIMENTS_DIR / "run_patch_attack_eval.py",
        "description": "Huấn luyện patch Adam EoT và quét kích thước patch",
        "is_dependency": False,
    },
    "step4_aqsm_esm": {
        "name": "Step 4: AQSM & Encoder Selection Monitor (ESM) (N=300)",
        "script": EXPERIMENTS_DIR / "run_aqsm_and_esm.py",
        "description": "Đánh giá đòn AQSM và phân phối suy giảm điểm ESM",
        "is_dependency": False,
    },
    "step5_query_attacks": {
        "name": "Step 5: Query Attacks (EQF, Vanishing, DQF) (N=300)",
        "script": EXPERIMENTS_DIR / "run_query_attacks_eval.py",
        "description": "Đánh giá 3 đòn tấn công thao túng query decoder/encoder",
        "is_dependency": False,
    },
    "step6_adaptive_attacks": {
        "name": "Step 6: White-Box Defense-Aware Adaptive Evasion (N=300)",
        "script": EXPERIMENTS_DIR / "run_adaptive_attacks_eval.py",
        "description": "Đánh giá đòn né tránh thích nghi 1D, Joint Maha và lambda sweep",
        "is_dependency": False,
    },
    "step9_defensive_baselines": {
        "name": "Step 9: Comparison with Defensive Baselines (TV & JPEG-Diff)",
        "script": EXPERIMENTS_DIR / "run_defensive_baselines.py",
        "description": "Đối sánh trực diện với Total Variation và JPEG Reconstruction",
        "is_dependency": False,
    },
    "step8_edge_latency": {
        "name": "Step 8: Edge Hardware Latency & Overhead Profile (50 runs)",
        "script": EXPERIMENTS_DIR / "run_edge_latency.py",
        "description": "Đo đạc độ trễ suy luận, trích xuất đặc trưng và Max-OR",
        "is_dependency": False,
    },
    "step10_epsilon_sweep": {
        "name": "Step 10: Epsilon Sensitivity Sweep (eps in [2, 4, 8, 16]/255)",
        "script": EXPERIMENTS_DIR / "run_epsilon_sweep.py",
        "description": "Đo độ suy giảm mAP và TPR của ESM/ActiveQ theo epsilon",
        "is_dependency": False,
    },
    "step11_mcnemar_tests": {
        "name": "Step 11: Paired McNemar Hypothesis Tests (Table 5 Replication)",
        "script": EXPERIMENTS_DIR / "run_mcnemar_tests.py",
        "description": "Kiểm định giả thuyết ghép cặp McNemar so sánh các kênh",
        "is_dependency": False,
    },
}


def run_single_task(task_key: str, task_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Thực thi một tác vụ thực nghiệm độc lập trong tiến trình con (subprocess).
    Tự động chuyển hướng stdout và stderr ra file log riêng biệt.
    """
    script_path = task_info["script"]
    task_name = task_info["name"]

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_log = LOGS_DIR / f"{task_key}_batch_{timestamp}.log"
    err_log = LOGS_DIR / f"{task_key}_batch_{timestamp}.err.log"

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    t0 = time.time()
    try:
        with open(out_log, "w", encoding="utf-8") as f_out, open(err_log, "w", encoding="utf-8") as f_err:
            proc = subprocess.Popen(
                [PYTHON_EXE, str(script_path)],
                cwd=str(PAPER3_DIR),
                env=env,
                stdout=f_out,
                stderr=f_err,
                text=True,
            )
            returncode = proc.wait()

        elapsed = time.time() - t0
        has_error = returncode != 0

        # Đọc tail lỗi nếu có
        err_tail = ""
        if has_error and err_log.exists():
            with open(err_log, "r", encoding="utf-8", errors="replace") as f_err:
                err_lines = f_err.readlines()
                err_tail = "".join(err_lines[-15:])

        return {
            "task_key": task_key,
            "task_name": task_name,
            "returncode": returncode,
            "elapsed_seconds": round(elapsed, 2),
            "out_log": str(out_log),
            "err_log": str(err_log),
            "success": not has_error,
            "error_tail": err_tail,
        }
    except Exception as e:
        elapsed = time.time() - t0
        return {
            "task_key": task_key,
            "task_name": task_name,
            "returncode": -1,
            "elapsed_seconds": round(elapsed, 2),
            "out_log": str(out_log),
            "err_log": str(err_log),
            "success": False,
            "error_tail": str(e),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Batch Parallel Runner for Paper 3 Guardrail Experiments (Max 4 Concurrency)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Số lượng tiến trình chạy đồng thời tối đa (mặc định: 4).",
    )
    parser.add_argument(
        "--skip-calibration",
        action="store_true",
        help="Bỏ qua Bước 1 nếu calibration_and_detectors.pkl đã tồn tại sẵn.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Chỉ định danh sách tasks cụ thể cần chạy (ví dụ: --tasks step2_controls_fpr step4_aqsm_esm).",
    )
    args = parser.parse_args()

    max_workers = min(max(1, args.max_workers), 8)

    print("=" * 95)
    print("   HỆ THỐNG ĐIỀU PHỐI CHẠY THỰC NGHIỆM BATCH PARALLEL (PAPER 3: DEFENSE GUARDRAIL)   ")
    print(f"   Python Executable: {PYTHON_EXE}")
    print(f"   Số luồng song song tối đa (Concurrency Pool) : {max_workers} WORKERS")
    print(f"   Thư mục lưu trữ logs: {LOGS_DIR}")
    print(f"   Thư mục lưu kết quả : {RESULTS_DIR}")
    print("=" * 95)

    total_start_time = time.time()
    execution_results: List[Dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # GIAI ĐOẠN 1: BƯỚC 1 CALIBRATION (BẮT BUỘC CHẠY TRƯỚC NẾU CHƯA CÓ CHECKPOINT)
    # -------------------------------------------------------------------------
    ckpt_file = WORKSPACE_DIR / "edge_sec_pro" / "results" / "checkpoints" / "calibration_and_detectors.pkl"
    should_run_step1 = not args.skip_calibration or not ckpt_file.exists()

    if should_run_step1:
        print("\n>>> [GIAI ĐOẠN 1] KHỞI CHẠY BƯỚC 1 (CALIBRATION & FIT DETECTORS - BẮT BUỘC TUẦN TỰ)...")
        t1_start = time.time()
        step1_info = TASK_REGISTRY["step1_calibration"]
        res_step1 = run_single_task("step1_calibration", step1_info)
        execution_results.append(res_step1)

        if not res_step1["success"]:
            print(f"\n[!] BƯỚC 1 THẤT BẠI (Mã lỗi: {res_step1['returncode']}). Toàn bộ các bước sau không thể tiếp tục!")
            print(f"    Chi tiết lỗi tại: {res_step1['err_log']}")
            if res_step1["error_tail"]:
                print("-" * 50)
                print(res_step1["error_tail"])
                print("-" * 50)
            sys.exit(1)
        else:
            print(f"[+] BƯỚC 1 HOÀN TẤT THÀNH CÔNG trong [{res_step1['elapsed_seconds']}s]!")
            print(f"    Checkpoint đã được tạo tại: {ckpt_file}")
    else:
        print(f"\n>>> [GIAI ĐOẠN 1] Bỏ qua Bước 1 vì checkpoint đã có sẵn: {ckpt_file.name}")

    # -------------------------------------------------------------------------
    # GIAI ĐOẠN 2: CHẠY ĐỒNG THỜI CÁC THỰC NGHIỆM ĐỘC LẬP (MAX CONCURRENCY = 4)
    # -------------------------------------------------------------------------
    parallel_tasks: Dict[str, Dict[str, Any]] = {}
    for key, val in TASK_REGISTRY.items():
        if key == "step1_calibration":
            continue
        if args.tasks is not None and key not in args.tasks:
            continue
        parallel_tasks[key] = val

    print(f"\n>>> [GIAI ĐOẠN 2] KHỞI CHẠY {len(parallel_tasks)} THỰC NGHIỆM SONG SONG (TỐI ĐA {max_workers} TIẾN TRÌNH CÙNG LÚC)...")
    print("-" * 95)
    for k, v in parallel_tasks.items():
        print(f"   * [{k:<26}] : {v['name']}")
    print("-" * 95)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(run_single_task, k, v): (k, v)
            for k, v in parallel_tasks.items()
        }

        print(f"\nĐã kích hoạt {min(len(parallel_tasks), max_workers)} tiến trình đầu tiên trong hàng đợi. Đang theo dõi tiến độ...\n")

        completed_count = 0
        total_parallel = len(parallel_tasks)

        for future in as_completed(future_to_task):
            task_key, task_info = future_to_task[future]
            completed_count += 1
            try:
                result = future.result()
                execution_results.append(result)
                status = "[SUCCESS]" if result["success"] else "[FAILED] "
                print(f"  {status} ({completed_count}/{total_parallel}) {task_info['name']}")
                print(f"            Thời gian: {result['elapsed_seconds']:6.2f}s | Log: {Path(result['out_log']).name}")

                if not result["success"]:
                    print(f"            [!] Lỗi phát sinh tại: {result['err_log']}")
                    if result["error_tail"]:
                        print("            " + "-" * 50)
                        for line in result["error_tail"].strip().split("\n")[-5:]:
                            print(f"            {line}")
                        print("            " + "-" * 50)
            except Exception as exc:
                print(f"  [ERROR] ({completed_count}/{total_parallel}) Task {task_key} sinh ngoại lệ: {exc}")

    # -------------------------------------------------------------------------
    # GIAI ĐOẠN 3: TỔNG KẾT BẢNG KẾT QUẢ ĐIỀU PHỐI
    # -------------------------------------------------------------------------
    total_elapsed = time.time() - total_start_time

    print("\n" + "=" * 95)
    print("                      BẢNG TỔNG KẾT TIẾN ĐỘ VÀ KẾT QUẢ BATCH RUNNER                    ")
    print("=" * 95)
    print(f"{'Task Key':<26} | {'Trạng Thái':<12} | {'Thời Gian':<12} | {'File Log Chính':<35}")
    print("-" * 95)

    all_success = True
    for res in execution_results:
        st_text = "THÀNH CÔNG" if res["success"] else "THẤT BẠI"
        if not res["success"]:
            all_success = False
        time_text = f"{res['elapsed_seconds']:.1f}s"
        log_name = Path(res["out_log"]).name
        print(f"{res['task_key']:<26} | {st_text:<12} | {time_text:<12} | {log_name:<35}")

    print("-" * 95)
    print(f"Tổng thời gian hoàn tất toàn bộ pipeline: {total_elapsed:.1f}s (~{total_elapsed/60.0:.2f} phút).")
    print(f"Thư mục lưu trữ Artifacts JSON/CSV    : {RESULTS_DIR}")
    print(f"Thư mục lưu trữ Run Logs               : {LOGS_DIR}")

    if all_success:
        print("\n>>> TẤT CẢ CÁC BƯỚC THỰC NGHIỆM ĐÃ HOÀN TẤT XUẤT SẮC 100%! <<<")
    else:
        print("\n[!] CẢNH BÁO: CÓ ÍT NHẤT MỘT THỰC NGHIỆM GẶP LỖI. VUI LÒNG KIỂM TRA FILE *.ERR.LOG!")

    print("=" * 95)


if __name__ == "__main__":
    main()
