# -*- coding: utf-8 -*-
"""
Script: paper_3_defense_guardrail.run_batch_experiments
Mục đích: Điều phối chạy toàn bộ bộ thực nghiệm Paper 3 một cách tự động, tối ưu hóa và trực quan:
1. Giai đoạn 1 (Tuần tự bắt buộc): Chạy Step 1 (run_calibration.py) với Live Stream Output & TQDM
   để người dùng theo dõi trực tiếp thanh tiến trình trích xuất 1200 ảnh và fit detectors.
2. Giai đoạn 2 (Song song tối đa 4 workers): Đưa toàn bộ các thực nghiệm độc lập còn lại vào
   hàng đợi (Queue) và chạy đồng thời tối đa 4 tiến trình (concurrency = 4).
   Tích hợp Live Heartbeat Ticker & Progress Monitor:
   - Tự động đọc tail log của từng worker đang chạy và hiển thị tiến độ thời gian thực (real-time).
   - Hiển thị thanh tiến trình TQDM tổng thể theo số tác vụ hoàn tất.
3. Hỗ trợ cờ `--stream`: Cho phép stream trực tiếp toàn bộ log/tqdm ra console cho từng tác vụ.
4. Quản lý log độc lập từng tiến trình ra file *.log và *.err.log, tổng kết bảng kết quả chi tiết.
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm

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


def get_last_log_snippet(log_path: Path) -> str:
    """Đọc dòng log / tiến độ mới nhất từ file log của worker đang chạy."""
    if not log_path.exists():
        return "Đang khởi tạo tiến trình..."
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
            if not lines:
                return "Đang nạp mô hình & dữ liệu..."
            # Lấy dòng mới nhất có ý nghĩa
            for l in reversed(lines[-8:]):
                if any(k in l for k in ["%", "it/s", "ảnh/s", "ĐANG ĐÁNH GIÁ", "Bắt đầu", "Epoch", "Batch", "Trích xuất", "ASR", "Hoàn tất"]):
                    # Cắt ngắn nếu quá dài
                    clean_l = l.split("] ")[-1] if "] " in l else l
                    return clean_l[:75] + ("..." if len(clean_l) > 75 else "")
            return lines[-1][:75] + ("..." if len(lines[-1]) > 75 else "")
    except Exception:
        return "Đang thực thi..."


def run_stream_task(task_key: str, task_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Chạy tác vụ ở chế độ Live Streaming: vừa in trực tiếp TQDM / log ra màn hình,
    vừa ghi đồng thời ra file log.
    Gộp stderr=subprocess.STDOUT để tránh triệt để OS pipe buffer deadlock trên Windows.
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
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            # Đọc và in theo thời gian thực từ stream đã gộp
            for line in iter(proc.stdout.readline, ""):
                sys.stdout.write(line)
                sys.stdout.flush()
                f_out.write(line)
                f_out.flush()

            proc.stdout.close()
            returncode = proc.wait()

        elapsed = time.time() - t0
        has_error = returncode != 0
        err_tail = ""
        if has_error:
            try:
                with open(out_log, "r", encoding="utf-8", errors="replace") as fo:
                    err_tail = "".join(fo.readlines()[-20:])
            except Exception:
                pass

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
        description="Batch Parallel Runner for Paper 3 Guardrail Experiments (Max 4 Concurrency with Live TQDM/Log)"
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
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Bật chế độ Live Stream in thẳng stdout/tqdm ra console (phù hợp khi chạy 1-2 tasks cụ thể).",
    )
    args = parser.parse_args()

    max_workers = min(max(1, args.max_workers), 8)

    print("=" * 95)
    print("   HỆ THỐNG ĐIỀU PHỐI CHẠY THỰC NGHIỆM BATCH PARALLEL (PAPER 3: DEFENSE GUARDRAIL)   ")
    print(f"   Python Executable : {PYTHON_EXE}")
    print(f"   Số luồng đồng thời: {max_workers} WORKERS")
    print(f"   Thư mục lưu Logs  : {LOGS_DIR}")
    print(f"   Thư mục lưu Data  : {RESULTS_DIR}")
    print("=" * 95)

    total_start_time = time.time()
    execution_results: List[Dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # GIAI ĐOẠN 1: BƯỚC 1 CALIBRATION (BẮT BUỘC CHẠY TRƯỚC NẾU CHƯA CÓ CHECKPOINT)
    # -------------------------------------------------------------------------
    ckpt_file = RESULTS_DIR / "checkpoints" / "calibration_and_detectors.pkl" if (RESULTS_DIR / "checkpoints" / "calibration_and_detectors.pkl").exists() else (WORKSPACE_DIR / "edge_sec_pro" / "results" / "checkpoints" / "calibration_and_detectors.pkl")
    should_run_step1 = not args.skip_calibration or not ckpt_file.exists()

    if should_run_step1:
        print("\n>>> [GIAI ĐOẠN 1] KHỞI CHẠY BƯỚC 1 (CALIBRATION & FIT DETECTORS - LIVE STREAM & TQDM)...")
        step1_info = TASK_REGISTRY["step1_calibration"]
        # Step 1 chạy đơn lẻ nên stream trực tiếp ra màn hình để người dùng thấy TQDM
        res_step1 = run_stream_task("step1_calibration", step1_info)
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
            print(f"\n[+] BƯỚC 1 HOÀN TẤT THÀNH CÔNG trong [{res_step1['elapsed_seconds']}s]!")
            print(f"    Checkpoint đã được tạo tại: {ckpt_file}\n")
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

    # Tách Step 11 nếu chạy cùng các tasks khác để đảm bảo Step 11 đọc kết quả tươi từ Step 4 và 5
    run_step11_after = ("step11_mcnemar_tests" in parallel_tasks) and (len(parallel_tasks) > 1)
    step11_info_deferred = None
    if run_step11_after:
        step11_info_deferred = parallel_tasks.pop("step11_mcnemar_tests")

    if not parallel_tasks and not run_step11_after:
        print("\n[i] Không có tác vụ nào cần thực thi tiếp. Kết thúc chương trình.")
        return

    print(f"\n>>> [GIAI ĐOẠN 2] KHỞI CHẠY {len(parallel_tasks)} THỰC NGHIỆM SONG SONG (TỐI ĐA {max_workers} TIẾN TRÌNH CÙNG LÚC)...")
    print("-" * 95)
    for k, v in parallel_tasks.items():
        print(f"   * [{k:<26}] : {v['name']}")
    print("-" * 95)

    if args.stream and len(parallel_tasks) == 1:
        # Nếu chỉ chạy 1 task và yêu cầu stream: chạy tuần tự stream trực tiếp
        t_key, t_info = list(parallel_tasks.items())[0]
        print(f"\n[Chế độ Stream] Đang chạy trực tiếp: {t_info['name']}...\n")
        res = run_stream_task(t_key, t_info)
        execution_results.append(res)
    else:
        # Chạy hàng đợi đa tiến trình với Live Progress Monitor
        task_queue = list(parallel_tasks.items())
        total_tasks = len(task_queue)
        active_processes: Dict[str, Dict[str, Any]] = {}
        completed_tasks: List[Dict[str, Any]] = []

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"

        print(f"\nĐang kích hoạt hàng đợi với {max_workers} slots song song. Cập nhật tiến độ theo thời gian thực (Live Dashboard):\n")

        pbar = tqdm(total=total_tasks, desc="  Tổng tiến độ Batch", unit="task", dynamic_ncols=True)
        last_heartbeat_time = 0

        while task_queue or active_processes:
            # 1. Đẩy tác vụ mới vào các slot còn trống
            while task_queue and len(active_processes) < max_workers:
                t_key, t_info = task_queue.pop(0)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                out_log = LOGS_DIR / f"{t_key}_batch_{timestamp}.log"
                err_log = LOGS_DIR / f"{t_key}_batch_{timestamp}.err.log"

                f_out = open(out_log, "w", encoding="utf-8")
                f_err = open(err_log, "w", encoding="utf-8")

                proc = subprocess.Popen(
                    [PYTHON_EXE, str(t_info["script"])],
                    cwd=str(PAPER3_DIR),
                    env=env,
                    stdout=f_out,
                    stderr=f_err,
                    text=True,
                )

                active_processes[t_key] = {
                    "proc": proc,
                    "info": t_info,
                    "f_out": f_out,
                    "f_err": f_err,
                    "out_log": out_log,
                    "err_log": err_log,
                    "t0": time.time(),
                }
                tqdm.write(f"  [+] [DISPATCH] Khởi chạy: {t_info['name']}")

            time.sleep(1.0)

            # 2. Kiểm tra các tiến trình đã hoàn thành
            finished_keys = []
            for t_key, p_data in active_processes.items():
                ret = p_data["proc"].poll()
                if ret is not None:
                    # Đã kết thúc
                    p_data["f_out"].close()
                    p_data["f_err"].close()
                    elapsed = time.time() - p_data["t0"]
                    success = ret == 0

                    err_tail = ""
                    if not success and p_data["err_log"].exists():
                        try:
                            with open(p_data["err_log"], "r", encoding="utf-8", errors="replace") as fe:
                                err_tail = "".join(fe.readlines()[-15:])
                        except Exception:
                            pass

                    res_item = {
                        "task_key": t_key,
                        "task_name": p_data["info"]["name"],
                        "returncode": ret,
                        "elapsed_seconds": round(elapsed, 2),
                        "out_log": str(p_data["out_log"]),
                        "err_log": str(p_data["err_log"]),
                        "success": success,
                        "error_tail": err_tail,
                    }
                    completed_tasks.append(res_item)
                    execution_results.append(res_item)
                    finished_keys.append(t_key)

                    status_str = "THÀNH CÔNG" if success else "THẤT BẠI"
                    mark = "[SUCCESS]" if success else "[FAILED] "
                    tqdm.write(f"  {mark} ({len(completed_tasks)}/{total_tasks}) {p_data['info']['name']} -> {status_str} trong [{elapsed:.1f}s]")
                    if not success and err_tail:
                        tqdm.write(f"            [!] Chi tiết lỗi tại: {p_data['err_log']}")
                    pbar.update(1)

            for fk in finished_keys:
                del active_processes[fk]

            # 3. Hiển thị Heartbeat Ticker mỗi 4 giây cho các tác vụ đang chạy
            now = time.time()
            if active_processes and (now - last_heartbeat_time >= 4.0):
                last_heartbeat_time = now
                running_summary = []
                for tk, pd in active_processes.items():
                    dur = int(now - pd["t0"])
                    snip = get_last_log_snippet(pd["out_log"])
                    running_summary.append(f"[{tk} ({dur}s): {snip}]")
                tqdm.write("    >>> [Đang chạy] " + " | ".join(running_summary[:2]))
                if len(running_summary) > 2:
                    tqdm.write("                    " + " | ".join(running_summary[2:]))

        pbar.close()

    # -------------------------------------------------------------------------
    # GIAI ĐOẠN 2B: CHẠY STEP 11 (MCNEMAR TESTS) DỰA TRÊN KẾT QUẢ TƯƠI TỪ BƯỚC 4 VÀ 5
    # -------------------------------------------------------------------------
    if step11_info_deferred is not None:
        print("\n>>> [GIAI ĐOẠN 2B] KHỞI CHẠY BƯỚC 11 (MCNEMAR TESTS) TỔNG HỢP TRỰC TIẾP TỪ MẪU TƯƠI BƯỚC 4 & 5...")
        res_step11 = run_stream_task("step11_mcnemar_tests", step11_info_deferred)
        execution_results.append(res_step11)
        if res_step11["success"]:
            print(f"[+] BƯỚC 11 HOÀN TẤT THÀNH CÔNG trong [{res_step11['elapsed_seconds']}s]!")
        else:
            print(f"[!] BƯỚC 11 THẤT BẠI. Chi tiết lỗi tại: {res_step11['err_log']}")

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
