# -*- coding: utf-8 -*-
"""
Module: paper_1_aqsm_mechanism.run_all_experiments
Mục đích: Trình điều phối thực nghiệm toàn diện duy nhất (Master Experiment Orchestrator)
cho Paper 1: Mechanistic Breakdown of Dynamic Query Selection (RT-DETR).

Thực thi khép kín toàn bộ chuỗi thực nghiệm:
  Task 0: Sinh bộ nhớ đệm đối kháng chuẩn Native 640x640 (AQSM, Vanishing, Task PGD).
  Task 1: Phẫu thuật can thiệp nhân quả 2x2 Factorial Tensor-Swap -> Table 3, Table 4, Figure 4.
  Task 2: Động lực học Encoder Memory & Query Selection Perturbation -> Table 5, Table 6, Figure 3, Figure 5.
  Task 3: Telemetry Deformable Attention (Entropy, FG Spatial Hit, Drift) -> Table 7, Figure 6.
  Task 4: Kiểm chứng tính đặc hiệu không gian & Bác bỏ Tautology -> Table 8.
  Task 5: Quét ngân sách đối kháng Matched-Budget L_inf Sweep -> Table 9, Figure 7.
  Task 6: Đo đạc độ trễ phần cứng tự động theo thiết bị (Hardware Latency Telemetry) -> Table 10.
  Task 8: Đánh giá hệ thống phòng thủ Encoder Selection Monitor (ESM) & Adaptive PGD -> Table 11, Figure 8/9.
  Task 7: Tự động kết xuất toàn bộ biểu đồ vector xuất bản chuẩn Elsevier EAAI (PDF & PNG).

Đặc tính chuẩn công nghiệp (Production Standards):
  - Chạy linh hoạt trên toàn bộ ảnh COCO val2017 (N=5,000 ảnh) hoặc tập con bất kỳ thông qua `--folder_image` hoặc `--max_samples`.
  - Quản lý bộ nhớ hoàn hảo: Giải phóng VRAM/RAM giữa các giai đoạn thông qua subprocess isolation.
  - Áp dụng cấu trúc TASK_REGISTRY module hóa cao, dễ bảo trì, mở rộng.
  - Không hardcode, không làm giả số liệu; xuất đầy đủ context trong JSON/CSV để điền vào bài báo.
"""

import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from paper_1_aqsm_mechanism.experiments.config import (
    ADV_CACHE_DIR,
    COCO_ANN_FILE,
    COCO_VAL_DIR,
    FIG_DIR,
    LOGS_DIR,
    MODEL_NAME,
    OUTPUTS_DIR,
    detect_hardware_environment,
    resolve_model_name,
    update_experiment_paths,
)
from paper_1_aqsm_mechanism.utils.io_helpers import format_duration, save_json


def run_subtask(
    script_path: Path,
    args_list: List[str],
    task_name: str,
    log_file: Optional[Path] = None,
) -> float:
    """
    Thực thi một subtask độc lập qua subprocess nhằm cách ly hoàn toàn bộ nhớ RAM/VRAM.
    """
    print("\n" + "=" * 80)
    print(f" >>> [BẮT ĐẦU] {task_name}")
    print(f" Lệnh: {sys.executable} {script_path.name} {' '.join(args_list)}")
    print("=" * 80)

    cmd = [sys.executable, str(script_path)] + args_list
    t0 = time.time()

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "w", encoding="utf-8") as f_log:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                f_log.write(line)
            proc.wait()
            ret_code = proc.returncode
    else:
        res = subprocess.run(cmd)
        ret_code = res.returncode

    elapsed = time.time() - t0
    if ret_code != 0:
        err_tail = ""
        if log_file and log_file.exists():
            try:
                with open(log_file, "r", encoding="utf-8", errors="replace") as f_check:
                    lines = f_check.readlines()
                    err_tail = "".join(lines[-25:])
            except Exception:
                pass
        err_msg = f"Task thất bại với mã lỗi {ret_code}: {task_name}"
        if err_tail.strip():
            err_msg += f"\n\n[TRÍCH XUẤT 25 DÒNG CUỐI TỪ NHẬT KÝ LOG: {log_file.name}]:\n{err_tail}"
        raise RuntimeError(err_msg)

    print(f"\n >>> [HOÀN THÀNH] {task_name} trong {format_duration(elapsed)} ({elapsed:.2f}s)")
    return elapsed


def _build_task_registry(
    base_dir: Path,
    args: argparse.Namespace,
    common_args: List[str],
) -> Dict[int, Dict[str, Any]]:
    """
    Đăng ký danh mục các Task thực nghiệm chuẩn hóa.
    """
    tasks_dir = base_dir / "experiments" / "tasks"
    logs_dir = Path(args.output_dir) / "logs"

    # Task 0 args
    def build_task0_args() -> List[str]:
        t0_args = [
            "--folder_image", str(args.folder_image),
            "--annotations", str(args.annotations),
            "--cache_dir", str(args.cache_dir),
            "--steps", str(args.pgd_steps),
            "--threads", str(args.threads),
            "--device", str(args.device),
        ]
        if args.include_sweep:
            t0_args.append("--include_sweep")
        if args.batch_size is not None:
            t0_args.extend(["--batch_size", str(args.batch_size)])
        if args.force_recompute:
            t0_args.append("--force_recompute")
        if args.max_samples is not None:
            t0_args.extend(["--max_samples", str(args.max_samples)])
        if getattr(args, "model", None) is not None:
            t0_args.extend(["--model", str(args.model)])
        return t0_args

    # Task 6 args
    def build_task6_args() -> List[str]:
        n_runs_val = str(min(300, args.max_samples)) if args.max_samples is not None else "300"
        t6_args = [
            "--folder_image", str(args.folder_image),
            "--annotations", str(args.annotations),
            "--output_dir", str(args.output_dir),
            "--threads", str(args.threads),
            "--device", str(args.device),
            "--n_runs", n_runs_val,
        ]
        if getattr(args, "model", None) is not None:
            t6_args.extend(["--model", str(args.model)])
        return t6_args

    # Task 7 args
    def build_task7_args() -> List[str]:
        return [
            "--output_dir", str(args.output_dir),
            "--fig_dir", str(args.fig_dir),
        ]

    # Task 9 args
    def build_task9_args() -> List[str]:
        return [
            "--output_dir", str(args.output_dir),
            "--tables_dir", str(Path(args.output_dir) / "tables"),
            "--verify_tex",
        ]

    return {
        0: {
            "name": "Task 0 (Cache Gen)",
            "title": "Task 0: Adversarial Cache Generation (Native 640x640)",
            "script": base_dir / "experiments" / "generate_adv_cache_640.py",
            "args_builder": build_task0_args,
            "log_file": logs_dir / "task0_cache_generation.log",
            "skip_condition": lambda: args.skip_cache_gen,
        },
        1: {
            "name": "Task 1 (Causal Surgery)",
            "title": "Task 1: Causal Surgery 2x2 Factorial Swap (Table 3, Table 4, Figure 4)",
            "script": tasks_dir / "task1_causal_surgery.py",
            "args_builder": lambda: list(common_args),
            "log_file": logs_dir / "task1_causal_surgery.log",
        },
        2: {
            "name": "Task 2 (Encoder Dynamics)",
            "title": "Task 2: Encoder Memory & Query Selection Dynamics (Table 5, Table 6)",
            "script": tasks_dir / "task2_encoder_query_dynamics.py",
            "args_builder": lambda: list(common_args),
            "log_file": logs_dir / "task2_encoder_query_dynamics.log",
        },
        3: {
            "name": "Task 3 (Attention Telemetry)",
            "title": "Task 3: Deformable Attention Telemetry (Table 7, Figure 6)",
            "script": tasks_dir / "task3_deformable_telemetry.py",
            "args_builder": lambda: list(common_args),
            "log_file": logs_dir / "task3_deformable_telemetry.log",
        },
        4: {
            "name": "Task 4 (Tautology Refutation)",
            "title": "Task 4: Controlled Spatial Misrouting & Tautology Refutation (Table 8)",
            "script": tasks_dir / "task4_controlled_misrouting.py",
            "args_builder": lambda: list(common_args),
            "log_file": logs_dir / "task4_controlled_misrouting.log",
        },
        5: {
            "name": "Task 5 (Matched Sweep)",
            "title": "Task 5: Matched-Budget Perturbation Sweep (Table 9, Figure 8)",
            "script": tasks_dir / "task5_matched_sweep.py",
            "args_builder": lambda: list(common_args),
            "log_file": logs_dir / "task5_matched_sweep.log",
        },
        6: {
            "name": "Task 6 (Latency Profiler)",
            "title": "Task 6: Hardware Execution Latency Profiler (Table 10)",
            "script": tasks_dir / "task6_latency_profiler.py",
            "args_builder": build_task6_args,
            "log_file": logs_dir / "task6_latency_profiler.log",
        },
        8: {
            "name": "Task 8 (Defense Evaluation)",
            "title": "Task 8: Empirical Defense ESM & Adaptive Evasion Evaluation (Table 11, Figure 9)",
            "script": tasks_dir / "task8_defense_evaluation.py",
            "args_builder": lambda: list(common_args),
            "log_file": logs_dir / "task8_defense_evaluation.log",
        },
        7: {
            "name": "Task 7 (Figure Generation)",
            "title": "Task 7: Generate Publication Figures (PDF/PNG Vector Graphics)",
            "script": tasks_dir / "task7_generate_figures.py",
            "args_builder": build_task7_args,
            "log_file": logs_dir / "task7_generate_figures.log",
        },
        9: {
            "name": "Task 9 (Table Generation)",
            "title": "Task 9: Generate Publication Tables (LaTeX snippets, Markdown, CSV)",
            "script": tasks_dir / "task9_generate_tables.py",
            "args_builder": build_task9_args,
            "log_file": logs_dir / "task9_generate_tables.log",
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Master Experiment Orchestrator: Chạy toàn bộ thực nghiệm Paper 1 (AQSM Mechanism)"
    )
    # 1. Tham số dữ liệu đầu vào & Mô hình
    parser.add_argument("--folder_image", type=str, default=str(COCO_VAL_DIR), help="Đường dẫn thư mục chứa ảnh COCO val2017 (5000 ảnh)")
    parser.add_argument("--annotations", type=str, default=str(COCO_ANN_FILE), help="Đường dẫn file instances_val2017.json")
    parser.add_argument("--model", type=str, default=str(MODEL_NAME), help="Kiến trúc hoặc checkpoint RT-DETR ('r18', 'r50', 'PekingU/rtdetr_r18vd', 'PekingU/rtdetr_r50vd')")
    parser.add_argument("--max_samples", type=int, default=1000, help="Giới hạn số lượng ảnh chạy thực nghiệm (mặc định 1000 ảnh, 5000 cho full COCO)")
    parser.add_argument("--num_images", type=int, default=None, help="Alias tương đương --max_samples (số lượng ảnh)")

    # 2. Tham số lưu trữ đầu ra
    parser.add_argument("--output_dir", type=str, default=str(OUTPUTS_DIR), help="Thư mục xuất kết quả thực nghiệm (summary JSON & raw CSV)")
    parser.add_argument("--cache_dir", type=str, default=str(ADV_CACHE_DIR), help="Thư mục lưu trữ bộ nhớ đệm mẫu đối kháng .npz")
    parser.add_argument("--fig_dir", type=str, default=str(FIG_DIR), help="Thư mục xuất hình ảnh biểu đồ vector PDF & PNG")

    # 3. Tham số phần cứng & thực thi
    parser.add_argument("--device", type=str, default="auto", help="Thiết bị tính toán (cuda / cpu / auto)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size suy luận (tự động tối ưu theo VRAM nếu None)")
    parser.add_argument("--threads", type=int, default=8, help="Số luồng CPU tối đa cho PyTorch")

    # 4. Tham số lựa chọn task
    parser.add_argument("--tasks", type=str, default="0,1,2,3,4,5,6,8,7,9", help="Danh sách các tasks cần chạy (phân tách bởi dấu phẩy, mặc định: 0,1,2,3,4,5,6,8,7,9)")
    parser.add_argument("--skip_cache_gen", action="store_true", default=False, help="Bỏ qua Task 0 nếu cache đã tồn tại")
    parser.add_argument("--force_recompute", action="store_true", default=False, help="Bắt buộc sinh lại cache từ đầu kể cả khi đã có")
    parser.add_argument("--include_sweep", action="store_true", default=False, help="Sinh thêm cache eps 4 và 8 cho Task 5")
    parser.add_argument("--pgd_steps", type=int, default=15, help="Số bước lặp PGD sinh mẫu đối kháng (mặc định 15)")
    args = parser.parse_args()

    if args.num_images is not None:
        args.max_samples = args.num_images

    chosen_model = resolve_model_name(args.model)
    update_experiment_paths(
        folder_image=args.folder_image,
        annotations=args.annotations,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        fig_dir=args.fig_dir,
        model=chosen_model,
    )

    t_global_start = time.time()
    hw_info = detect_hardware_environment(preferred_device=args.device)

    folder_p = Path(args.folder_image).resolve()
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    img_count = len([p for p in folder_p.iterdir() if p.suffix.lower() in valid_exts]) if folder_p.is_dir() else 0
    effective_count = min(img_count, args.max_samples) if args.max_samples is not None else img_count

    print("=" * 85)
    print(" 🚀 HỆ THỐNG ĐIỀU PHỐI THỰC NGHIỆM TOÀN DIỆN (RT-DETR MECHANISM BREAKDOWN)")
    print("=" * 85)
    print(f" Kiến trúc Model      : {chosen_model}")
    print(f" Thư mục ảnh đầu vào  : {folder_p} (Tìm thấy {img_count} ảnh | Chạy: {effective_count} ảnh)")
    print(f" File Annotations COCO: {Path(args.annotations).resolve()}")
    if args.max_samples is not None:
        print(f" Cấu hình Max Samples : {args.max_samples} ảnh")
    print(f" Thư mục đầu ra       : {Path(args.output_dir).resolve()}")
    print(f" Thư mục Cache        : {Path(args.cache_dir).resolve()}")
    print(f" Thư mục Figures      : {Path(args.fig_dir).resolve()}")
    print(f" Phần cứng nhận diện  : {hw_info['device_type'].upper()} ({hw_info.get('gpu_names', [hw_info.get('processor')])})")
    if hw_info["device_type"] == "cuda":
        print(f" Chi tiết GPU         : {hw_info['gpu_count']}x GPU | Tổng VRAM: {hw_info['total_vram_gb'][0]:.2f} GB | CUDA: {hw_info['cuda_version']}")
    else:
        print(f" Cấu hình CPU         : {hw_info['processor']} | Số luồng thiết lập: {args.threads}")
    print(f" Batch Size đề xuất   : {args.batch_size or hw_info['recommended_batch_size']}")
    print(f" Danh sách Tasks chạy : {args.tasks}")
    print("=" * 85)

    common_args = [
        "--folder_image", str(args.folder_image),
        "--annotations", str(args.annotations),
        "--output_dir", str(args.output_dir),
        "--cache_dir", str(args.cache_dir),
        "--threads", str(args.threads),
        "--device", str(args.device),
        "--model", str(chosen_model),
    ]
    if args.batch_size is not None:
        common_args.extend(["--batch_size", str(args.batch_size)])
    if args.max_samples is not None:
        common_args.extend(["--max_samples", str(args.max_samples)])

    base_dir = Path(__file__).resolve().parent
    registry = _build_task_registry(base_dir, args, common_args)

    tasks_to_run = [int(t.strip()) for t in args.tasks.split(",") if t.strip().isdigit()]
    task_durations: Dict[str, float] = {}
    task_statuses: Dict[str, str] = {}

    for task_id in tasks_to_run:
        if task_id not in registry:
            print(f"[CẢNH BÁO] Bỏ qua Task không hợp lệ: {task_id}")
            continue

        task_spec = registry[task_id]
        if "skip_condition" in task_spec and task_spec["skip_condition"]():
            print(f">>> Bỏ qua {task_spec['name']} theo cấu hình yêu cầu.")
            continue

        task_args = task_spec["args_builder"]()
        try:
            dur = run_subtask(
                script_path=task_spec["script"],
                args_list=task_args,
                task_name=task_spec["title"],
                log_file=task_spec["log_file"],
            )
            task_durations[task_spec["name"]] = dur
            task_statuses[task_spec["name"]] = "PASS"
        except Exception as e:
            task_statuses[task_spec["name"]] = "FAIL"
            print(f"\n[LỖI NGHIÊM TRỌNG] {task_spec['name']} thất bại: {e}")
            raise

    total_elapsed = time.time() - t_global_start

    # Xuất manifest tổng kết
    manifest_data = {
        "status": "SUCCESS",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_runtime_sec": round(total_elapsed, 2),
        "total_runtime_formatted": format_duration(total_elapsed),
        "dataset": {
            "image_folder": str(folder_p),
            "total_images": img_count,
            "evaluated_images": effective_count,
            "annotations_file": str(Path(args.annotations).resolve()),
        },
        "hardware_environment": hw_info,
        "task_execution_times_sec": task_durations,
        "task_statuses": task_statuses,
        "output_artifacts": {
            "summary_json_directory": str(Path(args.output_dir) / "summary_json"),
            "raw_csv_directory": str(Path(args.output_dir) / "raw_csv"),
            "figures_directory": str(Path(args.fig_dir).resolve()),
            "logs_directory": str(Path(args.output_dir) / "logs"),
            "table_mappings": {
                "Table 3 & Table 4": "causal_swap_results_aqsm.json / vanishing / taskpgd",
                "Table 5": "encoder_memory_analysis.json",
                "Table 6": "query_selection_dynamics.json",
                "Table 7": "deformable_attention_telemetry.json",
                "Table 8": "tautology_control_results.json",
                "Table 9": "matched_budget_sweep.json",
                "Table 10": "latency_profile.json",
                "Table 11": "defense_esm_evaluation.json",
            },
        },
    }

    manifest_path = Path(args.output_dir) / "final_experiment_manifest.json"
    save_json(manifest_data, manifest_path)

    print("\n" + "=" * 85)
    print(" 🎉 TOÀN BỘ CHUỖI THỰC NGHIỆM ĐÃ HOÀN TẤT THÀNH CÔNG VẺ VANG!")
    print("=" * 85)
    print(f" Tổng thời gian thực thi : {format_duration(total_elapsed)} ({total_elapsed:.2f}s)")
    print(f" Thư mục kết quả tổng hợp: {Path(args.output_dir) / 'summary_json'}")
    print(f" Thư mục số liệu thô CSV : {Path(args.output_dir) / 'raw_csv'}")
    print(f" Thư mục biểu đồ bài báo : {Path(args.fig_dir).resolve()}")
    print(f" File Manifest nghiệm thu: {manifest_path}")
    print("=" * 85)
    print(f"{'Tên Task':<35s} | {'Trạng thái':<10s} | {'Thời gian':<15s}")
    print("-" * 65)
    for task_name, dur in task_durations.items():
        status = task_statuses.get(task_name, "DONE")
        print(f"{task_name:<35s} | {status:<10s} | {format_duration(dur):>10s} ({dur:6.2f}s)")
    print("=" * 85)


if __name__ == "__main__":
    main()
