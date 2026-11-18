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
  Task 7: Tự động kết xuất toàn bộ biểu đồ vector xuất bản chuẩn Elsevier EAAI (PDF & PNG).
  Task 8: Đánh giá hệ thống phòng thủ Encoder Selection Monitor (ESM) & Adaptive PGD -> Table 11, Figure 8/9.

Đặc tính chuẩn công nghiệp (Production Standards):
  - Chạy linh hoạt trên toàn bộ ảnh COCO val2017 (N=5,000 ảnh) hoặc tập con bất kỳ thông qua `--folder_image`.
  - Cơ chế nạp theo lô lười (Lazy Batch Loading) ngăn ngừa 100% hiện tượng tràn RAM.
  - Tự động nhận diện thiết bị phần cứng (CUDA GPU / Multi-core CPU), tối ưu hóa luồng và batch size.
  - Quản lý bộ nhớ hoàn hảo: Giải phóng VRAM/RAM giữa các giai đoạn thông qua subprocess isolation.
  - Không hardcode, không làm giả số liệu; xuất đầy đủ context trong JSON/CSV để điền vào bài báo.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Thêm thư mục gốc vào PYTHONPATH
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
    RAW_CSV_DIR,
    SUMMARY_JSON_DIR,
    detect_hardware_environment,
    update_experiment_paths,
)


def format_duration(seconds: float) -> str:
    """Định dạng thời gian chạy thành hh:mm:ss."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}h:{m:02d}m:{s:02d}s"


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
        raise RuntimeError(f"Task thất bại với mã lỗi {ret_code}: {task_name}")

    print(f"\n >>> [HOÀN THÀNH] {task_name} trong {format_duration(elapsed)} ({elapsed:.2f}s)")
    return elapsed


def main():
    parser = argparse.ArgumentParser(
        description="Master Experiment Orchestrator: Chạy toàn bộ thực nghiệm Paper 1 (AQSM Mechanism)"
    )
    # 1. Tham số dữ liệu đầu vào
    parser.add_argument("--folder_image", type=str, default=str(COCO_VAL_DIR), help="Đường dẫn thư mục chứa ảnh COCO val2017 (5000 ảnh)")
    parser.add_argument("--annotations", type=str, default=str(COCO_ANN_FILE), help="Đường dẫn file instances_val2017.json")
    
    # 2. Tham số lưu trữ đầu ra
    parser.add_argument("--output_dir", type=str, default=str(OUTPUTS_DIR), help="Thư mục xuất kết quả thực nghiệm (summary JSON & raw CSV)")
    parser.add_argument("--cache_dir", type=str, default=str(ADV_CACHE_DIR), help="Thư mục lưu trữ bộ nhớ đệm mẫu đối kháng .npz")
    parser.add_argument("--fig_dir", type=str, default=str(FIG_DIR), help="Thư mục xuất hình ảnh biểu đồ vector PDF & PNG")

    # 3. Tham số phần cứng & thực thi
    parser.add_argument("--device", type=str, default="auto", help="Thiết bị tính toán (cuda / cpu / auto)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size suy luận (tự động tối ưu theo VRAM nếu None)")
    parser.add_argument("--threads", type=int, default=8, help="Số luồng CPU tối đa cho PyTorch")
    
    # 4. Tham số lựa chọn task
    parser.add_argument("--tasks", type=str, default="0,1,2,3,4,5,6,7,8", help="Danh sách các tasks cần chạy (phân tách bởi dấu phẩy, mặc định: 0,1,2,3,4,5,6,7,8)")
    parser.add_argument("--skip_cache_gen", action="store_true", default=False, help="Bỏ qua Task 0 nếu cache đã tồn tại")
    parser.add_argument("--force_recompute", action="store_true", default=False, help="Bắt buộc sinh lại cache từ đầu kể cả khi đã có")
    parser.add_argument("--pgd_steps", type=int, default=15, help="Số bước lặp PGD sinh mẫu đối kháng (mặc định 15)")
    args = parser.parse_args()

    # Cập nhật đường dẫn toàn cục
    update_experiment_paths(
        folder_image=args.folder_image,
        annotations=args.annotations,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        fig_dir=args.fig_dir,
    )

    t_global_start = time.time()
    hw_info = detect_hardware_environment(preferred_device=args.device)

    # Đếm số lượng ảnh hợp lệ trong folder
    folder_p = Path(args.folder_image).resolve()
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    img_count = len([p for p in folder_p.iterdir() if p.suffix.lower() in valid_exts]) if folder_p.is_dir() else 0

    print("=" * 85)
    print(" 🚀 HỆ THỐNG ĐIỀU PHỐI THỰC NGHIỆM TOÀN DIỆN (RT-DETR MECHANISM BREAKDOWN)")
    print("=" * 85)
    print(f" Thư mục ảnh đầu vào  : {folder_p} (Tìm thấy {img_count} ảnh)")
    print(f" File Annotations COCO: {Path(args.annotations).resolve()}")
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

    base_tasks_dir = Path(__file__).resolve().parent / "experiments" / "tasks"
    gen_cache_script = Path(__file__).resolve().parent / "experiments" / "generate_adv_cache_640.py"

    tasks_to_run = [int(t.strip()) for t in args.tasks.split(",") if t.strip().isdigit()]
    task_durations: Dict[str, float] = {}

    common_args = [
        "--folder_image", str(args.folder_image),
        "--annotations", str(args.annotations),
        "--output_dir", str(args.output_dir),
        "--cache_dir", str(args.cache_dir),
        "--threads", str(args.threads),
        "--device", str(args.device),
    ]
    if args.batch_size is not None:
        common_args.extend(["--batch_size", str(args.batch_size)])

    # -------------------------------------------------------------------------
    # TASK 0: Sinh bộ nhớ đệm đối kháng (Adversarial Cache Generation)
    # -------------------------------------------------------------------------
    if 0 in tasks_to_run and not args.skip_cache_gen:
        t0_args = [
            "--folder_image", str(args.folder_image),
            "--annotations", str(args.annotations),
            "--cache_dir", str(args.cache_dir),
            "--steps", str(args.pgd_steps),
            "--threads", str(args.threads),
            "--device", str(args.device),
            "--include_sweep",
        ]
        if args.batch_size is not None:
            t0_args.extend(["--batch_size", str(args.batch_size)])
        if args.force_recompute:
            t0_args.append("--force_recompute")

        dur = run_subtask(
            gen_cache_script,
            t0_args,
            "Task 0: Adversarial Cache Generation (Native 640x640, eps 4, 8, 16)",
            log_file=Path(args.output_dir) / "logs" / "task0_cache_generation.log",
        )
        task_durations["Task 0 (Cache Gen)"] = dur

    # -------------------------------------------------------------------------
    # TASK 1: Causal Surgery 2x2 Factorial Tensor-Swap -> Table 3, Table 4
    # -------------------------------------------------------------------------
    if 1 in tasks_to_run:
        dur = run_subtask(
            base_tasks_dir / "task1_causal_surgery.py",
            common_args,
            "Task 1: Causal Surgery 2x2 Factorial Swap (Table 3, Table 4, Figure 4)",
            log_file=Path(args.output_dir) / "logs" / "task1_causal_surgery.log",
        )
        task_durations["Task 1 (Causal Surgery)"] = dur

    # -------------------------------------------------------------------------
    # TASK 2: Encoder Memory & Query Selection Dynamics -> Table 5, Table 6
    # -------------------------------------------------------------------------
    if 2 in tasks_to_run:
        dur = run_subtask(
            base_tasks_dir / "task2_encoder_query_dynamics.py",
            common_args,
            "Task 2: Encoder Memory & Query Selection Dynamics (Table 5, Table 6)",
            log_file=Path(args.output_dir) / "logs" / "task2_encoder_query_dynamics.log",
        )
        task_durations["Task 2 (Encoder & Query Dynamics)"] = dur

    # -------------------------------------------------------------------------
    # TASK 3: Deformable Attention Telemetry -> Table 7
    # -------------------------------------------------------------------------
    if 3 in tasks_to_run:
        dur = run_subtask(
            base_tasks_dir / "task3_deformable_telemetry.py",
            common_args,
            "Task 3: Deformable Attention Telemetry (Table 7, Figure 6)",
            log_file=Path(args.output_dir) / "logs" / "task3_deformable_telemetry.log",
        )
        task_durations["Task 3 (Attention Telemetry)"] = dur

    # -------------------------------------------------------------------------
    # TASK 4: Controlled Spatial Misrouting (Tautology Refutation) -> Table 8
    # -------------------------------------------------------------------------
    if 4 in tasks_to_run:
        dur = run_subtask(
            base_tasks_dir / "task4_controlled_misrouting.py",
            common_args,
            "Task 4: Controlled Spatial Misrouting & Tautology Refutation (Table 8)",
            log_file=Path(args.output_dir) / "logs" / "task4_controlled_misrouting.log",
        )
        task_durations["Task 4 (Tautology Refutation)"] = dur

    # -------------------------------------------------------------------------
    # TASK 5: Matched-Budget Perturbation Sweep -> Table 9 & Figure 8
    # -------------------------------------------------------------------------
    if 5 in tasks_to_run:
        dur = run_subtask(
            base_tasks_dir / "task5_matched_sweep.py",
            common_args,
            "Task 5: Matched-Budget Perturbation Sweep (Table 9, Figure 8)",
            log_file=Path(args.output_dir) / "logs" / "task5_matched_sweep.log",
        )
        task_durations["Task 5 (Matched Sweep)"] = dur

    # -------------------------------------------------------------------------
    # TASK 6: Hardware Execution Latency Profiler -> Table 10
    # -------------------------------------------------------------------------
    if 6 in tasks_to_run:
        t6_args = [
            "--folder_image", str(args.folder_image),
            "--annotations", str(args.annotations),
            "--output_dir", str(args.output_dir),
            "--threads", str(args.threads),
            "--device", str(args.device),
            "--n_runs", "300",
        ]
        dur = run_subtask(
            base_tasks_dir / "task6_latency_profiler.py",
            t6_args,
            "Task 6: Hardware Execution Latency Profiler (Table 10)",
            log_file=Path(args.output_dir) / "logs" / "task6_latency_profiler.log",
        )
        task_durations["Task 6 (Latency Profiler)"] = dur

    # -------------------------------------------------------------------------
    # TASK 8: Empirical Defense & Adaptive Evasion Evaluation -> Table 11
    # -------------------------------------------------------------------------
    if 8 in tasks_to_run:
        dur = run_subtask(
            base_tasks_dir / "task8_defense_evaluation.py",
            common_args,
            "Task 8: Empirical Defense ESM & Adaptive Evasion Evaluation (Table 11, Figure 9)",
            log_file=Path(args.output_dir) / "logs" / "task8_defense_evaluation.log",
        )
        task_durations["Task 8 (Defense Evaluation)"] = dur

    # -------------------------------------------------------------------------
    # TASK 7: Sinh toàn bộ Figures vector xuất bản PDF / PNG
    # -------------------------------------------------------------------------
    if 7 in tasks_to_run:
        t7_args = [
            "--output_dir", str(args.output_dir),
            "--fig_dir", str(args.fig_dir),
        ]
        dur = run_subtask(
            base_tasks_dir / "task7_generate_figures.py",
            t7_args,
            "Task 7: Generate Publication Figures (PDF/PNG Vector Graphics)",
            log_file=Path(args.output_dir) / "logs" / "task7_generate_figures.log",
        )
        task_durations["Task 7 (Figure Generation)"] = dur

    total_elapsed = time.time() - t_global_start

    # -------------------------------------------------------------------------
    # TỔNG KẾT & XUẤT MANIFEST TOÀN DIỆN
    # -------------------------------------------------------------------------
    manifest_data = {
        "status": "SUCCESS",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_runtime_sec": round(total_elapsed, 2),
        "total_runtime_formatted": format_duration(total_elapsed),
        "dataset": {
            "image_folder": str(folder_p),
            "total_images": img_count,
            "annotations_file": str(Path(args.annotations).resolve()),
        },
        "hardware_environment": hw_info,
        "task_execution_times_sec": task_durations,
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
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 85)
    print(" 🎉 TOÀN BỘ CHUỖI THỰC NGHIỆM ĐÃ HOÀN TẤT THÀNH CÔNG VẺ VANG!")
    print("=" * 85)
    print(f" Tổng thời gian thực thi : {format_duration(total_elapsed)} ({total_elapsed:.2f}s)")
    print(f" Thư mục kết quả tổng hợp: {Path(args.output_dir) / 'summary_json'}")
    print(f" Thư mục số liệu thô CSV : {Path(args.output_dir) / 'raw_csv'}")
    print(f" Thư mục biểu đồ bài báo : {Path(args.fig_dir).resolve()}")
    print(f" File Manifest nghiệm thu: {manifest_path}")
    print("=" * 85)
    print("Bảng thời gian thực thi từng giai đoạn:")
    for task_name, dur in task_durations.items():
        print(f"  - {task_name:<45s}: {format_duration(dur):>10s} ({dur:7.2f}s)")
    print("=" * 85)


if __name__ == "__main__":
    main()
