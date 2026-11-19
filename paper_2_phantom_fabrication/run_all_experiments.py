# -*- coding: utf-8 -*-
"""
Module: paper_2_phantom_fabrication.run_all_experiments
Mục đích: Orchestrator trung tâm điều phối toàn bộ chuỗi thực nghiệm khoa học của Paper 2
(Task 0 -> Task 5) với cơ chế cách ly tiến trình con (Process Isolation) chống OOM:
  - Task 0: Tiền sinh Adversarial Cache float32 (Untargeted PGD, EQF, DQF)
  - Task 1: Multi-Seed Benchmark (RT-DETR vs YOLOv8n trên 3 seeds) -> Table 1 & Table 4
  - Task 2: Sensitivity Sweeps (Ngân sách nhiễu eps và số bước PGD) -> Table 2
  - Task 3: Downstream Cyber-Physical Impact (Cross-Class, Size breakdown, ByteTrack & AEB) -> Table 3
  - Task 4: Comprehensive Defense Evaluation (SQD-Filter vs Baselines) -> Table 4
  - Task 5: Xuất toàn bộ 4 biểu đồ vector IEEE/Elsevier (PDF & PNG) -> Figures 1-4
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
PACKAGE_DIR = Path(__file__).resolve().parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from paper_2_phantom_fabrication.core.config import (
    ADV_CACHE_DIR,
    FIG_DIR,
    OUTPUTS_DIR,
    RESULTS_DIR,
)
from paper_2_phantom_fabrication.utils.cli import create_base_parser
from paper_2_phantom_fabrication.utils.device import detect_hardware, setup_cpu_threads
from paper_2_phantom_fabrication.utils.io_helpers import format_duration, save_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Orchestrator")


TASK_REGISTRY: Dict[int, Dict[str, Any]] = {
    0: {
        "name": "Task 0: Pre-generate Adversarial Cache (float32 Memmap)",
        "script": "experiments/tasks/task0_generate_adv_cache.py",
        "description": "Tiền sinh bộ nhớ đệm đối kháng float32 chống tràn RAM",
    },
    1: {
        "name": "Task 1: Multi-Seed Benchmark (RT-DETR vs YOLOv8n)",
        "script": "experiments/tasks/task1_multiseed_benchmark.py",
        "description": "Đo đạc độ lệch chuẩn 3 seeds trên Clean, PGD, EQF, DQF",
    },
    2: {
        "name": "Task 2: Sensitivity Sweeps (Budget & Steps)",
        "script": "experiments/tasks/task2_sensitivity_sweeps.py",
        "description": "Quét ngân sách nhiễu eps và số bước lặp PGD (Table 2)",
    },
    3: {
        "name": "Task 3: Downstream Impact & Cross-Class Generalization",
        "script": "experiments/tasks/task3_downstream_impact.py",
        "description": "Đánh giá chuyển giao 5 lớp, phân rã kích thước, ByteTrack & AEB (Table 3)",
    },
    4: {
        "name": "Task 4: Defense Evaluation (SQD-Filter vs Baselines)",
        "script": "experiments/tasks/task4_defense_evaluation.py",
        "description": "Đánh giá chi tiết SQD-Filter và các giải pháp phòng thủ (Table 4)",
    },
    5: {
        "name": "Task 5: Generate Scientific Figures (Fig 1-4)",
        "script": "experiments/tasks/task5_generate_figures.py",
        "description": "Tự động xuất 4 đồ thị vector chuẩn xuất bản Elsevier",
    },
}


def run_subtask(
    task_id: int,
    script_rel_path: str,
    cli_args: List[str],
    log_dir: Path,
) -> Tuple[int, float]:
    """
    Thực thi 1 Task trong tiến trình Python con biệt lập nhằm giải phóng GPU/RAM tuyệt đối.
    """
    task_info = TASK_REGISTRY[task_id]
    script_path = (PACKAGE_DIR / script_rel_path).resolve()
    log_file = log_dir / f"task{task_id}_{Path(script_rel_path).stem}.log"

    cmd = [sys.executable, str(script_path)] + cli_args

    logger.info("=" * 80)
    logger.info(f"BẮT ĐẦU {task_info['name']}")
    logger.info(f"Mô tả: {task_info['description']}")
    logger.info(f"Lệnh : {' '.join(cmd)}")
    logger.info(f"Log  : {log_file}")
    logger.info("=" * 80)

    t0 = time.perf_counter()
    with open(log_file, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            cwd=str(PACKAGE_DIR),
            env=dict(os.environ, PYTHONUNBUFFERED="1"),
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            lf.write(line)
            lf.flush()
        proc.wait()

    dur = time.perf_counter() - t0
    ret_code = proc.returncode

    if ret_code != 0:
        logger.error(f"[LỖI] {task_info['name']} thất bại với mã thoát: {ret_code} (Sau {format_duration(dur)})")
    else:
        logger.info(f"[HOÀN TẤT] {task_info['name']} thành công sau {format_duration(dur)}")

    return ret_code, dur


def build_unified_telemetry(outputs_dir: Path) -> Path:
    """
    Tổng hợp kết quả từ các task thành file paper2_complete_telemetry.json hợp nhất.
    """
    t1_file = outputs_dir / "stage1_benchmark_telemetry.json"
    t2_file = outputs_dir / "stage2_sensitivity_telemetry.json"
    t3_file = outputs_dir / "stage2_downstream_telemetry.json"
    t4_file = outputs_dir / "stage1_defense_telemetry.json"

    unified: Dict[str, Any] = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "pipeline": "Paper 2 Zero-OOM Empirical Suite",
            "status": "completed",
        }
    }

    if t1_file.exists():
        with open(t1_file, "r", encoding="utf-8") as f:
            t1_data = json.load(f)
            unified["stage1_benchmark"] = t1_data.get("stage1_benchmark", t1_data)
            if "table4_defenses" in t1_data:
                unified["table4_defenses"] = t1_data["table4_defenses"]

    if t2_file.exists():
        with open(t2_file, "r", encoding="utf-8") as f:
            t2_data = json.load(f)
            unified["sensitivity_sweeps"] = t2_data.get("sensitivity_sweeps", t2_data)

    if t3_file.exists():
        with open(t3_file, "r", encoding="utf-8") as f:
            t3_data = json.load(f)
            unified["downstream_and_figures"] = t3_data.get("downstream_and_figures", t3_data)

    if t4_file.exists():
        with open(t4_file, "r", encoding="utf-8") as f:
            t4_data = json.load(f)
            unified["defense_evaluation"] = t4_data

    out_unified = outputs_dir / "paper2_complete_telemetry.json"
    save_json(unified, out_unified)

    # Đồng bộ sang results/ để tương thích ngược tuyệt đối
    res_unified = RESULTS_DIR / "paper2_complete_telemetry.json"
    save_json(unified, res_unified)

    return out_unified


def main():
    parser = argparse.ArgumentParser(
        description="Master Orchestrator - Paper 2 Empirical Experiment Suite (Task 0 -> Task 5)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--folder_image", type=str, required=True, help="Đường dẫn thư mục ảnh COCO (e.g. val2017)")
    parser.add_argument("--annotations", type=str, default="", help="Đường dẫn file annotations COCO JSON")
    parser.add_argument("--tasks", type=str, default="0,1,2,3,4,5", help="Danh sách task ID thực thi (e.g. '0,1,2,3,4,5' hoặc 'all')")
    parser.add_argument("--num_images", type=int, default=0, help="Số ảnh giới hạn (0 = chạy toàn bộ)")
    parser.add_argument("--device", type=str, default="auto", help="Thiết bị tính toán: auto, cuda, cpu")
    parser.add_argument("--batch_size", type=int, default=0, help="Batch size suy luận (0 = tự động)")
    parser.add_argument("--seeds", type=str, default="42,123,456", help="Danh sách seeds cho Task 1")
    parser.add_argument("--threads", type=int, default=4, help="Số luồng CPU")
    parser.add_argument("--cache_dir", type=str, default=str(ADV_CACHE_DIR), help="Thư mục bộ nhớ đệm đối kháng")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUTS_DIR), help="Thư mục lưu outputs/logs")
    parser.add_argument("--fig_dir", type=str, default=str(FIG_DIR), help="Thư mục xuất figures")
    parser.add_argument("--skip_cache_gen", action="store_true", default=False, help="Bỏ qua sinh cache nếu đã có")
    parser.add_argument("--force_recompute", action="store_true", default=False, help="Bắt buộc tính toán lại từ đầu")
    args = parser.parse_args()

    setup_cpu_threads(args.threads)
    hw_info = detect_hardware(args.device)

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Xác định danh sách task cần chạy
    if args.tasks.strip().lower() == "all":
        selected_tasks = sorted(list(TASK_REGISTRY.keys()))
    else:
        selected_tasks = sorted([int(t.strip()) for t in args.tasks.split(",") if t.strip()])

    total_start_time = time.perf_counter()
    task_manifest: List[Dict[str, Any]] = []

    print("=" * 80)
    print(" KHỞI ĐỘNG HỆ THỐNG ĐIỀU PHỐI THỰC NGHIỆM PAPER 2 (MASTER ORCHESTRATOR)")
    print(f" Thư mục ảnh COCO : {args.folder_image}")
    print(f" Thiết bị         : {hw_info['primary_device']} ({hw_info.get('num_gpus', 0)} GPUs)")
    print(f" Danh sách Tasks  : {selected_tasks}")
    print(f" Thư mục Cache    : {args.cache_dir}")
    print(f" Thư mục Output   : {out_dir}")
    print("=" * 80)

    # Tham số cơ bản chuyển tiếp tới các task con
    common_args = [
        "--folder_image", str(Path(args.folder_image).resolve()),
        "--num_images", str(args.num_images),
        "--device", str(args.device),
        "--batch_size", str(args.batch_size),
        "--threads", str(args.threads),
        "--cache_dir", str(Path(args.cache_dir).resolve()),
        "--output_dir", str(out_dir),
    ]
    if args.annotations:
        common_args.extend(["--annotations", str(Path(args.annotations).resolve())])
    if args.skip_cache_gen:
        common_args.append("--skip_cache_gen")
    if args.force_recompute:
        common_args.append("--force_recompute")

    # Vòng lặp thực thi từng task
    for tid in selected_tasks:
        if tid not in TASK_REGISTRY:
            logger.warning(f"Bỏ qua task không hợp lệ: {tid}")
            continue

        task_cli = list(common_args)
        if tid == 1:
            task_cli.extend(["--seeds", args.seeds])
        elif tid == 5:
            task_cli.extend(["--fig_dir", str(Path(args.fig_dir).resolve())])

        ret_code, dur = run_subtask(
            task_id=tid,
            script_rel_path=TASK_REGISTRY[tid]["script"],
            cli_args=task_cli,
            log_dir=log_dir,
        )

        task_manifest.append({
            "task_id": tid,
            "name": TASK_REGISTRY[tid]["name"],
            "return_code": ret_code,
            "duration_seconds": round(dur, 2),
            "status": "SUCCESS" if ret_code == 0 else "FAILED",
        })

        if ret_code != 0:
            logger.error(f"DỪNG TIẾN TRÌNH: Task {tid} gặp sự cố không thể tiếp tục.")
            break

    # Tổng hợp telemetry nếu Task 1, 2, 3 hoàn tất
    unified_json = build_unified_telemetry(out_dir)
    total_elapsed = time.perf_counter() - total_start_time

    # Lưu manifest
    manifest_file = out_dir / "execution_manifest.json"
    save_json({
        "timestamp": datetime.now().isoformat(),
        "total_elapsed_seconds": round(total_elapsed, 2),
        "tasks": task_manifest,
        "unified_telemetry": str(unified_json),
    }, manifest_file)

    # In bảng tổng kết
    print("\n" + "=" * 80)
    print(" BẢNG TỔNG KẾT THỰC THI THỰC NGHIỆM PAPER 2")
    print("=" * 80)
    print(f" {'Task':<6} | {'Tên Nhiệm Vụ':<48} | {'Thời Lượng':<10} | {'Trạng Thái'}")
    print("-" * 80)
    for m in task_manifest:
        status_str = "[OK]" if m["status"] == "SUCCESS" else "[LỖI]"
        print(f" {m['task_id']:<6} | {m['name'][:48]:<48} | {format_duration(m['duration_seconds']):<10} | {status_str}")
    print("-" * 80)
    print(f" Tổng thời gian: {format_duration(total_elapsed)}")
    print(f" File Telemetry: {unified_json}")
    print(f" Manifest Log  : {manifest_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
