# -*- coding: utf-8 -*-
"""
Module: paper_1_aqsm_mechanism.experiments.tasks.task6_latency_profiler
Mục đích: Task độc lập đo đạc thời gian thực thi phần cứng (Hardware Latency Telemetry qua n_runs lần suy luận) -> Phục vụ Table 10:
  - Tự động nhận diện thiết bị hiện tại (CUDA GPU vs Multi-core CPU) và trích xuất đúng metrics phần cứng tương ứng.
  - Phân tích độ trễ từng stage: Backbone -> Hybrid Encoder -> Query Selection -> Deformable Decoder -> End-to-End.

Đặc tính chuẩn công nghiệp (Production Standards):
  - Nhận tham số linh hoạt `--folder_image`, `--annotations`, `--output_dir`, `--n_runs`, `--warmup`, `--device`, `--threads`.
  - Tự động lấy mẫu ảnh từ folder COCO hoặc khởi tạo sample tensor chuẩn Native 640x640.
  - Ghi nhận raw CSV từng lần chạy và summary JSON Table 10.
"""

import argparse
import gc
import logging
import os
import sys
import time
from pathlib import Path

import torch

WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from paper_1_aqsm_mechanism.core.detr_wrapper import RTDETRModelWrapper
from paper_1_aqsm_mechanism.experiments.audit_logger import ExperimentAuditLogger
from paper_1_aqsm_mechanism.experiments.config import (
    COCO_ANN_FILE,
    COCO_VAL_DIR,
    LOGS_DIR,
    MODEL_NAME,
    OUTPUTS_DIR,
    RANDOM_SEED,
    configure_cpu_threads,
    detect_hardware_environment,
    get_device,
    update_experiment_paths,
)
from paper_1_aqsm_mechanism.experiments.latency_profiler import profile_pipeline_latency


def main():
    parser = argparse.ArgumentParser(description="Task 6: Hardware Execution Latency Profiler (Table 10)")
    parser.add_argument("--folder_image", type=str, default=str(COCO_VAL_DIR), help="Đường dẫn thư mục ảnh COCO")
    parser.add_argument("--annotations", type=str, default=str(COCO_ANN_FILE), help="Đường dẫn file instances_val2017.json")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUTS_DIR), help="Thư mục xuất kết quả")
    parser.add_argument("--n_runs", type=int, default=300, help="Số lần chạy đo đạc")
    parser.add_argument("--warmup", type=int, default=30, help="Số lần warm-up khởi động")
    parser.add_argument("--threads", type=int, default=8, help="Số luồng CPU cho task này")
    parser.add_argument("--device", type=str, default="auto", help="Thiết bị tính toán (cuda / cpu / auto)")
    args = parser.parse_args()

    update_experiment_paths(
        folder_image=args.folder_image,
        annotations=args.annotations,
        output_dir=args.output_dir,
    )

    configure_cpu_threads(args.threads)
    hw_info = detect_hardware_environment(preferred_device=args.device)
    device = get_device(preferred=args.device)

    audit_logger = ExperimentAuditLogger(task_name="task6_latency_profiler")
    audit_logger.log_metadata({
        "n_runs": args.n_runs,
        "n_warmup": args.warmup,
        "threads": args.threads,
        "device": str(device),
        "hardware_info": hw_info,
        "model_name": MODEL_NAME,
        "random_seed": RANDOM_SEED,
    })

    # Nạp mô hình RT-DETR
    wrapper = RTDETRModelWrapper(pretrained_model_name=MODEL_NAME, device=device)
    wrapper.eval()

    # Thử lấy ảnh thật từ folder_image hoặc tạo sample ngẫu nhiên
    sample_img = None
    folder_p = Path(args.folder_image)
    if folder_p.is_dir():
        imgs = sorted(list(folder_p.glob("*.jpg")))
        if imgs:
            from PIL import Image
            import torchvision.transforms.functional as TF
            with Image.open(imgs[0]) as img:
                rgb = img.convert("RGB").resize((640, 640))
                sample_img = TF.to_tensor(rgb).unsqueeze(0).to(device)

    if sample_img is None:
        sample_img = torch.randn(1, 3, 640, 640, device=device).clamp(0.0, 1.0)

    res = profile_pipeline_latency(
        wrapper=wrapper,
        sample_img=sample_img,
        n_runs=args.n_runs,
        n_warmup=args.warmup,
        audit_logger=audit_logger,
    )

    audit_logger.log_step(
        step_name="Latency_Profiling_Complete",
        step_idx=1,
        total_steps=1,
        metrics={"end_to_end_mean_ms": res["stages"]["End_to_End_Total"]["mean_ms"], "throughput_fps": res["throughput_fps"]},
    )
    audit_logger.close()
    del sample_img, wrapper
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print(f"\n[HOÀN THÀNH TOÀN BỘ] Task 6: Hardware Execution Latency Profiler thành công! Throughput: {res['throughput_fps']} FPS")


if __name__ == "__main__":
    main()
