# -*- coding: utf-8 -*-
"""
Module: paper_1_aqsm_mechanism.tools.run_multimodel_benchmark
Mục đích: Thực thi benchmark so sánh đa mô hình (Cross-Architecture Benchmark)
giữa các biến thể RT-DETR (ví dụ: RT-DETR-R18VD vs RT-DETR-R50VD).

Đặc tính chuẩn công nghiệp (Production Standards):
  - Hỗ trợ chạy lần lượt từng model với subprocess isolation hoặc in-process memory cleanup.
  - Đo lường Clean mAP, AQSM mAP, Task PGD mAP, Query Eviction Rate, Memory Cosine Similarity, và Latency.
  - Tự động xuất file tổng hợp:
      1. JSON: outputs/summary_json/multimodel_benchmark_summary.json
      2. LaTeX snippet: outputs/tables/table_multimodel_benchmark.tex
      3. Markdown table phục vụ xem nhanh.
  - 100% dữ liệu thực nghiệm thật, hỗ trợ streaming On-The-Fly (0 GB disk).
"""

import argparse
import gc
import json
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from paper_1_aqsm_mechanism.core.attack_generator import DETRAttackGenerator
from paper_1_aqsm_mechanism.core.coco_gt import (
    evaluate_coco_predictions,
    postprocess_rtdetr_predictions,
)
from paper_1_aqsm_mechanism.core.dataset_loader import FlexibleCOCODataset
from paper_1_aqsm_mechanism.core.detr_wrapper import RTDETRModelWrapper
from paper_1_aqsm_mechanism.core.mechanism_analyzer import (
    analyze_encoder_memory_batch,
    analyze_query_selection_batch,
    extract_encoder_components,
)
from paper_1_aqsm_mechanism.experiments.config import (
    COCO_ANN_FILE,
    COCO_VAL_DIR,
    OUTPUTS_DIR,
    configure_cpu_threads,
    detect_hardware_environment,
    get_device,
    get_model_tag,
    get_summary_json_dir,
    resolve_model_name,
    update_experiment_paths,
)
from paper_1_aqsm_mechanism.utils.cli import create_base_parser
from paper_1_aqsm_mechanism.utils.io_helpers import save_json


def benchmark_single_model(
    model_name_or_alias: str,
    folder_image: str,
    annotations: str,
    max_samples: int,
    batch_size: int,
    device: torch.device,
    eps_255: int = 16,
    pgd_steps: int = 15,
) -> Dict[str, Any]:
    """Đánh giá toàn diện một mô hình trên Clean, AQSM và Task PGD."""
    full_model_name = resolve_model_name(model_name_or_alias)
    model_tag = get_model_tag(full_model_name)
    print("\n" + "=" * 80)
    print(f" [BENCHMARK] Khởi chạy đánh giá mô hình: {full_model_name} ({model_tag})")
    print(f" Samples: {max_samples} | Batch size: {batch_size} | Device: {device}")
    print("=" * 80)

    # 1. Nạp Model Wrapper
    wrapper = RTDETRModelWrapper(pretrained_model_name=full_model_name, device=device)
    wrapper.eval()
    model = wrapper.model

    # 2. Nạp Dataset
    dataset = FlexibleCOCODataset(
        folder_image=folder_image,
        annotations_file=annotations,
        img_size=640,
        id2label=model.config.id2label,
        lazy_tensors=True,
        device=device,
        max_samples=max_samples,
    )
    clean_proxy = dataset.tensors
    gt_list = dataset.gt_list
    n_eval = len(dataset)

    attacker = DETRAttackGenerator(
        wrapper,
        epsilon=float(eps_255) / 255.0,
        alpha=(2.0 / 255.0),
        num_steps=pgd_steps,
    )

    results: Dict[str, Any] = {
        "model_name": full_model_name,
        "model_tag": model_tag,
        "n_samples": n_eval,
        "decoder_layers": getattr(model.config, "decoder_layers", 3),
        "hidden_dim": getattr(model.config, "d_model", 256),
    }

    # 3. Đánh giá Clean Baseline & Latency
    print(">>> 1/3: Đánh giá Clean Baseline...")
    clean_preds = []
    latencies_ms = []
    t0_clean = time.time()
    for i in tqdm(range(0, n_eval, batch_size), desc=f"{model_tag}_clean"):
        end_idx = min(i + batch_size, n_eval)
        b_clean = clean_proxy[i:end_idx].to(device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t_b0 = time.time()
        with torch.no_grad():
            out = wrapper(b_clean)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies_ms.append(((time.time() - t_b0) / (end_idx - i)) * 1000.0)

        clean_preds.extend(
            postprocess_rtdetr_predictions(out["logits"], out["pred_boxes"], mode="hf_multilabel")
        )

    clean_eval = evaluate_coco_predictions(clean_preds, gt_list)
    results["clean"] = {
        "mAP@50": round(float(clean_eval["mAP@50"]), 2),
        "mAP@50:95": round(float(clean_eval["mAP@50:95"]), 2),
        "recall_pct": round(float(clean_eval["recall_pct"]), 2),
        "mean_latency_ms": round(float(np.mean(latencies_ms)), 2),
    }
    print(f"   Clean mAP@50: {results['clean']['mAP@50']:.2f}% | Latency: {results['clean']['mean_latency_ms']:.2f} ms")

    # 4. Đánh giá AQSM Attack (Streaming On-The-Fly)
    print(">>> 2/3: Đánh giá AQSM Attack (eps=16/255)...")
    aqsm_preds = []
    aqsm_evictions = []
    aqsm_cosines = []
    for i in tqdm(range(0, n_eval, batch_size), desc=f"{model_tag}_aqsm"):
        end_idx = min(i + batch_size, n_eval)
        b_clean = clean_proxy[i:end_idx].to(device)
        b_adv = attacker.generate_query_selection_attack(b_clean, steps=pgd_steps)

        with torch.no_grad():
            c_comp = extract_encoder_components(model, b_clean)
            a_comp = extract_encoder_components(model, b_adv)
            out_adv = wrapper(b_adv)

            # Query dynamics
            q_metrics = analyze_query_selection_batch(
                c_comp["enc_outputs_class"],
                a_comp["enc_outputs_class"],
                top_k=300,
            )
            aqsm_evictions.extend(q_metrics["evicted_ratio"])

            # Memory cosine
            m_metrics = analyze_encoder_memory_batch(
                c_comp["output_memory"],
                a_comp["output_memory"],
                c_comp["spatial_shapes"],
            )
            aqsm_cosines.append(m_metrics["all_cosine_mean"])

        aqsm_preds.extend(
            postprocess_rtdetr_predictions(out_adv["logits"], out_adv["pred_boxes"], mode="hf_multilabel")
        )

    aqsm_eval = evaluate_coco_predictions(aqsm_preds, gt_list)
    results["aqsm"] = {
        "mAP@50": round(float(aqsm_eval["mAP@50"]), 2),
        "mAP@50:95": round(float(aqsm_eval["mAP@50:95"]), 2),
        "recall_pct": round(float(aqsm_eval["recall_pct"]), 2),
        "mean_evicted_pct": round(float(np.mean(aqsm_evictions) * 100.0), 2),
        "mean_memory_cosine": round(float(np.mean(aqsm_cosines)), 4),
    }
    print(f"   AQSM mAP@50: {results['aqsm']['mAP@50']:.2f}% | Eviction: {results['aqsm']['mean_evicted_pct']:.2f}% | Cosine: {results['aqsm']['mean_memory_cosine']:.4f}")

    # 5. Đánh giá Task PGD
    print(">>> 3/3: Đánh giá Task PGD Attack (eps=16/255)...")
    pgd_preds = []
    pgd_evictions = []
    pgd_cosines = []
    for i in tqdm(range(0, n_eval, batch_size), desc=f"{model_tag}_taskpgd"):
        end_idx = min(i + batch_size, n_eval)
        b_clean = clean_proxy[i:end_idx].to(device)
        b_adv = attacker.generate_standard_task_pgd(b_clean, steps=pgd_steps)

        with torch.no_grad():
            c_comp = extract_encoder_components(model, b_clean)
            a_comp = extract_encoder_components(model, b_adv)
            out_adv = wrapper(b_adv)

            q_metrics = analyze_query_selection_batch(
                c_comp["enc_outputs_class"],
                a_comp["enc_outputs_class"],
                top_k=300,
            )
            pgd_evictions.extend(q_metrics["evicted_ratio"])

            m_metrics = analyze_encoder_memory_batch(
                c_comp["output_memory"],
                a_comp["output_memory"],
                c_comp["spatial_shapes"],
            )
            pgd_cosines.append(m_metrics["all_cosine_mean"])

        pgd_preds.extend(
            postprocess_rtdetr_predictions(out_adv["logits"], out_adv["pred_boxes"], mode="hf_multilabel")
        )

    pgd_eval = evaluate_coco_predictions(pgd_preds, gt_list)
    results["task_pgd"] = {
        "mAP@50": round(float(pgd_eval["mAP@50"]), 2),
        "mAP@50:95": round(float(pgd_eval["mAP@50:95"]), 2),
        "recall_pct": round(float(pgd_eval["recall_pct"]), 2),
        "mean_evicted_pct": round(float(np.mean(pgd_evictions) * 100.0), 2),
        "mean_memory_cosine": round(float(np.mean(pgd_cosines)), 4),
    }
    print(f"   Task PGD mAP@50: {results['task_pgd']['mAP@50']:.2f}% | Eviction: {results['task_pgd']['mean_evicted_pct']:.2f}% | Cosine: {results['task_pgd']['mean_memory_cosine']:.4f}")

    # Thu hồi bộ nhớ
    del wrapper, model, attacker
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return results


def generate_multimodel_latex(all_results: Dict[str, Dict[str, Any]]) -> str:
    """Sinh bảng LaTeX so sánh các biến thể kiến trúc."""
    tex = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Cross-Architecture Robustness Benchmark across Real-Time DETR Variants under $\varepsilon=16/255$.}",
        r"\label{tab:multimodel_benchmark}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\textbf{Architecture} & \textbf{Decoder Layers} & \textbf{Clean AP$_{50}$} & \textbf{AQSM AP$_{50}$} & \textbf{Task PGD AP$_{50}$} & \textbf{Evicted Q (\%)} & \textbf{Mem Cosine} \\",
        r"\midrule",
    ]
    for tag, res in all_results.items():
        disp_name = "RT-DETR-R18VD" if "r18" in tag else ("RT-DETR-R50VD" if "r50" in tag else tag.upper())
        dec_layers = res.get("decoder_layers", 3)
        clean_ap = f"{res['clean']['mAP@50']:.2f}\\%"
        aqsm_ap = f"{res['aqsm']['mAP@50']:.2f}\\%"
        pgd_ap = f"{res['task_pgd']['mAP@50']:.2f}\\%"
        evict = f"{res['aqsm']['mean_evicted_pct']:.2f}\\%"
        cosine = f"{res['aqsm']['mean_memory_cosine']:.4f}"
        tex.append(f"{disp_name} & {dec_layers} & {clean_ap} & {aqsm_ap} & {pgd_ap} & {evict} & {cosine} \\\\")

    tex.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])
    return "\n".join(tex)


def main():
    parser = argparse.ArgumentParser(description="Multi-Model Comparative Benchmark for RT-DETR Architectures")
    parser.add_argument("--models", type=str, default="r18,r50", help="Danh sách models phân tách bởi dấu phẩy ('r18,r50')")
    parser.add_argument("--folder_image", type=str, default=str(COCO_VAL_DIR), help="Thư mục ảnh COCO")
    parser.add_argument("--annotations", type=str, default=str(COCO_ANN_FILE), help="File instances_val2017.json")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUTS_DIR), help="Thư mục xuất kết quả")
    parser.add_argument("--max_samples", type=int, default=1000, help="Số ảnh đánh giá cho mỗi model (mặc định 1000)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size suy luận")
    parser.add_argument("--device", type=str, default="auto", help="Thiết bị (auto / cuda / cpu)")
    parser.add_argument("--threads", type=int, default=8, help="Số luồng CPU")
    args = parser.parse_args()

    configure_cpu_threads(args.threads)
    hw_info = detect_hardware_environment(preferred_device=args.device)
    device = get_device(preferred=args.device)
    batch_size = args.batch_size if args.batch_size is not None else hw_info["recommended_batch_size"]

    models_list = [m.strip() for m in args.models.split(",") if m.strip()]
    print("=" * 85)
    print(" 🔬 TIẾN TRÌNH BENCHMARK ĐA KIẾN TRÚC MÔ HÌNH (CROSS-ARCHITECTURE BENCHMARK)")
    print(f" Danh sách mô hình : {models_list}")
    print(f" Cỡ mẫu mỗi model  : {args.max_samples} ảnh")
    print(f" Thiết bị tính toán: {device} | Batch Size: {batch_size}")
    print("=" * 85)

    all_results: Dict[str, Dict[str, Any]] = {}
    for m_spec in models_list:
        res = benchmark_single_model(
            model_name_or_alias=m_spec,
            folder_image=args.folder_image,
            annotations=args.annotations,
            max_samples=args.max_samples,
            batch_size=batch_size,
            device=device,
        )
        all_results[res["model_tag"]] = res

    # Lưu JSON tổng hợp
    out_dir = Path(args.output_dir)
    summary_dir = out_dir / "summary_json"
    summary_dir.mkdir(parents=True, exist_ok=True)
    json_path = summary_dir / "multimodel_benchmark_summary.json"
    save_json(all_results, json_path)
    print(f"\n>>> Đã xuất kết quả tổng hợp JSON: {json_path}")

    # Xuất LaTeX snippet
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    tex_str = generate_multimodel_latex(all_results)
    tex_path = tables_dir / "table_multimodel_benchmark.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex_str)
    print(f">>> Đã xuất snippet bảng LaTeX: {tex_path}")

    # In Markdown Table
    print("\n" + "=" * 85)
    print(" BẢNG SO SÁNH KẾT QUẢ ĐA KIẾN TRÚC MÔ HÌNH:")
    print("=" * 85)
    print("| Kiến trúc | Decoder Layers | Clean mAP@50 | AQSM mAP@50 | Task PGD mAP@50 | Evicted Q (%) | Mem Cosine |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for tag, res in all_results.items():
        disp_name = "RT-DETR-R18VD" if "r18" in tag else ("RT-DETR-R50VD" if "r50" in tag else tag.upper())
        dec = res.get("decoder_layers", 3)
        c_map = res["clean"]["mAP@50"]
        a_map = res["aqsm"]["mAP@50"]
        p_map = res["task_pgd"]["mAP@50"]
        ev = res["aqsm"]["mean_evicted_pct"]
        cos = res["aqsm"]["mean_memory_cosine"]
        print(f"| {disp_name} | {dec} | {c_map:.2f}% | {a_map:.2f}% | {p_map:.2f}% | {ev:.2f}% | {cos:.4f} |")
    print("=" * 85)


if __name__ == "__main__":
    main()
