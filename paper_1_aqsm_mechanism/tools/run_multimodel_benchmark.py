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
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import time
from pathlib import Path
from typing import Any, Dict

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
from paper_1_aqsm_mechanism.core.detr_wrapper import CROSS_DETECTOR_SPECS, CrossDetectorWrapper
from paper_1_aqsm_mechanism.experiments.config import (
    COCO_ANN_FILE,
    COCO_VAL_DIR,
    OUTPUTS_DIR,
    configure_cpu_threads,
    detect_hardware_environment,
    get_device,
    get_model_tag,
    resolve_model_name,
)
from paper_1_aqsm_mechanism.utils.io_helpers import save_json


def _get_postprocess_mode(wrapper: CrossDetectorWrapper) -> str:
    """Chọn chế độ hậu xử lý hộp/điểm phù hợp với từng họ kiến trúc detector."""
    if not wrapper.is_sigmoid or wrapper.family == "detr":
        return "detr_softmax"
    if wrapper.family in ("deformable_detr", "rtdetrv2"):
        return "per_query_max"
    return "hf_multilabel"


def _compute_batch_eviction_and_cosine(
    c_out: Any,
    a_out: Any,
    top_k: int,
) -> tuple:
    """Tính tỷ lệ Query Eviction (nếu có enc_outputs_class) và Memory Cosine Similarity."""
    evict_list = []
    c_enc_cls = getattr(c_out, "enc_outputs_class", None)
    a_enc_cls = getattr(a_out, "enc_outputs_class", None)
    if c_enc_cls is not None and a_enc_cls is not None:
        k_eff = min(int(top_k), int(c_enc_cls.shape[1]))
        c_top = torch.topk(torch.sigmoid(c_enc_cls).max(dim=-1).values, k=k_eff, dim=-1).indices
        a_top = torch.topk(torch.sigmoid(a_enc_cls).max(dim=-1).values, k=k_eff, dim=-1).indices
        for b_idx in range(c_top.shape[0]):
            c_set = set(c_top[b_idx].tolist())
            a_set = set(a_top[b_idx].tolist())
            evict_list.append(1.0 - (len(c_set & a_set) / float(k_eff)))

    cos_val = float("nan")
    c_mem = getattr(c_out, "encoder_last_hidden_state", None)
    a_mem = getattr(a_out, "encoder_last_hidden_state", None)
    if c_mem is not None and a_mem is not None:
        if isinstance(c_mem, (list, tuple)):
            c_mem_t = torch.cat([t.flatten(2).transpose(1, 2) if t.ndim == 4 else t for t in c_mem], dim=1)
            a_mem_t = torch.cat([t.flatten(2).transpose(1, 2) if t.ndim == 4 else t for t in a_mem], dim=1)
        else:
            c_mem_t = c_mem
            a_mem_t = a_mem
        cos_sim = torch.nn.functional.cosine_similarity(c_mem_t, a_mem_t, dim=-1)
        cos_val = float(cos_sim.mean().item())

    return evict_list, cos_val


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
    """Đánh giá toàn diện một mô hình trên Clean, AQSM (nếu có dynamic selection) và Task PGD."""
    key = str(model_name_or_alias).strip().lower()
    alias_hint = model_name_or_alias if key in CROSS_DETECTOR_SPECS else resolve_model_name(model_name_or_alias)

    # 1. Nạp CrossDetectorWrapper
    wrapper = CrossDetectorWrapper(model_name_or_alias=alias_hint, device=device)
    wrapper.eval()
    model = wrapper.model
    full_model_name = wrapper.pretrained_model_name
    model_tag = get_model_tag(full_model_name)
    pp_mode = _get_postprocess_mode(wrapper)

    print("\n" + "=" * 80)
    print(f" [BENCHMARK] Khởi chạy đánh giá mô hình: {full_model_name} ({model_tag})")
    print(f" Family: {wrapper.family} | Dynamic Selection: {wrapper.has_dynamic_selection} | Mode: {pp_mode}")
    print(f" Samples: {max_samples} | Batch size: {batch_size} | Device: {device}")
    print("=" * 80)

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
        "family": wrapper.family,
        "has_dynamic_selection": bool(wrapper.has_dynamic_selection),
        "postprocess_mode": pp_mode,
        "n_samples": n_eval,
        "decoder_layers": wrapper.decoder_layers,
        "hidden_dim": wrapper.d_model,
    }

    # 3. Đánh giá Clean Baseline & Latency
    print(">>> 1/3: Đánh giá Clean Baseline...")
    clean_preds = []
    latencies_ms = []
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
            postprocess_rtdetr_predictions(out["logits"], out["pred_boxes"], mode=pp_mode)
        )

    clean_eval = evaluate_coco_predictions(clean_preds, gt_list)
    results["clean"] = {
        "mAP@50": round(float(clean_eval["mAP@50"]), 2),
        "mAP@50:95": round(float(clean_eval["mAP@50:95"]), 2),
        "recall_pct": round(float(clean_eval["recall_pct"]), 2),
        "mean_latency_ms": round(float(np.mean(latencies_ms)), 2),
    }
    print(f"   Clean mAP@50: {results['clean']['mAP@50']:.2f}% | Latency: {results['clean']['mean_latency_ms']:.2f} ms")

    # 4. Đánh giá AQSM Attack (chỉ khả dụng trên mô hình có encoder top-K selection)
    if wrapper.has_dynamic_selection:
        print(">>> 2/3: Đánh giá AQSM Attack (eps=16/255)...")
        aqsm_preds = []
        aqsm_evictions = []
        aqsm_cosines = []
        for i in tqdm(range(0, n_eval, batch_size), desc=f"{model_tag}_aqsm"):
            end_idx = min(i + batch_size, n_eval)
            b_clean = clean_proxy[i:end_idx].to(device)
            b_adv = attacker.generate_query_selection_attack(b_clean, steps=pgd_steps)

            with torch.no_grad():
                c_out = model(pixel_values=b_clean, return_dict=True)
                a_out = model(pixel_values=b_adv, return_dict=True)
                ev_list, cos_val = _compute_batch_eviction_and_cosine(c_out, a_out, top_k=wrapper.num_queries)
                aqsm_evictions.extend(ev_list)
                if not np.isnan(cos_val):
                    aqsm_cosines.append(cos_val)

            aqsm_preds.extend(
                postprocess_rtdetr_predictions(a_out.logits, a_out.pred_boxes, mode=pp_mode)
            )

        aqsm_eval = evaluate_coco_predictions(aqsm_preds, gt_list)
        results["aqsm"] = {
            "mAP@50": round(float(aqsm_eval["mAP@50"]), 2),
            "mAP@50:95": round(float(aqsm_eval["mAP@50:95"]), 2),
            "recall_pct": round(float(aqsm_eval["recall_pct"]), 2),
            "mean_evicted_pct": round(float(np.mean(aqsm_evictions) * 100.0), 2) if aqsm_evictions else None,
            "mean_memory_cosine": round(float(np.mean(aqsm_cosines)), 4) if aqsm_cosines else None,
        }
        ev_disp = f"{results['aqsm']['mean_evicted_pct']:.2f}%" if results["aqsm"]["mean_evicted_pct"] is not None else "N/A"
        cos_disp = f"{results['aqsm']['mean_memory_cosine']:.4f}" if results["aqsm"]["mean_memory_cosine"] is not None else "N/A"
        print(f"   AQSM mAP@50: {results['aqsm']['mAP@50']:.2f}% | Eviction: {ev_disp} | Cosine: {cos_disp}")
    else:
        print(f">>> 2/3: Bỏ qua AQSM Attack vì {model_tag} dùng Static Queries (không có encoder top-K selection).")
        results["aqsm"] = None

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
            c_out = model(pixel_values=b_clean, return_dict=True)
            a_out = model(pixel_values=b_adv, return_dict=True)
            ev_list, cos_val = _compute_batch_eviction_and_cosine(c_out, a_out, top_k=wrapper.num_queries)
            pgd_evictions.extend(ev_list)
            if not np.isnan(cos_val):
                pgd_cosines.append(cos_val)

        pgd_preds.extend(
            postprocess_rtdetr_predictions(a_out.logits, a_out.pred_boxes, mode=pp_mode)
        )

    pgd_eval = evaluate_coco_predictions(pgd_preds, gt_list)
    results["task_pgd"] = {
        "mAP@50": round(float(pgd_eval["mAP@50"]), 2),
        "mAP@50:95": round(float(pgd_eval["mAP@50:95"]), 2),
        "recall_pct": round(float(pgd_eval["recall_pct"]), 2),
        "mean_evicted_pct": round(float(np.mean(pgd_evictions) * 100.0), 2) if pgd_evictions else None,
        "mean_memory_cosine": round(float(np.mean(pgd_cosines)), 4) if pgd_cosines else None,
    }
    ev_p = f"{results['task_pgd']['mean_evicted_pct']:.2f}%" if results["task_pgd"]["mean_evicted_pct"] is not None else "N/A"
    cos_p = f"{results['task_pgd']['mean_memory_cosine']:.4f}" if results["task_pgd"]["mean_memory_cosine"] is not None else "N/A"
    print(f"   Task PGD mAP@50: {results['task_pgd']['mAP@50']:.2f}% | Eviction: {ev_p} | Cosine: {cos_p}")

    # Thu hồi bộ nhớ
    del wrapper, model, attacker
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return results


def _format_arch_display_name(tag: str) -> str:
    t = tag.lower()
    if "rtdetr_v2_r18" in t or "rtdetrv2_r18" in t:
        return "RT-DETRv2-R18VD"
    if "rtdetr_v2_r50" in t or "rtdetrv2_r50" in t:
        return "RT-DETRv2-R50VD"
    if "r18" in t and "rtdetr" in t:
        return "RT-DETR-R18VD"
    if "r50" in t and "rtdetr" in t:
        return "RT-DETR-R50VD"
    if "two-stage" in t or "twostage" in t:
        return "Deformable-DETR (2-Stage)"
    if "deformable" in t:
        return "Deformable-DETR (1-Stage)"
    if "detr-resnet-50" in t or "detr_r50" in t:
        return "DETR-R50 (Static)"
    return tag.upper()


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
        disp_name = _format_arch_display_name(tag)
        dec_layers = res.get("decoder_layers", 3)
        clean_ap = f"{res['clean']['mAP@50']:.2f}\\%"
        aqsm_res = res.get("aqsm")
        if aqsm_res is not None:
            aqsm_ap = f"{aqsm_res['mAP@50']:.2f}\\%"
            evict = f"{aqsm_res['mean_evicted_pct']:.2f}\\%" if aqsm_res.get("mean_evicted_pct") is not None else "---"
            cosine = f"{aqsm_res['mean_memory_cosine']:.4f}" if aqsm_res.get("mean_memory_cosine") is not None else "---"
        else:
            aqsm_ap = "---"
            evict = "---"
            pgd_cos = res["task_pgd"].get("mean_memory_cosine")
            cosine = f"{pgd_cos:.4f}" if pgd_cos is not None else "---"
        pgd_ap = f"{res['task_pgd']['mAP@50']:.2f}\\%"
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
        disp_name = _format_arch_display_name(tag)
        dec = res.get("decoder_layers", 3)
        c_map = res["clean"]["mAP@50"]
        p_map = res["task_pgd"]["mAP@50"]
        aqsm_res = res.get("aqsm")
        if aqsm_res is not None:
            a_str = f"{aqsm_res['mAP@50']:.2f}%"
            ev_str = f"{aqsm_res['mean_evicted_pct']:.2f}%" if aqsm_res.get("mean_evicted_pct") is not None else "N/A"
            cos_str = f"{aqsm_res['mean_memory_cosine']:.4f}" if aqsm_res.get("mean_memory_cosine") is not None else "N/A"
        else:
            a_str = "N/A (Static)"
            ev_str = "N/A"
            pgd_cos = res["task_pgd"].get("mean_memory_cosine")
            cos_str = f"{pgd_cos:.4f}" if pgd_cos is not None else "N/A"
        print(f"| {disp_name} | {dec} | {c_map:.2f}% | {a_str} | {p_map:.2f}% | {ev_str} | {cos_str} |")
    print("=" * 85)


if __name__ == "__main__":
    main()
