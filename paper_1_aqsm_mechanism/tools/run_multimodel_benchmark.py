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
    compute_pooled_coco_bootstrap_ci,
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
    get_epsilon_l_inf,
    get_img_size,
    get_model_tag,
    get_pgd_num_steps,
    get_pgd_step_size,
    resolve_model_name,
    update_experiment_paths,
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


def _benchmark_cnn_nms_model(
    cnn_alias: str,
    folder_image: str,
    annotations: str,
    max_samples: int,
    batch_size: int,
    device: torch.device,
    eps_255: int = 16,
    pgd_steps: int = 15,
) -> Dict[str, Any]:
    """
    Benchmark phát hiện đối tượng thực nghiệm trên họ CNN+NMS (Torchvision RetinaNet/FasterRCNN)
    dưới Clean Baseline, Gaussian Noise (sigma=eps) và White-Box Task PGD L_inf.
    """
    import torchvision.models.detection as tv_det

    key = cnn_alias.strip().lower()
    if "fasterrcnn" in key:
        weights_enum = tv_det.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        cnn_model = tv_det.fasterrcnn_resnet50_fpn_v2(weights=weights_enum).to(device).eval()
        full_name = "torchvision/fasterrcnn_resnet50_fpn_v2"
        model_tag = "fasterrcnn_nms"
    else:
        weights_enum = tv_det.RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
        cnn_model = tv_det.retinanet_resnet50_fpn_v2(weights=weights_enum).to(device).eval()
        full_name = "torchvision/retinanet_resnet50_fpn_v2"
        model_tag = "retinanet_nms"

    cats_91 = weights_enum.meta.get("categories", [])
    id2label = {idx: name for idx, name in enumerate(cats_91)} if cats_91 else None
    img_size = int(get_img_size())

    dataset = FlexibleCOCODataset(
        folder_image=folder_image,
        annotations_file=annotations,
        img_size=img_size,
        id2label=id2label,
        lazy_tensors=True,
        device=device,
        max_samples=max_samples,
    )
    clean_proxy = dataset.tensors
    gt_list = dataset.gt_list
    n_eval = len(dataset)

    def _run_eval_pass(model_obj: torch.nn.Module, tensors_source: Any, desc_str: str) -> tuple:
        preds: list = []
        lats = []
        model_obj.eval()
        with torch.no_grad():
            for i in tqdm(range(0, n_eval, batch_size), desc=desc_str):
                end_idx = min(i + batch_size, n_eval)
                b_imgs = tensors_source[i:end_idx].to(device)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.time()
                outs = model_obj([b_imgs[k] for k in range(b_imgs.shape[0])])
                if device.type == "cuda":
                    torch.cuda.synchronize()
                lats.append(((time.time() - t0) / max(1, b_imgs.shape[0])) * 1000.0)
                for o in outs:
                    bx = (o["boxes"].detach().cpu().numpy() / float(img_size)).clip(0.0, 1.0)
                    sc = o["scores"].detach().cpu().numpy().astype(np.float64)
                    lb = o["labels"].detach().cpu().numpy().astype(np.int64)
                    preds.append({"boxes": bx.astype(np.float64), "scores": sc, "labels": lb})
        return preds, float(np.mean(lats)) if lats else 0.0

    # 1. Clean Baseline
    clean_preds, clean_lat = _run_eval_pass(cnn_model, clean_proxy, f"{model_tag}_clean")
    clean_eval = evaluate_coco_predictions(clean_preds, gt_list, num_classes=91)
    clean_ci = compute_pooled_coco_bootstrap_ci(clean_preds, gt_list, n_boot=300, num_classes=91)

    # 2. Gaussian Noise Control
    eps_val = float(eps_255) / 255.0
    rng_g = torch.Generator(device="cpu").manual_seed(42)
    gauss_preds_list = []
    cnn_model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, n_eval, batch_size), desc=f"{model_tag}_gaussian"):
            end_idx = min(i + batch_size, n_eval)
            b_imgs = clean_proxy[i:end_idx].to(device)
            noise = torch.randn(b_imgs.shape, generator=rng_g, dtype=b_imgs.dtype).to(device) * eps_val
            b_noisy = torch.clamp(b_imgs + torch.clamp(noise, -eps_val, eps_val), 0.0, 1.0)
            outs = cnn_model([b_noisy[k] for k in range(b_noisy.shape[0])])
            for o in outs:
                bx = (o["boxes"].detach().cpu().numpy() / float(img_size)).clip(0.0, 1.0)
                gauss_preds_list.append({
                    "boxes": bx.astype(np.float64),
                    "scores": o["scores"].detach().cpu().numpy().astype(np.float64),
                    "labels": o["labels"].detach().cpu().numpy().astype(np.int64),
                })
    gauss_eval = evaluate_coco_predictions(gauss_preds_list, gt_list, num_classes=91)

    # 3. White-Box Task PGD on CNN+NMS
    step_alpha = float(get_pgd_step_size())
    pgd_preds = []
    pgd_lats = []
    for i in tqdm(range(0, n_eval, batch_size), desc=f"{model_tag}_taskpgd"):
        end_idx = min(i + batch_size, n_eval)
        b_orig = clean_proxy[i:end_idx].to(device)
        b_adv = b_orig.clone().detach()

        valid_pairs = []
        for local_k, img_idx in enumerate(range(i, end_idx)):
            gt_b = gt_list[img_idx]
            crowd = gt_b.get("iscrowd", np.zeros(len(gt_b["labels"]), dtype=bool))
            reg_boxes = gt_b["boxes_xyxy"][~crowd] * float(img_size)
            reg_labels = gt_b["labels"][~crowd]
            if len(reg_boxes) > 0:
                valid_pairs.append((
                    local_k,
                    {
                        "boxes": torch.from_numpy(reg_boxes).to(device=device, dtype=torch.float32),
                        "labels": torch.from_numpy(reg_labels).to(device=device, dtype=torch.int64),
                    },
                ))

        if valid_pairs:
            cnn_model.train()
            for mod in cnn_model.modules():
                if isinstance(mod, torch.nn.BatchNorm2d) or "FrozenBatchNorm2d" in type(mod).__name__:
                    mod.eval()
            for _ in range(pgd_steps):
                b_adv.requires_grad_(True)
                img_inputs = [b_adv[lk] for lk, _ in valid_pairs]
                tgt_inputs = [tgt for _, tgt in valid_pairs]
                loss_dict = cnn_model(img_inputs, tgt_inputs)
                loss_tot = sum(loss_dict.values()) if isinstance(loss_dict, dict) else loss_dict
                cnn_model.zero_grad()
                if b_adv.grad is not None:
                    b_adv.grad.zero_()
                loss_tot.backward()
                with torch.no_grad():
                    gs = b_adv.grad.sign() if b_adv.grad is not None else torch.zeros_like(b_adv)
                    b_adv = b_adv + step_alpha * gs
                    delta = torch.clamp(b_adv - b_orig, min=-eps_val, max=eps_val)
                    b_adv = torch.clamp(b_orig + delta, min=0.0, max=1.0).detach()

        cnn_model.eval()
        with torch.no_grad():
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            outs = cnn_model([b_adv[k] for k in range(b_adv.shape[0])])
            if device.type == "cuda":
                torch.cuda.synchronize()
            pgd_lats.append(((time.time() - t0) / max(1, b_adv.shape[0])) * 1000.0)
            for o in outs:
                bx = (o["boxes"].detach().cpu().numpy() / float(img_size)).clip(0.0, 1.0)
                pgd_preds.append({
                    "boxes": bx.astype(np.float64),
                    "scores": o["scores"].detach().cpu().numpy().astype(np.float64),
                    "labels": o["labels"].detach().cpu().numpy().astype(np.int64),
                })

    pgd_eval = evaluate_coco_predictions(pgd_preds, gt_list, num_classes=91)
    pgd_ci = compute_pooled_coco_bootstrap_ci(pgd_preds, gt_list, n_boot=300, num_classes=91)

    del cnn_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return {
        "model_name": full_name,
        "model_tag": model_tag,
        "family": "cnn_nms",
        "has_dynamic_selection": False,
        "postprocess_mode": "torchvision_nms",
        "n_samples": n_eval,
        "decoder_layers": 0,
        "hidden_dim": 256,
        "clean": {
            "mAP@50": round(float(clean_eval["mAP@50"]), 2),
            "mAP@50_ci95": clean_ci["mAP@50_ci95"],
            "mAP@50:95": round(float(clean_eval["mAP@50:95"]), 2),
            "recall_pct": round(float(clean_eval["recall_pct"]), 2),
            "mean_latency_ms": round(clean_lat, 2),
        },
        "gaussian_noise": {
            "mAP@50": round(float(gauss_eval["mAP@50"]), 2),
            "mAP@50:95": round(float(gauss_eval["mAP@50:95"]), 2),
        },
        "aqsm": None,
        "task_pgd": {
            "mAP@50": round(float(pgd_eval["mAP@50"]), 2),
            "mAP@50_ci95": pgd_ci["mAP@50_ci95"],
            "mAP@50:95": round(float(pgd_eval["mAP@50:95"]), 2),
            "recall_pct": round(float(pgd_eval["recall_pct"]), 2),
            "mean_latency_ms": round(float(np.mean(pgd_lats)) if pgd_lats else 0.0, 2),
            "mean_evicted_pct": None,
            "mean_memory_cosine": None,
        },
    }


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
    if key in ("retinanet", "retinanet_nms", "fasterrcnn", "fasterrcnn_nms"):
        return _benchmark_cnn_nms_model(
            cnn_alias=key,
            folder_image=folder_image,
            annotations=annotations,
            max_samples=max_samples,
            batch_size=batch_size,
            device=device,
            eps_255=eps_255,
            pgd_steps=pgd_steps,
        )
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
        img_size=get_img_size(),
        id2label=model.config.id2label,
        lazy_tensors=True,
        device=device,
        max_samples=max_samples,
    )
    clean_proxy = dataset.tensors
    gt_list = dataset.gt_list
    n_eval = len(dataset)

    base_alpha = float(get_pgd_step_size())
    step_alpha = base_alpha if int(eps_255) >= 8 else min(base_alpha, 1.0 / 255.0)
    attacker = DETRAttackGenerator(
        wrapper,
        epsilon=float(eps_255) / 255.0,
        alpha=step_alpha,
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
    clean_ci = compute_pooled_coco_bootstrap_ci(clean_preds, gt_list, n_boot=300)
    results["clean"] = {
        "mAP@50": round(float(clean_eval["mAP@50"]), 2),
        "mAP@50_ci95": clean_ci["mAP@50_ci95"],
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
        aqsm_ci = compute_pooled_coco_bootstrap_ci(aqsm_preds, gt_list, n_boot=300)
        results["aqsm"] = {
            "mAP@50": round(float(aqsm_eval["mAP@50"]), 2),
            "mAP@50_ci95": aqsm_ci["mAP@50_ci95"],
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
        b_hf_labels = dataset.hf_labels[i:end_idx] if getattr(dataset, "hf_labels", None) is not None else None
        b_adv = attacker.generate_standard_task_pgd(b_clean, hf_labels=b_hf_labels, steps=pgd_steps)

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
    pgd_ci = compute_pooled_coco_bootstrap_ci(pgd_preds, gt_list, n_boot=300)
    results["task_pgd"] = {
        "mAP@50": round(float(pgd_eval["mAP@50"]), 2),
        "mAP@50_ci95": pgd_ci["mAP@50_ci95"],
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
    if "retinanet" in t:
        return "RetinaNet-R50-FPN (CNN+NMS)"
    if "fasterrcnn" in t:
        return "FasterRCNN-R50-FPN (CNN+NMS)"
    return tag.upper()


def generate_multimodel_latex(all_results: Dict[str, Dict[str, Any]], eps_255: int = 16) -> str:
    """Sinh bảng LaTeX so sánh các biến thể kiến trúc."""
    tex = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{Cross-Architecture Robustness Benchmark across Real-Time DETR Variants under $\varepsilon={eps_255}/255$.}}",
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
    parser.add_argument("--eps_255", type=int, default=None, help="Biên độ nhiễu L_inf trên thang 255 (mặc định đọc từ cấu hình)")
    parser.add_argument("--pgd_steps", type=int, default=None, help="Số bước lặp PGD (mặc định đọc từ cấu hình)")
    args = parser.parse_args()

    eff_eps_255 = int(args.eps_255) if args.eps_255 is not None else int(round(get_epsilon_l_inf() * 255.0))
    eff_pgd_steps = int(args.pgd_steps) if args.pgd_steps is not None else int(get_pgd_num_steps())
    update_experiment_paths(
        folder_image=args.folder_image,
        annotations=args.annotations,
        output_dir=args.output_dir,
        pgd_steps=eff_pgd_steps,
        eps_l_inf=float(eff_eps_255) / 255.0,
    )

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
            eps_255=eff_eps_255,
            pgd_steps=eff_pgd_steps,
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
    tex_str = generate_multimodel_latex(all_results, eps_255=eff_eps_255)
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
