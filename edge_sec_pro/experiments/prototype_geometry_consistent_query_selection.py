# -*- coding: utf-8 -*-
"""
Script: prototype_geometry_consistent_query_selection.py
Mục đích: Thiết kế và chạy thử nghiệm nguyên mẫu (prototype):
1. Geometry-Consistent Query Selection (GC-QS / Dual-Gated Query Selection)
2. Reference Shift Consistency Guard (RSC-Guard)

Đặt tại giao tiếp Encoder-Decoder của RT-DETR để:
- Ngăn chặn hiện tượng Spatial Target Misrouting do AQSM gây ra.
- Kiểm chứng thực nghiệm mức độ khôi phục mAP và độ trễ tính toán (latency overhead).
- Độc lập với pipeline chính (chạy thử nghiệm đánh giá trước).
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

torch.set_num_threads(8)

WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from edge_sec_pro.core.config import CHECKPOINTS_DIR, RESULTS_DIR
from edge_sec_pro.core.coco_gt import (
    load_coco_val2017_gt_cohort,
    postprocess_rtdetr_predictions,
    evaluate_coco_predictions,
)
from edge_sec_pro.transformer_monitors.detr_wrapper import RTDETRModelWrapper
from edge_sec_pro.transformer_monitors.rtdetr_mechanism_analyzer import (
    extract_encoder_components,
    run_decoder_from_components,
    capture_deformable_attention_internals,
    _summarize_attention_state,
)


def compute_geometry_consistent_query_indices(
    output_memory: torch.Tensor,               # [B, 2100, D]
    enc_outputs_class: torch.Tensor,           # [B, 2100, C]
    enc_outputs_coord_logits: torch.Tensor,    # [B, 2100, 4]
    anchors: torch.Tensor,                     # [1, 2100, 4]
    spatial_shapes_list: List[Tuple[int, int]],
    num_queries: int = 300,
    alpha_cls: float = 0.60,
    sigma_spatial: float = 0.10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Cơ chế Dual-Gated Geometry-Consistent Query Selection (GC-QS):
    Kết hợp giữa điểm số phân loại (Classification Confidence) và
    tính nhất quán hình học không gian (Geometric Spatial Support & Feature Energy).

    Returns:
        gc_indices: [B, num_queries] - Danh sách chỉ số đã được chọn lọc nhất quán.
        rsc_scores: [B] - Điểm bất thường Reference Shift Consistency Guard (RSC-Guard).
    """
    bsz, num_cand, _ = output_memory.shape
    device = output_memory.device

    # 1. Điểm phân loại nguyên bản
    probs = torch.sigmoid(enc_outputs_class)
    s_cls, _ = probs.max(dim=-1)  # [B, 2100]

    # 2. Tọa độ tâm anchor chuẩn hóa
    anchors_xy = torch.sigmoid(anchors[0, :, :2])  # [2100, 2]
    pred_boxes = torch.sigmoid(enc_outputs_coord_logits)  # [B, 2100, 4]
    pred_centers = pred_boxes[:, :, :2]  # [B, 2100, 2]
    pred_wh = pred_boxes[:, :, 2:]       # [B, 2100, 2]

    # 3. Tính độ hợp lý hình học (Geometric Plausibility)
    # Hộp có diện tích hợp lý (không quá nhỏ < 0.0005 và không choán toàn ảnh > 0.95)
    box_area = pred_wh[:, :, 0] * pred_wh[:, :, 1]  # [B, 2100]
    plausibility = torch.clamp(box_area / 0.002, 0.0, 1.0) * torch.clamp((0.95 - box_area) / 0.05, 0.0, 1.0)

    # 4. Tính năng lượng đặc trưng (Feature Energy Salience)
    feat_norm = torch.norm(output_memory, dim=-1)  # [B, 2100]
    min_fn = feat_norm.min(dim=-1, keepdim=True).values
    max_fn = feat_norm.max(dim=-1, keepdim=True).values
    norm_feat = (feat_norm - min_fn) / torch.clamp(max_fn - min_fn, min=1e-6)  # [B, 2100]

    # 5. Hỗ trợ mật độ không gian cục bộ (Spatial Consensus Density)
    # Lấy top-300 ứng viên có điểm phân loại sơ bộ để tính mật độ lân cận
    _, top_pre = torch.topk(s_cls, k=300, dim=1)  # [B, 300]
    
    # Tính ma trận khoảng cách giữa các candidate và top_pre candidates
    # pred_centers: [B, 2100, 2], sub_centers: [B, 300, 2]
    sub_centers = torch.gather(pred_centers, 1, top_pre.unsqueeze(-1).expand(-1, -1, 2))
    sub_scores = torch.gather(s_cls, 1, top_pre)  # [B, 300]

    diff = pred_centers.unsqueeze(2) - sub_centers.unsqueeze(1)  # [B, 2100, 300, 2]
    dist_sq = torch.sum(diff ** 2, dim=-1)  # [B, 2100, 300]
    kernel = torch.exp(-dist_sq / (2.0 * (sigma_spatial ** 2)))  # [B, 2100, 300]
    spatial_support = torch.sum(kernel * sub_scores.unsqueeze(1), dim=-1)  # [B, 2100]
    
    min_sp = spatial_support.min(dim=-1, keepdim=True).values
    max_sp = spatial_support.max(dim=-1, keepdim=True).values
    norm_spatial = (spatial_support - min_sp) / torch.clamp(max_sp - min_sp, min=1e-6)

    # 6. Điểm số kết hợp Dual-Gated Consistency Score
    # s_final = s_cls * [alpha * plausibility + beta * spatial_support + gamma * norm_feat]
    geom_factor = 0.50 * norm_spatial + 0.30 * plausibility + 0.20 * norm_feat
    s_final = (alpha_cls * s_cls + (1.0 - alpha_cls) * (s_cls * geom_factor))

    # Chọn Top-K queries theo điểm nhất quán hình học
    _, gc_indices = torch.topk(s_final, k=num_queries, dim=1)

    # 7. Tính chỉ số giám sát trực tuyến RSC-Guard:
    # Đo độ lệch tán xạ không gian giữa các reference points được chọn
    sel_boxes = torch.gather(pred_boxes, 1, gc_indices.unsqueeze(-1).expand(-1, -1, 4))
    centers = sel_boxes[:, :, :2]  # [B, num_queries, 2]
    mean_c = centers.mean(dim=1, keepdim=True)
    spatial_dispersion = torch.norm(centers - mean_c, dim=-1).mean(dim=-1)  # [B]
    
    # Đo độ bất nhất giữa điểm phân loại và năng lượng đặc trưng tại các queries được chọn
    sel_cls = torch.gather(s_cls, 1, gc_indices)
    sel_feat = torch.gather(norm_feat, 1, gc_indices)
    discordance = torch.clamp(sel_cls - sel_feat, min=0.0).mean(dim=-1)  # [B]
    
    rsc_scores = spatial_dispersion + 0.5 * discordance

    return gc_indices, rsc_scores


def run_prototype_evaluation(n_eval: int = 50, batch_size: int = 10) -> Dict[str, Any]:
    print("=" * 80)
    print("THỬ NGHIỆM NGUYÊN MẪU: GEOMETRY-CONSISTENT QUERY SELECTION (GC-QS)")
    print(f"Kiểm chứng trên N={n_eval} ảnh COCO val2017 Ground Truth (Official Instances)")
    print("=" * 80)

    device = torch.device("cpu")
    wrapper = RTDETRModelWrapper(pretrained_model_name="PekingU/rtdetr_r18vd", device=device)
    model = wrapper.model
    model.eval()

    # 1. Nạp COCO Ground Truth Cohort
    print("[1/4] Nạp dữ liệu COCO GT...")
    cohort = load_coco_val2017_gt_cohort(
        n_cal=1200,
        n_test=600,
        eval_start=0,
        n_eval=n_eval,
        seed=42,
        id2label=model.config.id2label,
        load_tensors=True,
    )
    clean_imgs = cohort["tensors"]
    gt_list = cohort["gt_list"]
    gt_boxes_batch = [gt["boxes_xyxy"] for gt in gt_list]

    # 2. Nạp Adversarial Cache AQSM (eps=16/255)
    print("[2/4] Nạp Adversarial Cache AQSM...")
    cache_file = CHECKPOINTS_DIR / "adv_cache" / f"adv_cache_aqsm_eps16_n300.npz"
    if not cache_file.exists():
        raise FileNotFoundError(f"Chưa có file cache: {cache_file}")
    cache_data = np.load(cache_file)
    delta_f16 = cache_data["delta_f16"][:n_eval]
    adv_imgs = torch.clamp(clean_imgs + torch.from_numpy(delta_f16.astype(np.float32)), 0.0, 1.0)

    # 3. Chạy 4 nhánh đánh giá đối chứng:
    # (1) Clean Baseline (Standard Selection)
    # (2) Clean Baseline + GC-QS (Đảm bảo không suy giảm hiệu năng sạch)
    # (3) AQSM Attack (Standard Selection - Bị tấn công không phòng thủ)
    # (4) AQSM Attack + GC-QS (Áp dụng cơ chế đề xuất)
    branches = [
        "clean_standard",
        "clean_gcqs",
        "aqsm_standard",
        "aqsm_gcqs",
    ]
    preds_dict: Dict[str, List[Dict[str, np.ndarray]]] = {b: [] for b in branches}
    attn_records_dict: Dict[str, List[Dict[str, Any]]] = {b: [] for b in branches}
    latencies: Dict[str, List[float]] = {b: [] for b in branches}
    rsc_guard_scores: Dict[str, List[float]] = {"clean": [], "aqsm": []}

    print("\n[3/4] Chạy suy luận và đo lường phục hồi hiệu năng...")
    for start in tqdm(range(0, n_eval, batch_size), desc="Prototype Runs"):
        end = min(n_eval, start + batch_size)
        c_sub = clean_imgs[start:end]
        a_sub = adv_imgs[start:end]
        gt_sub = gt_boxes_batch[start:end]
        b_len = end - start

        with torch.no_grad():
            c_comp = extract_encoder_components(model, c_sub)
            a_comp = extract_encoder_components(model, a_sub)

            # (1) Clean Standard
            t0 = time.time()
            with capture_deformable_attention_internals(model) as cap:
                dec_clean_std = run_decoder_from_components(
                    model=model,
                    target=c_comp["target"],
                    source_flatten=c_comp["source_flatten"],
                    init_reference_points=c_comp["init_reference_points"],
                    spatial_shapes=c_comp["spatial_shapes"],
                    spatial_shapes_list=c_comp["spatial_shapes_list"],
                    level_start_index=c_comp["level_start_index"],
                )
            latencies["clean_standard"].append((time.time() - t0) / b_len)
            preds_dict["clean_standard"].extend(
                postprocess_rtdetr_predictions(dec_clean_std["logits"], dec_clean_std["pred_boxes"], mode="hf_multilabel", max_dets=100)
            )
            attn_records_dict["clean_standard"].extend(
                _summarize_attention_state(cap, dec_clean_std["logits"].cpu().numpy(), gt_sub)["per_image"]
            )

            # (2) Clean with GC-QS
            t0 = time.time()
            gc_ind_c, rsc_c = compute_geometry_consistent_query_indices(
                output_memory=c_comp["output_memory"],
                enc_outputs_class=c_comp["enc_outputs_class"],
                enc_outputs_coord_logits=c_comp["enc_outputs_coord_logits"],
                anchors=c_comp["anchors"],
                spatial_shapes_list=c_comp["spatial_shapes_list"],
                num_queries=model.config.num_queries,
            )
            ref_dim = c_comp["enc_outputs_coord_logits"].shape[-1]
            mem_dim = c_comp["output_memory"].shape[-1]
            ref_gc_c = c_comp["enc_outputs_coord_logits"].gather(1, gc_ind_c.unsqueeze(-1).expand(-1, -1, ref_dim)).detach()
            tgt_gc_c = c_comp["output_memory"].gather(1, gc_ind_c.unsqueeze(-1).expand(-1, -1, mem_dim)).detach()

            with capture_deformable_attention_internals(model) as cap:
                dec_clean_gcqs = run_decoder_from_components(
                    model=model,
                    target=tgt_gc_c,
                    source_flatten=c_comp["source_flatten"],
                    init_reference_points=ref_gc_c,
                    spatial_shapes=c_comp["spatial_shapes"],
                    spatial_shapes_list=c_comp["spatial_shapes_list"],
                    level_start_index=c_comp["level_start_index"],
                )
            latencies["clean_gcqs"].append((time.time() - t0) / b_len)
            preds_dict["clean_gcqs"].extend(
                postprocess_rtdetr_predictions(dec_clean_gcqs["logits"], dec_clean_gcqs["pred_boxes"], mode="hf_multilabel", max_dets=100)
            )
            attn_records_dict["clean_gcqs"].extend(
                _summarize_attention_state(cap, dec_clean_gcqs["logits"].cpu().numpy(), gt_sub)["per_image"]
            )
            rsc_guard_scores["clean"].extend(rsc_c.cpu().tolist())

            # (3) AQSM Standard
            t0 = time.time()
            with capture_deformable_attention_internals(model) as cap:
                dec_aqsm_std = run_decoder_from_components(
                    model=model,
                    target=a_comp["target"],
                    source_flatten=a_comp["source_flatten"],
                    init_reference_points=a_comp["init_reference_points"],
                    spatial_shapes=a_comp["spatial_shapes"],
                    spatial_shapes_list=a_comp["spatial_shapes_list"],
                    level_start_index=a_comp["level_start_index"],
                )
            latencies["aqsm_standard"].append((time.time() - t0) / b_len)
            preds_dict["aqsm_standard"].extend(
                postprocess_rtdetr_predictions(dec_aqsm_std["logits"], dec_aqsm_std["pred_boxes"], mode="hf_multilabel", max_dets=100)
            )
            attn_records_dict["aqsm_standard"].extend(
                _summarize_attention_state(cap, dec_aqsm_std["logits"].cpu().numpy(), gt_sub)["per_image"]
            )

            # (4) AQSM with GC-QS (Phòng thủ đề xuất)
            t0 = time.time()
            gc_ind_a, rsc_a = compute_geometry_consistent_query_indices(
                output_memory=a_comp["output_memory"],
                enc_outputs_class=a_comp["enc_outputs_class"],
                enc_outputs_coord_logits=a_comp["enc_outputs_coord_logits"],
                anchors=a_comp["anchors"],
                spatial_shapes_list=a_comp["spatial_shapes_list"],
                num_queries=model.config.num_queries,
            )
            ref_gc_a = a_comp["enc_outputs_coord_logits"].gather(1, gc_ind_a.unsqueeze(-1).expand(-1, -1, ref_dim)).detach()
            tgt_gc_a = a_comp["output_memory"].gather(1, gc_ind_a.unsqueeze(-1).expand(-1, -1, mem_dim)).detach()

            with capture_deformable_attention_internals(model) as cap:
                dec_aqsm_gcqs = run_decoder_from_components(
                    model=model,
                    target=tgt_gc_a,
                    source_flatten=a_comp["source_flatten"],
                    init_reference_points=ref_gc_a,
                    spatial_shapes=a_comp["spatial_shapes"],
                    spatial_shapes_list=a_comp["spatial_shapes_list"],
                    level_start_index=a_comp["level_start_index"],
                )
            latencies["aqsm_gcqs"].append((time.time() - t0) / b_len)
            preds_dict["aqsm_gcqs"].extend(
                postprocess_rtdetr_predictions(dec_aqsm_gcqs["logits"], dec_aqsm_gcqs["pred_boxes"], mode="hf_multilabel", max_dets=100)
            )
            attn_records_dict["aqsm_gcqs"].extend(
                _summarize_attention_state(cap, dec_aqsm_gcqs["logits"].cpu().numpy(), gt_sub)["per_image"]
            )
            rsc_guard_scores["aqsm"].extend(rsc_a.cpu().tolist())

    # 4. Đánh giá COCO mAP và thống kê kết quả
    print("\n[4/4] Tính toán số liệu định lượng tổng hợp...")
    summary_results: Dict[str, Any] = {"n_eval": n_eval, "branches": {}}

    for b in branches:
        m = evaluate_coco_predictions(preds_dict[b], gt_list)
        recs = attn_records_dict[b]
        fg_hit = float(np.nanmean([r["mean_fg_spatial_hit"] for r in recs]) * 100.0)
        fg_hit_act = float(np.nanmean([r["mean_fg_spatial_hit_active_q"] for r in recs]) * 100.0)
        entropy = float(np.nanmean([r["mean_entropy_norm"] for r in recs]))
        lat_ms = float(np.mean(latencies[b]) * 1000.0)

        summary_results["branches"][b] = {
            "mAP@50": m["mAP@50"],
            "mAP@75": m["mAP@75"],
            "mAP@50:95": m["mAP@50:95"],
            "fg_spatial_hit_all": fg_hit,
            "fg_spatial_hit_act": fg_hit_act,
            "attn_entropy": entropy,
            "mean_latency_ms": lat_ms,
        }

    # Đánh giá năng lực phát hiện của RSC-Guard
    clean_rsc = np.array(rsc_guard_scores["clean"])
    aqsm_rsc = np.array(rsc_guard_scores["aqsm"])
    # Ngưỡng tại 95th percentile của clean
    tau_rsc = float(np.percentile(clean_rsc, 95.0))
    tpr_rsc = float(np.mean(aqsm_rsc >= tau_rsc) * 100.0)
    fpr_rsc = float(np.mean(clean_rsc >= tau_rsc) * 100.0)

    summary_results["rsc_guard_monitor"] = {
        "threshold_tau_95": tau_rsc,
        "clean_fpr_pct": fpr_rsc,
        "aqsm_tpr_pct": tpr_rsc,
        "clean_mean_score": float(np.mean(clean_rsc)),
        "aqsm_mean_score": float(np.mean(aqsm_rsc)),
    }

    print("\n" + "=" * 80)
    print("KẾT QUẢ THỰC NGHIỆM ĐỐI CHỨNG PROTOTYPE:")
    print("=" * 80)
    for b in branches:
        sb = summary_results["branches"][b]
        print(
            f"  {b:<16}: mAP@50 = {sb['mAP@50']:5.2f}% | "
            f"mAP@50:95 = {sb['mAP@50:95']:5.2f}% | "
            f"FG Hit = {sb['fg_spatial_hit_all']:4.1f}% | "
            f"Latency = {sb['mean_latency_ms']:5.1f} ms"
        )
    print("-" * 80)
    print(
        f"  RSC-Guard Monitor: Clean FPR = {fpr_rsc:.1f}% | "
        f"AQSM Detection TPR = {tpr_rsc:.1f}% (Ngưỡng tau={tau_rsc:.4f})"
    )
    print("=" * 80)

    # Lưu kết quả JSON
    out_file = RESULTS_DIR / "rtdetr_gc_query_selection_prototype_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2, ensure_ascii=False)
    print(f"\n--> Đã lưu kết quả thực nghiệm nguyên mẫu: {out_file}")

    return summary_results


if __name__ == "__main__":
    run_prototype_evaluation(n_eval=50, batch_size=10)
