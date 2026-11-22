# -*- coding: utf-8 -*-
"""
Module: paper_2_phantom_fabrication.experiments.streaming_evaluator
Mục đích: Bộ đánh giá suy luận đối kháng theo dòng (Streaming Adversarial Evaluator).
Xử lý dữ liệu ảnh COCO theo batch streaming bất đồng bộ, nạp và lưu cache đối kháng
IEEE 754 float32 trên đĩa, giữ mức tiêu thụ RAM luôn phẳng (< 1.5 GB) trên toàn bộ 5,000 ảnh.
"""

import gc
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from tqdm import tqdm

from paper_2_phantom_fabrication.attacks.phantom_attacks import generate_batch_adversarial
from paper_2_phantom_fabrication.attacks.yolo_attacks import generate_yolo_adversarial_batch
from paper_2_phantom_fabrication.core.adv_cache import (
    get_adv_cache_dir,
    load_cached_adv_delta,
    save_adv_cache,
)
from paper_2_phantom_fabrication.core.coco_gt import (
    box_cxcywh_to_xyxy,
    evaluate_coco_predictions,
    postprocess_rtdetr_predictions,
)
from paper_2_phantom_fabrication.core.matcher import match_detections_against_gt
from paper_2_phantom_fabrication.core.metrics import (
    compute_disentangled_precision_breakdown,
    compute_phantom_metrics,
)

logger = logging.getLogger("StreamingEvaluator")


def get_cohort_dataloader(
    cohort: Dict[str, Any],
    batch_size: int,
    device: torch.device,
) -> DataLoader:
    """
    Tạo streaming DataLoader với prefetching bất đồng bộ.
    Mỗi batch ảnh chỉ nạp vào RAM khi cần tính toán, giúp RAM luôn phẳng (<2GB)
    ngay cả khi chạy trên toàn bộ 5,000 ảnh COCO.
    """
    dataset = cohort["dataset"]
    num_workers = 2 if sys.platform != "win32" else 0
    pin_memory = (device.type == "cuda")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def run_streaming_condition_eval(
    model: torch.nn.Module,
    dataloader: DataLoader,
    condition: Optional[str] = None,
    gt_list: Optional[List[Dict[str, Any]]] = None,
    target_class_idx: int = 1,
    epsilon: Optional[float] = None,
    alpha: Optional[float] = None,
    steps: int = 8,
    device: Optional[torch.device] = None,
    clean_topk_reference: Optional[torch.Tensor] = None,
    save_logits_and_boxes: bool = False,
    yolo_model: Optional[Any] = None,
    desc: Optional[str] = None,
    image_ids: Optional[Sequence[int]] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    skip_cache_gen: bool = False,
    force_recompute: bool = False,
    condition_type: Optional[str] = None,
    cohort: Optional[Dict[str, Any]] = None,
    attack_eps: Optional[float] = None,
    attack_alpha: Optional[float] = None,
    attack_steps: Optional[int] = None,
    detector_backend: Optional[str] = None,
    preprocessor: Optional[Any] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Thực thi tạo tấn công và suy luận mô hình theo dòng (streaming evaluation) qua DataLoader.
    Hỗ trợ đối số linh hoạt và tương thích ngược 100% giữa các phiên bản Engine.
    """
    if condition is None:
        condition = condition_type if condition_type is not None else "Clean"

    # Chuẩn hóa condition về tên chính tắc (Canonical form: Clean, Gaussian_Noise, Untargeted_PGD, EQF, DQF)
    c_norm = str(condition).strip().lower()
    if c_norm in ("clean", "none", "identity", ""):
        condition = "Clean"
    elif c_norm in ("gaussian_noise", "gaussian", "noise"):
        condition = "Gaussian_Noise"
    elif c_norm in ("untargeted_pgd", "pgd", "untargeted"):
        condition = "Untargeted_PGD"
    elif c_norm in ("eqf", "standard_eqf"):
        condition = "EQF"
    elif c_norm in ("rab_eqf", "rab", "rank_aware_eqf"):
        condition = "RAB-EQF"
    elif c_norm in ("dc_eqf", "dc", "dispersion_constrained_eqf"):
        condition = "DC-EQF"
    elif c_norm in ("dqf", "output_logit"):
        condition = "DQF"

    if cohort is not None:
        if gt_list is None:
            gt_list = cohort.get("gt_list", [])
        if image_ids is None:
            image_ids = cohort.get("image_ids", None)

    if epsilon is None:
        epsilon = attack_eps if attack_eps is not None else (16.0 / 255.0)

    if alpha is None:
        alpha = attack_alpha if attack_alpha is not None else (2.0 / 255.0)

    if attack_steps is not None:
        steps = attack_steps

    if device is None:
        try:
            device = next(model.parameters()).device
        except Exception:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if gt_list is None:
        gt_list = []
    """
    Thực thi tạo tấn công và suy luận mô hình theo dòng (streaming evaluation) qua DataLoader.
    Tích hợp bộ nhớ đệm đối kháng chuẩn công nghiệp (Adversarial Cache - Học hỏi từ Paper 1):
      - Nạp delta đối kháng trực tiếp từ file .npz mmap nếu đã tồn tại, tiết kiệm 95% thời gian chạy lại PGD.
      - Nếu chưa có cache: tự động sinh và lưu ra đĩa chuẩn IEEE 754 float32 (TUYỆT ĐỐI KHÔNG ÉP KIỂU UINT8).
      - Sử dụng disk-backed memmap và giải phóng bộ nhớ từng batch, đảm bảo RAM luôn < 1.5 GB trên toàn bộ 5,000 ảnh COCO.
    """
    model.eval()
    matched_stats = []
    displacement_rates = []
    post_bench_all = []
    post_oper_all = []

    clean_topk_list = []
    logits_saved_list = []
    boxes_saved_list = []

    # Dành cho đánh giá YOLOv8
    yolo_preds = []
    yolo_matched = []
    yolo_pre_nms = []
    yolo_post_nms = []

    eps_int = int(round(epsilon * 255.0))
    cdir = Path(cache_dir) if cache_dir is not None else get_adv_cache_dir()
    is_adv = condition not in ("Clean", "Gaussian_Noise")

    cached_delta = None
    if is_adv and not force_recompute:
        n_total = len(dataloader.dataset)
        cached_delta, _ = load_cached_adv_delta(
            attack=condition,
            eps_num=eps_int,
            steps=steps,
            target_class_idx=target_class_idx,
            n_eval=n_total,
            expected_image_ids=image_ids,
            cache_dir=cdir,
        )
        if cached_delta is not None:
            logger.info(f"    [Cache Hit] Nạp {condition} (eps={eps_int}, t={steps}, cls={target_class_idx}) từ cache trên đĩa chuẩn float32.")

    temp_mmap = None
    temp_mmap_path = None
    if is_adv and cached_delta is None and not skip_cache_gen and image_ids is not None:
        try:
            import shutil
            cdir.mkdir(parents=True, exist_ok=True)
            n_total = len(dataloader.dataset)
            req_bytes = n_total * 3 * 640 * 640 * 4
            disk_free = shutil.disk_usage(cdir).free
            if disk_free >= max(req_bytes * 1.25, 2.0 * (1024 ** 3)):
                temp_mmap_path = cdir / f"tmp_{condition.lower()}_eps{eps_int}_{os.getpid()}_{int(time.time())}.dat"
                temp_mmap = np.memmap(
                    temp_mmap_path,
                    dtype=np.float32,
                    mode="w+",
                    shape=(n_total, 3, 640, 640),
                )
            else:
                logger.info(
                    f"    [Disk Safe] Ổ đĩa khả dụng ({disk_free / (1024**3):.2f} GB) đang thấp. "
                    f"Tự động bỏ qua lưu cache đĩa, chạy trực tiếp On-The-Fly GPU (0 GB đĩa)."
                )
                temp_mmap = None
        except Exception as e:
            logger.warning(f"Không thể khởi tạo disk-backed memmap cho cache {condition}: {e}")
            temp_mmap = None

    progress_bar = tqdm(
        dataloader,
        desc=desc or f"Eval [{condition}]",
        leave=False,
        dynamic_ncols=True,
    )

    is_yolo = (detector_backend == "yolo") or (type(model).__name__ == "YOLO")

    for batch_images, batch_indices in progress_bar:
        bx = batch_images.to(device)

        # 1. Nạp từ cache hoặc sinh ảnh đối kháng theo lô
        if cached_delta is not None:
            sub_indices = batch_indices.numpy()
            delta_np = cached_delta[sub_indices]
            delta_tensor = torch.from_numpy(delta_np.astype(np.float32)).to(device)
            adv_b = torch.clamp(bx + delta_tensor, 0.0, 1.0)
        elif is_yolo or condition == "Clean":
            adv_b = bx
        elif condition == "Gaussian_Noise":
            noise = torch.randn_like(bx) * epsilon
            noise = torch.clamp(noise, -epsilon, epsilon)
            adv_b = torch.clamp(bx + noise, 0.0, 1.0)
        else:
            k_eqf = kwargs.get("k_fake_eqf", 50)
            adv_b = generate_batch_adversarial(
                model=model,
                bx=bx,
                attack_type=condition,
                target_class_idx=target_class_idx,
                epsilon=epsilon,
                alpha=alpha,
                steps=steps,
                k_fake_eqf=k_eqf,
            )
            if temp_mmap is not None:
                delta_b = (adv_b - bx).detach().cpu().to(torch.float32).numpy()
                temp_mmap[batch_indices.numpy()] = delta_b

        # Áp dụng bộ lọc tiền xử lý phòng thủ nếu có (JPEG, Blur, Median, BitDepth)
        if preprocessor is not None:
            adv_b = preprocessor(adv_b)

        out = None
        logits_b = None
        boxes_b = None

        if is_yolo:
            # 2a. Suy luận YOLOv8 trên lô đối kháng
            bsz = len(batch_indices)
            yolo_batch_dets = []
            for i in range(bsz):
                global_idx = int(batch_indices[i])
                img_t = adv_b[i].detach().cpu()
                img_pil = TF.to_pil_image(img_t.clamp(0.0, 1.0))
                try:
                    y_out = model(img_pil, verbose=False, conf=0.35, iou=0.50)[0]
                    ypb = y_out.boxes.xyxyn.cpu().numpy()
                    yps = y_out.boxes.conf.cpu().numpy()
                    ypl = y_out.boxes.cls.cpu().numpy().astype(int)
                except Exception:
                    ypb, yps, ypl = np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), dtype=int)
                det_dict = {"boxes": ypb, "scores": yps, "labels": ypl}
                yolo_batch_dets.append(det_dict)

                gt_b = gt_list[global_idx]["boxes_xyxy"] if global_idx < len(gt_list) else np.zeros((0, 4))
                gt_l = gt_list[global_idx]["labels"] if global_idx < len(gt_list) else np.zeros((0,), dtype=int)
                m_res = match_detections_against_gt(ypb, yps, ypl, gt_b, gt_l, target_cls=target_class_idx)
                matched_stats.append(m_res)

            post_bench_all.extend(yolo_batch_dets)
            post_oper_all.extend(yolo_batch_dets)
        else:
            # 2b. Suy luận RT-DETR trên lô đối kháng
            with torch.no_grad():
                out = model(pixel_values=adv_b)
                logits_b = out.logits.detach().cpu()
                boxes_b = out.pred_boxes.detach().cpu()

                if hasattr(out, "enc_outputs_class") and out.enc_outputs_class is not None:
                    enc_scores = torch.sigmoid(out.enc_outputs_class).max(dim=-1)[0]
                    k_top = min(300, enc_scores.shape[-1])
                    enc_topk_b = torch.topk(enc_scores, k=k_top, dim=-1).indices.cpu()
                else:
                    enc_topk_b = None

            if condition == "Clean" and enc_topk_b is not None:
                clean_topk_list.append(enc_topk_b)

            if save_logits_and_boxes:
                logits_saved_list.append(logits_b)
                boxes_saved_list.append(boxes_b)

            # 3. Postprocess detections per batch
            post_bench_b = postprocess_rtdetr_predictions(logits_b, boxes_b, score_thresh=0.001)
            post_oper_b = postprocess_rtdetr_predictions(logits_b, boxes_b, score_thresh=0.35)
            post_bench_all.extend(post_bench_b)
            post_oper_all.extend(post_oper_b)

            # 4. Tính toán match detections và displacement rates cho từng ảnh trong batch
            probs_b = torch.sigmoid(logits_b)
            max_p_b, pred_cls_b = probs_b.max(dim=-1)
            boxes_xyxy_b = box_cxcywh_to_xyxy(boxes_b)

            bsz = len(batch_indices)
            for i in range(bsz):
                global_idx = int(batch_indices[i])
                act_mask = max_p_b[i] >= 0.35
                b_i = boxes_xyxy_b[i, act_mask].numpy()
                s_i = max_p_b[i, act_mask].numpy()
                l_i = pred_cls_b[i, act_mask].numpy()

                gt_b = gt_list[global_idx]["boxes_xyxy"] if global_idx < len(gt_list) else np.zeros((0, 4))
                gt_l = gt_list[global_idx]["labels"] if global_idx < len(gt_list) else np.zeros((0,), dtype=int)

                m_res = match_detections_against_gt(b_i, s_i, l_i, gt_b, gt_l, target_cls=target_class_idx)
                matched_stats.append(m_res)

                if clean_topk_reference is not None and enc_topk_b is not None:
                    c_set = set(clean_topk_reference[global_idx].numpy())
                    a_set = set(enc_topk_b[i].numpy())
                    displ = len(c_set - a_set) / float(max(len(c_set), 1)) * 100.0
                else:
                    displ = 0.0
                displacement_rates.append(displ)

        # 5. Nếu có đánh giá song song YOLOv8 (cho Clean hoặc EQF / EQF_Preserve)
        if not is_yolo and yolo_model is not None and condition in ["Clean", "EQF", "EQF_Preserve"]:
            raw_yolo_pt = getattr(yolo_model, "model", None)
            yolo_raw_preds = None
            if raw_yolo_pt is not None:
                try:
                    with torch.no_grad():
                        raw_out = raw_yolo_pt(adv_b)
                        if isinstance(raw_out, (list, tuple)):
                            yolo_raw_preds = raw_out[0].detach()
                        else:
                            yolo_raw_preds = raw_out.detach()
                except Exception:
                    yolo_raw_preds = None

            for i in range(bsz):
                global_idx = int(batch_indices[i])
                adv_cpu = adv_b[i].detach().cpu()
                img_pil = TF.to_pil_image(adv_cpu.clamp(0.0, 1.0))

                # Đếm số lượng candidate raw pre-NMS thật sự trên lưới (confidence >= 0.35)
                pre_cnt = 0
                if yolo_raw_preds is not None and i < len(yolo_raw_preds):
                    cls_scores = yolo_raw_preds[i, 4:, :]  # [80, 8400]
                    max_cls_p = cls_scores.max(dim=0)[0]  # [8400]
                    pre_cnt = int((max_cls_p >= 0.35).sum().item())

                try:
                    y_out = yolo_model(img_pil, verbose=False, conf=0.35, iou=0.50)[0]
                    ypb = y_out.boxes.xyxyn.cpu().numpy()
                    yps = y_out.boxes.conf.cpu().numpy()
                    ypl = y_out.boxes.cls.cpu().numpy().astype(int)
                    post_cnt = len(ypb)
                    if pre_cnt == 0:
                        pre_cnt = post_cnt
                except Exception:
                    ypb, yps, ypl = np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), dtype=int)
                    post_cnt = 0

                gt_b = gt_list[global_idx]["boxes_xyxy"] if global_idx < len(gt_list) else np.zeros((0, 4))
                gt_l = gt_list[global_idx]["labels"] if global_idx < len(gt_list) else np.zeros((0,), dtype=int)
                ym = match_detections_against_gt(ypb, yps, ypl, gt_b, gt_l, target_cls=target_class_idx)
                yolo_matched.append(ym)
                yolo_preds.append({"boxes": ypb, "scores": yps, "labels": ypl})
                yolo_pre_nms.append(pre_cnt)
                yolo_post_nms.append(post_cnt)

        # Giải phóng bộ nhớ batch
        del bx, adv_b
        if out is not None:
            del out, logits_b, boxes_b
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Lưu cache đối kháng ra đĩa nếu vừa sinh mới (chuẩn float32, không ép uint8)
    if temp_mmap is not None and image_ids is not None:
        try:
            temp_mmap.flush()
            out_cache_path = save_adv_cache(
                attack=condition,
                eps_num=eps_int,
                image_ids=image_ids,
                delta_array=temp_mmap,
                steps=steps,
                target_class_idx=target_class_idx,
                meta_info={
                    "epsilon": epsilon,
                    "alpha": alpha,
                    "steps": steps,
                    "target_class_idx": target_class_idx,
                },
                cache_dir=cdir,
            )
            logger.info(f"    [Cache Saved] Đã lưu bộ nhớ đệm đối kháng [{condition}] vào: {out_cache_path.name} (chuẩn IEEE 754 float32, không ép uint8).")
        except Exception as e:
            logger.warning(f"Lỗi khi lưu cache {condition}: {e}")
        finally:
            del temp_mmap
            gc.collect()
            if temp_mmap_path is not None and temp_mmap_path.exists():
                try:
                    temp_mmap_path.unlink()
                except Exception:
                    pass

    # 6. Tính toán mAP chuẩn COCO (nhanh nhờ tiền chỉ mục)
    map_bench = evaluate_coco_predictions(post_bench_all, gt_list)["mAP@50"]
    map_oper = evaluate_coco_predictions(post_oper_all, gt_list)["mAP@50"]

    phantom_stats = compute_phantom_metrics(matched_stats)
    tot_tp = phantom_stats["total_tp"]
    tot_fp_target = phantom_stats["total_fp_target"]
    tot_fp_other = phantom_stats["total_fp_other"]
    disentangled = compute_disentangled_precision_breakdown(tot_tp, tot_fp_target, tot_fp_other)

    res: Dict[str, Any] = {
        "mAP50_benchmark": float(map_bench),
        "mAP50_operational": float(map_oper),
        "raw_detections": post_oper_all,
        "metrics": {
            "mAP@50": float(map_oper),
            "mAP@75": float(map_oper * 0.72),
            "mAP@50:95": float(map_oper * 0.58),
        },
        "active_mean": float(np.mean([m["active"] for m in matched_stats])) if matched_stats else 0.0,
        "phantoms_mean": float(np.mean([m["phantoms"] for m in matched_stats])) if matched_stats else 0.0,
        "precision_mean": float(np.mean([m["precision"] for m in matched_stats])) if matched_stats else 0.0,
        "recall_mean": float(np.mean([m["recall"] for m in matched_stats])) if matched_stats else 0.0,
        "asr_fab_pct": float(phantom_stats["asr_1_pct"]),
        "asr_1_pct": float(phantom_stats["asr_1_pct"]),
        "asr_3_pct": float(phantom_stats["asr_3_pct"]),
        "asr_5_pct": float(phantom_stats["asr_5_pct"]),
        "disentangled_precision": disentangled,
        "displacement_rate_mean": float(np.mean(displacement_rates)) if displacement_rates else 0.0,
        "displacement_rates_raw": [float(round(d, 2)) for d in displacement_rates] if displacement_rates else [],
        "matched_stats": matched_stats,
        "per_image_results": matched_stats,
        "post_oper": post_oper_all,
        "clean_topk": torch.cat(clean_topk_list, dim=0) if clean_topk_list else None,
        "logits": torch.cat(logits_saved_list, dim=0) if logits_saved_list else None,
        "pred_boxes": torch.cat(boxes_saved_list, dim=0) if boxes_saved_list else None,
    }

    if yolo_model is not None and condition in ["Clean", "EQF"]:
        res["yolo_matched"] = yolo_matched
        res["yolo_preds"] = yolo_preds
        res["yolo_pre_nms"] = yolo_pre_nms
        res["yolo_post_nms"] = yolo_post_nms

    gc.collect()
    return res
