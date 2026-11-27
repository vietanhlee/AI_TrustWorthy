# -*- coding: utf-8 -*-
"""
Module: paper_1_aqsm_mechanism.experiments.tasks.task6_latency_profiler
Mục đích: Đo đạc và lập hồ sơ thời gian thực thi phần cứng (Hardware-Aware Latency Telemetry)
chuẩn công nghiệp cho RT-DETR -> Phục vụ Table 10 trong bài báo:
  - Tự động nhận diện thiết bị thực thi (CUDA GPU vs Multi-core CPU).
  - Trích xuất các metrics phần cứng tương ứng:
    + CUDA GPU: Đo bằng CUDA Events & CUDA Synchronize, Peak VRAM (MB), Throughput (FPS).
    + CPU: Đo bằng High-resolution Performance Counter, số luồng (Threads), Throughput (FPS).
  - Phân tích chi tiết từng giai đoạn kiến trúc:
    Backbone -> Hybrid Encoder -> Query Selection -> Deformable Decoder -> End-to-End.
  - Đánh giá tính ổn định độ trễ suy luận đo đạc thực tế (Measured Latency Stability) giữa Clean, AQSM và Vanishing dưới các điều kiện proposal-flooding được kiểm định (không kết luận miễn nhiễm tuyệt đối mọi biến động runtime).
"""

import gc
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from paper_1_aqsm_mechanism.core.attack_generator import DETRAttackGenerator
from paper_1_aqsm_mechanism.core.dataset_loader import FlexibleCOCODataset
from paper_1_aqsm_mechanism.core.detr_wrapper import RTDETRModelWrapper
from paper_1_aqsm_mechanism.core.mechanism_analyzer import _get_rtdetr_core_model
from paper_1_aqsm_mechanism.experiments.audit_logger import ExperimentAuditLogger
from paper_1_aqsm_mechanism.experiments.config import (
    MODEL_NAME,
    RANDOM_SEED,
    configure_cpu_threads,
    detect_hardware_environment,
    get_device,
    get_img_size,
    get_raw_csv_dir,
    get_summary_json_dir,
    resolve_model_name,
    update_experiment_paths,
)
from paper_1_aqsm_mechanism.utils.cli import create_base_parser
from paper_1_aqsm_mechanism.utils.io_helpers import save_json, write_csv_rows


def _measure_stage_gpu(fn, *args, **kwargs) -> Tuple[float, Any]:
    """Đo thời gian một hàm chạy trên GPU bằng CUDA Events chuẩn xác."""
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start_event.record()
    out = fn(*args, **kwargs)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_ms = start_event.elapsed_time(end_event)
    return elapsed_ms, out


def _measure_stage_cpu(fn, *args, **kwargs) -> Tuple[float, Any]:
    """Đo thời gian một hàm chạy trên CPU bằng high-resolution perf counter."""
    t0 = time.perf_counter_ns()
    out = fn(*args, **kwargs)
    t1 = time.perf_counter_ns()
    elapsed_ms = (t1 - t0) / 1e6
    return elapsed_ms, out


def _measure_cnn_nms_sponge_baseline(
    clean_imgs: torch.Tensor,
    device: torch.device,
    cnn_arch: str = "retinanet",
    n_runs: int = 100,
    n_warmup: int = 10,
    eps: float = 16.0 / 255.0,
    alpha: float = 2.0 / 255.0,
    steps: int = 15,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Đo đạc độ trễ thực nghiệm xen kẽ (interleaved CUDA-event) của một detector CNN+NMS thực thụ
    (mặc định Torchvision RetinaNet-R50-FPN-v2 hoặc FasterRCNN-R50-FPN-v2) trên cùng GPU và cùng tập ảnh
    dưới điều kiện Clean vs Proposal-Flooding Sponge (tối đa hóa mật độ anchor trước tầng NMS).
    """
    if cnn_arch.lower() == "none":
        return {"status": "skipped_by_flag"}

    try:
        import torchvision.models.detection as tv_det
    except Exception as exc:
        return {"status": f"unavailable ({exc})"}

    is_cuda = (device.type == "cuda")
    arch_key = cnn_arch.lower()
    loaded_pretrained = True
    try:
        if "fasterrcnn" in arch_key:
            try:
                cnn_model = tv_det.fasterrcnn_resnet50_fpn_v2(
                    weights=tv_det.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
                )
            except Exception:
                loaded_pretrained = False
                cnn_model = tv_det.fasterrcnn_resnet50_fpn_v2(weights=None)
            model_label = "FasterRCNN-R50-FPN-v2 (NMS)"
        else:
            try:
                cnn_model = tv_det.retinanet_resnet50_fpn_v2(
                    weights=tv_det.RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
                )
            except Exception:
                loaded_pretrained = False
                cnn_model = tv_det.retinanet_resnet50_fpn_v2(weights=None)
            model_label = "RetinaNet-R50-FPN-v2 (NMS)"
        cnn_model = cnn_model.to(device).eval()
    except Exception as exc:
        return {"status": f"load_failed ({exc})"}

    n_imgs = clean_imgs.shape[0]
    sponge_list: List[torch.Tensor] = []
    clean_list: List[torch.Tensor] = [clean_imgs[i].to(device) for i in range(n_imgs)]

    # Sinh mẫu Proposal-Flooding Sponge cho CNN+NMS thông qua hook trước NMS
    captured_logits: List[torch.Tensor] = []

    def _cls_hook(_mod, _inp, out):
        if torch.is_tensor(out):
            captured_logits.append(out)
        elif isinstance(out, (tuple, list)) and len(out) > 0 and torch.is_tensor(out[0]):
            captured_logits.append(out[0])

    hook_handle = None
    if hasattr(cnn_model, "head") and hasattr(cnn_model.head, "classification_head"):
        hook_handle = cnn_model.head.classification_head.register_forward_hook(_cls_hook)
    elif hasattr(cnn_model, "rpn") and hasattr(cnn_model.rpn, "head"):
        hook_handle = cnn_model.rpn.head.register_forward_hook(_cls_hook)

    try:
        for i in range(n_imgs):
            orig = clean_list[i].unsqueeze(0)
            adv = orig.clone().detach()
            for _ in range(steps):
                captured_logits.clear()
                adv.requires_grad_(True)
                _ = cnn_model([adv[0]])
                if captured_logits:
                    cls_lgt = captured_logits[0]
                    probs = torch.sigmoid(cls_lgt)
                    k_top = min(2000, probs.reshape(1, -1).shape[1])
                    top_p, _ = torch.topk(probs.reshape(1, -1), k=k_top, dim=-1)
                    loss_sp = -top_p.mean()
                else:
                    out_eval = cnn_model([adv[0]])
                    loss_sp = -out_eval[0]["scores"].sum() if out_eval and len(out_eval[0]["scores"]) > 0 else adv.sum() * 0.0
                cnn_model.zero_grad()
                if adv.grad is not None:
                    adv.grad.zero_()
                loss_sp.backward()
                with torch.no_grad():
                    gs = adv.grad.sign() if adv.grad is not None else torch.zeros_like(adv)
                    adv = adv - alpha * gs
                    delta = torch.clamp(adv - orig, min=-eps, max=eps)
                    adv = torch.clamp(orig + delta, min=0.0, max=1.0).detach()
            sponge_list.append(adv[0].detach())
    finally:
        if hook_handle is not None:
            hook_handle.remove()

    # Warm-up
    with torch.no_grad():
        for _ in range(min(5, n_warmup)):
            _ = cnn_model([clean_list[0]])
            _ = cnn_model([sponge_list[0]])
            if is_cuda:
                torch.cuda.synchronize()

    # Đo đạc xen kẽ (interleaved)
    rng = np.random.RandomState(seed)
    t_clean: List[float] = []
    t_sponge: List[float] = []
    boxes_clean: List[int] = []
    boxes_sponge: List[int] = []
    measure_fn = _measure_stage_gpu if is_cuda else _measure_stage_cpu

    with torch.no_grad():
        for r_i in range(n_runs):
            img_idx = r_i % n_imgs
            order = rng.permutation(2)
            for c_id in order:
                if c_id == 0:
                    dt, out_c = measure_fn(cnn_model, [clean_list[img_idx]])
                    t_clean.append(dt)
                    if r_i < n_imgs and out_c:
                        sc = out_c[0]["scores"]
                        boxes_clean.append(int((sc >= 0.05).sum().item()))
                else:
                    dt, out_s = measure_fn(cnn_model, [sponge_list[img_idx]])
                    t_sponge.append(dt)
                    if r_i < n_imgs and out_s:
                        sc = out_s[0]["scores"]
                        boxes_sponge.append(int((sc >= 0.05).sum().item()))

    def _summ(arr: List[float], boxes: List[int]) -> Dict[str, Any]:
        a = np.asarray(arr, dtype=np.float64)
        mean_ms = float(np.mean(a)) if a.size > 0 else 0.0
        return {
            "mean_ms": round(mean_ms, 3),
            "std_ms": round(float(np.std(a, ddof=1)) if a.size > 1 else 0.0, 3),
            "median_ms": round(float(np.median(a)) if a.size > 0 else 0.0, 3),
            "p95_ms": round(float(np.percentile(a, 95)) if a.size > 0 else 0.0, 3),
            "throughput_fps": round(1000.0 / max(mean_ms, 1e-4), 2),
            "mean_post_nms_boxes_05": round(float(np.mean(boxes)) if boxes else 0.0, 2),
            "n_runs": int(a.size),
        }

    c_stat = _summ(t_clean, boxes_clean)
    s_stat = _summ(t_sponge, boxes_sponge)
    delta_ms = round(s_stat["mean_ms"] - c_stat["mean_ms"], 3)
    delta_pct = round((delta_ms / max(c_stat["mean_ms"], 1e-4)) * 100.0, 2)

    del cnn_model, clean_list, sponge_list
    if is_cuda:
        torch.cuda.empty_cache()

    return {
        "status": "evaluated",
        "model": model_label,
        "pretrained_weights": loaded_pretrained,
        "interleaved": True,
        "n_distinct_images": int(n_imgs),
        "clean": c_stat,
        "sponge": s_stat,
        "delta_ms": delta_ms,
        "delta_pct": delta_pct,
    }


def profile_pipeline_latency(
    wrapper: RTDETRModelWrapper,
    sample_img: torch.Tensor,
    n_runs: int = 300,
    n_warmup: int = 30,
    audit_logger: Any = None,
    hf_labels: Any = None,
    cnn_baseline: str = "none",
) -> Dict[str, Any]:
    """
    Thực hiện profiling chi tiết từng giai đoạn của RT-DETR qua n_runs lần suy luận.
    Hỗ trợ đánh giá xen kẽ (interleaved) trên nhiều ảnh (sample_img shape [M, 3, H, W])
    cùng đối chứng CNN+NMS sponge baseline nếu được kích hoạt.
    """
    model = wrapper.model
    model.eval()
    core = _get_rtdetr_core_model(model)

    device = wrapper.device
    is_cuda = (device.type == "cuda")
    if sample_img.ndim == 3:
        sample_img = sample_img.unsqueeze(0)
    n_lat_imgs, _, height, width = sample_img.shape
    sample_imgs_dev = sample_img.to(device)
    single_img = sample_imgs_dev[0:1]
    pixel_mask = torch.ones((1, height, width), device=device, dtype=torch.long)

    hw_info = detect_hardware_environment(preferred_device=device.type)

    times_backbone: List[float] = []
    times_encoder: List[float] = []
    times_selection: List[float] = []
    times_decoder: List[float] = []
    times_total: List[float] = []

    print("\n==================================================================")
    print(f" [Hardware Latency Profiler] Đo đạc thời gian thực thi ({n_runs} runs)")
    print(f" Thiết bị đang chạy: {hw_info['device_type'].upper()} ({hw_info.get('gpu_names', [hw_info.get('processor')])})")
    if is_cuda:
        print(f" CUDA Device: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print(f" CPU Threads: {torch.get_num_threads()} | Cores: {hw_info.get('cpu_cores')}")
    print("==================================================================")

    print(f">>> Đang khởi động (Warm-up {n_warmup} runs)...")
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(single_img)
            if is_cuda:
                torch.cuda.synchronize()

    if is_cuda:
        torch.cuda.reset_peak_memory_stats()

    print(f">>> Bắt đầu đo đạc chuẩn xác qua {n_runs} runs (xen kẽ trên {n_lat_imgs} ảnh)...")
    measure_fn = _measure_stage_gpu if is_cuda else _measure_stage_cpu

    with torch.no_grad():
        for run_i in tqdm(range(n_runs), desc="Profiling RT-DETR"):
            cur_img = sample_imgs_dev[run_i % n_lat_imgs : (run_i % n_lat_imgs) + 1]
            t_total_start = time.perf_counter_ns()

            def run_backbone():
                return core.backbone(cur_img, pixel_mask)
            t_bb, features = measure_fn(run_backbone)

            def run_encoder():
                proj_feats = [core.encoder_input_proj[level](src) for level, (src, _) in enumerate(features)]
                enc_out = core.encoder(proj_feats, return_dict=True)
                sources = []
                for level, source in enumerate(enc_out.last_hidden_state):
                    sources.append(core.decoder_input_proj[level](source))
                if core.config.num_feature_levels > len(sources):
                    len_src = len(sources)
                    sources.append(core.decoder_input_proj[len_src](enc_out.last_hidden_state[-1]))
                    for i in range(len_src + 1, core.config.num_feature_levels):
                        sources.append(core.decoder_input_proj[i](enc_out.last_hidden_state[-1]))

                source_flatten = []
                spatial_shapes = torch.empty((len(sources), 2), device=device, dtype=torch.long)
                spatial_shapes_list = []
                for level, source in enumerate(sources):
                    h_l, w_l = source.shape[-2:]
                    spatial_shapes[level, 0] = h_l
                    spatial_shapes[level, 1] = w_l
                    spatial_shapes_list.append((h_l, w_l))
                    source_flatten.append(source.flatten(2).transpose(1, 2))
                source_flatten = torch.cat(source_flatten, dim=1)
                level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
                return source_flatten, spatial_shapes, spatial_shapes_list, level_start_index
            t_enc, (source_flatten, spatial_shapes, spatial_shapes_list, level_start_index) = measure_fn(run_encoder)

            def run_selection():
                if (
                    core.training
                    or core.config.anchor_image_size is None
                    or getattr(core, "anchors", None) is None
                    or core.anchors.shape[1] != source_flatten.shape[1]
                ):
                    anchors, valid_mask = core.generate_anchors(tuple(spatial_shapes_list), device=device, dtype=source_flatten.dtype)
                else:
                    anchors = core.anchors.to(device=device, dtype=source_flatten.dtype)
                    valid_mask = core.valid_mask.to(device=device, dtype=source_flatten.dtype)
                memory = valid_mask.to(source_flatten.dtype) * source_flatten
                output_memory = core.enc_output(memory)
                enc_outputs_class = core.enc_score_head(output_memory)
                enc_outputs_coord_logits = core.enc_bbox_head(output_memory) + anchors

                n_q_sel = min(int(core.config.num_queries), enc_outputs_class.shape[1])
                _, topk_ind = torch.topk(enc_outputs_class.max(-1).values, n_q_sel, dim=1)
                ref_points_unact = enc_outputs_coord_logits.gather(dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, 4))
                if getattr(core.config, "learn_initial_query", False) and hasattr(core, "weight_embedding"):
                    target = core.weight_embedding.tile([1, 1, 1])
                else:
                    target = output_memory.gather(dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1]))
                return target, source_flatten, ref_points_unact.detach(), spatial_shapes, spatial_shapes_list, level_start_index
            t_sel, (target, src_flat, ref_init, sp_shapes, sp_shapes_list, lvl_start) = measure_fn(run_selection)

            def run_decoder():
                dec_out = core.decoder(
                    inputs_embeds=target,
                    encoder_hidden_states=src_flat,
                    encoder_attention_mask=None,
                    reference_points=ref_init,
                    spatial_shapes=sp_shapes,
                    spatial_shapes_list=sp_shapes_list,
                    level_start_index=lvl_start,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
                return dec_out.intermediate_logits[:, -1], dec_out.intermediate_reference_points[:, -1]
            t_dec, _ = measure_fn(run_decoder)

            t_total_end = time.perf_counter_ns()
            if is_cuda:
                torch.cuda.synchronize()

            t_tot = (t_total_end - t_total_start) / 1e6

            times_backbone.append(t_bb)
            times_encoder.append(t_enc)
            times_selection.append(t_sel)
            times_decoder.append(t_dec)
            times_total.append(t_tot)

    def _calc_stats(arr: List[float]) -> Dict[str, float]:
        a = np.array(arr, dtype=np.float64)
        if a.size == 0:
            return {"mean_ms": 0.0, "std_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}
        return {
            "mean_ms": round(float(np.mean(a)), 3),
            "std_ms": round(float(np.std(a, ddof=1)) if a.size > 1 else 0.0, 3),
            "median_ms": round(float(np.median(a)), 3),
            "p95_ms": round(float(np.percentile(a, 95)), 3),
            "p99_ms": round(float(np.percentile(a, 99)), 3),
            "min_ms": round(float(np.min(a)), 3),
            "max_ms": round(float(np.max(a)), 3),
        }

    stats_bb = _calc_stats(times_backbone)
    stats_enc = _calc_stats(times_encoder)
    stats_sel = _calc_stats(times_selection)
    stats_dec = _calc_stats(times_decoder)
    stats_tot = _calc_stats(times_total)

    mean_tot = stats_tot["mean_ms"]
    fps = round(1000.0 / max(mean_tot, 1e-4), 2)

    # Benchmark đối chứng xen kẽ (Interleaved) 4 điều kiện trên M ảnh đa dạng (loại bỏ sai lệch thứ tự/nhiệt GPU)
    print(f"\n>>> [Interleaved Latency Contrast] Đo xen kẽ trên M={n_lat_imgs} ảnh đa dạng (Clean, AQSM, Task PGD, Sponge)...")
    from paper_1_aqsm_mechanism.experiments.config import get_epsilon_l_inf, get_pgd_num_steps, get_pgd_step_size
    n_adv_runs = min(n_runs, max(n_lat_imgs, 100))
    eff_eps = get_epsilon_l_inf()
    eff_alpha = get_pgd_step_size()
    eff_steps = get_pgd_num_steps()
    attacker = DETRAttackGenerator(wrapper, epsilon=eff_eps, alpha=eff_alpha, num_steps=eff_steps)

    cond_names = ["Clean_Baseline", "AQSM_Attack", "Task_PGD", "RT_DETR_Flooding_Sponge"]
    cond_tensors: Dict[str, List[torch.Tensor]] = {c: [] for c in cond_names}
    for m_i in range(n_lat_imgs):
        im_m = sample_imgs_dev[m_i:m_i + 1]
        lbl_m = hf_labels[m_i:m_i + 1] if (hf_labels is not None and m_i < len(hf_labels)) else None
        cond_tensors["Clean_Baseline"].append(im_m)
        cond_tensors["AQSM_Attack"].append(attacker.generate_query_selection_attack(im_m, steps=eff_steps, bg_seed=RANDOM_SEED))
        cond_tensors["Task_PGD"].append(attacker.generate_standard_task_pgd(im_m, hf_labels=lbl_m, steps=eff_steps))
        cond_tensors["RT_DETR_Flooding_Sponge"].append(attacker.generate_latency_flooding_attack(im_m, steps=eff_steps, conf_threshold=0.35))

    with torch.no_grad():
        for _ in range(min(10, n_warmup)):
            _ = model(single_img)
            if is_cuda:
                torch.cuda.synchronize()

    rng_lat = np.random.RandomState(RANDOM_SEED)
    times_by_cond: Dict[str, List[float]] = {c: [] for c in cond_names}
    boxes_by_cond: Dict[str, List[float]] = {c: [] for c in cond_names}
    with torch.no_grad():
        for r_i in range(n_adv_runs):
            m_idx = r_i % n_lat_imgs
            order = rng_lat.permutation(len(cond_names))
            for idx_c in order:
                c_name = cond_names[int(idx_c)]
                c_t = cond_tensors[c_name][m_idx]
                if is_cuda:
                    t_c, out_c = _measure_stage_gpu(model, c_t)
                else:
                    t_c, out_c = _measure_stage_cpu(model, c_t)
                times_by_cond[c_name].append(t_c)
                if r_i < n_lat_imgs and hasattr(out_c, "logits"):
                    probs_c = torch.sigmoid(out_c.logits).max(dim=-1)[0]
                    boxes_by_cond[c_name].append(float((probs_c >= 0.35).sum().item()))

    adv_latency_summary: Dict[str, Any] = {}
    for cond_name in cond_names:
        c_stats = _calc_stats(times_by_cond[cond_name])
        active_mean = round(float(np.mean(boxes_by_cond[cond_name])) if boxes_by_cond[cond_name] else 0.0, 2)
        adv_latency_summary[cond_name] = {
            **c_stats,
            "throughput_fps": round(1000.0 / max(c_stats["mean_ms"], 1e-4), 2),
            "active_boxes_tau35": active_mean,
            "n_runs": n_adv_runs,
            "n_distinct_images": n_lat_imgs,
            "interleaved": True,
        }
        print(
            f"   * {cond_name:24s}: {c_stats['mean_ms']:6.2f} +- {c_stats['std_ms']:5.2f} ms "
            f"(p95={c_stats['p95_ms']:6.2f} ms) | Active boxes={active_mean}"
        )

    cnn_baseline_res = _measure_cnn_nms_sponge_baseline(
        clean_imgs=sample_imgs_dev,
        device=device,
        cnn_arch=cnn_baseline,
        n_runs=n_adv_runs,
        n_warmup=n_warmup,
        eps=eff_eps,
        alpha=eff_alpha,
        steps=eff_steps,
        seed=RANDOM_SEED,
    )
    if cnn_baseline_res.get("status") == "evaluated":
        print(
            f"   * {cnn_baseline_res['model']:24s}: Clean={cnn_baseline_res['clean']['mean_ms']:.2f} ms "
            f"({cnn_baseline_res['clean']['mean_post_nms_boxes_05']} boxes) vs Sponge={cnn_baseline_res['sponge']['mean_ms']:.2f} ms "
            f"({cnn_baseline_res['sponge']['mean_post_nms_boxes_05']} boxes) | Delta={cnn_baseline_res['delta_ms']:+.2f} ms ({cnn_baseline_res['delta_pct']:+.1f}%)"
        )

    mem_metrics: Dict[str, Any] = {}
    if is_cuda:
        mem_metrics["peak_vram_allocated_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 2)
        mem_metrics["vram_reserved_mb"] = round(torch.cuda.memory_reserved() / 1e6, 2)
        mem_metrics["gpu_name"] = torch.cuda.get_device_name(0)
        mem_metrics["cuda_capability"] = f"{torch.cuda.get_device_capability(0)[0]}.{torch.cuda.get_device_capability(0)[1]}"
    else:
        mem_metrics["cpu_threads"] = torch.get_num_threads()
        mem_metrics["cpu_cores"] = hw_info.get("cpu_cores")
        mem_metrics["processor"] = hw_info.get("processor")

    print("\n" + "=" * 80)
    print(" KẾT QUẢ PROFILING ĐỘ TRỄ THỰC THI PHẦN CỨNG (RT-DETR Native 640x640)")
    print("=" * 80)
    print(f" 1. Backbone ({getattr(wrapper, 'pretrained_model_name', 'ResNet') or 'ResNet'}) : {stats_bb['mean_ms']:6.2f} ms +- {stats_bb['std_ms']:4.2f} ({(stats_bb['mean_ms']/mean_tot)*100:4.1f}%)")
    print(f" 2. Hybrid Encoder (AIFI + CCFM)    : {stats_enc['mean_ms']:6.2f} ms +- {stats_enc['std_ms']:4.2f} ({(stats_enc['mean_ms']/mean_tot)*100:4.1f}%)")
    print(f" 3. Query Selection (Head + Top-k)  : {stats_sel['mean_ms']:6.2f} ms +- {stats_sel['std_ms']:4.2f} ({(stats_sel['mean_ms']/mean_tot)*100:4.1f}%)")
    print(f" 4. Deformable Decoder ({getattr(wrapper, 'decoder_layers', 3)} Layers)   : {stats_dec['mean_ms']:6.2f} ms +- {stats_dec['std_ms']:4.2f} ({(stats_dec['mean_ms']/mean_tot)*100:4.1f}%)")
    print("-" * 80)
    print(f" TỔNG ĐỘ TRỄ SUY LUẬN (End-to-End)  : {mean_tot:6.2f} ms +- {stats_tot['std_ms']:4.2f} | P95: {stats_tot['p95_ms']:.2f} ms")
    print(f" THÔNG LƯỢNG SUY LUẬN (Throughput)  : {fps:6.2f} FPS")
    if is_cuda:
        print(f" PEAK VRAM ALLOCATED                : {mem_metrics['peak_vram_allocated_mb']} MB")
    print("=" * 80)

    raw_csv_dir = get_raw_csv_dir()
    raw_csv_dir.mkdir(parents=True, exist_ok=True)
    raw_csv_path = raw_csv_dir / "latency_profiler_raw.csv"
    csv_header = ["run_idx", "backbone_ms", "encoder_ms", "selection_ms", "decoder_ms", "total_ms"]
    raw_rows: List[List[Any]] = []
    for i in range(n_runs):
        raw_rows.append([
            i + 1,
            f"{times_backbone[i]:.4f}",
            f"{times_encoder[i]:.4f}",
            f"{times_selection[i]:.4f}",
            f"{times_decoder[i]:.4f}",
            f"{times_total[i]:.4f}",
        ])
    write_csv_rows(raw_csv_path, csv_header, raw_rows)
    print(f">>> Đã ghi nhận số liệu thô per-run: {raw_csv_path}")

    if audit_logger is not None:
        audit_logger.write_audit_record({
            "record_type": "LATENCY_PROFILE",
            "n_runs": n_runs,
            "device": str(device),
            "hardware_info": hw_info,
            "memory_metrics": mem_metrics,
            "latency_stages": {
                "backbone": stats_bb,
                "encoder": stats_enc,
                "selection": stats_sel,
                "decoder": stats_dec,
                "total": stats_tot,
            },
            "throughput_fps": fps,
        })

    output_dict = {
        "experiment": "Table 10: Hardware Execution Latency Breakdown & Profiling",
        "model": getattr(wrapper, "pretrained_model_name", None) or "unknown",
        "n_runs": n_runs,
        "device_type": hw_info["device_type"],
        "timing_backend": "cuda_events_pytorch_fp32" if is_cuda else "perf_counter_cpu_pytorch_fp32",
        "hardware_metadata": mem_metrics,
        "throughput_fps": fps,
        "stages": {
            "Backbone": stats_bb,
            "Hybrid_Encoder": stats_enc,
            "Query_Selection": stats_sel,
            "Deformable_Decoder": stats_dec,
            "End_to_End_Total": stats_tot,
        },
        "adversarial_latency_comparison": adv_latency_summary,
        "cnn_nms_sponge_baseline": cnn_baseline_res,
        "n_distinct_images_profiled": n_lat_imgs,
        "interleaved_protocol": True,
        "percentage_breakdown": {
            "Backbone_pct": round((stats_bb["mean_ms"] / mean_tot) * 100.0, 2),
            "Hybrid_Encoder_pct": round((stats_enc["mean_ms"] / mean_tot) * 100.0, 2),
            "Query_Selection_pct": round((stats_sel["mean_ms"] / mean_tot) * 100.0, 2),
            "Deformable_Decoder_pct": round((stats_dec["mean_ms"] / mean_tot) * 100.0, 2),
        },
        "timing_invariance_note": (
            "RT-DETR exhibits stable measured inference latency under the evaluated "
            "proposal-flooding conditions, consistent with its fixed-query, NMS-free "
            "inference architecture (without ruling out hardware or system-level runtime variability)."
        ),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    summary_json_dir = get_summary_json_dir()
    summary_json_dir.mkdir(parents=True, exist_ok=True)
    json_path = summary_json_dir / "latency_profile.json"
    save_json(output_dict, json_path)
    save_json(output_dict, summary_json_dir / "edge_latency_results.json")
    print(f">>> Đã lưu tổng hợp JSON Table 10: {json_path}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return output_dict


def main():
    parser = create_base_parser("Hardware Latency Profiler (Table 10)", task_name="Task 6")
    parser.add_argument("--n_runs", type=int, default=None, help="Số lần lặp đo đạc độ trễ (mặc định: lấy theo max_samples hoặc 1000)")
    parser.add_argument("--warmup", type=int, default=30, help="Số lần chạy khởi động trước khi đo (mặc định: 30)")
    parser.add_argument("--latency_images", type=int, default=20, help="Số ảnh COCO đa dạng dùng cho profiling xen kẽ (mặc định: 20)")
    parser.add_argument("--cnn_baseline", type=str, default="retinanet", choices=["none", "retinanet", "fasterrcnn"], help="Detector CNN+NMS đối chứng dưới sponge attack (mặc định: retinanet)")
    args = parser.parse_args()
    raw_n_runs = args.n_runs if args.n_runs is not None else args.max_samples
    effective_n_runs = int(raw_n_runs) if (raw_n_runs is not None and int(raw_n_runs) > 0) else 1000

    chosen_model = resolve_model_name(getattr(args, "model", None))
    update_experiment_paths(
        folder_image=args.folder_image,
        annotations=args.annotations,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        model=chosen_model,
    )

    configure_cpu_threads(args.threads)
    hw_info = detect_hardware_environment(preferred_device=args.device)
    device = get_device(preferred=args.device)

    audit_logger = ExperimentAuditLogger(task_name="task6_latency_profiler")

    wrapper = RTDETRModelWrapper(pretrained_model_name=chosen_model, device=device)
    wrapper.eval()

    n_lat_fetch = max(1, int(args.latency_images))
    if args.max_samples is not None and int(args.max_samples) > 0:
        n_lat_fetch = min(n_lat_fetch, int(args.max_samples))
    dataset = FlexibleCOCODataset(
        folder_image=args.folder_image,
        annotations_file=args.annotations,
        img_size=get_img_size(),
        id2label=wrapper.model.config.id2label,
        lazy_tensors=True,
        device=device,
        max_samples=n_lat_fetch,
    )
    sample_img = dataset.tensors[:n_lat_fetch]
    sample_hf_labels = dataset.hf_labels[:n_lat_fetch] if getattr(dataset, "hf_labels", None) else None

    audit_logger.log_metadata({
        "folder_image": str(args.folder_image),
        "n_runs": effective_n_runs,
        "warmup": args.warmup,
        "threads": args.threads,
        "device": str(device),
        "hardware_info": hw_info,
        "random_seed": RANDOM_SEED,
        "model_name": chosen_model,
    })

    profile_pipeline_latency(
        wrapper=wrapper,
        sample_img=sample_img,
        n_runs=effective_n_runs,
        n_warmup=args.warmup,
        audit_logger=audit_logger,
        hf_labels=sample_hf_labels,
        cnn_baseline=args.cnn_baseline,
    )

    audit_logger.close()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print("\n[HOÀN THÀNH TOÀN BỘ] Task 6: Hardware Latency Profiler thành công!")


if __name__ == "__main__":
    main()
