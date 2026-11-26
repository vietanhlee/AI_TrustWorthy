# -*- coding: utf-8 -*-
"""
Module: paper_1_aqsm_mechanism.core.mechanism_analyzer
Mục đích: Bộ trích xuất và phân tích cơ chế nội tại của RT-DETR phục vụ giải trình khoa học:
1. Causal Mechanism Surgery (Encoder Memory M vs Query Prior Q) chuẩn hóa khớp 100% HuggingFace `RTDetrModel`.
2. Phân tích Encoder Memory (Cosine Similarity, Relative L2 Perturbation theo scale S3/S4/S5 và vùng Vật thể/Nền).
3. Phân tích Query Selection (Top-K Overlap, Spearman/Kendall Rank Correlation, Object Coverage Rate, Eviction).
4. Phân tích Deformable Attention (Hook trực tiếp `sampling_locations`, `reference_points`, `attention_weights`,
   Entropy phân tán chú ý, bán kính phân tán và Foreground Spatial Hit Rate).
"""

from contextlib import contextmanager
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats
import torch
import torch.nn.functional as F


# Phân đoạn token theo feature scale tại input 640x640:
# S3: 80x80 = 6400 tokens [0:6400]
# S4: 40x40 = 1600 tokens [6400:8000]
# S5: 20x20 = 400 tokens [8000:8400]
SCALE_NAMES: Tuple[str, ...] = ("S3", "S4", "S5")


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _get_rtdetr_core_model(model: torch.nn.Module) -> torch.nn.Module:
    """Truy xuất chính xác instance `RTDetrModel` từ wrapper hoặc `RTDetrForObjectDetection`."""
    cur = model
    for _ in range(3):
        if hasattr(cur, "backbone") and hasattr(cur, "decoder"):
            return cur
        if hasattr(cur, "model"):
            cur = cur.model
        else:
            break
    raise AttributeError(f"Không tìm thấy RTDetrModel hợp lệ trong {type(model)}")


def extract_encoder_components(
    model: torch.nn.Module,
    pixel_values: torch.Tensor,
    enable_grad: bool = False,
) -> Dict[str, Any]:
    """
    Chạy phần Backbone + Hybrid Encoder + Encoder Proposal Head của `RTDetrForObjectDetection`,
    trích xuất chính xác từng tensor làm đầu vào cho Deformable Decoder.

    Args:
        model: Instance `RTDetrForObjectDetection` (hoặc wrapper chứa `.model`).
        pixel_values: Tensor ảnh [B, 3, H, W] trong [0, 1].
        enable_grad: Nếu True, giữ nguyên đồ thị gradient phục vụ adaptive white-box attacks.
    """
    core = _get_rtdetr_core_model(model)

    device = pixel_values.device
    bsz, _, height, width = pixel_values.shape
    pixel_mask = torch.ones((bsz, height, width), device=device, dtype=torch.long)

    grad_ctx = torch.enable_grad() if enable_grad else torch.no_grad()
    _out_t = (lambda t: t) if enable_grad else (lambda t: t.detach())

    with grad_ctx:
        features = core.backbone(pixel_values, pixel_mask)
        proj_feats = [core.encoder_input_proj[level](src) for level, (src, _) in enumerate(features)]
        encoder_outputs = core.encoder(proj_feats, return_dict=True)

        sources = []
        for level, source in enumerate(encoder_outputs.last_hidden_state):
            sources.append(core.decoder_input_proj[level](source))
        if core.config.num_feature_levels > len(sources):
            len_src = len(sources)
            sources.append(core.decoder_input_proj[len_src](encoder_outputs.last_hidden_state[-1]))
            for i in range(len_src + 1, core.config.num_feature_levels):
                sources.append(core.decoder_input_proj[i](encoder_outputs.last_hidden_state[-1]))

        source_flatten = []
        spatial_shapes_list: List[Tuple[int, int]] = []
        spatial_shapes = torch.empty((len(sources), 2), device=device, dtype=torch.long)
        for level, source in enumerate(sources):
            h_l, w_l = source.shape[-2:]
            spatial_shapes[level, 0] = h_l
            spatial_shapes[level, 1] = w_l
            spatial_shapes_list.append((h_l, w_l))
            source_flatten.append(source.flatten(2).transpose(1, 2))
        source_flatten_t = torch.cat(source_flatten, dim=1)
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )

        if (
            core.training
            or core.config.anchor_image_size is None
            or getattr(core, "anchors", None) is None
            or core.anchors.shape[1] != source_flatten_t.shape[1]
        ):
            anchors, valid_mask = core.generate_anchors(
                tuple(spatial_shapes_list),
                device=device,
                dtype=source_flatten_t.dtype,
            )
        else:
            anchors = core.anchors.to(device=device, dtype=source_flatten_t.dtype)
            valid_mask = core.valid_mask.to(device=device, dtype=source_flatten_t.dtype)

        memory = valid_mask.to(source_flatten_t.dtype) * source_flatten_t
        output_memory = core.enc_output(memory)
        enc_outputs_class = core.enc_score_head(output_memory)
        enc_outputs_coord_logits = core.enc_bbox_head(output_memory) + anchors

        n_q_select = min(int(core.config.num_queries), enc_outputs_class.shape[1])
        _, topk_ind = torch.topk(
            enc_outputs_class.max(-1).values,
            n_q_select,
            dim=1,
        )
        reference_points_unact = enc_outputs_coord_logits.gather(
            dim=1,
            index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_coord_logits.shape[-1]),
        )
        if core.config.learn_initial_query:
            target = core.weight_embedding.tile([bsz, 1, 1])
        else:
            target = output_memory.gather(
                dim=1,
                index=topk_ind.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1]),
            )

        init_reference_points = _out_t(reference_points_unact)

    return {
        "encoder_last_hidden_state": [_out_t(t) for t in encoder_outputs.last_hidden_state],
        # Multi-scale spatial tokens S: S = 8,400 at native 640x640 (P3: 6400, P4: 1600, P5: 400); S = 2,100 at 320x320
        "source_flatten": _out_t(source_flatten_t),            # M: [B, S, 256]
        "output_memory": _out_t(output_memory),                # [B, S, 256]
        "enc_outputs_class": _out_t(enc_outputs_class),        # [B, S, 80]
        "enc_outputs_coord_logits": _out_t(enc_outputs_coord_logits),  # [B, S, 4]
        "anchors": _out_t(anchors),                            # [1, S, 4]
        "topk_ind": topk_ind.detach(),                          # I: [B, 300]
        "target": _out_t(target),                              # Q_feat: [B, 300, 256]
        "init_reference_points": init_reference_points,         # Q_ref: [B, 300, 4]
        "spatial_shapes": spatial_shapes,
        "spatial_shapes_list": spatial_shapes_list,
        "level_start_index": level_start_index,
    }


def run_decoder_from_components(
    model: torch.nn.Module,
    target: torch.Tensor,
    source_flatten: torch.Tensor,
    init_reference_points: torch.Tensor,
    spatial_shapes: torch.Tensor,
    spatial_shapes_list: List[Tuple[int, int]],
    level_start_index: torch.Tensor,
    output_attentions: bool = False,
    enable_grad: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Chạy riêng Deformable Decoder từ các thành phần (M, target, init_reference_points) đã tách.
    """
    core = _get_rtdetr_core_model(model)

    grad_ctx = torch.enable_grad() if enable_grad else torch.no_grad()
    with grad_ctx:
        dec_out = core.decoder(
            inputs_embeds=target,
            encoder_hidden_states=source_flatten,
            encoder_attention_mask=None,
            reference_points=init_reference_points,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            level_start_index=level_start_index,
            output_attentions=output_attentions,
            output_hidden_states=False,
            return_dict=True,
        )
        logits = dec_out.intermediate_logits[:, -1]
        pred_boxes = dec_out.intermediate_reference_points[:, -1]

    return {
        "logits": logits,
        "pred_boxes": pred_boxes,
        "intermediate_reference_points": dec_out.intermediate_reference_points,
        "initial_reference_points": dec_out.initial_reference_points,
        "cross_attentions": dec_out.cross_attentions if output_attentions else None,
    }


def compute_scale_slices(spatial_shapes_list: Sequence[Tuple[int, int]]) -> Dict[str, slice]:
    """Tạo slice index cho các scale S3, S4, S5 trong vector flattened [S] (S=8,400 ở 640x640; S=2,100 ở 320x320)."""
    slices: Dict[str, slice] = {}
    cursor = 0
    for idx, (h_l, w_l) in enumerate(spatial_shapes_list):
        name = SCALE_NAMES[idx] if idx < len(SCALE_NAMES) else f"S{idx + 3}"
        n_tok = int(h_l * w_l)
        slices[name] = slice(cursor, cursor + n_tok)
        cursor += n_tok
    return slices


def compute_token_fg_masks(
    anchors_sigmoid: np.ndarray,
    gt_boxes_xyxy: np.ndarray,
) -> np.ndarray:
    """
    Xác định mask [N_tokens] (S tokens: S=8,400 ở 640x640; 2,100 ở 320x320) xem tâm grid anchor `(cx, cy)` có rơi vào
    bất kỳ Ground Truth bbox `[x1, y1, x2, y2]` nào của ảnh hay không.
    """
    n_tok = anchors_sigmoid.shape[0]
    if len(gt_boxes_xyxy) == 0:
        return np.zeros(n_tok, dtype=bool)
    cx = anchors_sigmoid[:, 0 : 1]  # [S, 1]
    cy = anchors_sigmoid[:, 1 : 2]  # [S, 1]
    in_x = (cx >= gt_boxes_xyxy[None, :, 0]) & (cx <= gt_boxes_xyxy[None, :, 2])
    in_y = (cy >= gt_boxes_xyxy[None, :, 1]) & (cy <= gt_boxes_xyxy[None, :, 3])
    return np.any(in_x & in_y, axis=1)


def analyze_encoder_memory_batch(
    clean_comp: Dict[str, Any],
    adv_comp: Dict[str, Any],
    gt_boxes_batch: Sequence[np.ndarray],
) -> List[Dict[str, Any]]:
    """
    Mục 3: Phân tích độ biến đổi Encoder Memory giữa Clean và Attacked theo:
    - Toàn bộ S tokens (`source_flatten` trước cross-attention: S=8,400 ở native 640x640) và từng scale (S3, S4, S5).
    - Phân tầng vùng Vật thể (Foreground tokens có tâm nằm trong GT box) vs vùng Nền (Background).
    """
    m_clean = clean_comp["source_flatten"].detach().cpu().numpy()  # [B, S, 256]
    m_adv = adv_comp["source_flatten"].detach().cpu().numpy()
    enc_h_clean = [t.detach().cpu().numpy() for t in clean_comp["encoder_last_hidden_state"]]
    enc_h_adv = [t.detach().cpu().numpy() for t in adv_comp["encoder_last_hidden_state"]]

    anchors_sig = torch.sigmoid(clean_comp["anchors"][0]).detach().cpu().numpy()  # [S, 4]
    scale_slices = compute_scale_slices(clean_comp["spatial_shapes_list"])

    bsz = m_clean.shape[0]
    records: List[Dict[str, Any]] = []

    def _pair_metrics(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, float]:
        # a, b shape [M_tok, D]
        if len(a) == 0:
            return float("nan"), float("nan"), float("nan")
        diff = b - a
        norm_a = np.linalg.norm(a, axis=-1)
        norm_b = np.linalg.norm(b, axis=-1)
        cos = np.sum(a * b, axis=-1) / np.maximum(norm_a * norm_b, 1e-8)
        rel_tok = np.linalg.norm(diff, axis=-1) / np.maximum(norm_a, 1e-8)
        rel_frob = float(np.linalg.norm(diff) / max(float(np.linalg.norm(a)), 1e-8))
        return float(np.mean(cos)), float(np.mean(rel_tok)), rel_frob

    for b in range(bsz):
        gt_b = gt_boxes_batch[b]
        fg_mask = compute_token_fg_masks(anchors_sig, gt_b)
        bg_mask = ~fg_mask

        cos_all, rel_all, frob_all = _pair_metrics(m_clean[b], m_adv[b])
        cos_fg, rel_fg, frob_fg = _pair_metrics(m_clean[b][fg_mask], m_adv[b][fg_mask])
        cos_bg, rel_bg, frob_bg = _pair_metrics(m_clean[b][bg_mask], m_adv[b][bg_mask])

        per_scale: Dict[str, Dict[str, float]] = {}
        for s_idx, (s_name, sl) in enumerate(scale_slices.items()):
            mc_s = m_clean[b][sl]
            ma_s = m_adv[b][sl]
            c_s, r_s, f_s = _pair_metrics(mc_s, ma_s)

            fg_s = fg_mask[sl]
            bg_s = ~fg_s
            c_s_fg, r_s_fg, _ = _pair_metrics(mc_s[fg_s], ma_s[fg_s])
            c_s_bg, r_s_bg, _ = _pair_metrics(mc_s[bg_s], ma_s[bg_s])

            # Thêm chỉ số trên feature map ngay trước decoder_input_proj (`encoder_last_hidden_state`)
            if s_idx < len(enc_h_clean):
                hc = enc_h_clean[s_idx][b].reshape(enc_h_clean[s_idx].shape[1], -1).T
                ha = enc_h_adv[s_idx][b].reshape(enc_h_adv[s_idx].shape[1], -1).T
                c_enc, r_enc, _ = _pair_metrics(hc, ha)
            else:
                c_enc, r_enc = float("nan"), float("nan")

            per_scale[s_name] = {
                "cosine": c_s,
                "rel_l2": r_s,
                "rel_frob": f_s,
                "fg_cosine": c_s_fg,
                "fg_rel_l2": r_s_fg,
                "bg_cosine": c_s_bg,
                "bg_rel_l2": r_s_bg,
                "pre_proj_cosine": c_enc,
                "pre_proj_rel_l2": r_enc,
            }

        records.append({
            "n_fg_tokens": int(np.sum(fg_mask)),
            "n_bg_tokens": int(np.sum(bg_mask)),
            "all_cosine": cos_all,
            "all_rel_l2": rel_all,
            "all_rel_frob": frob_all,
            "fg_cosine": cos_fg,
            "fg_rel_l2": rel_fg,
            "fg_rel_frob": frob_fg,
            "bg_cosine": cos_bg,
            "bg_rel_l2": rel_bg,
            "bg_rel_frob": frob_bg,
            "per_scale": per_scale,
        })

    return records


def _box_iou_np(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float64)
    a1 = np.maximum(0.0, boxes1[:, 2] - boxes1[:, 0]) * np.maximum(0.0, boxes1[:, 3] - boxes1[:, 1])
    a2 = np.maximum(0.0, boxes2[:, 2] - boxes2[:, 0]) * np.maximum(0.0, boxes2[:, 3] - boxes2[:, 1])
    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = np.clip(rb - lt, a_min=0.0, a_max=None)
    inter = wh[:, :, 0] * wh[:, :, 1]
    return inter / np.maximum(a1[:, None] + a2[None, :] - inter, 1e-12)


def _cxcywh_to_xyxy_np(boxes: np.ndarray) -> np.ndarray:
    cx, cy, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    return np.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], axis=-1)


def compute_spatial_clustering_metrics(boxes_or_centers: np.ndarray) -> Dict[str, float]:
    """
    Tính các chỉ số gom cụm không gian (Spatial Clustering) và phân tán (Spatial Dispersion):
    1. Average Nearest Neighbor Distance (ANND):
       ANND = (1 / N) * sum_{i=1}^N min_{j != i} ||c_i - c_j||_2
       Khi các anchor bị gom cụm cục bộ vào các điểm mù / nền, ANND sụt giảm rõ rệt.
    2. Spatial Variance & Standard Deviation (sigma_x, sigma_y, total_variance):
       Đo lường mức độ bao phủ toàn diện của các anchors trên toàn bộ canvas [0, 1]^2.
    """
    pts = np.asarray(boxes_or_centers, dtype=np.float64)
    if pts.ndim == 2 and pts.shape[1] >= 4:
        # Nếu truyền vào [N, 4], lấy tâm [cx, cy]
        centers = pts[:, :2]
    elif pts.ndim == 2 and pts.shape[1] == 2:
        centers = pts
    else:
        return {"annd": float("nan"), "spatial_variance": float("nan"), "spatial_std_x": float("nan"), "spatial_std_y": float("nan")}

    N = len(centers)
    if N < 2:
        return {"annd": float("nan"), "spatial_variance": 0.0, "spatial_std_x": 0.0, "spatial_std_y": 0.0}

    # Ma trận khoảng cách Euclid N x N
    diff = centers[:, None, :] - centers[None, :, :]  # [N, N, 2]
    dist_mat = np.sqrt(np.sum(diff ** 2, axis=-1))    # [N, N]
    np.fill_diagonal(dist_mat, np.inf)

    # Nearest neighbor distance cho từng anchor
    min_nnd = np.min(dist_mat, axis=1)  # [N]
    annd = float(np.mean(min_nnd))

    # Spatial variance quanh trọng tâm
    mean_center = np.mean(centers, axis=0, keepdims=True)
    sq_dist_to_mean = np.sum((centers - mean_center) ** 2, axis=-1)
    spatial_variance = float(np.mean(sq_dist_to_mean))
    ddof_c = 1 if centers.shape[0] > 1 else 0
    std_x = float(np.std(centers[:, 0], ddof=ddof_c))
    std_y = float(np.std(centers[:, 1], ddof=ddof_c))

    return {
        "annd": round(annd, 6),
        "spatial_variance": round(spatial_variance, 6),
        "spatial_std_x": round(std_x, 6),
        "spatial_std_y": round(std_y, 6),
    }


def compute_directional_misrouting_metrics(
    c_ref_centers: np.ndarray,
    target_centers: np.ndarray,
) -> Dict[str, float]:
    """
    Đo tính định hướng (directional spatial bias) của phép dịch chuyển tọa độ anchor
    giữa Clean [Q, 2] và trạng thái đích [Q, 2] trong không gian [0, 1]^2:
      - mean_shift_vector_norm: chuẩn vector tịnh tiến trung bình ||mean(delta_c)||_2
        (xấp xỉ 0 nếu đẳng hướng, dương nếu đẩy có hướng hệ thống).
      - directional_resultant_length: độ dài tổng hợp đơn vị R = ||mean(u_i)||_2 in [0, 1]
        (Mean Resultant Length chuẩn thống kê hướng).
      - radial_outward_bias: trung bình tích vô hướng của vector dịch chuyển theo tia
        hướng từ tâm ảnh (0.5, 0.5) ra rìa (dương = đẩy ra rìa, âm = co về tâm).
    """
    if len(c_ref_centers) == 0 or len(target_centers) == 0:
        return {
            "mean_shift_vector_norm": 0.0,
            "directional_resultant_length": 0.0,
            "radial_outward_bias": 0.0,
        }

    c0 = np.asarray(c_ref_centers[:, :2], dtype=np.float64)
    c1 = np.asarray(target_centers[:, :2], dtype=np.float64)
    deltas = c1 - c0
    mean_shift_vec = np.mean(deltas, axis=0)
    mean_shift_norm = float(np.linalg.norm(mean_shift_vec))

    norms = np.linalg.norm(deltas, axis=1)
    nz_mask = norms > 1e-8
    if np.any(nz_mask):
        unit_vecs = deltas[nz_mask] / norms[nz_mask, None]
        resultant_length = float(np.linalg.norm(np.mean(unit_vecs, axis=0)))
    else:
        resultant_length = 0.0

    # Xu hướng đẩy ra biên so với co về tâm khung ảnh (0.5, 0.5)
    radial_vecs = c0 - np.array([[0.5, 0.5]], dtype=np.float64)
    radial_norms = np.linalg.norm(radial_vecs, axis=1)
    r_mask = radial_norms > 1e-8
    if np.any(r_mask):
        radial_unit = radial_vecs[r_mask] / radial_norms[r_mask, None]
        radial_proj = np.sum(deltas[r_mask] * radial_unit, axis=1)
        radial_bias = float(np.mean(radial_proj))
    else:
        radial_bias = 0.0

    return {
        "mean_shift_vector_norm": round(mean_shift_norm, 6),
        "directional_resultant_length": round(resultant_length, 6),
        "radial_outward_bias": round(radial_bias, 6),
    }


def analyze_query_selection_batch(
    clean_comp: Dict[str, Any],
    adv_comp: Dict[str, Any],
    gt_boxes_batch: Sequence[np.ndarray],
) -> List[Dict[str, Any]]:
    """
    Mục 4: Phân tích quá trình Query Selection (S proposals -> Top-300; S=8,400 tại 640x640):
    - Top-K Overlap (K=50, 100, 300) và Jaccard index.
    - Spearman rho và Kendall tau trên toàn bộ S proposals + nhóm foreground tokens.
    - Object Coverage Rate (tỷ lệ GT objects được phủ bởi >=1 proposal trong Top-300):
      theo 3 tiêu chí (Anchor Center-in-GT, Encoder Box IoU >= 0.50, Encoder Box IoU >= 0.30).
    - Thống kê phổ điểm s_(1), s_(50), s_(300), Dropoff và tỷ lệ GT tokens trong Top-300.
    """
    c_scores_all = torch.sigmoid(clean_comp["enc_outputs_class"]).max(-1).values.detach().cpu().numpy()
    a_scores_all = torch.sigmoid(adv_comp["enc_outputs_class"]).max(-1).values.detach().cpu().numpy()

    c_boxes_all = torch.sigmoid(clean_comp["enc_outputs_coord_logits"]).detach().cpu().numpy()
    a_boxes_all = torch.sigmoid(adv_comp["enc_outputs_coord_logits"]).detach().cpu().numpy()
    c_topk = clean_comp["topk_ind"].detach().cpu().numpy()
    a_topk = adv_comp["topk_ind"].detach().cpu().numpy()

    anchors_sig = torch.sigmoid(clean_comp["anchors"][0]).detach().cpu().numpy()
    bsz = c_scores_all.shape[0]
    records: List[Dict[str, Any]] = []

    for b in range(bsz):
        sc = c_scores_all[b]  # [S] (S=8,400 ở 640x640)
        sa = a_scores_all[b]  # [S]
        ic = c_topk[b]        # [300]
        ia = a_topk[b]        # [300]
        gt_b = gt_boxes_batch[b]

        set_c_300 = set(ic.tolist())
        set_a_300 = set(ia.tolist())
        inter_300 = len(set_c_300 & set_a_300)
        union_300 = len(set_c_300 | set_a_300)

        set_c_100 = set(ic[:100].tolist())
        set_a_100 = set(ia[:100].tolist())
        set_c_50 = set(ic[:50].tolist())
        set_a_50 = set(ia[:50].tolist())

        sp_all, _ = stats.spearmanr(sc, sa)
        kt_all, _ = stats.kendalltau(sc, sa)

        # Rank (0 = cao nhất) trên toàn bộ S proposals (8,400 ở 640x640)
        rank_c = stats.rankdata(-sc, method="ordinal") - 1
        rank_a = stats.rankdata(-sa, method="ordinal") - 1

        # Top-15 clean proposal eviction ratio
        top15_c = ic[:15]
        evicted_top15 = float(np.mean(~np.isin(top15_c, ia)))
        median_rank_shift_top15 = float(np.median(rank_a[top15_c] - rank_c[top15_c]))

        fg_mask = compute_token_fg_masks(anchors_sig, gt_b)
        if int(np.sum(fg_mask)) >= 3:
            sp_fg, _ = stats.spearmanr(sc[fg_mask], sa[fg_mask])
            fg_rank_shift = float(np.median(rank_a[fg_mask] - rank_c[fg_mask]))
        else:
            sp_fg = float("nan")
            fg_rank_shift = float("nan")

        # Tỷ lệ các query được chọn có anchor tâm nằm trong vùng GT box
        fg_ratio_clean_300 = float(np.mean(fg_mask[ic])) if len(gt_b) > 0 else float("nan")
        fg_ratio_adv_300 = float(np.mean(fg_mask[ia])) if len(gt_b) > 0 else float("nan")

        # Object Coverage Rate trên các COCO GT boxes thật của ảnh
        if len(gt_b) > 0:
            c_sel_xyxy = np.clip(_cxcywh_to_xyxy_np(c_boxes_all[b][ic]), 0.0, 1.0)
            a_sel_xyxy = np.clip(_cxcywh_to_xyxy_np(a_boxes_all[b][ia]), 0.0, 1.0)

            c_anc_cx = anchors_sig[ic, 0 : 1]
            c_anc_cy = anchors_sig[ic, 1 : 2]
            a_anc_cx = anchors_sig[ia, 0 : 1]
            a_anc_cy = anchors_sig[ia, 1 : 2]

            cov_center_c = float(np.mean(np.any(
                (c_anc_cx >= gt_b[None, :, 0]) & (c_anc_cx <= gt_b[None, :, 2])
                & (c_anc_cy >= gt_b[None, :, 1]) & (c_anc_cy <= gt_b[None, :, 3]),
                axis=0,
            )))
            cov_center_a = float(np.mean(np.any(
                (a_anc_cx >= gt_b[None, :, 0]) & (a_anc_cx <= gt_b[None, :, 2])
                & (a_anc_cy >= gt_b[None, :, 1]) & (a_anc_cy <= gt_b[None, :, 3]),
                axis=0,
            )))

            iou_c = _box_iou_np(c_sel_xyxy, gt_b).max(axis=0)  # [N_gt]
            iou_a = _box_iou_np(a_sel_xyxy, gt_b).max(axis=0)  # [N_gt]
            cov_iou50_c = float(np.mean(iou_c >= 0.50))
            cov_iou50_a = float(np.mean(iou_a >= 0.50))
            cov_iou30_c = float(np.mean(iou_c >= 0.30))
            cov_iou30_a = float(np.mean(iou_a >= 0.30))
        else:
            cov_center_c = cov_center_a = float("nan")
            cov_iou50_c = cov_iou50_a = float("nan")
            cov_iou30_c = cov_iou30_a = float("nan")

        # Phổ điểm đã sắp xếp Top-N_q
        n_q_eff = max(1, len(ic))
        sc_sorted = np.sort(sc)[::-1]
        sa_sorted = np.sort(sa)[::-1]
        idx_s50 = min(49, len(sc_sorted) - 1)
        idx_s_nq = min(n_q_eff - 1, len(sc_sorted) - 1)
        c_margin = float(np.mean(sc_sorted[:n_q_eff]) - np.mean(sc_sorted[n_q_eff:])) if len(sc_sorted) > n_q_eff else 0.0
        a_margin = float(np.mean(sa_sorted[:n_q_eff]) - np.mean(sa_sorted[n_q_eff:])) if len(sa_sorted) > n_q_eff else 0.0

        # Thống kê gom cụm không gian ANND và Spatial Dispersion cho Top-N_q anchors
        clean_spatial = compute_spatial_clustering_metrics(anchors_sig[ic])
        adv_spatial = compute_spatial_clustering_metrics(anchors_sig[ia])

        records.append({
            "n_gt": int(len(gt_b)),
            "top300_overlap": float(inter_300 / float(n_q_eff)),
            "top300_jaccard": float(inter_300 / max(union_300, 1)),
            "top100_overlap": float(len(set_c_100 & set_a_100) / float(min(100, n_q_eff))),
            "top50_overlap": float(len(set_c_50 & set_a_50) / float(min(50, n_q_eff))),
            "n_proposals": int(len(sc)),
            "spearman_all_proposals": float(sp_all) if not np.isnan(sp_all) else 0.0,
            "kendall_all_proposals": float(kt_all) if not np.isnan(kt_all) else 0.0,
            # Backward compatibility aliases
            "spearman_2100": float(sp_all) if not np.isnan(sp_all) else 0.0,
            "kendall_2100": float(kt_all) if not np.isnan(kt_all) else 0.0,
            "spearman_fg_tokens": float(sp_fg) if not np.isnan(sp_fg) else float("nan"),
            "median_rank_shift_fg_tokens": fg_rank_shift,
            "evicted_top15_ratio": evicted_top15,
            "median_rank_shift_top15": median_rank_shift_top15,
            # Tách bạch rõ 2 nhóm metric theo chuẩn phản biện:
            # 1. gt_object_coverage_center (axis=0): Tỷ lệ GT objects có >=1 anchor tâm nằm trong
            "gt_object_coverage_center_clean": cov_center_c,
            "gt_object_coverage_center_adv": cov_center_a,
            # 2. anchor_foreground_ratio (axis=1): Tỷ lệ trong Top-N_q anchors có tâm nằm trong GT
            "anchor_foreground_ratio_clean": fg_ratio_clean_300,
            "anchor_foreground_ratio_adv": fg_ratio_adv_300,
            # Thước đo gom cụm không gian ANND & Spatial Dispersion
            "clean_annd": clean_spatial["annd"],
            "adv_annd": adv_spatial["annd"],
            "clean_spatial_variance": clean_spatial["spatial_variance"],
            "adv_spatial_variance": adv_spatial["spatial_variance"],
            "clean_spatial_std_x": clean_spatial["spatial_std_x"],
            "clean_spatial_std_y": clean_spatial["spatial_std_y"],
            "adv_spatial_std_x": adv_spatial["spatial_std_x"],
            "adv_spatial_std_y": adv_spatial["spatial_std_y"],
            # Backward compatibility aliases
            "fg_query_ratio_clean_top300": fg_ratio_clean_300,
            "fg_query_ratio_adv_top300": fg_ratio_adv_300,
            "cov_center_clean": cov_center_c,
            "cov_center_adv": cov_center_a,
            "cov_iou50_clean": cov_iou50_c,
            "cov_iou50_adv": cov_iou50_a,
            "cov_iou30_clean": cov_iou30_c,
            "cov_iou30_adv": cov_iou30_a,
            "clean_s1": float(sc_sorted[0]),
            "clean_s50": float(sc_sorted[idx_s50]),
            "clean_s300": float(sc_sorted[idx_s_nq]),
            "clean_dropoff": float(sc_sorted[0] / (sc_sorted[idx_s_nq] + 1e-4)),
            "clean_enc_margin": c_margin,
            "adv_s1": float(sa_sorted[0]),
            "adv_s50": float(sa_sorted[idx_s50]),
            "adv_s300": float(sa_sorted[idx_s_nq]),
            "adv_dropoff": float(sa_sorted[0] / (sa_sorted[idx_s_nq] + 1e-4)),
            "adv_enc_margin": a_margin,
        })

    return records


@contextmanager
def capture_deformable_attention_internals(model: torch.nn.Module):
    """
    Context manager gắn forward hook lên `decoder.layers.{l}.encoder_attn`
    để trích xuất chính xác `reference_points` [B, 300, 3, 4],
    `sampling_locations` [B, 300, 8, 3, 4, 2] và `attention_weights` [B, 300, 8, 3, 4]
    theo đúng công thức của `RTDetrMultiscaleDeformableAttention.forward`.
    """
    core = _get_rtdetr_core_model(model)

    captured: List[Dict[str, torch.Tensor]] = []
    handles = []

    def make_hook(layer_idx: int):
        def _hook(module, args, kwargs, output):
            hidden_states = kwargs.get("hidden_states", args[0] if len(args) > 0 else None)
            position_embeddings = kwargs.get("position_embeddings", None)
            reference_points = kwargs.get("reference_points", None)
            if hidden_states is None or reference_points is None:
                return
            hs = hidden_states + position_embeddings if position_embeddings is not None else hidden_states
            bsz, num_q, _ = hs.shape
            offsets = module.sampling_offsets(hs).view(
                bsz, num_q, module.n_heads, module.n_levels, module.n_points, 2
            )
            attn_w = module.attention_weights(hs).view(
                bsz, num_q, module.n_heads, module.n_levels * module.n_points
            )
            attn_w = F.softmax(attn_w, dim=-1).view(
                bsz, num_q, module.n_heads, module.n_levels, module.n_points
            )
            num_coords = reference_points.shape[-1]
            if num_coords == 4:
                samp_loc = (
                    reference_points[:, :, None, :, None, :2]
                    + offsets / module.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
                )
            else:
                samp_loc = reference_points[:, :, None, :, None, :] + offsets
            captured.append({
                "layer": layer_idx,
                "reference_points": reference_points.detach().cpu(),
                "sampling_locations": samp_loc.detach().cpu(),
                "attention_weights": attn_w.detach().cpu(),
                "sampling_offsets": offsets.detach().cpu(),
            })

        return _hook

    for l_idx, layer in enumerate(core.decoder.layers):
        h = layer.encoder_attn.register_forward_hook(make_hook(l_idx), with_kwargs=True)
        handles.append(h)

    try:
        yield captured
    finally:
        for h in handles:
            h.remove()


def _summarize_attention_state(
    captured_layers: List[Dict[str, torch.Tensor]],
    logits_batch: np.ndarray,
    gt_boxes_batch: Sequence[np.ndarray],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Tính các chỉ số phân phối Deformable Attention trên 1 trạng thái (Clean hoặc Attacked)
    tách theo từng tầng decoder và trung bình toàn decoder:
    - Attention Entropy chuẩn hóa H(w)/log(12)
    - Top-1 Weight Concentration (trọng số max trên 12 điểm)
    - Sampling Spatial Dispersion (độ lệch chuẩn vị trí mẫu quanh tâm mẫu của query)
    - Attention-Weighted Foreground Spatial Hit Rate (tỷ trọng attention rơi vào vùng GT box)
    - Out-of-bounds sampling rate
    - Tỷ trọng attention phân bổ lên các scale S3 / S4 / S5
    """
    n_layers = len(captured_layers)
    bsz = logits_batch.shape[0]

    per_image_records: List[Dict[str, Any]] = []
    for b in range(bsz):
        gt_b = gt_boxes_batch[b]  # [N_gt, 4]
        probs_q = _sigmoid_np(logits_batch[b]).max(axis=-1)  # [300]
        active_q = probs_q >= 0.35
        n_active_q = int(np.sum(active_q))

        layer_metrics: Dict[str, Dict[str, float]] = {}
        all_ent = []
        all_top1 = []
        all_disp = []
        all_fg_hit = []
        all_fg_hit_act = []
        all_oob = []
        all_s3 = []
        all_s4 = []
        all_s5 = []

        for l_idx in range(n_layers):
            cap = captured_layers[l_idx]
            w = cap["attention_weights"][b].numpy()   # [N_q, H, L, K]
            loc = cap["sampling_locations"][b].numpy()  # [N_q, H, L, K, 2]
            ref = cap["reference_points"][b].numpy()    # [N_q, 1 or L, 4]

            n_q, n_h = w.shape[0], w.shape[1]
            n_pts = w.shape[2] * w.shape[3]
            log_n_pts = float(np.log(max(n_pts, 2)))

            # 1. Entropy chuẩn hóa trên L*K điểm của mỗi (query, head)
            w_flat = w.reshape(n_q, n_h, n_pts)
            w_safe = np.clip(w_flat, 1e-12, 1.0)
            ent = -np.sum(w_safe * np.log(w_safe), axis=-1) / log_n_pts  # [N_q, H]
            mean_ent = float(np.mean(ent))
            mean_ent_act = float(np.mean(ent[active_q])) if n_active_q > 0 else float("nan")

            # 2. Max weight concentration
            top1_w = float(np.mean(np.max(w_flat, axis=-1)))

            # 3. Spatial dispersion radius quanh tâm lấy mẫu có trọng số của mỗi (query, head)
            loc_flat = loc.reshape(n_q, n_h, n_pts, 2)
            mean_loc = np.sum(loc_flat * w_flat[..., None], axis=2, keepdims=True)  # [N_q, H, 1, 2]
            sq_dist = np.sum((loc_flat - mean_loc) ** 2, axis=-1)  # [N_q, H, n_pts]
            disp_radius = np.sqrt(np.sum(w_flat * sq_dist, axis=-1))  # [N_q, H]
            mean_disp = float(np.mean(disp_radius))

            # 4. Phân bố trọng số theo scale S3 (idx=0), S4 (1), S5 (2)
            w_scale = np.sum(w, axis=(0, 1, 3)) / (float(n_q) * float(n_h))  # [3]

            # 5. Out-of-bounds rate
            oob = np.any((loc_flat < 0.0) | (loc_flat > 1.0), axis=-1)  # [N_q, H, n_pts]
            mean_oob = float(np.mean(oob))

            # 6. Foreground Spatial Hit Rate (attention-weighted xác suất điểm mẫu rơi vào GT box)
            if len(gt_b) > 0:
                px = loc_flat[..., 0 : 1]  # [N_q, H, n_pts, 1]
                py = loc_flat[..., 1 : 2]
                gt_e = gt_b[None, None, None, :, :]  # [1, 1, 1, N_gt, 4]
                inside = np.any(
                    (px >= gt_e[..., 0]) & (px <= gt_e[..., 2])
                    & (py >= gt_e[..., 1]) & (py <= gt_e[..., 3]),
                    axis=-1,
                )  # [N_q, H, n_pts]
                fg_hit_per_q = np.mean(np.sum(w_flat * inside, axis=-1), axis=-1)  # [N_q]
                fg_hit_all = float(np.mean(fg_hit_per_q))
                fg_hit_act = float(np.mean(fg_hit_per_q[active_q])) if n_active_q > 0 else float("nan")
                # Tỷ lệ riêng reference point rơi trong GT box
                ref_xy = ref[:, 0, :2]  # [N_q, 2]
                ref_in_gt = float(np.mean(np.any(
                    (ref_xy[:, 0:1] >= gt_b[None, :, 0]) & (ref_xy[:, 0:1] <= gt_b[None, :, 2])
                    & (ref_xy[:, 1:2] >= gt_b[None, :, 1]) & (ref_xy[:, 1:2] <= gt_b[None, :, 3]),
                    axis=1,
                )))
            else:
                fg_hit_all = fg_hit_act = ref_in_gt = float("nan")

            s3_mass = float(w_scale[0]) if len(w_scale) > 0 else 0.0
            s4_mass = float(w_scale[1]) if len(w_scale) > 1 else 0.0
            s5_mass = float(w_scale[2]) if len(w_scale) > 2 else 0.0

            layer_metrics[f"L{l_idx + 1}"] = {
                "entropy_norm": mean_ent,
                "entropy_norm_active_q": mean_ent_act,
                "top1_weight": top1_w,
                "dispersion_radius": mean_disp,
                "fg_spatial_hit_all": fg_hit_all,
                "fg_spatial_hit_active_q": fg_hit_act,
                "ref_in_gt_ratio": ref_in_gt,
                "oob_rate": mean_oob,
                "scale_mass_s3": s3_mass,
                "scale_mass_s4": s4_mass,
                "scale_mass_s5": s5_mass,
            }

            all_ent.append(mean_ent)
            all_top1.append(top1_w)
            all_disp.append(mean_disp)
            all_fg_hit.append(fg_hit_all)
            all_fg_hit_act.append(fg_hit_act)
            all_oob.append(mean_oob)
            all_s3.append(s3_mass)
            all_s4.append(s4_mass)
            all_s5.append(s5_mass)

        per_image_records.append({
            "n_active_queries": n_active_q,
            "mean_entropy_norm": float(np.nanmean(all_ent)),
            "mean_top1_weight": float(np.nanmean(all_top1)),
            "mean_dispersion_radius": float(np.nanmean(all_disp)),
            "mean_fg_spatial_hit": float(np.nanmean(all_fg_hit)),
            "mean_fg_spatial_hit_active_q": float(np.nanmean(all_fg_hit_act)) if not np.all(np.isnan(all_fg_hit_act)) else float("nan"),
            "mean_oob_rate": float(np.nanmean(all_oob)),
            "mean_scale_mass_s3": float(np.nanmean(all_s3)),
            "mean_scale_mass_s4": float(np.nanmean(all_s4)),
            "mean_scale_mass_s5": float(np.nanmean(all_s5)),
            "per_layer": layer_metrics,
        })

    return {"per_image": per_image_records}


def analyze_deformable_attention_batch(
    model: torch.nn.Module,
    clean_imgs: torch.Tensor,
    adv_imgs: torch.Tensor,
    gt_boxes_batch: Sequence[np.ndarray],
) -> List[Dict[str, Any]]:
    """
    Mục 5: So sánh trực tiếp trạng thái Deformable Attention giữa Clean và Attacked
    trên 3 tầng decoder (đo lường cả biến thiên phân phối toàn cục lẫn dịch chuyển
    từng vị trí reference point / sampling location theo cặp Clean -> Adv).
    """
    model.eval()
    bsz = clean_imgs.shape[0]

    with torch.no_grad():
        with capture_deformable_attention_internals(model) as cap_clean:
            out_c = model(pixel_values=clean_imgs, return_dict=True)
        with capture_deformable_attention_internals(model) as cap_adv:
            out_a = model(pixel_values=adv_imgs, return_dict=True)

    logits_c = out_c.logits.detach().cpu().numpy()
    logits_a = out_a.logits.detach().cpu().numpy()
    init_ref_c = torch.sigmoid(out_c.init_reference_points).detach().cpu().numpy()  # [B, 300, 4]
    init_ref_a = torch.sigmoid(out_a.init_reference_points).detach().cpu().numpy()

    sum_c = _summarize_attention_state(cap_clean, logits_c, gt_boxes_batch)["per_image"]
    sum_a = _summarize_attention_state(cap_adv, logits_a, gt_boxes_batch)["per_image"]

    n_layers = len(cap_clean)
    records: List[Dict[str, Any]] = []

    for b in range(bsz):
        probs_c = _sigmoid_np(logits_c[b]).max(axis=-1)
        fg_q = probs_c >= 0.35
        if int(np.sum(fg_q)) == 0:
            top10 = np.argsort(-probs_c)[:min(10, len(probs_c))]
            fg_q = np.zeros(len(probs_c), dtype=bool)
            fg_q[top10] = True

        # Shift của initial reference points (lưu ý: ở AQSM thứ tự 300 slot có thể bị hoán đổi)
        init_xy_shift_all = float(np.mean(np.linalg.norm(init_ref_a[b, :, :2] - init_ref_c[b, :, :2], axis=-1)))
        init_xy_shift_fg_slots = float(np.mean(np.linalg.norm(init_ref_a[b, fg_q, :2] - init_ref_c[b, fg_q, :2], axis=-1)))

        ref_shifts_l = []
        loc_shifts_l = []
        offset_l2_l = []
        w_l1_shift_l = []
        loc_shifts_fg_slots_l = []

        for l_idx in range(n_layers):
            rc = cap_clean[l_idx]["reference_points"][b].numpy()      # [300, 3, 4]
            ra = cap_adv[l_idx]["reference_points"][b].numpy()
            lc = cap_clean[l_idx]["sampling_locations"][b].numpy()    # [300, 8, 3, 4, 2]
            la = cap_adv[l_idx]["sampling_locations"][b].numpy()
            oc = cap_clean[l_idx]["sampling_offsets"][b].numpy()
            oa = cap_adv[l_idx]["sampling_offsets"][b].numpy()
            wc = cap_clean[l_idx]["attention_weights"][b].numpy()
            wa = cap_adv[l_idx]["attention_weights"][b].numpy()

            # L2 shift of (x, y) reference center
            ref_shift = np.linalg.norm(ra[:, 0, :2] - rc[:, 0, :2], axis=-1)  # [300]
            # L2 shift of sampling locations
            loc_shift = np.linalg.norm(la - lc, axis=-1)  # [300, 8, 3, 4]
            # L2 shift of learned sampling offsets (trực tiếp đo phản ứng nội tại của decoder head)
            off_shift = np.linalg.norm(oa - oc, axis=-1)
            # L1 tổng biến thiên trọng số attention trên 12 điểm
            w_shift = np.sum(np.abs(wa - wc), axis=(-2, -1))  # [300, 8]

            ref_shifts_l.append(float(np.mean(ref_shift)))
            loc_shifts_l.append(float(np.mean(loc_shift)))
            loc_shifts_fg_slots_l.append(float(np.mean(loc_shift[fg_q])))
            offset_l2_l.append(float(np.mean(off_shift)))
            w_l1_shift_l.append(float(np.mean(w_shift)))

        rec = {
            "clean": sum_c[b],
            "adv": sum_a[b],
            "delta_entropy_norm": sum_a[b]["mean_entropy_norm"] - sum_c[b]["mean_entropy_norm"],
            "delta_top1_weight": sum_a[b]["mean_top1_weight"] - sum_c[b]["mean_top1_weight"],
            "delta_dispersion_radius": sum_a[b]["mean_dispersion_radius"] - sum_c[b]["mean_dispersion_radius"],
            "delta_fg_spatial_hit": sum_a[b]["mean_fg_spatial_hit"] - sum_c[b]["mean_fg_spatial_hit"],
            "init_ref_xy_shift_all": init_xy_shift_all,
            "init_ref_xy_shift_fg_slots": init_xy_shift_fg_slots,
            "layer_ref_xy_shift": ref_shifts_l,
            "layer_sampling_loc_shift": loc_shifts_l,
            "layer_sampling_loc_shift_fg_slots": loc_shifts_fg_slots_l,
            "layer_sampling_offset_shift": offset_l2_l,
            "layer_attn_weight_tv_shift": w_l1_shift_l,
            "mean_ref_xy_shift": float(np.mean(ref_shifts_l)),
            "mean_sampling_loc_shift": float(np.mean(loc_shifts_l)),
            "mean_sampling_loc_shift_fg_slots": float(np.mean(loc_shifts_fg_slots_l)),
            "mean_sampling_offset_shift": float(np.mean(offset_l2_l)),
            "mean_attn_weight_tv_shift": float(np.mean(w_l1_shift_l)),
        }
        records.append(rec)

    return records
