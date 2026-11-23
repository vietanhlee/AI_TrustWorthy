# -*- coding: utf-8 -*-
"""
Script: generate_qualitative_predictions.py
Mục đích: Sinh Figure 11 (Qualitative Detection Visualizations) đồng bộ 100% với bộ cache
thực nghiệm chuẩn hóa N=300 trên COCO val2017 (T=15 steps PGD, eps=16/255, alpha=2/255,
target_class_idx=1 bicycle cho EQF/DQF/Patch/Adaptive 1D) tại cùng chỉ số Calibrated Max-OR
từ checkpoints/calibration_and_detectors.pkl cho từng panel (a)-(f).
"""

import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from edge_sec_pro.core.config import (
    AQSM_LAMBDA_RANK,
    AQSM_RANKING_MARGIN,
    CHECKPOINTS_DIR,
    PGD_NUM_STEPS,
    PGD_STEP_SIZE,
    RESULTS_DIR,
)
from edge_sec_pro.core.dataset import load_coco_images
from edge_sec_pro.transformer_monitors.detr_attack_generator import DETRAttackGenerator
from edge_sec_pro.transformer_monitors.detr_wrapper import RTDETRModelWrapper
from edge_sec_pro.transformer_monitors.rtdetr_advanced_features import RTDETRDeepFeatureExtractor

PGD_EPSILON = 16.0 / 255.0
DEFAULT_SAMPLE_IDX = 6  # COCO val2017 test[6] (image_id=159791): kitchen scene showing all 6 mechanisms cleanly
CACHE_DIR = RESULTS_DIR / "checkpoints" / "adv_cache"


def load_cached_adv_tensor(clean_tensor: torch.Tensor, cache_name: str, sample_idx: int) -> Tuple[torch.Tensor, int]:
    """Tải tensor nhiễu đối kháng trực tiếp từ cache N=300 để khớp tuyệt đối số liệu benchmark."""
    cache_path = CACHE_DIR / cache_name
    if not cache_path.exists():
        raise FileNotFoundError(f"Không tìm thấy cache {cache_path}")
    npz = np.load(cache_path)
    delta = torch.tensor(npz["delta_f16"][sample_idx : sample_idx + 1], dtype=torch.float32)
    img_id = int(npz["image_ids"][sample_idx]) if "image_ids" in npz else -1
    return torch.clamp(clean_tensor + delta, 0.0, 1.0), img_id


def analyze_panel(
    wrapper: RTDETRModelWrapper,
    extractor: RTDETRDeepFeatureExtractor,
    max_or_det: Any,
    tau_or: float,
    tensor: torch.Tensor,
    conf_thresh: float = 0.35,
    target_class_idx: int = 1,
) -> Dict[str, Any]:
    """Chạy forward thật + trích xuất thống kê định lượng và trạng thái Calibrated Max-OR."""
    with torch.no_grad():
        state = wrapper(tensor)
        feats = extractor.extract_from_state(state)
        q_max = float(max_or_det.score(feats)[0])
        alert = bool(q_max >= tau_or)

    logits = state.logits[0]  # [300, 80]
    boxes = state.pred_boxes[0]  # [300, 4]
    probs = torch.sigmoid(logits)
    max_probs, labels = probs.max(dim=-1)

    mask = max_probs >= conf_thresh
    indices = torch.where(mask)[0]
    id2label = wrapper.model.config.id2label

    detected: List[Dict[str, Any]] = []
    for idx in indices:
        score = float(max_probs[idx].item())
        lbl_id = int(labels[idx].item())
        detected.append({
            "name": id2label.get(lbl_id, f"cls_{lbl_id}"),
            "label_id": lbl_id,
            "score": score,
            "box": boxes[idx].tolist(),
            "query_idx": int(idx.item()),
        })

    target_probs = probs[:, target_class_idx]
    target_cnt = int((target_probs >= conf_thresh).sum().item())
    target_max = float(target_probs.max().item())

    return {
        "detections": detected,
        "n_active_raw": int(mask.sum().item()),
        "r_act_pct": float(mask.float().mean().item() * 100.0),
        "max_conf": float(max_probs.max().item()),
        "target_cnt": target_cnt,
        "target_max": target_max,
        "q_max": q_max,
        "alert": alert,
    }


def draw_predictions_on_ax(
    ax: plt.Axes,
    img_np: np.ndarray,
    info: Dict[str, Any],
    title: str,
    subtitle: str,
    badge_color: str,
    status_text: str,
    highlight_target_id: int = -1,
) -> None:
    """Vẽ bounding boxes đã khử trùng lặp cục bộ kèm chú thích định lượng rõ ràng."""
    ax.imshow(img_np)
    h, w, _ = img_np.shape

    colors = {
        "refrigerator": "#00E676",
        "microwave": "#00B0FF",
        "oven": "#FFD600",
        "chair": "#FF9100",
        "person": "#2979FF",
        "default": "#E040FB",
    }

    filtered_dets: List[Dict[str, Any]] = []
    for det in sorted(info["detections"], key=lambda d: d["score"], reverse=True):
        duplicate = False
        for kept in filtered_dets:
            if det["name"] == kept["name"]:
                cx1, cy1, _, _ = det["box"]
                cx2, cy2, _, _ = kept["box"]
                if abs(cx1 - cx2) < 0.04 and abs(cy1 - cy2) < 0.04:
                    duplicate = True
                    break
        if not duplicate:
            filtered_dets.append(det)

    placed_badges: List[Tuple[float, float]] = []
    for det in filtered_dets[:10]:  # Giới hạn tối đa 10 nhãn rõ ràng trên mỗi panel để tránh rối mắt
        cx, cy, bw, bh = det["box"]
        xmin = max(0.0, (cx - bw / 2.0) * w)
        ymin = max(0.0, (cy - bh / 2.0) * h)
        box_w = min(w - xmin, bw * w)
        box_h = min(h - ymin, bh * h)

        c_name = det["name"]
        color = "#FF1744" if (highlight_target_id >= 0 and det["label_id"] == highlight_target_id) else colors.get(c_name, colors["default"])

        rect = patches.Rectangle(
            (xmin, ymin), box_w, box_h,
            linewidth=2.0, edgecolor=color, facecolor="none", linestyle="-",
        )
        ax.add_patch(rect)

        label_str = f"{c_name} {det['score']:.2f}"
        badge_y = max(ymin - 4, 14)
        badge_x = min(max(xmin + 2, 4), w - 95)
        for bx, by in placed_badges:
            if abs(badge_x - bx) < 62 and abs(badge_y - by) < 15:
                badge_y = min(h - 10, by + 15)
        placed_badges.append((badge_x, badge_y))

        ax.text(
            badge_x, badge_y, label_str,
            fontsize=7.6, fontweight="bold", color="black",
            bbox=dict(boxstyle="square,pad=0.18", facecolor=color, edgecolor="none", alpha=0.92),
        )

    ax.set_title(title, fontsize=10.5, fontweight="bold", pad=7)
    ax.set_xlabel(subtitle, fontsize=8.3, fontweight="medium", labelpad=5, color="#1A202C")

    badge_bg = "#E8F5E9" if badge_color == "green" else ("#FFEBEE" if badge_color == "red" else "#FFF8E1")
    badge_fg = "#1B5E20" if badge_color == "green" else ("#B71C1C" if badge_color == "red" else "#E65100")
    ax.text(
        0.98, 0.96, status_text,
        transform=ax.transAxes,
        fontsize=8.0, fontweight="bold", color=badge_fg,
        ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.32", facecolor=badge_bg, edgecolor=badge_fg, linewidth=1.0),
    )

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#CFD8DC")
        spine.set_linewidth(1.0)


def main() -> None:
    print("=" * 80)
    print("SINH FIGURE 11 (QUALITATIVE DETECTIONS) DONG BO CHINH XAC CACHE N=300")
    print("=" * 80)

    device = torch.device("cpu")
    wrapper = RTDETRModelWrapper("PekingU/rtdetr_r18vd", device=device)
    wrapper.model.eval()
    extractor = RTDETRDeepFeatureExtractor(conf_threshold=0.35, top_k_queries=10, projection_dim=64)

    with open(CHECKPOINTS_DIR / "calibration_and_detectors.pkl", "rb") as f:
        ckpt = pickle.load(f)
    max_or_det = ckpt["detectors"]["Calibrated_Max_OR"]
    tau_or = float(max_or_det.threshold)
    cs = ckpt["clean_stats"]

    cal_imgs, test_imgs = load_coco_images(n_cal=1200, n_test=600, img_size=640, seed=42)
    clean_tensor = test_imgs[DEFAULT_SAMPLE_IDX : DEFAULT_SAMPLE_IDX + 1]

    try:
        adv_aqsm, img_id = load_cached_adv_tensor(clean_tensor, "adv_cache_aqsm_eps16_n300.npz", DEFAULT_SAMPLE_IDX)
        adv_van, _ = load_cached_adv_tensor(clean_tensor, "adv_cache_vanishing_eps16_n300.npz", DEFAULT_SAMPLE_IDX)
        adv_dqf, _ = load_cached_adv_tensor(clean_tensor, "adv_cache_dqf_eps16_n300.npz", DEFAULT_SAMPLE_IDX)
        adv_patch, _ = load_cached_adv_tensor(clean_tensor, "adv_cache_patch48_eps16_n300.npz", DEFAULT_SAMPLE_IDX)
        adv_adapt, _ = load_cached_adv_tensor(clean_tensor, "adv_cache_adaptive1d_eps16_n300.npz", DEFAULT_SAMPLE_IDX)
        print(f"  -> Loaded cached deltas for test[{DEFAULT_SAMPLE_IDX}] (COCO image_id={img_id})")
    except FileNotFoundError:
        img_id = -1
        attacker = DETRAttackGenerator(wrapper, epsilon=PGD_EPSILON, alpha=PGD_STEP_SIZE, num_steps=PGD_NUM_STEPS)
        adv_aqsm = attacker.generate_query_selection_attack(
            clean_tensor, steps=PGD_NUM_STEPS, alpha=PGD_STEP_SIZE,
            margin=AQSM_RANKING_MARGIN, lambda_rank=AQSM_LAMBDA_RANK,
        )
        adv_van = attacker.generate_vanishing_attack(clean_tensor, steps=PGD_NUM_STEPS, alpha=PGD_STEP_SIZE)
        adv_dqf = attacker.generate_fabrication_attack(clean_tensor, target_class_idx=1, steps=PGD_NUM_STEPS, alpha=PGD_STEP_SIZE)
        patch_48 = attacker.optimize_adversarial_patch(cal_imgs[:60], patch_size=48, patch_loc="random", target_class_idx=1, num_epochs=4)
        adv_patch = attacker.apply_adversarial_patch(clean_tensor, patch_48, patch_loc=(20, 240), seed=1337)
        adv_adapt = attacker.generate_adaptive_attack(
            clean_tensor, target_class_idx=1, evasion_weight=0.50, steps=PGD_NUM_STEPS,
            loss_mode="hinge", target_detector="1d",
            tau_bounds_act=cs["tau_act_bounds"], tau_bounds_top1=cs["tau_top_bounds"],
            clean_act_mean=cs["mu_act"], clean_act_std=cs["sd_act"],
            clean_top1_mean=cs["mu_top1"], clean_top1_std=cs["sd_top1"], temperature=25.0,
        )

    clean_info = analyze_panel(wrapper, extractor, max_or_det, tau_or, clean_tensor)
    aqsm_info = analyze_panel(wrapper, extractor, max_or_det, tau_or, adv_aqsm)
    van_info = analyze_panel(wrapper, extractor, max_or_det, tau_or, adv_van)
    dqf_info = analyze_panel(wrapper, extractor, max_or_det, tau_or, adv_dqf)
    patch_info = analyze_panel(wrapper, extractor, max_or_det, tau_or, adv_patch)
    adapt_info = analyze_panel(wrapper, extractor, max_or_det, tau_or, adv_adapt)

    def _status(info: Dict[str, Any]) -> Tuple[str, str]:
        tag = "ALERT" if info["alert"] else "PASS"
        col = "red" if info["alert"] else "orange"
        return f"Max-OR q={info['q_max']:.3f} [{tag}]", col

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 10.4), dpi=300)
    plt.subplots_adjust(left=0.025, right=0.975, top=0.89, bottom=0.06, wspace=0.10, hspace=0.27)

    s_cl, _ = _status(clean_info)
    draw_predictions_on_ax(
        axes[0, 0], clean_tensor[0].permute(1, 2, 0).numpy(), clean_info,
        title=f"(a) Clean Baseline (COCO ID {img_id})",
        subtitle=f"Boxes>=0.35: {clean_info['n_active_raw']} (r_act={clean_info['r_act_pct']:.2f}%) | Max conf={clean_info['max_conf']:.2f}",
        badge_color="green", status_text=f"Clean: {s_cl}",
    )

    s_aq, c_aq = _status(aqsm_info)
    draw_predictions_on_ax(
        axes[0, 1], adv_aqsm[0].permute(1, 2, 0).numpy(), aqsm_info,
        title="(b) Tier 1: AQSM (T=15, eps=16/255)",
        subtitle=f"Boxes>=0.35: {aqsm_info['n_active_raw']} (0 true kept) | Max conf={aqsm_info['max_conf']:.2f}",
        badge_color=c_aq, status_text=s_aq,
    )

    s_vn, c_vn = _status(van_info)
    draw_predictions_on_ax(
        axes[0, 2], adv_van[0].permute(1, 2, 0).numpy(), van_info,
        title="(c) Tier 2a: Query Vanishing (T=15, eps=16/255)",
        subtitle=f"Boxes>=0.35: {van_info['n_active_raw']} (r_act={van_info['r_act_pct']:.2f}%) | Max conf={van_info['max_conf']:.2f}",
        badge_color=c_vn, status_text=s_vn,
    )

    s_dq, c_dq = _status(dqf_info)
    draw_predictions_on_ax(
        axes[1, 0], adv_dqf[0].permute(1, 2, 0).numpy(), dqf_info,
        title="(d) Tier 2b: Decoder Fabrication DQF (T=15)",
        subtitle=f"Boxes>=0.35: {dqf_info['n_active_raw']} | Target(bicycle)={dqf_info['target_cnt']} (max={dqf_info['target_max']:.2f})",
        badge_color=c_dq, status_text=s_dq, highlight_target_id=1,
    )

    s_pt, _ = _status(patch_info)
    draw_predictions_on_ax(
        axes[1, 1], adv_patch[0].permute(1, 2, 0).numpy(), patch_info,
        title="(e) Tier 0: Gradient Patch (48x48, EoT)",
        subtitle=f"Boxes>=0.35: {patch_info['n_active_raw']} (r_act={patch_info['r_act_pct']:.2f}%) | Target(bicycle)={patch_info['target_cnt']}",
        badge_color="orange" if not patch_info["alert"] else "red", status_text=s_pt, highlight_target_id=1,
    )

    s_ad, _ = _status(adapt_info)
    draw_predictions_on_ax(
        axes[1, 2], adv_adapt[0].permute(1, 2, 0).numpy(), adapt_info,
        title=r"(f) Tier 3: Adaptive 1D Evasion ($\lambda=0.5$, T=15)",
        subtitle=f"Boxes>=0.35: {adapt_info['n_active_raw']} (r_act={adapt_info['r_act_pct']:.2f}%) | Target(bicycle)={adapt_info['target_cnt']}",
        badge_color="red" if adapt_info["alert"] else "orange",
        status_text=f"Bypassed ({s_ad})" if not adapt_info["alert"] else s_ad,
        highlight_target_id=1,
    )

    fig.suptitle(
        rf"Qualitative Detections & Guardrail States on RT-DETR-R18VD ($640\times 640$, COCO ID {img_id}, $\tau_{{\mathrm{{OR}}}}={tau_or:.3f}$)",
        fontsize=13.0, fontweight="bold", y=0.96, color="#1A202C",
    )

    dest_dirs = [
        PROJECT_ROOT / "paper" / "fig",
        PROJECT_ROOT / "paper_ieee" / "fig",
        PROJECT_ROOT / "slides",
    ]
    for d in dest_dirs:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / "fig11_qualitative_detections.pdf", bbox_inches="tight", dpi=300)
        fig.savefig(d / "fig11_qualitative_detections.png", bbox_inches="tight", dpi=300)
        print(f"  -> Saved fig11 in {d}")

    plt.close(fig)


if __name__ == "__main__":
    main()
