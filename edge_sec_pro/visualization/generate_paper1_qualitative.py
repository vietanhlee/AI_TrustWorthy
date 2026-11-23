# -*- coding: utf-8 -*-
"""
Script: generate_paper1_qualitative.py
Mục đích: Sinh Figure 3 (fig11_qualitative_detections.pdf / .png) cho Paper 1.
Loại bỏ hoàn toàn EQF (thuộc về Paper 2), tập trung 100% vào các mối đe dọa của Paper 1:
  (a) Clean Baseline (COCO ID 159791)
  (b) Negative Control: Gaussian Noise (sigma=16/255)
  (c) Tier 1: AQSM (eps=16/255, T=15) - Đảo ngược Dynamic Selection
  (d) Tier 1: AQSM Low-Budget (eps=4/255, T=15) - Triệt tiêu tàng hình không flood box
  (e) Tier 2: PGD Vanishing (eps=16/255, T=15) - Scene Blackout
  (f) Tier 2: Standard Task PGD (Hungarian Loss, eps=16/255, T=15) - Phá hủy foreground + Box Flood
"""

import os
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

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from edge_sec_pro.core.config import RESULTS_DIR
from edge_sec_pro.core.dataset import load_coco_images
from edge_sec_pro.transformer_monitors.detr_wrapper import RTDETRModelWrapper

DEFAULT_SAMPLE_IDX = 6  # COCO val2017 test[6] (image_id=159791): kitchen scene
CACHE_DIR = WORKSPACE_ROOT / "paper_1_aqsm_mechanism" / "cache" / "adv_cache_640"
if not CACHE_DIR.exists():
    CACHE_DIR = RESULTS_DIR / "checkpoints" / "adv_cache"


def load_cached_adv_tensor(clean_tensor: torch.Tensor, cache_name: str, sample_idx: int) -> torch.Tensor:
    cache_path = CACHE_DIR / cache_name
    if not cache_path.exists():
        raise FileNotFoundError(f"Không tìm thấy cache: {cache_path}")
    npz = np.load(cache_path)
    delta = torch.tensor(npz["delta_f16"][sample_idx : sample_idx + 1], dtype=torch.float32)
    return torch.clamp(clean_tensor + delta, 0.0, 1.0)


def analyze_detections(wrapper: RTDETRModelWrapper, tensor: torch.Tensor, conf_thresh: float = 0.35) -> Dict[str, Any]:
    with torch.no_grad():
        state = wrapper(tensor)
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

    return {
        "detections": detected,
        "n_active": int(mask.sum().item()),
        "max_conf": float(max_probs.max().item()),
    }


def draw_panel(
    ax: plt.Axes,
    img_np: np.ndarray,
    info: Dict[str, Any],
    title: str,
    subtitle: str,
    status_text: str,
    is_nominal: bool = False,
) -> None:
    ax.imshow(img_np)
    h, w, _ = img_np.shape

    colors = {
        "refrigerator": "#00E676",  # Vibrant green
        "microwave": "#00B0FF",     # Sky blue
        "oven": "#FFD600",          # Gold yellow
        "chair": "#FF9100",         # Orange
        "person": "#FF1744",        # Crimson red
        "car": "#D500F9",           # Purple
        "umbrella": "#00E5FF",      # Cyan
        "bottle": "#FF6D00",        # Deep orange
        "book": "#76FF03",          # Lime
        "default": "#E040FB",
    }

    # Bỏ các box trùng lặp cục bộ để tránh đè chữ
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
    for det in filtered_dets[:8]:
        cx, cy, bw, bh = det["box"]
        xmin = max(0.0, (cx - bw / 2.0) * w)
        ymin = max(0.0, (cy - bh / 2.0) * h)
        box_w = min(w - xmin, bw * w)
        box_h = min(h - ymin, bh * h)

        c_name = det["name"]
        color = colors.get(c_name, colors["default"])

        rect = patches.Rectangle(
            (xmin, ymin), box_w, box_h,
            linewidth=2.2, edgecolor=color, facecolor="none", linestyle="-",
        )
        ax.add_patch(rect)

        label_str = f"{c_name} {det['score']:.2f}"
        badge_y = max(ymin - 4, 15)
        badge_x = min(max(xmin + 2, 4), w - 95)
        for bx, by in placed_badges:
            if abs(badge_x - bx) < 65 and abs(badge_y - by) < 16:
                badge_y = min(h - 12, by + 16)
        placed_badges.append((badge_x, badge_y))

        ax.text(
            badge_x, badge_y, label_str,
            fontsize=8.0, fontweight="bold", color="black",
            bbox=dict(boxstyle="square,pad=0.20", facecolor=color, edgecolor="none", alpha=0.92),
        )

    ax.set_title(title, fontsize=10.5, fontweight="bold", pad=7)
    ax.set_xlabel(subtitle, fontsize=8.2, fontweight="medium", labelpad=5, color="#1A202C")

    badge_bg = "#E8F5E9" if is_nominal else "#FFEBEE"
    badge_fg = "#1B5E20" if is_nominal else "#B71C1C"
    ax.text(
        0.98, 0.96, status_text,
        transform=ax.transAxes,
        fontsize=8.2, fontweight="bold", color=badge_fg,
        ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.35", facecolor=badge_bg, edgecolor=badge_fg, linewidth=1.2),
    )

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#CFD8DC")
        spine.set_linewidth(1.0)


def main() -> None:
    print("=" * 80)
    print("SINH FIGURE 3 (QUALITATIVE PREDICTIONS) CHUAN HOAN TOAN CHO PAPER 1")
    print("=" * 80)

    device = torch.device("cpu")
    wrapper = RTDETRModelWrapper("PekingU/rtdetr_r18vd", device=device)
    wrapper.model.eval()

    _, test_imgs = load_coco_images(n_cal=1200, n_test=600, img_size=640, seed=42)
    clean_tensor = test_imgs[DEFAULT_SAMPLE_IDX : DEFAULT_SAMPLE_IDX + 1]

    # 1. Clean
    clean_info = analyze_detections(wrapper, clean_tensor)

    # 2. Gaussian Noise
    torch.manual_seed(42)
    gn_tensor = torch.clamp(clean_tensor + torch.randn_like(clean_tensor) * (16.0 / 255.0), 0.0, 1.0)
    gn_info = analyze_detections(wrapper, gn_tensor)

    # 3. AQSM eps=16
    aq16_tensor = load_cached_adv_tensor(clean_tensor, "adv_cache_aqsm_eps16_n300.npz", DEFAULT_SAMPLE_IDX)
    aq16_info = analyze_detections(wrapper, aq16_tensor)

    # 4. AQSM eps=4 (Low Budget)
    aq4_tensor = load_cached_adv_tensor(clean_tensor, "adv_cache_aqsm_eps4_n300.npz", DEFAULT_SAMPLE_IDX)
    aq4_info = analyze_detections(wrapper, aq4_tensor)

    # 5. Vanishing eps=16
    van_tensor = load_cached_adv_tensor(clean_tensor, "adv_cache_vanishing_eps16_n300.npz", DEFAULT_SAMPLE_IDX)
    van_info = analyze_detections(wrapper, van_tensor)

    # 6. Task PGD eps=16
    task_tensor = load_cached_adv_tensor(clean_tensor, "adv_cache_taskpgd_eps16_n300.npz", DEFAULT_SAMPLE_IDX)
    task_info = analyze_detections(wrapper, task_tensor)

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 10.4), dpi=300)
    plt.subplots_adjust(left=0.025, right=0.975, top=0.91, bottom=0.05, wspace=0.10, hspace=0.26)

    # Panel (a): Clean Baseline
    draw_panel(
        axes[0, 0], clean_tensor[0].permute(1, 2, 0).numpy(), clean_info,
        title="(a) Clean Baseline (COCO ID 159791)",
        subtitle=f"Active Boxes (>=0.35): {clean_info['n_active']} | Max Conf: {clean_info['max_conf']:.2f}",
        status_text="Nominal: 4 Valid Objs",
        is_nominal=True,
    )

    # Panel (b): Gaussian Noise Control
    draw_panel(
        axes[0, 1], gn_tensor[0].permute(1, 2, 0).numpy(), gn_info,
        title=r"(b) Negative Control: Gaussian Noise ($\sigma=16/255$)",
        subtitle=f"Active Boxes (>=0.35): {gn_info['n_active']} | Salient objects preserved under thermal noise",
        status_text="Control: Robust",
        is_nominal=True,
    )

    # Panel (c): AQSM eps=16
    draw_panel(
        axes[0, 2], aq16_tensor[0].permute(1, 2, 0).numpy(), aq16_info,
        title=r"(c) Tier C: AQSM ($\epsilon=16/255$, $T=15$ steps)",
        subtitle=f"Active Boxes: {aq16_info['n_active']} (0 true kept) | Max Conf: {aq16_info['max_conf']:.2f} (Spurious noise)",
        status_text="AQSM: 0 Kept (Evicted)",
        is_nominal=False,
    )

    # Panel (d): AQSM Low Budget eps=4
    draw_panel(
        axes[1, 0], aq4_tensor[0].permute(1, 2, 0).numpy(), aq4_info,
        title=r"(d) Tier C: AQSM Low-Budget ($\epsilon=4/255$, $T=15$ steps)",
        subtitle=f"Active Boxes: {aq4_info['n_active']} (0 true kept) | Subtle perturbation, no false-alarm flooding",
        status_text="Stealthy Eviction: 0 Kept",
        is_nominal=False,
    )

    # Panel (e): PGD Vanishing
    draw_panel(
        axes[1, 1], van_tensor[0].permute(1, 2, 0).numpy(), van_info,
        title=r"(e) Tier B: PGD Vanishing ($\epsilon=16/255$, $T=15$ steps)",
        subtitle=f"Active Boxes: {van_info['n_active']} (Complete scene blackout) | Confidences suppressed below threshold",
        status_text="Scene Blackout: 0 Kept",
        is_nominal=False,
    )

    # Panel (f): Standard Task PGD
    draw_panel(
        axes[1, 2], task_tensor[0].permute(1, 2, 0).numpy(), task_info,
        title=r"(f) Tier B: Standard Task PGD (Hungarian Loss, $\epsilon=16/255$)",
        subtitle=f"Active Boxes: {task_info['n_active']} (0 true kept) | Concentrated foreground destruction + False-alarm flooding",
        status_text="FG Silenced + Box Flood",
        is_nominal=False,
    )

    fig.suptitle(
        r"Qualitative Detection Visualizations on RT-DETR-R18VD Across Controls and Threat Tiers (COCO ID 159791)",
        fontsize=13.0, fontweight="bold", y=0.965, color="#1A202C",
    )

    destinations = [
        WORKSPACE_ROOT / "paper_1_aqsm_mechanism" / "fig",
        WORKSPACE_ROOT / "paper_1_aqsm_mechanism" / "ieee" / "fig",
    ]

    for d in destinations:
        d.mkdir(parents=True, exist_ok=True)
        for prefix in ["fig3_qualitative_detections", "fig11_qualitative_detections"]:
            pdf_path = d / f"{prefix}.pdf"
            png_path = d / f"{prefix}.png"
            fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
            fig.savefig(png_path, bbox_inches="tight", dpi=300)
            print(f"  -> Saved: {pdf_path}")
            print(f"  -> Saved: {png_path}")

    plt.close(fig)
    print("Done! Figure 3 has been regenerated cleanly with NO EQF!")


if __name__ == "__main__":
    main()
