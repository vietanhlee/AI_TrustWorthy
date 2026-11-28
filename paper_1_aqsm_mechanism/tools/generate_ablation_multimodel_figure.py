#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: paper_1_aqsm_mechanism.tools.generate_ablation_multimodel_figure
Mục đích: Tự động sinh Figure 9 (fig9_ablation_and_multimodel.pdf / .png) từ JSON thực nghiệm:
  - Panel (a): Tripartite Loss Ablation & Baselines (Clean, Gaussian, Mem-Isolated, 7 AQSM variants, Jayasundara [27], Pure-Memory) trên cả R18VD (N=300) và R50VD (N=200).
  - Panel (b): Cross-Architecture Benchmark trên unified COCO cohort (N=300) qua 6 kiến trúc (RT-DETR-R18/R50, RT-DETRv2, Deformable-DETR, Vanilla DETR, RetinaNet-NMS).
"""

import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": ["DejaVu Serif", "serif", "sans-serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 9.5,
    "axes.labelsize": 10.5,
    "axes.titlesize": 11,
    "xtick.labelsize": 8.2,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def generate_fig9(r18_dir: Path, r50_dir: Path, fig_dir: Path) -> None:
    ab18 = json.loads((r18_dir / "ablation_and_baselines_summary.json").read_text(encoding="utf-8"))
    ab50 = json.loads((r50_dir / "ablation_and_baselines_summary.json").read_text(encoding="utf-8"))
    mm = json.loads((r18_dir / "multimodel_benchmark_summary.json").read_text(encoding="utf-8"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 4.4), gridspec_kw={"width_ratios": [1.35, 1.0]})

    # ------------------------------------------------------------------
    # Panel (a): Tripartite Loss Ablation & Baselines (R18 N=300 vs R50 N=200)
    # ------------------------------------------------------------------
    items = [
        ("Clean", ab18["clean_baseline"], ab50["clean_baseline"]),
        ("Gaussian\nNoise", ab18["gaussian_noise_control"], ab50["gaussian_noise_control"]),
        ("Mem-Isol.\nQuery", ab18["memory_isolated_query_control"], ab50["memory_isolated_query_control"]),
        (r"$\mathcal{L}_{\mathrm{elev}}$", ab18["tripartite_loss_ablation"]["L_elev"], ab50["tripartite_loss_ablation"]["L_elev"]),
        (r"$\mathcal{L}_{\mathrm{rank}}$", ab18["tripartite_loss_ablation"]["L_rank"], ab50["tripartite_loss_ablation"]["L_rank"]),
        (r"$\mathcal{L}_{\mathrm{suppr}}+\mathcal{L}_{\mathrm{rank}}$", ab18["tripartite_loss_ablation"]["L_suppr+L_rank"], ab50["tripartite_loss_ablation"]["L_suppr+L_rank"]),
        (r"$\mathcal{L}_{\mathrm{rank}}+\mathcal{L}_{\mathrm{elev}}$", ab18["tripartite_loss_ablation"]["L_rank+L_elev"], ab50["tripartite_loss_ablation"]["L_rank+L_elev"]),
        ("Full\nAQSM", ab18["tripartite_loss_ablation"]["Full_Tripartite"], ab50["tripartite_loss_ablation"]["Full_Tripartite"]),
        (r"$\mathcal{L}_{\mathrm{suppr}}$", ab18["tripartite_loss_ablation"]["L_suppr"], ab50["tripartite_loss_ablation"]["L_suppr"]),
        (r"$\mathcal{L}_{\mathrm{suppr}}+\mathcal{L}_{\mathrm{elev}}$", ab18["tripartite_loss_ablation"]["L_suppr+L_elev"], ab50["tripartite_loss_ablation"]["L_suppr+L_elev"]),
        ("Attn [27]", ab18["encoder_attention_baseline_27"], ab50["encoder_attention_baseline_27"]),
        ("Pure\nMemory", ab18["pure_memory_control"], ab50["pure_memory_control"]),
    ]

    x = np.arange(len(items))
    w = 0.38
    y18 = [float(it[1]["mAP@50"]) for it in items]
    y50 = [float(it[2]["mAP@50"]) for it in items]

    b1 = ax1.bar(x - w / 2, y18, w, label=r"RT-DETR-R18VD ($N=300$)", color="#2563eb", edgecolor="black", alpha=0.88)
    b2 = ax1.bar(x + w / 2, y50, w, label=r"RT-DETR-R50VD ($N=200$)", color="#ea580c", edgecolor="black", alpha=0.88)

    ax1.set_xticks(x)
    ax1.set_xticklabels([it[0] for it in items], rotation=28, ha="right", fontsize=7.8)
    ax1.set_ylabel("COCO mAP@50 (%)", fontweight="bold")
    ax1.set_title("(a) Tripartite Ablation & Pixel-Space Controls", fontweight="bold")
    ax1.set_ylim(0, 86.0)
    ax1.grid(axis="y", linestyle="--", alpha=0.45)
    ax1.legend(loc="upper right", frameon=True)

    for rect in list(b1) + list(b2):
        h = rect.get_height()
        lbl = f"{h:.1f}" if h >= 10 else (f"{h:.2f}" if h < 1.0 else f"{h:.1f}")
        ax1.annotate(lbl,
                     xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=6.4, rotation=90)

    # ------------------------------------------------------------------
    # Panel (b): Cross-Architecture Benchmark (Unified N=300)
    # ------------------------------------------------------------------
    arch_specs = [
        ("rtdetr_r18vd", "RT-DETR\nR18VD"),
        ("rtdetr_r50vd", "RT-DETR\nR50VD"),
        ("rtdetr_v2_r18vd", "RT-DETRv2\nR18VD"),
        ("deformable_detr_with_box_refine_two_stage", "Deform.\nDETR (2S)"),
        ("detr_resnet_50", "Vanilla\nDETR-R50"),
        ("retinanet_nms", "RetinaNet\n(CNN+NMS)"),
    ]

    x2 = np.arange(len(arch_specs))
    w2 = 0.25
    clean_vals = [float(mm[k]["clean"]["mAP@50"]) for k, _ in arch_specs]
    aqsm_vals = [float(mm[k]["aqsm"]["mAP@50"]) if mm[k].get("aqsm") is not None else 0.0 for k, _ in arch_specs]
    pgd_vals = [float(mm[k]["task_pgd"]["mAP@50"]) for k, _ in arch_specs]

    c_bars = ax2.bar(x2 - w2, clean_vals, w2, label="Clean Baseline", color="#16a34a", edgecolor="black", alpha=0.88)
    a_bars = ax2.bar(x2, aqsm_vals, w2, label="AQSM (Encoder Probe)", color="#dc2626", edgecolor="black", alpha=0.88)
    p_bars = ax2.bar(x2 + w2, pgd_vals, w2, label="Standard Task PGD", color="#f59e0b", edgecolor="black", alpha=0.88)

    ax2.set_xticks(x2)
    ax2.set_xticklabels([lbl for _, lbl in arch_specs], fontsize=8.0)
    ax2.set_ylabel("COCO mAP@50 (%)", fontweight="bold")
    ax2.set_title(r"(b) Cross-Architecture Generalization ($N=300$)", fontweight="bold")
    ax2.set_ylim(0, 86.0)
    ax2.grid(axis="y", linestyle="--", alpha=0.45)
    ax2.legend(loc="upper right", frameon=True)

    for idx, (k, _) in enumerate(arch_specs):
        # Clean annotation
        hc = clean_vals[idx]
        ax2.annotate(f"{hc:.1f}", xy=(c_bars[idx].get_x() + w2 / 2, hc),
                     xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=7.0)
        # AQSM annotation (or N/A)
        if mm[k].get("aqsm") is not None:
            ha = aqsm_vals[idx]
            ax2.annotate(f"{ha:.1f}", xy=(a_bars[idx].get_x() + w2 / 2, ha),
                         xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=7.0)
        else:
            a_bars[idx].set_alpha(0.15)
            ax2.annotate("N/A", xy=(a_bars[idx].get_x() + w2 / 2, 1.5),
                         ha="center", va="bottom", fontsize=6.5, color="#64748b", rotation=90)
        # Task PGD annotation
        hp = pgd_vals[idx]
        ax2.annotate(f"{hp:.1f}", xy=(p_bars[idx].get_x() + w2 / 2, hp),
                     xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=7.0)

    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig9_ablation_and_multimodel.pdf")
    plt.savefig(fig_dir / "fig9_ablation_and_multimodel.png")
    plt.close()
    print("[OK] Generated fig9_ablation_and_multimodel.pdf/.png")


if __name__ == "__main__":
    base = Path(__file__).resolve().parent.parent
    generate_fig9(base / "outputs" / "summary_json_r18", base / "outputs" / "summary_json_r50", base / "fig")
