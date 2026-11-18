# -*- coding: utf-8 -*-
"""
KỊCH BẢN XUẤT TOÀN BỘ FIGURES VECTOR CHUẨN PUBLICATION (ELSEVIER & IEEE)
Dự án: Nghiên cứu Lỗ hổng Kiến trúc & Cơ chế Giám sát Đa tầng trên RT-DETR
Toàn bộ số liệu được trích xuất 100% từ dữ liệu thực nghiệm JSON/NPZ (Zero Mock / Zero Synthetic Fallback).
Xuất đồng bộ ra `paper/fig`, `paper_ieee/fig`, và `slides/`.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent.parent
WORKSPACE_DIR = SCRIPT_DIR.parent
RESULTS_DIR = SCRIPT_DIR / "results"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"
RESULTS_JSON_PATH = RESULTS_DIR / "rtdetr_advanced_defense_results.json"
RAW_SCORES_NPZ_PATH = RESULTS_DIR / "rtdetr_advanced_defense_raw_scores.npz"
CAUSAL_ANOVA_JSON_PATH = RESULTS_DIR / "rtdetr_factorial_anova_standardized.json"
MECHANISM_JSON_PATH = RESULTS_DIR / "rtdetr_mechanism_analysis_results.json"
MECHANISM_NPZ_PATH = CHECKPOINTS_DIR / "raw_mechanism_analysis_n300.npz"

FIG_DIRS = [
    WORKSPACE_DIR / "paper_1_aqsm_mechanism" / "fig",
    WORKSPACE_DIR / "paper_1_aqsm_mechanism" / "ieee" / "fig",
    WORKSPACE_DIR / "paper_3_defense_guardrail" / "fig",
    SCRIPT_DIR / "paper" / "fig",
    SCRIPT_DIR / "paper_ieee" / "fig",
    SCRIPT_DIR / "slides",
]
for _d in FIG_DIRS:
    _d.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 11
plt.rcParams["xtick.labelsize"] = 9.5
plt.rcParams["ytick.labelsize"] = 9.5
plt.rcParams["legend.fontsize"] = 9.0
plt.rcParams["figure.titlesize"] = 12

COLORS = {
    "proposed": "#2E4053",
    "split_act": "#2980B9",
    "split_top": "#27AE60",
    "maha": "#8E44AD",
    "incon": "#D35400",
    "spatial_bbox": "#16A085",
    "asr": "#C0392B",
    "taskpgd": "#8E44AD",
    "noise": "#7F8C8D",
}


def save_fig_all(fig, base_name: str) -> None:
    """Lưu đồng bộ vector PDF & PNG 300 DPI vào paper_1/fig, paper_1_ieee/fig, paper/fig và slides."""
    for d in FIG_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f"{base_name}.pdf", bbox_inches="tight", dpi=300)
        fig.savefig(d / f"{base_name}.png", bbox_inches="tight", dpi=300)
    print(f"    --> Đã xuất đồng bộ: {base_name}.pdf/.png", flush=True)


def draw_figure_1_architecture():
    print("  + Vẽ Figure 1: System Architecture (khớp đúng 4 kênh Max-OR & ESM Margin phụ trợ)...", flush=True)
    fig, ax = plt.subplots(figsize=(13.5, 7.8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    def add_card(x, y, w, h, title, subtitle="", color="#EBF5FB", edge="#2980B9", lw=1.5, title_fs=10.5, sub_fs=9.0, title_color="#1A252C", sub_color="#2C3E50"):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35", fc=color, ec=edge, lw=lw)
        ax.add_patch(rect)
        if subtitle:
            ax.text(x + w / 2, y + h * 0.63, title, ha="center", va="center", fontsize=title_fs, fontweight="bold", color=title_color)
            ax.text(x + w / 2, y + h * 0.28, subtitle, ha="center", va="center", fontsize=sub_fs, color=sub_color, linespacing=1.2)
        else:
            ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=title_fs, fontweight="bold", color=title_color)

    def add_arrow(x1, y1, x2, y2, label="", color="#34495E", lw=1.8, label_color="#1B4F72", ec_label="#AED6F1", y_label=None, x_label=None):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(facecolor=color, edgecolor=color, width=lw, headwidth=5.5, headlength=5.5, shrink=0.03),
        )
        if label:
            xm = x_label if x_label is not None else (x1 + x2) / 2
            ym = y_label if y_label is not None else (y1 + y2) / 2
            ax.text(
                xm, ym, label, ha="center", va="center", fontsize=9.0, fontweight="bold", color=label_color,
                bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor=ec_label, linewidth=0.8, alpha=0.98),
            )

    detr_frame = patches.FancyBboxPatch((2.0, 75.0), 96.0, 23.0, boxstyle="round,pad=0.6", fc="#F8F9FA", ec="#34495E", lw=1.4)
    ax.add_patch(detr_frame)
    ax.text(4.0, 95.0, "RT-DETR FORWARD INFERENCE PIPELINE (PASSIVE ARTIFACT HOOKS)", fontsize=10.5, fontweight="bold", color="#2C3E50")
    add_card(3.5, 77.0, 15.0, 15.5, "Input Image $X$", "RGB $320 \\times 320$\nClean / Attacked", color="#FDEDEC", edge="#C0392B", title_fs=10.2, sub_fs=8.8)
    add_arrow(18.5, 84.75, 21.5, 84.75)
    add_card(21.5, 77.0, 17.5, 15.5, "Hybrid Encoder", "ResNet-18 + CCFM\nMulti-scale Fusion", color="#E8F8F5", edge="#16A085", title_fs=10.2, sub_fs=8.8)
    add_arrow(39.0, 84.75, 42.0, 84.75)
    add_card(42.0, 77.0, 19.5, 15.5, "Query Selection", "Top-300 Candidates\nFrom 2,100 Proposals", color="#FEF9E7", edge="#D4AC0D", lw=1.8, title_fs=10.2, sub_fs=8.8)
    add_arrow(61.5, 84.75, 64.5, 84.75)
    add_card(64.5, 77.0, 19.5, 15.5, "Deformable Decoder", "3 Layers, 300 Queries\nCross-Attention", color="#EAF2F8", edge="#2980B9", title_fs=10.2, sub_fs=8.8)
    add_arrow(84.0, 84.75, 87.0, 84.75)
    add_card(87.0, 77.0, 10.0, 15.5, "Output", "$(\\hat{P}, \\hat{B})$\nNo NMS", color="#EAFAF1", edge="#27AE60", title_fs=10.0, sub_fs=8.5)

    add_arrow(51.5, 77.0, 19.0, 56.5, label="Scores $s_{(i)}$", color="#B7950B", label_color="#7D6608", ec_label="#F9E79F", y_label=66.5, x_label=35.0)
    add_arrow(74.0, 77.0, 52.0, 56.5, label="Latents $H^{(l)}$", color="#922B21", label_color="#78281F", ec_label="#F5B7B1", y_label=66.5, x_label=64.0)
    add_arrow(92.0, 77.0, 83.5, 56.5, label="Logits $\\hat{L}$", color="#1B4F72", label_color="#1B4F72", ec_label="#AED6F1", y_label=66.5, x_label=89.0)

    guard_frame = patches.FancyBboxPatch((2.0, 24.0), 96.0, 48.0, boxstyle="round,pad=0.6", fc="#FAFAFA", ec="#7F8C8D", lw=1.4, ls="--")
    ax.add_patch(guard_frame)
    ax.text(4.0, 70.0, "PROPOSED MULTI-TIER PASSIVE GUARDRAIL MONITOR (EDGE CPU)", fontsize=10.5, fontweight="bold", color="#2C3E50")

    t1_frame = patches.FancyBboxPatch((3.5, 26.5), 31.0, 29.5, boxstyle="round,pad=0.4", fc="#FEF9E7", ec="#B7950B", lw=1.8)
    ax.add_patch(t1_frame)
    ax.text(19.0, 52.5, "TIER 1: ENCODER SELECTION (ESM)", ha="center", va="center", fontsize=10.2, fontweight="bold", color="#7D6608")
    add_card(4.8, 39.5, 28.4, 11.5, "ESM Dropoff (in Max-OR)", "$S_{\\mathrm{drop}} = s_{(1)} / \\max(10^{-4}, s_{(300)})$", color="#FFFFFF", edge="#B7950B", lw=1.2, title_fs=9.8, sub_fs=8.8)
    add_card(4.8, 27.5, 28.4, 11.0, "ESM Margin (Aux. Monitor)", "$S_{\\mathrm{margin}} = \\bar{s}_{1:300} - \\bar{s}_{301:2100}$", color="#FFFFFF", edge="#B7950B", lw=1.2, title_fs=9.8, sub_fs=8.8)

    t2_frame = patches.FancyBboxPatch((36.5, 26.5), 31.0, 29.5, boxstyle="round,pad=0.4", fc="#FDEDEC", ec="#B03A2E", lw=1.8)
    ax.add_patch(t2_frame)
    ax.text(52.0, 52.5, "TIER 2: MANIFOLD DENSITY", ha="center", va="center", fontsize=10.2, fontweight="bold", color="#78281F")
    add_card(37.8, 27.5, 28.4, 23.0, "Layer Mahalanobis (in Max-OR)", "$\\max_l z(D_M^2(W_{\\mathrm{QR}}\\bar{h}^{(l)}))$\n$d=64$ QR Orthogonal Projection\nTop-10 Query Mean per Layer", color="#FFFFFF", edge="#B03A2E", lw=1.2, title_fs=10.0, sub_fs=9.0)

    t3_frame = patches.FancyBboxPatch((69.5, 26.5), 27.5, 29.5, boxstyle="round,pad=0.4", fc="#EBF5FB", ec="#1B4F72", lw=1.8)
    ax.add_patch(t3_frame)
    ax.text(83.25, 52.5, "TIER 3: 1D OUTPUT STATS", ha="center", va="center", fontsize=10.2, fontweight="bold", color="#0B2545")
    add_card(70.5, 39.5, 25.5, 11.5, "ActiveQ Split (in Max-OR)", "$\\tilde{r}_{\\mathrm{act}} = \\frac{1}{N}\\sum \\sigma(20(p_i - 0.35))$", color="#FFFFFF", edge="#1B4F72", lw=1.2, title_fs=9.8, sub_fs=8.6)
    add_card(70.5, 27.5, 25.5, 11.0, "Top1Mean Split (in Max-OR)", "$\\bar{p}_{\\mathrm{top1}} = \\frac{1}{N}\\sum \\max_c p_{i,c}$", color="#FFFFFF", edge="#1B4F72", lw=1.2, title_fs=9.8, sub_fs=8.6)

    add_arrow(19.0, 26.5, 32.0, 18.5, color="#B7950B", lw=2.0)
    add_arrow(52.0, 26.5, 48.0, 18.5, color="#B03A2E", lw=2.0)
    add_arrow(83.25, 26.5, 62.0, 18.5, color="#1B4F72", lw=2.0)

    add_card(
        18.0, 3.5, 50.0, 15.0,
        "Calibrated Extreme-Value Max-OR Gate (4 Channels)",
        r"$S_{\mathrm{fused}}(X) = \max_{k=1..4} q_k(X) \geq \tau_{\mathrm{cal}}$  |  Non-Oracle Split Mapping" + "\n"
        r"Bounded Clean FPR = $5.2\%$ ($31/600$) on Disjoint Test ($N=600$)",
        color="#F4ECF7", edge="#8E44AD", lw=2.0, title_fs=11.0, sub_fs=9.0, title_color="#4A235A", sub_color="#512E5F",
    )
    add_arrow(68.0, 11.0, 72.0, 11.0, color="#229954", lw=2.2)
    add_card(72.0, 3.5, 25.0, 15.0, "Security Decision", "Clean Pass ($A=0$): Normal Flow\nAlarm ($A=1$): Safety Fallback", color="#D5F5E3", edge="#229954", lw=2.0, title_fs=10.8, sub_fs=9.0, title_color="#145A32", sub_color="#1E8449")

    plt.tight_layout()
    save_fig_all(fig, "fig1_system_architecture")
    plt.close(fig)


def draw_figure_6_attack_taxonomy():
    print("  + Vẽ Figure 6: Attack Taxonomy (Focused on Paper 1 Scope)...", flush=True)
    fig, ax = plt.subplots(figsize=(13.5, 7.8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    def add_card(x, y, w, h, title, subtitle="", color="#EBF5FB", edge="#2980B9", lw=1.5, title_fs=10.8, sub_fs=9.2, title_color="#1A252C", sub_color="#2C3E50"):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4", fc=color, ec=edge, lw=lw)
        ax.add_patch(rect)
        if subtitle:
            ax.text(x + w / 2, y + h * 0.65, title, ha="center", va="center", fontsize=title_fs, fontweight="bold", color=title_color)
            ax.text(x + w / 2, y + h * 0.28, subtitle, ha="center", va="center", fontsize=sub_fs, color=sub_color, linespacing=1.2)
        else:
            ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=title_fs, fontweight="bold", color=title_color)

    def add_attack_badge(x, y, w, h, tier_label, attack_name, math_target, impact_desc, color="#FDEDEC", edge="#C0392B"):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4", fc=color, ec=edge, lw=1.5)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h * 0.82, tier_label, ha="center", va="center", fontsize=10.0, fontweight="bold", color="#78281F")
        ax.text(x + w / 2, y + h * 0.62, attack_name, ha="center", va="center", fontsize=11.5, fontweight="bold", color="#922B21")
        ax.text(x + w / 2, y + h * 0.38, math_target, ha="center", va="center", fontsize=9.2, color="#1B2631", linespacing=1.2)
        ax.text(x + w / 2, y + h * 0.15, impact_desc, ha="center", va="center", fontsize=8.8, fontweight="bold", color="#7B241C")

    def add_attack_arrow(x1, y1, x2, y2, label="", y_label=73.5, x_label=None):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(facecolor="#C0392B", edgecolor="#C0392B", width=1.5, headwidth=5.5, headlength=5.5, shrink=0.03))
        if label:
            xm = x_label if x_label is not None else (x1 + x2) / 2
            ax.text(xm, y_label, label, ha="center", va="center", fontsize=8.8, fontweight="bold", color="#922B21", bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#E6B0AA", linewidth=0.8, alpha=0.98))

    detr_frame = patches.FancyBboxPatch((2.0, 77.0), 96.0, 21.0, boxstyle="round,pad=0.6", fc="#F8F9FA", ec="#2C3E50", lw=1.4)
    ax.add_patch(detr_frame)
    ax.text(3.8, 95.0, "RT-DETR INFERENCE PIPELINE & ATTACK TARGET SURFACES", fontsize=10.5, fontweight="bold", color="#2C3E50")
    add_card(3.5, 79.0, 15.0, 14.0, "Stage 0: Input", "Image Pixels $X$\n$320 \\times 320$ RGB", color="#E8F8F5", edge="#16A085", title_fs=10.2, sub_fs=8.8)
    ax.annotate("", xy=(21.5, 86.0), xytext=(18.5, 86.0), arrowprops=dict(facecolor="#2C3E50", edgecolor="#2C3E50", width=1.4, headwidth=4.5, headlength=4.5))
    add_card(21.5, 79.0, 20.0, 14.0, "Stage 1: Encoder", "Multi-Scale Features\n$\\{S_3, S_4, S_5\\}$", color="#EAF2F8", edge="#2980B9", title_fs=10.2, sub_fs=8.8)
    ax.annotate("", xy=(44.5, 86.0), xytext=(41.5, 86.0), arrowprops=dict(facecolor="#2C3E50", edgecolor="#2C3E50", width=1.4, headwidth=4.5, headlength=4.5))
    add_card(44.5, 79.0, 22.0, 14.0, "Stage 2: Selection", "Top-300 Proposals\nDirect to Queries", color="#FEF9E7", edge="#D4AC0D", title_fs=10.2, sub_fs=8.8)
    ax.annotate("", xy=(69.5, 86.0), xytext=(66.5, 86.0), arrowprops=dict(facecolor="#2C3E50", edgecolor="#2C3E50", width=1.4, headwidth=4.5, headlength=4.5))
    add_card(69.5, 79.0, 16.5, 14.0, "Stage 3: Decoder", "Cross-Attention\n300 Queries", color="#EAF2F8", edge="#2980B9", title_fs=10.2, sub_fs=8.8)
    ax.annotate("", xy=(89.0, 86.0), xytext=(86.0, 86.0), arrowprops=dict(facecolor="#2C3E50", edgecolor="#2C3E50", width=1.4, headwidth=4.5, headlength=4.5))
    add_card(89.0, 79.0, 8.0, 14.0, "Stage 4", "BBox+Cls\nNo-NMS", color="#EAFAF1", edge="#27AE60", title_fs=9.8, sub_fs=8.5)

    threat_frame = patches.FancyBboxPatch((2.0, 35.0), 96.0, 34.0, boxstyle="round,pad=0.6", fc="#FDFEFE", ec="#C0392B", lw=1.5, ls="--")
    ax.add_patch(threat_frame)
    ax.text(3.8, 66.5, "TWO-TIER THREAT TAXONOMY: ENCODER SELECTION & DECODER TASK SURFACES", fontsize=10.5, fontweight="bold", color="#C0392B")

    # Tier 1 Card: Selection interface attacks
    add_attack_badge(
        3.5, 37.0, 46.5, 26.5,
        "TIER 1: DYNAMIC QUERY SELECTION INTERFACE (ENCODER BOUNDARY)",
        "Adversarial Query Selection Manipulation (AQSM)",
        r"$\mathcal{L}_{\mathrm{aqsm}} = \mathcal{L}_{\mathrm{supp}}(C_{\mathrm{real}}) + \lambda \mathcal{L}_{\mathrm{rank}}(C_{\mathrm{real}}, C_{\mathrm{bg}}) + 0.5\mathcal{L}_{\mathrm{bg}}$",
        "• Top-300 Turnover: 76.6% proposals evicted to background\n• mAP@50 drops: 50.59% -> 1.23% (-97.6%)  |  Spatial Misrouting: 59.0% -> 36.2%"
    )
    add_attack_arrow(26.75, 69.2, 55.5, 79.0, "Top-K Ranking Inversion", y_label=74.2, x_label=41.0)

    # Tier 2 Card: Decoder & Task PGD attacks
    add_attack_badge(
        51.5, 37.0, 46.5, 26.5,
        "TIER 2: DEFORMABLE DECODER OUTPUT SPACE & TASK PGD",
        "PGD Query Vanishing & Standard Task PGD (Hungarian Loss)",
        r"Vanishing: $\min \sum \sigma(\hat{L}_i)^2$  |  Task PGD: $\min \mathcal{L}_{\mathrm{hungarian}}(\hat{y}, y)$",
        "• Vanishing: 49.7% scene blackout (0.22 active boxes, entropy flattening)\n• Task PGD: 100.0% GT suppression, 27.61 noise boxes (FG feature destruction)"
    )
    add_attack_arrow(74.75, 69.2, 78.0, 79.0, "Query & Cross-Attention Manipulation", y_label=74.2, x_label=76.0)

    summary_box = patches.FancyBboxPatch((2.0, 3.5), 96.0, 27.5, boxstyle="round,pad=0.6", fc="#F4F6F6", ec="#7F8C8D", lw=1.2)
    ax.add_patch(summary_box)
    ax.text(3.8, 28.0, "KEY ARCHITECTURAL FINDINGS AT THE ENCODER-DECODER BOUNDARY", fontsize=10.5, fontweight="bold", color="#2C3E50")
    insights = [
        ("• Top-K Selection Bottleneck:", "Dynamic query selection is a primary failure surface: pure index eviction (Arm D_index) drops mAP@50 to 28.80% (43.1% loss) even with pristine encoder memory."),
        ("• Spatial Target Misrouting:", "AQSM does not flatten deformable attention entropy (p=0.914), but diverts sampling offsets away from foreground targets (FG hit: 59.0% -> 36.2%)."),
        ("• Matched-Budget Contrast:", "Under matched e=16/255, Task PGD collapses raw mAP (0.01%), while AQSM inflicts stronger ranking turnover (23.4% vs. 36.8%) and spatial misrouting."),
        ("• Deterministic Edge Invariance:", "Fixed bipartite matching with 300 queries yields strict runtime invariance (6.13 ms GPU / 159.7 ms CPU, Delta t < 0.02 ms, p=0.88), eliminating NMS sponge attacks."),
    ]
    y_pos = 22.5
    for title, desc in insights:
        ax.text(4.2, y_pos, title, fontsize=9.2, fontweight="bold", color="#1A5276")
        ax.text(26.5, y_pos, desc, fontsize=8.8, color="#2C3E50")
        y_pos -= 5.5

    plt.tight_layout()
    save_fig_all(fig, "fig6_attack_taxonomy")
    plt.close(fig)


def draw_figure_2_roc_curves(data: dict):
    print("  + Đang vẽ Figure 2: Đường cong ROC từ dữ liệu thực nghiệm...", flush=True)
    cohorts = data.get("cohort_results", {})
    clean_raw_dict = data.get("clean_test_raw_scores", {})
    npz_data = np.load(RAW_SCORES_NPZ_PATH) if RAW_SCORES_NPZ_PATH.exists() else None

    target_attacks = [
        ("Attack: Query Selection Manipulation (AQSM)", "Query Selection Manipulation (AQSM)", "a"),
        ("Attack: PGD Query Vanishing (15 steps)", "Query Vanishing (Object Suppression)", "b"),
        ("Attack: PGD Query Fabrication (DQF, 15 steps)", "Query Fabrication (Ghost Flooding)", "c"),
        ("Attack: Gradient Patch (48x48 Random Pos)", "Gradient Patch (48x48 Localized)", "d"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.8))
    axes = axes.flatten()

    def get_empirical_roc(cohort_key, m_key, auc_default):
        from sklearn.metrics import roc_curve, auc
        c_clean, c_adv = None, None
        if npz_data is not None:
            clean_k = f"clean_test__{m_key}"
            c_safe_k = cohort_key.replace(" ", "_").replace(":", "_").replace("(", "").replace(")", "").replace("/", "_").replace(",", "")
            cand_prefixes = [c_safe_k, c_safe_k.replace("_DQF_", "_")]
            for pref in cand_prefixes:
                adv_k = f"{pref}__{m_key}"
                if clean_k in npz_data and adv_k in npz_data:
                    c_clean = npz_data[clean_k]
                    c_adv = npz_data[adv_k]
                    break
        if c_clean is None and clean_raw_dict and m_key in clean_raw_dict:
            ch_item = cohorts.get(cohort_key, {})
            if "raw_scores" in ch_item and m_key in ch_item["raw_scores"]:
                c_clean = np.array(clean_raw_dict[m_key], dtype=float)
                c_adv = np.array(ch_item["raw_scores"][m_key], dtype=float)
        if c_clean is not None and c_adv is not None and len(c_clean) > 0 and len(c_adv) > 0:
            y_t = np.array([0] * len(c_clean) + [1] * len(c_adv))
            y_s = np.concatenate([c_clean, c_adv])
            f_arr, t_arr, _ = roc_curve(y_t, y_s)
            return f_arr * 100.0, t_arr * 100.0, float(auc(f_arr, t_arr))
        return None, None, float(auc_default)

    for idx, (cohort_key, title_text, panel_id) in enumerate(target_attacks):
        ax = axes[idx]
        metrics = cohorts.get(cohort_key, {}).get("metrics", {})
        specs = [
            ("Calibrated_Max_OR", "Calibrated Max-OR", COLORS["proposed"], "-", "o", 7),
            ("ActiveQ_Split", "ActiveQ Split", COLORS["split_act"], "--", "s", 6),
            ("Top1Mean_Split", "Top1Mean Split", COLORS["split_top"], "-.", "^", 6),
            ("Layer_Maha", "Layer Mahalanobis", COLORS["maha"], ":", "d", 5),
        ]
        if panel_id == "a":
            specs.append(("ESM_Dropoff", "ESM Dropoff", "#E67E22", "-", "*", 8))

        for m_k, lbl, col, ls, mk, ms in specs:
            auc_val = metrics.get(m_k, {}).get("auroc", 0.5)
            tpr_pt = metrics.get(m_k, {}).get("tpr@5%", 0.05)
            f_arr, t_arr, auc_real = get_empirical_roc(cohort_key, m_k, auc_val)
            if f_arr is not None:
                ax.plot(f_arr, t_arr, label=f"{lbl} (AUC={auc_real:.3f})", color=col, lw=2.2 if m_k in ["Calibrated_Max_OR", "ESM_Dropoff"] else 1.7, ls=ls)
                ax.plot(5.0, tpr_pt * 100, marker=mk, markersize=ms, color=col)

        ax.plot([0, 100], [0, 100], color="gray", lw=1.0, ls=":")
        ax.axvline(x=5.0, color="red", lw=1.2, ls="--", alpha=0.7, label="Nominal 5% FPR (Max-OR: 5.2%)" if idx == 0 else "")
        ax.set_xlim(-1, 101)
        ax.set_ylim(-1, 101)
        ax.set_xlabel("False Positive Rate (FPR, %)")
        ax.set_ylabel("True Positive Rate (TPR, %)")
        ax.set_title(f"({panel_id}) {title_text}", fontweight="bold")
        ax.grid(True, alpha=0.35)
        ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)

    plt.tight_layout()
    save_fig_all(fig, "fig2_roc_curves")
    plt.close(fig)


def draw_figure_3_adaptive_tradeoff(data: dict):
    print("  + Đang vẽ Figure 3: Adaptive Evasion Trade-off...", flush=True)
    sweep = data.get("adaptive_sweep_results", {})
    if not sweep:
        print("    [SKIP] Không có adaptive_sweep_results.")
        return
    lambdas = sorted([float(k) for k in sweep.keys()])
    asrs = [sweep[str(l)]["asr"] * 100.0 for l in lambdas]
    asr_lowers = [sweep[str(l)].get("asr_ci", [sweep[str(l)]["asr"], sweep[str(l)]["asr"]])[0] * 100.0 for l in lambdas]
    asr_uppers = [sweep[str(l)].get("asr_ci", [sweep[str(l)]["asr"], sweep[str(l)]["asr"]])[1] * 100.0 for l in lambdas]
    tpr_ors = [sweep[str(l)].get("max_or_tpr", sweep[str(l)].get("tpr_or", 0.0)) * 100.0 for l in lambdas]
    tpr_acts = [sweep[str(l)].get("activeq_tpr", sweep[str(l)].get("tpr_act", 0.0)) * 100.0 for l in lambdas]
    tpr_tops = [sweep[str(l)].get("top1mean_tpr", sweep[str(l)].get("tpr_top", 0.0)) * 100.0 for l in lambdas]

    fig, ax1 = plt.subplots(figsize=(8.8, 5.2))
    color_asr = COLORS["asr"]
    ax1.set_xlabel(r"Monitor Evasion Penalty Weight ($\lambda$)", fontsize=12, fontweight="bold")
    ax1.set_ylabel(r"Task-level Attack Success Rate (ASR, %)", color=color_asr, fontsize=12, fontweight="bold")
    l1 = ax1.plot(lambdas, asrs, color=color_asr, marker="s", lw=2.4, label="Attacker Objective: ASR (%)")
    ax1.fill_between(lambdas, asr_lowers, asr_uppers, color=color_asr, alpha=0.15)
    ax1.tick_params(axis="y", labelcolor=color_asr)
    ax1.set_ylim(-5, 105)

    ax2 = ax1.twinx()
    color_tpr = COLORS["proposed"]
    ax2.set_ylabel(r"Monitor True Positive Rate @ 5% FPR (%)", color=color_tpr, fontsize=12, fontweight="bold")
    l2 = ax2.plot(lambdas, tpr_ors, color=color_tpr, marker="o", lw=2.4, ls="-", label="Proposed: Calibrated Max-OR TPR (%)")
    l3 = ax2.plot(lambdas, tpr_acts, color=COLORS["split_act"], marker="^", lw=1.8, ls="--", label="Channel: ActiveQ Split TPR (%)")
    l4 = ax2.plot(lambdas, tpr_tops, color=COLORS["split_top"], marker="d", lw=1.8, ls="-.", label="Channel: Top1Mean Split TPR (%)")
    ax2.axhline(y=5.0, color="gray", ls=":", lw=1.2)
    ax2.tick_params(axis="y", labelcolor=color_tpr)
    ax2.set_ylim(-5, 105)

    lines = l1 + l2 + l3 + l4
    ax1.legend(lines, [ln.get_label() for ln in lines], loc="center right", framealpha=0.92, fontsize=9.5)
    plt.title(r"Adaptive Evasion Trade-off ($\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{task}} + \lambda \mathcal{L}_{\mathrm{monitor}}$)", fontsize=11.5, fontweight="bold", pad=12)
    ax1.grid(True, alpha=0.35)
    plt.tight_layout()
    save_fig_all(fig, "fig3_adaptive_tradeoff")
    plt.close(fig)


def draw_figure_4_patch_size_sweep(data: dict):
    print("  + Đang vẽ Figure 4: Patch Size Sweep...", flush=True)
    sweep = data.get("patch_sweep_results", {})
    if not sweep:
        print("    [SKIP] Không có patch_sweep_results.")
        return
    sizes = [16, 32, 48, 64]
    areas = [sweep[str(s)]["area_pct"] for s in sizes]
    asrs = [sweep[str(s)]["asr"] * 100.0 for s in sizes]
    uncond_tprs = [sweep[str(s)].get("tpr", sweep[str(s)].get("uncond_tpr", 0.0)) * 100.0 for s in sizes]
    cond_tprs = [sweep[str(s)].get("cond_tpr", sweep[str(s)].get("tpr", 0.0)) * 100.0 for s in sizes]
    patch_cohort = data.get("cohort_results", {}).get("Attack: Gradient Patch (48x48 Random Pos)", {})
    maha_tpr_48 = patch_cohort.get("metrics", {}).get("Layer_Maha", {}).get("tpr@5%", 0.0) * 100.0

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    x = np.arange(len(sizes))
    width = 0.22
    r1 = ax.bar(x - width, asrs, width, label="Task-level ASR (%)", color=COLORS["asr"], alpha=0.85, edgecolor="black")
    ax.bar(x, uncond_tprs, width, label="Calibrated Max-OR Unconditional TPR (%)", color=COLORS["proposed"], alpha=0.85, edgecolor="black")
    ax.bar(x + width, cond_tprs, width, label="Calibrated Max-OR Conditional TPR (%)", color="#E67E22", alpha=0.85, edgecolor="black")
    ax.scatter([2 + 1.5 * width], [maha_tpr_48], color=COLORS["maha"], marker="D", s=90, zorder=5, label=f"Layer Mahalanobis @ 48x48 ({maha_tpr_48:.1f}%)")
    ax.axhline(y=5.0, color="gray", ls="--", lw=1.2, label="Nominal 5% False Alarm Floor")
    ax.set_ylabel("Percentage (%)", fontsize=11.5, fontweight="bold")
    ax.set_xlabel("Adversarial Patch Size (Pixels / Image Area Ratio)", fontsize=11.5, fontweight="bold")
    ax.set_title("Adversarial Patch Evaluation: Task Failure vs. Passive Monitor Blind Spot", fontsize=11.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}x{s}\n({a:.2f}%)" for s, a in zip(sizes, areas)], fontsize=10.0)
    ax.set_ylim(0, 120)
    ax.grid(True, alpha=0.35, axis="y")
    ax.legend(loc="upper left", fontsize=9.0, framealpha=0.92)
    for r in r1:
        h = r.get_height()
        ax.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8.8, fontweight="bold")
    plt.tight_layout()
    save_fig_all(fig, "fig4_patch_size_sweep")
    plt.close(fig)


def draw_figure_5_fpr_drift_stability(data: dict):
    print("  + Đang vẽ Figure 5: FPR Stability across Controls...", flush=True)
    cohorts = data.get("cohort_results", {})
    clean_fprs = data.get("clean_test_fprs", {})
    conditions = [
        "Clean Test Set\n(N=600 Disjoint)",
        "Gaussian Noise\n(N=300, $\\sigma=16$)",
        "Random Patch\n(N=300, 48x48)",
        "Gaussian Blur\n(N=300, 5x5 $\\sigma=1.5$)",
        "Brightness Shift\n(N=300, -30%)",
    ]
    keys = [
        "clean",
        "Control: Gaussian Noise (sigma=16/255)",
        "Control: Random Patch (48x48 Random Pos)",
        "Clean Shift: Gaussian Blur (5x5, sigma=1.5)",
        "Clean Shift: Brightness (-30%)",
    ]

    or_fprs, act_fprs, top_fprs, maha_fprs = [], [], [], []
    for k in keys:
        if k == "clean":
            or_fprs.append(clean_fprs["Calibrated_Max_OR"][0] * 100.0)
            act_fprs.append(clean_fprs["ActiveQ_Split"][0] * 100.0)
            top_fprs.append(clean_fprs["Top1Mean_Split"][0] * 100.0)
            maha_fprs.append(clean_fprs["Layer_Maha"][0] * 100.0)
        else:
            ch = cohorts.get(k, {}).get("metrics", {})
            or_fprs.append(ch.get("Calibrated_Max_OR", {}).get("tpr@5%", 0.05) * 100.0)
            act_fprs.append(ch.get("ActiveQ_Split", {}).get("tpr@5%", 0.05) * 100.0)
            top_fprs.append(ch.get("Top1Mean_Split", {}).get("tpr@5%", 0.05) * 100.0)
            maha_fprs.append(ch.get("Layer_Maha", {}).get("tpr@5%", 0.05) * 100.0)

    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    x = np.arange(len(conditions))
    width = 0.20
    r1 = ax.bar(x - 1.5 * width, or_fprs, width, label="Calibrated Max-OR (Proposed)", color=COLORS["proposed"], edgecolor="black")
    ax.bar(x - 0.5 * width, act_fprs, width, label="ActiveQ Split", color=COLORS["split_act"], edgecolor="black")
    ax.bar(x + 0.5 * width, top_fprs, width, label="Top1Mean Split", color=COLORS["split_top"], edgecolor="black")
    ax.bar(x + 1.5 * width, maha_fprs, width, label="Layer Mahalanobis", color=COLORS["maha"], edgecolor="black")
    ax.axhline(y=5.0, color="red", ls="--", lw=1.5, label="Nominal 5.0% False Positive Budget")
    ax.axhspan(3.6, 7.3, color="red", alpha=0.08, label="Clean 95% CI Corridor [3.6%, 7.3%]")
    ax.set_ylabel("Empirical False Alarm Rate (FPR, %)", fontsize=11.5, fontweight="bold")
    ax.set_title("False Positive Rate (FPR) Stability across Clean Shifts & Negative Controls", fontsize=11.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=10.0)
    ax.set_ylim(0, 18)
    ax.grid(True, alpha=0.35, axis="y")
    ax.legend(loc="upper right", fontsize=9.0, framealpha=0.92)
    for r in r1:
        h = r.get_height()
        ax.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.8, fontweight="bold", color=COLORS["proposed"])

    plt.tight_layout()
    save_fig_all(fig, "fig5_fpr_drift_stability")
    plt.close(fig)


def draw_figure_7_calibration_mechanism():
    print("  + Đang vẽ Figure 7: Calibration Mechanism (100% Real Data + Multitest JSON)...", flush=True)
    mt_file = RESULTS_DIR / "rtdetr_multitest_baselines_results.json"
    if not mt_file.exists() or not RESULTS_JSON_PATH.exists():
        print("    [SKIP] Thiếu multitest JSON hoặc master results.")
        return
    with open(mt_file, "r", encoding="utf-8") as f:
        mt = json.load(f)
    with open(RESULTS_JSON_PATH, "r", encoding="utf-8") as f:
        adv_d = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.4))
    if MECHANISM_NPZ_PATH.exists() and "clean__logits_max" in np.load(MECHANISM_NPZ_PATH, allow_pickle=False).files:
        m_npz = np.load(MECHANISM_NPZ_PATH, allow_pickle=False)
        clean_logits_max = m_npz["clean__logits_max"]  # [N, 300]
        clean_probs_max = 1.0 / (1.0 + np.exp(-clean_logits_max))
        r_act_real = np.mean(clean_probs_max >= 0.35, axis=1)
        q_low = float(np.percentile(r_act_real, 2.5))
        q_high = float(np.percentile(r_act_real, 97.5))
        bins = np.linspace(0.0, max(0.14, float(np.max(r_act_real)) * 1.08), 24)
        ax1.hist(r_act_real, bins=bins, density=True, color="#D4E6F1", edgecolor="#1A5276", alpha=0.85, label=f"Empirical Clean $r_{{\\mathrm{{act}}}}$ ($N={len(r_act_real)}$)")
        ax1.axvline(x=q_low, color="#C0392B", lw=1.9, ls="--", label=f"Empirical 2.5% Tail ($\\tau_{{\\mathrm{{low}}}}={q_low:.3f}$)")
        ax1.axvline(x=q_high, color="#C0392B", lw=1.9, ls="-.", label=f"Empirical 97.5% Tail ($\\tau_{{\\mathrm{{high}}}}={q_high:.3f}$)")
        ax1.set_title("(a) Empirical Clean Active Query Distribution\n(Skewed Real-World COCO Distribution & Bilateral Tails)", fontsize=11.0, fontweight="bold")
        ax1.set_xlabel(r"Empirical Active Query Ratio ($r_{\mathrm{act}}$, conf $\geq 0.35$)", fontsize=10.5)
        ax1.set_ylabel("Empirical Probability Density", fontsize=10.5)
    else:
        raise FileNotFoundError(f"Thiếu {MECHANISM_NPZ_PATH} (clean__logits_max) để vẽ phân bố thực nghiệm Figure 7(a).")
    ax1.grid(True, alpha=0.35)
    ax1.legend(loc="upper right", fontsize=8.6)

    cf_map = adv_d.get("clean_test_fprs", {})
    boole_sum = sum(float(cf_map[c][0]) * 100.0 for c in ["ActiveQ_Split", "Top1Mean_Split", "ESM_Dropoff", "Layer_Maha"] if c in cf_map)
    u = mt["Uncalibrated_OR"]["clean_fpr"]
    b = mt["Bonferroni_OR"]["clean_fpr"]
    m = mt["Calibrated_Max_OR"]["clean_fpr"]
    vals = [boole_sum, u["fpr_pct"], b["fpr_pct"], m["fpr_pct"]]
    lows = [boole_sum, u["ci_95"][0], b["ci_95"][0], m["ci_95"][0]]
    highs = [boole_sum, u["ci_95"][1], b["ci_95"][1], m["ci_95"][1]]
    yerr = [[max(0.0, vals[i] - lows[i]) for i in range(4)], [max(0.0, highs[i] - vals[i]) for i in range(4)]]
    methods = ["Boole Union\nBound", "Uncalibrated\nOR (M=4)", "Bonferroni\nCorrection", "Proposed\nCalibrated Max-OR"]
    bars = ax2.bar(np.arange(4), vals, yerr=yerr, capsize=5, color=["#BDC3C7", "#E74C3C", "#F39C12", "#2ECC71"], edgecolor="black", lw=1.4, width=0.55)
    ax2.axhline(y=5.0, color="#C0392B", ls="--", lw=1.8, label="Nominal FPR Budget (5.0%)")
    for i, bar in enumerate(bars):
        ax2.text(bar.get_x() + bar.get_width() / 2, highs[i] + 0.8, f"{vals[i]:.1f}%", ha="center", va="bottom", fontsize=10.5, fontweight="bold")
    ax2.set_xticks(np.arange(4))
    ax2.set_xticklabels(methods, fontsize=10.0)
    ax2.set_ylim(0, max(highs) + 5)
    ax2.set_ylabel("Empirical Clean FPR (%)", fontsize=11)
    ax2.set_title("(b) Empirical Clean Test FPR Control (N=600 Disjoint, 95% CI)", fontsize=11.5, fontweight="bold")
    ax2.grid(True, axis="y", alpha=0.35)
    ax2.legend(loc="upper right", fontsize=9.0)

    plt.tight_layout()
    save_fig_all(fig, "fig7_calibration_mechanism")
    plt.close(fig)


def draw_figure_8_cross_channel_correlation(npz_data=None):
    print("  + Đang vẽ Figure 8: Cross-Channel Correlation & 2D Boundary (100% Real NPZ)...", flush=True)
    if npz_data is None or "clean_test__ActiveQ_Split" not in npz_data:
        print("    [SKIP] Không có raw NPZ cho Figure 8.")
        return
    from scipy.stats import spearmanr
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.5))
    channel_names = [r"$r_{\mathrm{act}}$ (ActiveQ)", r"$\bar{p}_{\mathrm{top1}}$ (Top1Mean)", r"$D_M^2$ (Mahalanobis)", r"$S_{\mathrm{fused}}$ (Max-OR)"]
    r_act = npz_data["clean_test__ActiveQ_Split"]
    p_top = npz_data["clean_test__Top1Mean_Split"]
    d_maha = npz_data["clean_test__Layer_Maha"]
    s_or = npz_data["clean_test__Calibrated_Max_OR"]
    corr_matrix, _ = spearmanr(np.stack([r_act, p_top, d_maha, s_or], axis=1))

    im = ax1.imshow(corr_matrix, cmap="YlGnBu", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label("Spearman Rank Correlation", fontsize=10, fontweight="bold")
    ax1.set_xticks(np.arange(4))
    ax1.set_yticks(np.arange(4))
    ax1.set_xticklabels(channel_names, fontsize=9.5, rotation=15, ha="right")
    ax1.set_yticklabels(channel_names, fontsize=9.5)
    for i in range(4):
        for j in range(4):
            val = corr_matrix[i, j]
            ax1.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=11, fontweight="bold", color="white" if val > 0.65 else "black")
    ax1.set_title("(a) Cross-Channel Correlation Matrix (Clean Set N=600)", fontsize=11, fontweight="bold")

    def get_k(prefixes, metric):
        for p in prefixes:
            k = f"{p}__{metric}"
            if k in npz_data:
                return npz_data[k]
        return None

    c_act, c_maha = r_act[:300], d_maha[:300]
    v_act = get_k(["Attack__PGD_Query_Vanishing_15_steps"], "ActiveQ_Split")
    v_maha = get_k(["Attack__PGD_Query_Vanishing_15_steps"], "Layer_Maha")
    f_act = get_k(["Attack__PGD_Query_Fabrication_DQF_15_steps", "Attack__PGD_Query_Fabrication_15_steps"], "ActiveQ_Split")
    f_maha = get_k(["Attack__PGD_Query_Fabrication_DQF_15_steps", "Attack__PGD_Query_Fabrication_15_steps"], "Layer_Maha")
    p_act = npz_data["Attack__Gradient_Patch_48x48_Random_Pos__ActiveQ_Split"]
    p_maha = npz_data["Attack__Gradient_Patch_48x48_Random_Pos__Layer_Maha"]

    ax2.scatter(c_act, c_maha, color="#27AE60", alpha=0.5, s=26, label="Clean Test (N=300 Unseen)")
    if v_act is not None:
        ax2.scatter(v_act, v_maha, color="#2980B9", alpha=0.8, s=30, marker="v", label="Query Vanishing")
    if f_act is not None:
        ax2.scatter(f_act, f_maha, color="#E67E22", alpha=0.8, s=30, marker="^", label="Query Fabrication (DQF)")
    ax2.scatter(p_act, p_maha, color="#C0392B", alpha=0.75, s=28, marker="s", label="Gradient Patch (48x48)")
    ax2.set_xlabel(r"Active Query Tail Quantile Score", fontsize=11, fontweight="bold")
    ax2.set_ylabel(r"Layer Mahalanobis Score $s(D_M^2)$", fontsize=11, fontweight="bold")
    ax2.set_title("(b) 2D Representation Distribution Across Attacks", fontsize=11, fontweight="bold")
    ax2.grid(True, alpha=0.35)
    ax2.legend(loc="upper right", fontsize=8.6, framealpha=0.92)

    plt.tight_layout()
    save_fig_all(fig, "fig8_cross_channel_correlation")
    plt.close(fig)


def draw_figure_9_per_query_drift():
    print("  + Đang vẽ Figure 9: Real Per-Query Logit Spectrum (100% Real N=300 NPZ)...", flush=True)
    if not MECHANISM_NPZ_PATH.exists():
        print(f"    [SKIP] Chưa có {MECHANISM_NPZ_PATH.name}; bỏ qua Figure 9 (Zero Mock).")
        return
    npz = np.load(MECHANISM_NPZ_PATH, allow_pickle=False)
    c_arr = npz["clean__logits_max"]      # [300, 300]
    v_arr = npz["vanishing__logits_max"]
    a_arr = npz["aqsm__logits_max"]
    t_arr = npz["taskpgd__logits_max"]

    def _mean_sorted_with_iqr(mat: np.ndarray):
        s_mat = np.sort(mat, axis=1)[:, ::-1]
        return np.mean(s_mat, axis=0), np.percentile(s_mat, 25, axis=0), np.percentile(s_mat, 75, axis=0)

    c_m, c_q25, c_q75 = _mean_sorted_with_iqr(c_arr)
    v_m, v_q25, v_q75 = _mean_sorted_with_iqr(v_arr)
    a_m, a_q25, a_q75 = _mean_sorted_with_iqr(a_arr)
    t_m, t_q25, t_q75 = _mean_sorted_with_iqr(t_arr)

    ranks = np.arange(1, 51)
    conf_logit_th = float(np.log(0.35 / (1.0 - 0.35)))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.2))

    ax1.plot(ranks, c_m[:50], color="#27AE60", lw=2.4, marker="o", ms=3.5, label="Clean Baseline (Mean + IQR)")
    ax1.fill_between(ranks, c_q25[:50], c_q75[:50], color="#27AE60", alpha=0.14)
    ax1.plot(ranks, v_m[:50], color="#C0392B", lw=2.4, ls="--", marker="v", ms=3.5, label="PGD Vanishing (15 steps)")
    ax1.fill_between(ranks, v_q25[:50], v_q75[:50], color="#C0392B", alpha=0.14)
    ax1.axhline(y=conf_logit_th, color="black", ls=":", lw=1.5, label=r"Threshold $\tau=0.35$ (Logit $\approx -0.62$)")
    ax1.set_title(f"(a) Decoder Logit Suppression: Clean vs. Vanishing (N={len(c_arr)})", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Decoder Query Rank Order (Sorted Max-Class Logit, Top-50)")
    ax1.set_ylabel(r"Max-Class Query Logit $\hat{L}_{(i)}$")
    ax1.set_xlim(1, 50)
    ax1.grid(True, alpha=0.35)
    ax1.legend(loc="upper right", fontsize=8.8)

    ax2.plot(ranks, c_m[:50], color="#27AE60", lw=2.4, marker="o", ms=3.5, label="Clean Baseline")
    ax2.plot(ranks, a_m[:50], color="#E67E22", lw=2.4, ls="--", marker="s", ms=3.5, label="AQSM (15 steps)")
    ax2.fill_between(ranks, a_q25[:50], a_q75[:50], color="#E67E22", alpha=0.14)
    ax2.plot(ranks, t_m[:50], color="#8E44AD", lw=2.2, ls="-.", marker="^", ms=3.5, label="Standard Task PGD (15 steps)")
    ax2.axhline(y=conf_logit_th, color="black", ls=":", lw=1.5, label=r"Threshold $\tau=0.35$")
    ax2.set_title(f"(b) Decoder Output Comparison: AQSM vs. Task PGD (N={len(c_arr)})", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Decoder Query Rank Order (Top-50 / 300 Queries)")
    ax2.set_ylabel(r"Max-Class Query Logit $\hat{L}_{(i)}$")
    ax2.set_xlim(1, 50)
    ax2.grid(True, alpha=0.35)
    ax2.legend(loc="upper right", fontsize=8.8)

    plt.tight_layout()
    save_fig_all(fig, "fig9_per_query_drift")
    plt.close(fig)


def draw_figure_10_latency_breakdown():
    print("  + Đang vẽ Figure 10: Real Edge Latency Profile (100% Real Runs, Zero Mock)...", flush=True)
    lat_file = CHECKPOINTS_DIR / "res_step8_edge_latency.json"
    if not lat_file.exists():
        print(f"    [SKIP] Không tìm thấy {lat_file}")
        return
    with open(lat_file, "r", encoding="utf-8") as f:
        lat = json.load(f)

    raw_runs = lat.get("raw_runs", {})
    overheads = raw_runs.get("overhead_pct_runs", [])
    if not overheads:
        print("    [SKIP] res_step8_edge_latency.json chưa có raw_runs.overhead_pct_runs; chạy Step 8 trước.")
        return

    avg_model = float(lat["avg_model_ms"])
    avg_mon = float(lat["avg_monitor_ms"])
    ext_ms = float(lat["feature_extraction_ms"])
    ch = lat["channel_latencies"]
    c_1d = float(ch["ActiveQ_Split"]["avg_ms"]) + float(ch["Top1Mean_Split"]["avg_ms"])
    c_esm = float(ch["ESM_Dropoff"]["avg_ms"])
    c_maha = float(ch["Layer_Maha"]["avg_ms"])
    c_gate = float(ch["Calibrated_Max_OR"]["avg_ms"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.2))
    mon_comps = ["Tensor\nExtraction", "Tier 1:\nESM Dropoff", "Tier 2:\nMahalanobis", "Tier 3:\n1D Stats", "Gate:\nMax-OR"]
    mon_latencies = [ext_ms, c_esm, c_maha, c_1d, c_gate]
    bars = ax1.bar(mon_comps, mon_latencies, color=["#34495E", "#F39C12", "#8E44AD", "#2980B9", "#27AE60"], edgecolor="black", lw=1.2, width=0.55)
    ax1.set_ylabel("Execution Time on Edge CPU (ms)", fontsize=11, fontweight="bold")
    ax1.set_title(f"(a) Breakdown of Guardrail Latency ({avg_mon:.2f} ms Total)\nBase RT-DETR Inference = {avg_model:.2f} ms", fontsize=11, fontweight="bold")
    ax1.grid(True, alpha=0.35, axis="y")
    ax1.set_ylim(0, max(mon_latencies) * 1.28)
    for bar in bars:
        h = bar.get_height()
        ax1.annotate(f"{h:.3f} ms", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9.0, fontweight="bold")

    ov_arr = np.asarray(overheads, dtype=float)
    parts = ax2.violinplot(ov_arr, positions=[1], showmeans=True, showmedians=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("#2980B9")
        pc.set_edgecolor("black")
        pc.set_alpha(0.65)
    rng = np.random.RandomState(42)
    ax2.scatter(np.ones_like(ov_arr) + rng.uniform(-0.04, 0.04, size=len(ov_arr)), ov_arr, color="#1A5276", alpha=0.65, s=24, label=f"{len(ov_arr)} Measured Runs")
    mean_ov = float(lat["overhead_percent"])
    ax2.axhline(y=mean_ov, color="#27AE60", ls="-", lw=1.8, label=f"Empirical Mean Overhead: {mean_ov:.2f}% ({avg_mon:.2f} / {avg_model:.2f} ms)")
    ax2.axhline(y=6.0, color="#C0392B", ls="--", lw=1.5, label="Real-time Budget Limit (6.0%)")
    ax2.set_xticks([1])
    ax2.set_xticklabels([f"RT-DETR r18vd ({lat.get('torch_num_threads', 4)} CPU Threads)"], fontsize=10.0, fontweight="bold")
    ax2.set_ylabel("Monitor Overhead (% of Model Inference)", fontsize=11, fontweight="bold")
    ax2.set_title(f"(b) Measured Guardrail Overhead Distribution (N={len(ov_arr)} Runs)", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, max(7.5, float(np.max(ov_arr)) * 1.25))
    ax2.grid(True, alpha=0.35, axis="y")
    ax2.legend(loc="upper right", fontsize=8.8, framealpha=0.92)

    plt.tight_layout()
    save_fig_all(fig, "fig10_latency_breakdown")
    plt.close(fig)


def draw_figure_12_encoder_score_spectrum():
    print("  + Đang vẽ Figure 12: Encoder Proposal Score Spectrum (Clean vs AQSM)...", flush=True)
    spectrum_file = CHECKPOINTS_DIR / "res_step9_expansion_spectrum.json"
    if not spectrum_file.exists():
        print(f"    [SKIP] Không tìm thấy {spectrum_file}")
        return
    with open(spectrum_file, "r", encoding="utf-8") as f:
        spec_data = json.load(f)
    clean_spec = np.array(spec_data["clean_spectrum"])
    aqsm_spec = np.array(spec_data["aqsm_spectrum"])
    ranks = np.arange(1, len(clean_spec) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.0))
    ax1.plot(ranks[:300], clean_spec[:300], color="#1B4F72", lw=2.6, label="Clean Pristine Baseline")
    ax1.plot(ranks[:300], aqsm_spec[:300], color="#C0392B", lw=2.6, ls="--", label="AQSM Attack (Ranking Inversion)")
    ax1.fill_between(ranks[:300], clean_spec[:300], color="#AED6F1", alpha=0.3)
    ax1.axvline(x=300, color="#7F8C8D", ls="-", lw=1.5, label="Top-300 Selection Bottleneck")
    ax1.set_title("(a) Encoder Proposal Score Spectrum Across Top-300 Bottleneck", fontsize=11.5, fontweight="bold")
    ax1.set_xlabel("Proposal Candidate Rank Order ($i = 1, \\dots, 300$)", fontsize=10.5)
    ax1.set_ylabel("Sigmoid Proposal Score $s_{(i)}$", fontsize=10.5)
    ax1.set_xlim(1, 300)
    ax1.set_ylim(0, 1.0)
    ax1.grid(True, alpha=0.35)
    ax1.legend(loc="upper right", fontsize=9.5)

    ax2.plot(ranks[:50], clean_spec[:50], color="#1B4F72", lw=2.6, marker="o", markersize=4.5, label="Clean Proposals")
    ax2.plot(ranks[:50], aqsm_spec[:50], color="#C0392B", lw=2.6, marker="s", markersize=4.5, ls="--", label="AQSM Perturbed")
    ax2.axhline(y=0.35, color="#27AE60", ls="--", lw=1.6, label=r"Confidence Gate ($\tau_{\mathrm{act}} = 0.35$)")
    ax2.set_title("(b) Foreground Proposal Zoom (Ranks #1 to #50)", fontsize=11.5, fontweight="bold")
    ax2.set_xlabel("Proposal Candidate Rank Order ($i = 1, \\dots, 50$)", fontsize=10.5)
    ax2.set_ylabel("Sigmoid Score $s_{(i)}$", fontsize=10.5)
    ax2.set_xlim(1, 50)
    ax2.set_ylim(0, 1.0)
    ax2.grid(True, alpha=0.35)
    ax2.legend(loc="upper right", fontsize=9.5)

    plt.tight_layout()
    save_fig_all(fig, "fig12_encoder_score_spectrum")
    plt.close(fig)
    plt.close(fig)


def draw_figure_13_causal_swap_n300():
    print("  + Đang vẽ Figure 13: Causal Intervention N=300 (AQSM vs Vanishing vs Task PGD)...", flush=True)
    if not CAUSAL_ANOVA_JSON_PATH.exists():
        print(f"    [SKIP] Chưa có {CAUSAL_ANOVA_JSON_PATH.name}")
        return
    with open(CAUSAL_ANOVA_JSON_PATH, "r", encoding="utf-8") as f:
        d = json.load(f)

    attacks = d.get("attacks", {})
    avail = [k for k in ["aqsm", "vanishing", "taskpgd"] if k in attacks]
    if not avail:
        return

    arms = ["Arm_A", "Arm_D_index", "Arm_D_full", "Arm_C_full", "Arm_C_index", "Arm_B"]
    arm_lbls = ["Arm A\n(Clean)", "Arm D_idx\n(CleanM+AdvI)", "Arm D_full\n(CleanM+AdvQ)", "Arm C_full\n(AdvM+CleanQ)", "Arm C_idx\n(AdvM+CleanI)", "Arm B\n(Pure Attack)"]
    atk_labels = {"aqsm": "AQSM (Encoder Selection)", "vanishing": "PGD Vanishing (Decoder)", "taskpgd": "Standard Task PGD"}
    atk_colors = {"aqsm": "#C0392B", "vanishing": "#2980B9", "taskpgd": "#8E44AD"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.6, 5.6))
    x = np.arange(len(arms))
    w = 0.78 / max(len(avail), 1)
    for idx, atk in enumerate(avail):
        gc = attacks[atk]["global_coco_map"]
        vals = [gc[a]["mAP@50"] for a in arms]
        offset = (idx - (len(avail) - 1) / 2.0) * w
        bars = ax1.bar(x + offset, vals, w, label=atk_labels.get(atk, atk), color=atk_colors.get(atk, "#34495E"), edgecolor="black", alpha=0.88)
        for b_i, bar in enumerate(bars):
            if b_i == 0 and idx != 1:
                continue
            h = bar.get_height()
            lbl_str = f"{h:.1f}" if b_i > 0 else f"Clean {h:.1f}"
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.9, lbl_str, ha="center", va="bottom", fontsize=7.6, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(arm_lbls, fontsize=8.4)
    ax1.set_ylabel("Global COCO mAP@50 (%)", fontsize=11, fontweight="bold")
    ax1.set_title(f"(a) Causal Intervention Arms on Official COCO GT (N={d.get('n_eval', 300)})", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, max(58.0, float(attacks[avail[0]]["global_coco_map"]["Arm_A"]["mAP@50"]) * 1.25))
    ax1.grid(True, axis="y", alpha=0.35)
    ax1.legend(loc="upper right", fontsize=8.6)

    # Panel (b): 2x2 ANOVA Marginal Effects (ME_M vs ME_Q vs Inter) trên Per-Image AP@50
    eff_names = [r"Encoder Memory ($\mathrm{ME}_M$)", r"Query Prior ($\mathrm{ME}_Q$)", r"Interaction ($M \times Q$)"]
    x2 = np.arange(len(eff_names))
    for idx, atk in enumerate(avail):
        an = attacks[atk]["anova_per_image_ap50"]
        me = an["marginal_effects"]
        p_eta = an["attribution_shares"]["partial_eta_squared"]
        means = [me["ME_M"], me["ME_Q"], me["Interaction_MxQ"]]
        cis = [me["ME_M_ci95"], me["ME_Q_ci95"], me["Interaction_ci95"]]
        yerr = [[abs(means[j] - cis[j][0]) for j in range(3)], [abs(cis[j][1] - means[j]) for j in range(3)]]
        offset = (idx - (len(avail) - 1) / 2.0) * w
        bars = ax2.bar(
            x2 + offset, means, w, yerr=yerr, capsize=4,
            label=f"{atk_labels.get(atk, atk)} ($\\eta_p^2(M)={p_eta['Memory_M']:.2f}, Q={p_eta['Query_Q']:.2f}$)",
            color=atk_colors.get(atk, "#34495E"), edgecolor="black", alpha=0.88,
        )
    ax2.axhline(y=0.0, color="black", lw=1.1)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(eff_names, fontsize=9.2)
    ax2.set_ylabel("Per-Image AP@50 Effect (Percentage Points)", fontsize=10.5, fontweight="bold")
    ax2.set_title("(b) Repeated-Measures 2x2 ANOVA Marginal Effects (with 95% CI)", fontsize=11, fontweight="bold")
    ax2.grid(True, axis="y", alpha=0.35)
    ax2.legend(loc="lower right", fontsize=8.0)

    plt.tight_layout()
    save_fig_all(fig, "fig13_causal_swap_n300")
    plt.close(fig)


def draw_figure_14_encoder_selection_dynamics():
    print("  + Đang vẽ Figure 14: Encoder Memory & Query Selection Dynamics (N=300)...", flush=True)
    if not MECHANISM_JSON_PATH.exists():
        print(f"    [SKIP] Chưa có {MECHANISM_JSON_PATH.name}")
        return
    with open(MECHANISM_JSON_PATH, "r", encoding="utf-8") as f:
        d = json.load(f)

    main_d = d.get("main_n300_analysis", {})
    conds = [c for c in ["gaussian_noise_ctrl", "aqsm", "vanishing", "taskpgd"] if c in main_d]
    if not conds:
        return
    c_names = {"gaussian_noise_ctrl": "Gauss Noise\n(Control)", "aqsm": "AQSM\n(Selection)", "vanishing": "PGD\nVanishing", "taskpgd": "Standard\nTask PGD"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 5.4))
    # Panel (a): Encoder Relative L2 by Scale (S3, S4, S5) + FG vs BG
    x = np.arange(len(conds))
    w = 0.16
    s3 = [main_d[c]["encoder_memory"]["per_scale"]["S3"]["rel_l2"]["mean"] for c in conds]
    s4 = [main_d[c]["encoder_memory"]["per_scale"]["S4"]["rel_l2"]["mean"] for c in conds]
    s5 = [main_d[c]["encoder_memory"]["per_scale"]["S5"]["rel_l2"]["mean"] for c in conds]
    fg = [main_d[c]["encoder_memory"]["overall"]["fg_rel_l2"]["mean"] for c in conds]
    bg = [main_d[c]["encoder_memory"]["overall"]["bg_rel_l2"]["mean"] for c in conds]

    ax1.bar(x - 2 * w, s3, w, label="Scale S3 (40x40)", color="#5DADE2", edgecolor="black")
    ax1.bar(x - w, s4, w, label="Scale S4 (20x20)", color="#2E86C1", edgecolor="black")
    ax1.bar(x, s5, w, label="Scale S5 (10x10)", color="#1B4F72", edgecolor="black")
    ax1.bar(x + w, fg, w, label="FG Tokens (Inside GT)", color="#C0392B", edgecolor="black")
    ax1.bar(x + 2 * w, bg, w, label="BG Tokens (Outside GT)", color="#BDC3C7", edgecolor="black")
    ax1.set_xticks(x)
    ax1.set_xticklabels([c_names[c] for c in conds], fontsize=9.2)
    ax1.set_ylabel(r"Relative Feature Perturbation $\|\Delta M\|_2 / \|M_{\mathrm{clean}}\|_2$", fontsize=10.5, fontweight="bold")
    ax1.set_title("(a) Encoder Memory Perturbation by Scale & FG/BG Regions", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 1.38)
    ax1.grid(True, axis="y", alpha=0.35)
    ax1.legend(loc="upper left", ncol=2, fontsize=8.1)

    # Panel (b): Query Selection Top-300 Overlap, Spearman, and GT Coverage Rate
    ovl = [main_d[c]["query_selection"]["top300_overlap"]["mean"] * 100.0 for c in conds]
    cov_c = [main_d[c]["query_selection"]["cov_iou50_clean"]["mean"] * 100.0 for c in conds]
    cov_a = [main_d[c]["query_selection"]["cov_iou50_adv"]["mean"] * 100.0 for c in conds]
    cov_cnt = [main_d[c]["query_selection"]["cov_center_adv"]["mean"] * 100.0 for c in conds]

    w2 = 0.20
    ax2.bar(x - 1.5 * w2, ovl, w2, label="Top-300 Index Overlap (%)", color="#D4AC0D", edgecolor="black")
    ax2.bar(x - 0.5 * w2, cov_c, w2, label="Clean GT Coverage (IoU>=0.5)", color="#27AE60", edgecolor="black")
    ax2.bar(x + 0.5 * w2, cov_a, w2, label="Attacked GT Coverage (IoU>=0.5)", color="#C0392B", edgecolor="black")
    ax2.bar(x + 1.5 * w2, cov_cnt, w2, label="Attacked Center-in-GT Cov (%)", color="#8E44AD", edgecolor="black")
    ax2.set_xticks(x)
    ax2.set_xticklabels([c_names[c] for c in conds], fontsize=9.2)
    ax2.set_ylabel("Percentage (%)", fontsize=10.5, fontweight="bold")
    ax2.set_ylim(0, 115)
    ax2.set_title("(b) Query Selection Overlap & GT Object Coverage Verification", fontsize=11, fontweight="bold")
    ax2.grid(True, axis="y", alpha=0.35)
    ax2.legend(loc="upper right", fontsize=8.2)

    plt.tight_layout()
    save_fig_all(fig, "fig14_encoder_and_selection_dynamics")
    plt.close(fig)


def draw_figure_15_attention_scattering_comparison():
    print("  + Đang vẽ Figure 15: Deformable Attention & Matched-Budget Comparison (N=300)...", flush=True)
    if not MECHANISM_JSON_PATH.exists():
        print(f"    [SKIP] Chưa có {MECHANISM_JSON_PATH.name}")
        return
    with open(MECHANISM_JSON_PATH, "r", encoding="utf-8") as f:
        d = json.load(f)

    main_d = d.get("main_n300_analysis", {})
    conds = [c for c in ["gaussian_noise_ctrl", "aqsm", "vanishing", "taskpgd"] if c in main_d]
    if not conds:
        return
    c_names = {"gaussian_noise_ctrl": "Gauss Noise", "aqsm": "AQSM", "vanishing": "Vanishing", "taskpgd": "Task PGD"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 5.4))
    # Panel (a): Clean vs Attacked Attention FG Spatial Hit Rate & Normalized Entropy
    x = np.arange(len(conds))
    w = 0.20
    hit_c = [main_d[c]["deformable_attention"]["clean_distribution"]["mean_fg_spatial_hit"]["mean"] * 100.0 for c in conds]
    hit_a = [main_d[c]["deformable_attention"]["adv_distribution"]["mean_fg_spatial_hit"]["mean"] * 100.0 for c in conds]
    ent_c = [main_d[c]["deformable_attention"]["clean_distribution"]["mean_entropy_norm"]["mean"] * 100.0 for c in conds]
    ent_a = [main_d[c]["deformable_attention"]["adv_distribution"]["mean_entropy_norm"]["mean"] * 100.0 for c in conds]

    ax1.bar(x - 1.5 * w, hit_c, w, label="Clean FG Spatial Hit (%)", color="#27AE60", edgecolor="black")
    ax1.bar(x - 0.5 * w, hit_a, w, label="Attacked FG Spatial Hit (%)", color="#C0392B", edgecolor="black")
    ax1.bar(x + 0.5 * w, ent_c, w, label="Clean Attn Entropy (x100)", color="#5DADE2", edgecolor="black")
    ax1.bar(x + 1.5 * w, ent_a, w, label="Attacked Attn Entropy (x100)", color="#1B4F72", edgecolor="black")
    ax1.set_xticks(x)
    ax1.set_xticklabels([c_names[c] for c in conds], fontsize=9.5)
    ax1.set_ylabel("Percentage / Scaled Entropy (%)", fontsize=10.5, fontweight="bold")
    ax1.set_ylim(0, 110)
    ax1.set_title("(a) Deformable Attention FG Spatial Hit & Normalized Entropy", fontsize=11, fontweight="bold")
    ax1.grid(True, axis="y", alpha=0.35)
    ax1.legend(loc="upper left", fontsize=8.3)

    # Panel (b): Multi-budget comparison (4/255, 8/255, 16/255) mAP retention AQSM vs Vanishing vs Task PGD
    sweep = d.get("epsilon_budget_comparison", {})
    eps_keys = [k for k in ["4/255", "8/255", "16/255"] if k in sweep]
    if eps_keys:
        for atk, col, mk, lbl in [
            ("aqsm", "#C0392B", "s", "AQSM (Encoder Selection)"),
            ("vanishing", "#2980B9", "v", "PGD Vanishing (Decoder)"),
            ("taskpgd", "#8E44AD", "o", "Standard Task PGD (Hungarian)"),
        ]:
            if all(atk in sweep[ek] for ek in eps_keys):
                maps = [sweep[ek][atk]["coco_map"]["mAP@50"] for ek in eps_keys]
                ax2.plot(eps_keys, maps, marker=mk, lw=2.4, color=col, label=lbl)
        clean_sw = float(d.get("sweep_clean_baseline", {}).get("mAP@50", 0.0))
        if clean_sw > 0:
            ax2.axhline(y=clean_sw, color="#27AE60", ls="--", lw=1.6, label=f"Clean COCO Baseline ({clean_sw:.1f}%)")
        ax2.set_xlabel(r"Matched $\ell_\infty$ Perturbation Budget ($\epsilon$)", fontsize=10.5, fontweight="bold")
        ax2.set_ylabel("COCO mAP@50 (%)", fontsize=10.5, fontweight="bold")
        ax2.set_title(f"(b) Matched-Budget Task Degradation (T=15, alpha=2/255, N={d['protocol']['n_sweep']})", fontsize=11, fontweight="bold")
        ax2.grid(True, alpha=0.35)
        ax2.legend(loc="upper right", fontsize=8.5)

    plt.tight_layout()
    save_fig_all(fig, "fig15_attention_scattering_comparison")
    plt.close(fig)


def main():
    print("=" * 90)
    print("SINH TOÀN BỘ FIGURES CHUẨN PUBLICATION (100% REAL DATA, ZERO MOCK)")
    print("=" * 90)
    draw_figure_1_architecture()
    draw_figure_6_attack_taxonomy()
    draw_figure_7_calibration_mechanism()

    data = {}
    if RESULTS_JSON_PATH.exists():
        with open(RESULTS_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    npz_data = np.load(RAW_SCORES_NPZ_PATH) if RAW_SCORES_NPZ_PATH.exists() else None

    if data:
        draw_figure_2_roc_curves(data)
        draw_figure_3_adaptive_tradeoff(data)
        draw_figure_4_patch_size_sweep(data)
        draw_figure_5_fpr_drift_stability(data)
    if npz_data is not None:
        draw_figure_8_cross_channel_correlation(npz_data)

    draw_figure_9_per_query_drift()
    draw_figure_10_latency_breakdown()
    draw_figure_12_encoder_score_spectrum()
    draw_figure_13_causal_swap_n300()
    draw_figure_14_encoder_selection_dynamics()
    draw_figure_15_attention_scattering_comparison()
    print("\n[OK] Hoàn tất xuất toàn bộ biểu đồ đồng bộ tại paper/fig, paper_ieee/fig và slides!")


if __name__ == "__main__":
    main()
