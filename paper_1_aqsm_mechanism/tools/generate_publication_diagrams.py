# -*- coding: utf-8 -*-
"""
Generate publication-quality Figure 1 & Figure 2:
  - Figure 1: RT-DETR Pipeline with Dynamic Query Selection Bottleneck & ESM Guardrail (640x640)
  - Figure 2: Three-Tier Threat Taxonomy contrasting CNNs, Output-Space DETR, and AQSM (640x640)
Guarantees clean typography, generous padding, and zero text overlap.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Chế độ headless bắt buộc cho Linux / Kaggle / Docker
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_fig1_pipeline(output_pdf: Path, output_png: Path):
    fig, ax = plt.subplots(figsize=(14.0, 8.2), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Font definitions
    title_font = {"fontsize": 11, "fontweight": "bold", "family": "serif"}
    subtitle_font = {"fontsize": 9.5, "fontweight": "bold", "family": "serif"}
    body_font = {"fontsize": 8.5, "family": "serif"}
    small_font = {"fontsize": 7.8, "family": "serif"}

    # Top Section: Forward Inference Pipeline
    bg_pipe = patches.FancyBboxPatch((1.5, 46), 97, 51, boxstyle="round,pad=1,rounding_size=1.5",
                                     edgecolor="#2b5c8f", facecolor="#f4f7fb", linewidth=1.5)
    ax.add_patch(bg_pipe)
    ax.text(3.5, 94.0, "RT-DETR FORWARD INFERENCE PIPELINE & QUERY SELECTION VULNERABILITY (NATIVE 640×640)",
            fontsize=11, fontweight="bold", color="#1a365d", family="serif")

    # Block 1: Input Image
    b1 = patches.FancyBboxPatch((3.5, 50), 14, 40, boxstyle="round,pad=0.5,rounding_size=1",
                                edgecolor="#4a5568", facecolor="#ffffff", linewidth=1.2)
    ax.add_patch(b1)
    ax.text(10.5, 86.5, "Input Image $X$", ha="center", va="center", **subtitle_font)
    ax.text(10.5, 82.0, "RGB $640 \\times 640$", ha="center", va="center", color="#2b6cb0", **body_font)
    ax.text(10.5, 77.0, "Clean Sample or\nAdv. Perturbation\n$X_{\\mathrm{adv}} = X + \\delta$\n($\\|\\delta\\|_\\infty \\leq 16/255$)",
            ha="center", va="top", color="#4a5568", **small_font)

    # Arrow 1 -> 2
    ax.annotate("", xy=(21, 70), xytext=(17.5, 70),
                arrowprops=dict(arrowstyle="-|>", lw=1.8, color="#2d3748", mutation_scale=15))

    # Block 2: Hybrid Encoder
    b2 = patches.FancyBboxPatch((21.5, 50), 17.5, 40, boxstyle="round,pad=0.5,rounding_size=1",
                                edgecolor="#2b6cb0", facecolor="#ffffff", linewidth=1.2)
    ax.add_patch(b2)
    ax.text(30.25, 86.5, "Hybrid Encoder", ha="center", va="center", **subtitle_font)
    ax.text(30.25, 82.0, "ResNet-18 + CCFM", ha="center", va="center", color="#2b6cb0", **body_font)
    ax.text(30.25, 77.0, "Multi-Scale Fusion:\n• $S_3: 80 \\times 80 = 6,400$\n• $S_4: 40 \\times 40 = 1,600$\n• $S_5: 20 \\times 20 = 400$\nTotal Tokens: $S = 8,400$",
            ha="center", va="top", color="#4a5568", **small_font)

    # Arrow 2 -> 3
    ax.annotate("", xy=(42.5, 70), xytext=(39.0, 70),
                arrowprops=dict(arrowstyle="-|>", lw=1.8, color="#2d3748", mutation_scale=15))

    # Block 3: Dynamic Query Selection Bottleneck
    b3 = patches.FancyBboxPatch((43, 49), 20.5, 42, boxstyle="round,pad=0.5,rounding_size=1",
                                edgecolor="#c53030", facecolor="#fff5f5", linewidth=1.8)
    ax.add_patch(b3)
    ax.text(53.25, 88.0, "Query Selection", ha="center", va="center", color="#9b2c2c", **subtitle_font)
    ax.text(53.25, 84.0, "8,400 → 300 Bottleneck", ha="center", va="center", color="#c53030", fontweight="bold", **small_font)
    ax.text(53.25, 80.0, "Top-$N_q$ Admission (3.57%)", ha="center", va="center", color="#4a5568", **small_font)

    # Pathways
    p_clean = patches.FancyBboxPatch((44.5, 66.5), 17.5, 10.5, boxstyle="round,pad=0.2,rounding_size=0.5",
                                     edgecolor="#2f855a", facecolor="#f0fff4", linewidth=1)
    ax.add_patch(p_clean)
    ax.text(53.25, 73.0, "Clean: Saliency Ranking", ha="center", va="center", color="#22543d", fontweight="bold", **small_font)
    ax.text(53.25, 69.0, "Top-15 FG Anchors Admitted", ha="center", va="center", color="#276749", **small_font)

    p_adv = patches.FancyBboxPatch((44.5, 51.5), 17.5, 12.5, boxstyle="round,pad=0.2,rounding_size=0.5",
                                   edgecolor="#e53e3e", facecolor="#fff5f5", linewidth=1)
    ax.add_patch(p_adv)
    ax.text(53.25, 60.5, "AQSM: Query Eviction", ha="center", va="center", color="#9b2c2c", fontweight="bold", **small_font)
    ax.text(53.25, 55.5, "Margin Inversion (Eq. 3)\n87.9% Slots Evicted to Clutter",
            ha="center", va="center", color="#c53030", **small_font)

    # Arrow 3 -> 4
    ax.annotate("", xy=(67.0, 70), xytext=(63.5, 70),
                arrowprops=dict(arrowstyle="-|>", lw=1.8, color="#2d3748", mutation_scale=15))

    # Block 4: Deformable Decoder
    b4 = patches.FancyBboxPatch((67.5, 50), 16.5, 40, boxstyle="round,pad=0.5,rounding_size=1",
                                edgecolor="#2b6cb0", facecolor="#ffffff", linewidth=1.2)
    ax.add_patch(b4)
    ax.text(75.75, 86.5, "Deformable Decoder", ha="center", va="center", **subtitle_font)
    ax.text(75.75, 82.0, "3 Layers, 300 Queries", ha="center", va="center", color="#2b6cb0", **body_font)
    ax.text(75.75, 77.0, "Cross-Attention:\n• Ref. Points: $B_{\\mathrm{init}}$\n• Offsets: $\\Delta p_k$\nSpatial Misrouting:\nFG Hit: $62.3\\% \\to 38.3\\%$\n(Entropy concentrated)",
            ha="center", va="top", color="#4a5568", **small_font)

    # Arrow 4 -> 5
    ax.annotate("", xy=(87.5, 70), xytext=(84.0, 70),
                arrowprops=dict(arrowstyle="-|>", lw=1.8, color="#2d3748", mutation_scale=15))

    # Block 5: Prediction Head
    b5 = patches.FancyBboxPatch((88.0, 50), 9.5, 40, boxstyle="round,pad=0.5,rounding_size=1",
                                edgecolor="#2f855a", facecolor="#ffffff", linewidth=1.2)
    ax.add_patch(b5)
    ax.text(92.75, 86.5, "Predictions", ha="center", va="center", **subtitle_font)
    ax.text(92.75, 82.0, "No NMS", ha="center", va="center", color="#2f855a", fontweight="bold", **body_font)
    ax.text(92.75, 77.0, "Classes $\\hat{P}$\nBoxes $\\hat{B}$\nDeterministic\nEdge Latency\n(Sponge Proof)",
            ha="center", va="top", color="#4a5568", **small_font)

    # Bottom Section: ESM Guardrail
    bg_guard = patches.FancyBboxPatch((1.5, 3), 97, 39, boxstyle="round,pad=1,rounding_size=1.5",
                                      edgecolor="#2c7a7b", facecolor="#f0fdf4", linewidth=1.5)
    ax.add_patch(bg_guard)
    ax.text(3.5, 38.0, "PASSIVE RUNTIME GUARDRAIL: ENCODER SELECTION MONITOR (ESM) ON EDGE COMPUTING PLATFORMS",
            fontsize=10.5, fontweight="bold", color="#234e52", family="serif")

    # Feature 1: Dropoff Ratio
    f1 = patches.FancyBboxPatch((4, 7), 24, 28, boxstyle="round,pad=0.5,rounding_size=1",
                                edgecolor="#2c7a7b", facecolor="#ffffff", linewidth=1.2)
    ax.add_patch(f1)
    ax.text(16, 31.0, "Proposal Dropoff Ratio $D(X)$", ha="center", va="center", color="#234e52", **subtitle_font)
    ax.text(16, 26.0, "$D(X) = \\frac{s_{(1)}^{\\mathrm{enc}} + \\epsilon_0}{s_{(300)}^{\\mathrm{enc}} + \\epsilon_0}$",
            ha="center", va="center", color="#0d9488", fontsize=10.5, fontweight="bold")
    ax.text(16, 21.0, "Clean Baseline: $35.64 \\pm 23.63$\nAQSM Eviction: $6.70 \\pm 3.24$\nCompression: $\\mathbf{5.32\\times}$ Collapse",
            ha="center", va="top", color="#4a5568", **small_font)
    ax.text(16, 10.0, "Calibrated Threshold: $\\tau_D$ (Quantile 5%)", ha="center", va="center", color="#2c7a7b", **small_font)

    # Feature 2: Selection Margin
    f2 = patches.FancyBboxPatch((31, 7), 24, 28, boxstyle="round,pad=0.5,rounding_size=1",
                                edgecolor="#2c7a7b", facecolor="#ffffff", linewidth=1.2)
    ax.add_patch(f2)
    ax.text(43, 31.0, "Selection Score Margin $M(X)$", ha="center", va="center", color="#234e52", **subtitle_font)
    ax.text(43, 26.0, "$M(X) = s_{(1)}^{\\mathrm{enc}} - s_{(300)}^{\\mathrm{enc}}$",
            ha="center", va="center", color="#0d9488", fontsize=10.5, fontweight="bold")
    ax.text(43, 21.0, "Clean Baseline: $0.7961 \\pm 0.130$\nAQSM Inversion: $0.6571 \\pm 0.092$\nMargin Gap: $\\Delta = -0.1390$",
            ha="center", va="top", color="#4a5568", **small_font)
    ax.text(43, 10.0, "Calibrated Threshold: $\\tau_M$ (Quantile 5%)", ha="center", va="center", color="#2c7a7b", **small_font)

    # Fusion Gate: Disjunctive Max-OR Rule
    fgate = patches.FancyBboxPatch((58, 7), 21, 28, boxstyle="round,pad=0.5,rounding_size=1",
                                   edgecolor="#dd6b20", facecolor="#fffaf0", linewidth=1.2)
    ax.add_patch(fgate)
    ax.text(68.5, 31.0, "Calibrated Max-OR Gate", ha="center", va="center", color="#9c4221", **subtitle_font)
    ax.text(68.5, 26.0, "$S_{\\mathrm{fused}} = \\max(S_D, S_M)$",
            ha="center", va="center", color="#c05621", fontsize=9.5, fontweight="bold")
    ax.text(68.5, 21.0, "Union Bound:\n$P(E_D \\cup E_M) \\leq 5\\% + 5\\% = 10\\%$\nEmpirical Clean FPR: $10.0\\%$\nZero Run-time Latency Overhead",
            ha="center", va="top", color="#7b341e", **small_font)
    ax.text(68.5, 10.0, "Fused Cutoff: $\\tau = 0.975$", ha="center", va="center", color="#9c4221", **small_font)

    # Final Security Decision
    sbox = patches.FancyBboxPatch((82, 7), 15.5, 28, boxstyle="round,pad=0.5,rounding_size=1",
                                  edgecolor="#2b6cb0", facecolor="#ffffff", linewidth=1.2)
    ax.add_patch(sbox)
    ax.text(89.75, 31.0, "Security Decision", ha="center", va="center", **subtitle_font)

    spass = patches.FancyBboxPatch((83.5, 20.5), 12.5, 7.5, boxstyle="round,pad=0.2,rounding_size=0.5",
                                   edgecolor="#2f855a", facecolor="#f0fff4", linewidth=1)
    ax.add_patch(spass)
    ax.text(89.75, 25.5, "Clean Pass", ha="center", va="center", color="#22543d", fontweight="bold", **small_font)
    ax.text(89.75, 22.0, "Normal Flow ($A=0$)", ha="center", va="center", color="#276749", **small_font)

    salarm = patches.FancyBboxPatch((83.5, 9.5), 12.5, 8.5, boxstyle="round,pad=0.2,rounding_size=0.5",
                                    edgecolor="#c53030", facecolor="#fff5f5", linewidth=1)
    ax.add_patch(salarm)
    ax.text(89.75, 15.0, "Security Alarm", ha="center", va="center", color="#9b2c2c", fontweight="bold", **small_font)
    ax.text(89.75, 11.5, "Fallback Safe ($A=1$)", ha="center", va="center", color="#c53030", **small_font)

    # Connecting Arrows
    ax.annotate("", xy=(58, 21.0), xytext=(55, 21.0),
                arrowprops=dict(arrowstyle="-|>", lw=1.5, color="#dd6b20", mutation_scale=12))
    ax.annotate("", xy=(82, 21.0), xytext=(79, 21.0),
                arrowprops=dict(arrowstyle="-|>", lw=1.5, color="#2b6cb0", mutation_scale=12))

    # Tap arrow from Query Selection to ESM
    ax.annotate("", xy=(30, 35.0), xytext=(45, 49.0),
                arrowprops=dict(arrowstyle="-|>", lw=1.5, ls="--", color="#2c7a7b",
                                connectionstyle="arc3,rad=-0.15", mutation_scale=12))
    ax.text(36.5, 43.5, "Passive Logit Tap", ha="center", va="center", color="#2c7a7b", fontweight="bold", **small_font)

    plt.tight_layout()
    plt.savefig(output_pdf)
    plt.savefig(output_png)
    plt.close()
    print(f">>> Đã sinh Figure 1: {output_pdf}")


def draw_fig2_taxonomy(output_pdf: Path, output_png: Path):
    fig, ax = plt.subplots(figsize=(13.0, 7.5), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    subtitle_font = {"fontsize": 10.0, "fontweight": "bold", "family": "serif"}
    body_font = {"fontsize": 9.0, "family": "serif"}
    small_font = {"fontsize": 8.0, "family": "serif"}

    # Main Card
    bg_tax = patches.FancyBboxPatch((1.5, 3), 97, 94, boxstyle="round,pad=1,rounding_size=1.5",
                                    edgecolor="#4a5568", facecolor="#f8fafc", linewidth=1.5)
    ax.add_patch(bg_tax)
    ax.text(3.5, 93.0, "TAXONOMY OF ADVERSARIAL THREAT MODELS ACROSS DETECTION PARADIGMS (NATIVE 640×640)",
            fontsize=11.5, fontweight="bold", color="#1e293b", family="serif")

    # Column 1: Conventional CNN Attacks
    c1 = patches.FancyBboxPatch((4, 6), 29, 83, boxstyle="round,pad=0.5,rounding_size=1",
                                edgecolor="#64748b", facecolor="#ffffff", linewidth=1.2)
    ax.add_patch(c1)
    ax.text(18.5, 84.5, "Tier A: CNN Threat Models", ha="center", va="center", color="#334155", **subtitle_font)
    ax.text(18.5, 80.5, "Dense Anchor Grids + NMS", ha="center", va="center", color="#64748b", **body_font)

    ax.text(18.5, 76.0,
            "• Target: Anchor grid probabilities\n  or bounding-box IoU overlap.\n\n"
            "• Attack Method: Dense sponge attacks\n  (flooding candidate proposals),\n  adversarial patch on feature maps.\n\n"
            "• Failure Mode: Quadratic NMS delay\n  surge (+242% on Jetson, +232% on Pi 5)\n  stalling real-time control loops.\n\n"
            "• Interface: Heuristic post-processing.",
            ha="center", va="top", color="#475569", **small_font)

    b1_bot = patches.FancyBboxPatch((5.5, 9), 26, 14, boxstyle="round,pad=0.2,rounding_size=0.5",
                                    edgecolor="#94a3b8", facecolor="#f1f5f9", linewidth=1)
    ax.add_patch(b1_bot)
    ax.text(18.5, 18.0, "Availability Risk (Sponge)", ha="center", va="center", color="#334155", fontweight="bold", **small_font)
    ax.text(18.5, 13.0, "NMS execution time explodes;\nLatency non-deterministic.", ha="center", va="center", color="#64748b", **small_font)

    # Column 2: Vanilla DETR Output-Space Attacks
    c2 = patches.FancyBboxPatch((35.5, 6), 29, 83, boxstyle="round,pad=0.5,rounding_size=1",
                                edgecolor="#3b82f6", facecolor="#ffffff", linewidth=1.2)
    ax.add_patch(c2)
    ax.text(50.0, 84.5, "Tier B: Vanilla DETR Attacks", ha="center", va="center", color="#1d4ed8", **subtitle_font)
    ax.text(50.0, 80.5, "Static Queries + Hungarian Loss", ha="center", va="center", color="#3b82f6", **body_font)

    ax.text(50.0, 76.0,
            "• Target: Learned static embeddings\n  or bipartite Hungarian matching.\n\n"
            "• Attack Method: Standard Task PGD\n  (classification loss + box GIoU loss)\n  or Query Vanishing on logits.\n\n"
            "• Failure Mode: Global scene blackout\n  or chaotic false-alarm flooding\n  (up to ~29.4 false alarms per image).\n\n"
            "• Interface: Decoder prediction heads.",
            ha="center", va="top", color="#475569", **small_font)

    b2_bot = patches.FancyBboxPatch((37.0, 9), 26, 14, boxstyle="round,pad=0.2,rounding_size=0.5",
                                    edgecolor="#60a5fa", facecolor="#eff6ff", linewidth=1)
    ax.add_patch(b2_bot)
    ax.text(50.0, 18.0, "Task Loss Disruption", ha="center", va="center", color="#1e40af", fontweight="bold", **small_font)
    ax.text(50.0, 13.0, "Static query pool preserved;\nAttacks global attention weights.", ha="center", va="center", color="#3b82f6", **small_font)

    # Column 3: RT-DETR Dynamic Query Selection Manipulation (AQSM)
    c3 = patches.FancyBboxPatch((67, 6), 29, 83, boxstyle="round,pad=0.5,rounding_size=1",
                                edgecolor="#dc2626", facecolor="#fff5f5", linewidth=1.5)
    ax.add_patch(c3)
    ax.text(81.5, 84.5, "Tier C: AQSM (This Work)", ha="center", va="center", color="#b91c1c", **subtitle_font)
    ax.text(81.5, 80.5, "Dynamic 8,400 → 300 Bottleneck", ha="center", va="center", color="#dc2626", fontweight="bold", **body_font)

    ax.text(81.5, 76.0,
            "• Target: Hybrid Encoder selection logits\n  at native 640×640 (S = 8,400 tokens).\n\n"
            "• Attack Method: Tripartite loss\n  (margin inversion + background elevation)\n  without requiring decoder backprop.\n\n"
            "• Failure Mode: Spatial Target Misrouting\n  (Sampling Hit 63.5% → 43.2%, recall 29.1%)\n  with minimal clutter (<5 false boxes).\n\n"
            "• Interface: Dynamic query selection.",
            ha="center", va="top", color="#475569", **small_font)

    b3_bot = patches.FancyBboxPatch((68.5, 9), 26, 14, boxstyle="round,pad=0.2,rounding_size=0.5",
                                    edgecolor="#f87171", facecolor="#fef2f2", linewidth=1)
    ax.add_patch(b3_bot)
    ax.text(81.5, 18.0, "Integrity Risk (Misrouting)", ha="center", va="center", color="#991b1b", fontweight="bold", **small_font)
    ax.text(81.5, 13.0, "Latency remains O(1) constant;\nSilently misroutes reference anchors.", ha="center", va="center", color="#dc2626", **small_font)

    plt.tight_layout()
    plt.savefig(output_pdf)
    plt.savefig(output_png)
    plt.close()
    print(f">>> Đã sinh Figure 2: {output_pdf}")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    fig_dir = base_dir / "fig"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # User requested to keep original Figure 1
    # draw_fig1_pipeline(fig_dir / "fig1_system_architecture.pdf", fig_dir / "fig1_system_architecture.png")
    draw_fig2_taxonomy(fig_dir / "fig2_threat_taxonomy.pdf", fig_dir / "fig2_threat_taxonomy.png")
