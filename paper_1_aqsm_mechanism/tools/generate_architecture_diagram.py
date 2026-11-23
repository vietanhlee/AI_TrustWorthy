# -*- coding: utf-8 -*-
"""
Generate publication-quality Figure 1: RT-DETR Pipeline with Dynamic Query Selection Bottleneck
and Encoder Selection Monitor (ESM) Guardrail at Native 640x640 Resolution.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Chế độ headless bắt buộc cho Linux / Kaggle / Docker
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_system_architecture(output_pdf: Path, output_png: Path):
    fig, ax = plt.subplots(figsize=(13.0, 6.8), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Font setup
    title_font = {"fontsize": 11, "fontweight": "bold", "family": "serif"}
    subtitle_font = {"fontsize": 9.5, "fontweight": "bold", "family": "serif"}
    body_font = {"fontsize": 8.5, "family": "serif"}
    small_font = {"fontsize": 7.5, "family": "serif"}

    # Top Section: Main Inference Pipeline
    # Background card for Pipeline
    bg_pipe = patches.FancyBboxPatch((1.5, 45), 97, 52, boxstyle="round,pad=1,rounding_size=1.5",
                                     edgecolor="#2b5c8f", facecolor="#f4f7fb", linewidth=1.5)
    ax.add_patch(bg_pipe)
    ax.text(3, 94.5, "RT-DETR FORWARD INFERENCE PIPELINE & QUERY SELECTION VULNERABILITY (NATIVE 640×640)",
            fontsize=11, fontweight="bold", color="#1a365d", family="serif")

    # Block 1: Input Image
    b1 = patches.FancyBboxPatch((3.5, 52), 14, 38, boxstyle="round,pad=0.5,rounding_size=1",
                                edgecolor="#4a5568", facecolor="#ffffff", linewidth=1.2)
    ax.add_patch(b1)
    ax.text(10.5, 86, "Input Image $X$", ha="center", **subtitle_font)
    ax.text(10.5, 80, "RGB $640 \\times 640$", ha="center", color="#2b6cb0", **body_font)
    ax.text(10.5, 74, "Clean Input or\nAdv. Perturbation\n$X_{\\mathrm{adv}} = X + \\delta$\n($\\|\\delta\\|_\\infty \\leq 16/255$)",
            ha="center", color="#4a5568", **small_font)

    # Arrow 1 -> 2
    ax.annotate("", xy=(21, 71), xytext=(17.5, 71),
                arrowprops=dict(arrowstyle="-|>", lw=1.8, color="#2d3748", mutation_scale=15))

    # Block 2: Hybrid Encoder
    b2 = patches.FancyBboxPatch((21.5, 52), 17, 38, boxstyle="round,pad=0.5,rounding_size=1",
                                edgecolor="#2b6cb0", facecolor="#ffffff", linewidth=1.2)
    ax.add_patch(b2)
    ax.text(30, 86, "Hybrid Encoder", ha="center", **subtitle_font)
    ax.text(30, 80, "ResNet-18 + CCFM", ha="center", color="#2b6cb0", **body_font)
    ax.text(30, 74, "Multi-Scale Fusion:\n• $S_3: 80 \\times 80 = 6,400$\n• $S_4: 40 \\times 40 = 1,600$\n• $S_5: 20 \\times 20 = 400$\nTotal: $S = 8,400$ tokens",
            ha="center", color="#4a5568", **small_font)

    # Arrow 2 -> 3
    ax.annotate("", xy=(42.5, 71), xytext=(38.5, 71),
                arrowprops=dict(arrowstyle="-|>", lw=1.8, color="#2d3748", mutation_scale=15))

    # Block 3: Dynamic Query Selection Bottleneck (KEY SURFACE)
    b3 = patches.FancyBboxPatch((43, 50), 20, 42, boxstyle="round,pad=0.5,rounding_size=1",
                                edgecolor="#c53030", facecolor="#fff5f5", linewidth=1.8)
    ax.add_patch(b3)
    ax.text(53, 88.5, "Query Selection", ha="center", color="#9b2c2c", **subtitle_font)
    ax.text(53, 83.5, "8,400 → 300 Bottleneck", ha="center", color="#c53030", fontweight="bold", **small_font)
    ax.text(53, 78.5, "Top-$N_q$ Admission (3.57%)", ha="center", color="#4a5568", **small_font)

    # Pathways inside Block 3
    p_clean = patches.FancyBboxPatch((44.5, 66), 17, 9, boxstyle="round,pad=0.2,rounding_size=0.5",
                                     edgecolor="#2f855a", facecolor="#f0fff4", linewidth=1)
    ax.add_patch(p_clean)
    ax.text(53, 71.5, "Clean: Saliency Ranking", ha="center", color="#22543d", fontweight="bold", **small_font)
    ax.text(53, 67.5, "Top-15 FG Anchors Admitted", ha="center", color="#276749", **small_font)

    p_adv = patches.FancyBboxPatch((44.5, 53), 17, 10.5, boxstyle="round,pad=0.2,rounding_size=0.5",
                                   edgecolor="#e53e3e", facecolor="#fff5f5", linewidth=1)
    ax.add_patch(p_adv)
    ax.text(53, 59.5, "AQSM: Query Eviction", ha="center", color="#9b2c2c", fontweight="bold", **small_font)
    ax.text(53, 55, r"Margin Inversion ($\mathcal{L}_{\mathrm{AQSM}}$)" + "\n96.9% Top-15 FG Evicted",
            ha="center", color="#c53030", **small_font)

    # Arrow 3 -> 4
    ax.annotate("", xy=(66.5, 71), xytext=(63, 71),
                arrowprops=dict(arrowstyle="-|>", lw=1.8, color="#2d3748", mutation_scale=15))

    # Block 4: Deformable Decoder
    b4 = patches.FancyBboxPatch((67, 52), 16.5, 38, boxstyle="round,pad=0.5,rounding_size=1",
                                edgecolor="#2b6cb0", facecolor="#ffffff", linewidth=1.2)
    ax.add_patch(b4)
    ax.text(75.25, 86, "Deformable Decoder", ha="center", **subtitle_font)
    ax.text(75.25, 80, "3/6 Layers, 300 Queries", ha="center", color="#2b6cb0", **body_font)
    ax.text(75.25, 74, "Cross-Attention:\n• Ref. Points: $B_{\\mathrm{init}}$\n• Sampling Offsets: $\\Delta p$\nSpatial Misrouting:\nFG Hit: $63.5\\% \\to 43.2\\%$\n(Sharp weights, displaced)",
            ha="center", color="#4a5568", **small_font)

    # Arrow 4 -> 5
    ax.annotate("", xy=(87, 71), xytext=(83.5, 71),
                arrowprops=dict(arrowstyle="-|>", lw=1.8, color="#2d3748", mutation_scale=15))

    # Block 5: Prediction Head (No NMS)
    b5 = patches.FancyBboxPatch((87.5, 52), 9.5, 38, boxstyle="round,pad=0.5,rounding_size=1",
                                edgecolor="#2f855a", facecolor="#ffffff", linewidth=1.2)
    ax.add_patch(b5)
    ax.text(92.25, 86, "Predictions", ha="center", **subtitle_font)
    ax.text(92.25, 80, "No NMS", ha="center", color="#2f855a", fontweight="bold", **body_font)
    ax.text(92.25, 72, "Classes $\\hat{P}$\nBoxes $\\hat{B}$\nDeterministic\nEdge Latency\n(Sponge Proof)",
            ha="center", color="#4a5568", **small_font)

    # -------------------------------------------------------------
    # Bottom Section: Encoder Selection Monitor (ESM) Guardrail
    # -------------------------------------------------------------
    bg_guard = patches.FancyBboxPatch((1.5, 4), 97, 36, boxstyle="round,pad=1,rounding_size=1.5",
                                      edgecolor="#2c7a7b", facecolor="#f0fdf4", linewidth=1.5)
    ax.add_patch(bg_guard)
    ax.text(3, 36.5, "PASSIVE RUNTIME GUARDRAIL: ENCODER SELECTION MONITOR (ESM) ON EDGE COMPUTING PLATFORMS",
            fontsize=10.5, fontweight="bold", color="#234e52", family="serif")

    # Feature 1: Dropoff Ratio
    f1 = patches.FancyBboxPatch((4, 8), 24, 25, boxstyle="round,pad=0.5,rounding_size=1",
                                edgecolor="#2c7a7b", facecolor="#ffffff", linewidth=1.2)
    ax.add_patch(f1)
    ax.text(16, 29, "Proposal Dropoff Ratio $D(X)$", ha="center", color="#234e52", **subtitle_font)
    ax.text(16, 23.5, "$D(X) = \\frac{s_{(1)}^{\\mathrm{enc}} + \\epsilon_0}{s_{(300)}^{\\mathrm{enc}} + \\epsilon_0}$",
            ha="center", color="#0d9488", fontsize=10, fontweight="bold")
    ax.text(16, 17, "Clean Baseline: $35.64 \\pm 23.63$\nAQSM Eviction: $6.70 \\pm 3.24$\nCompression: $\\mathbf{5.32\\times}$ Collapse",
            ha="center", color="#4a5568", **small_font)
    ax.text(16, 10.5, "Calibrated Cutoff: $\\tau_D$ (Quantile 5%)", ha="center", color="#2c7a7b", **small_font)

    # Feature 2: Selection Margin
    f2 = patches.FancyBboxPatch((31, 8), 24, 25, boxstyle="round,pad=0.5,rounding_size=1",
                                edgecolor="#2c7a7b", facecolor="#ffffff", linewidth=1.2)
    ax.add_patch(f2)
    ax.text(43, 29, "Selection Score Margin $M(X)$", ha="center", color="#234e52", **subtitle_font)
    ax.text(43, 23.5, "$M(X) = s_{(1)}^{\\mathrm{enc}} - s_{(300)}^{\\mathrm{enc}}$",
            ha="center", color="#0d9488", fontsize=10, fontweight="bold")
    ax.text(43, 17, "Clean Baseline: $0.7961 \\pm 0.130$\nAQSM Inversion: $0.6571 \\pm 0.092$\nMargin Gap: $\\Delta = -0.1390$",
            ha="center", color="#4a5568", **small_font)
    ax.text(43, 10.5, "Calibrated Cutoff: $\\tau_M$ (Quantile 5%)", ha="center", color="#2c7a7b", **small_font)

    # Fusion Gate: Disjunctive Max-OR Rule
    fgate = patches.FancyBboxPatch((58, 8), 21, 25, boxstyle="round,pad=0.5,rounding_size=1",
                                   edgecolor="#dd6b20", facecolor="#fffaf0", linewidth=1.2)
    ax.add_patch(fgate)
    ax.text(68.5, 29, "Calibrated Max-OR Gate", ha="center", color="#9c4221", **subtitle_font)
    ax.text(68.5, 23.5, "$S_{\\mathrm{fused}} = \\max(S_D, S_M)$",
            ha="center", color="#dd6b20", fontsize=9.5, fontweight="bold")
    ax.text(68.5, 17, "Union Bound:\n$\\mathbb{P}(E_D \\cup E_M) \\leq 5\\% + 5\\% = 10\\%$\nEmpirical Clean FPR: $6.57\\%$\nZero Run-time Latency Overhead",
            ha="center", color="#4a5568", **small_font)
    ax.text(68.5, 10.5, "Fused Threshold: $\\tau = 0.975$", ha="center", color="#9c4221", **small_font)

    # Decision Block
    fdec = patches.FancyBboxPatch((82, 8), 15, 25, boxstyle="round,pad=0.5,rounding_size=1",
                                  edgecolor="#2b6cb0", facecolor="#ffffff", linewidth=1.2)
    ax.add_patch(fdec)
    ax.text(89.5, 29, "Security Decision", ha="center", color="#2b6cb0", **subtitle_font)
    # Sub-box Pass
    spass = patches.FancyBboxPatch((83.5, 18), 12, 7.5, boxstyle="round,pad=0.2,rounding_size=0.5",
                                   edgecolor="#2f855a", facecolor="#f0fff4", linewidth=1)
    ax.add_patch(spass)
    ax.text(89.5, 22.5, "Clean Pass", ha="center", color="#22543d", fontweight="bold", **small_font)
    ax.text(89.5, 19.5, "Normal Flow ($A=0$)", ha="center", color="#276749", **small_font)

    # Sub-box Alarm
    salarm = patches.FancyBboxPatch((83.5, 9.5), 12, 7.5, boxstyle="round,pad=0.2,rounding_size=0.5",
                                    edgecolor="#c53030", facecolor="#fff5f5", linewidth=1)
    ax.add_patch(salarm)
    ax.text(89.5, 14, "Security Alarm", ha="center", color="#9b2c2c", fontweight="bold", **small_font)
    ax.text(89.5, 11, "Fallback Safe ($A=1$)", ha="center", color="#c53030", **small_font)

    # Connecting Arrows from features to gate, gate to decision
    ax.annotate("", xy=(58, 20.5), xytext=(55, 20.5),
                arrowprops=dict(arrowstyle="-|>", lw=1.5, color="#dd6b20", mutation_scale=12))
    ax.annotate("", xy=(82, 20.5), xytext=(79, 20.5),
                arrowprops=dict(arrowstyle="-|>", lw=1.5, color="#2b6cb0", mutation_scale=12))

    # Tap arrow from Query Selection to ESM
    ax.annotate("", xy=(30, 33), xytext=(45, 50),
                arrowprops=dict(arrowstyle="-|>", lw=1.5, ls="--", color="#2c7a7b",
                                connectionstyle="arc3,rad=-0.15", mutation_scale=12))
    ax.text(37, 42, "Passive Logit Tap", ha="center", color="#2c7a7b", fontweight="bold", **small_font)

    plt.tight_layout()
    plt.savefig(output_pdf)
    plt.savefig(output_png)
    plt.close()
    print(f">>> Đã tạo Figure 1 mới: {output_pdf} và {output_png}")

if __name__ == "__main__":
    fig_dir = Path("paper_1_aqsm_mechanism/fig")
    draw_system_architecture(fig_dir / "fig1_system_architecture.pdf", fig_dir / "fig1_system_architecture.png")
