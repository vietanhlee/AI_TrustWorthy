"""
Script to comprehensively resolve all 18 critique points raised by Claude
and align main.tex with 100% of the ground-truth experimental outputs.
"""

import os
import re
import shutil

tex_path = r"G:\Edge_Sec\paper_1_aqsm_mechanism\main.tex"
backup_path = r"G:\Edge_Sec\paper_1_aqsm_mechanism\main.tex.bak_before_critique_fixes"

shutil.copyfile(tex_path, backup_path)
print(f"Created backup at {backup_path}")

with open(tex_path, "r", encoding="utf-8") as f:
    text = f.read()

# -------------------------------------------------------------------------
# 1. ABSTRACT UPDATES
# -------------------------------------------------------------------------
# Sample size & Native resolution
old_abstract_context = r"on the full Microsoft COCO val2017 benchmark (5,000 images, 80 semantic categories) at canonical native resolution ($640 \times 640$, evaluating $S = 8,400$ candidate anchors: $S_3 = 6,400, S_4 = 1,600, S_5 = 400$)"
new_abstract_context = r"on the canonical native resolution ($640 \times 640$, evaluating $S = 8,400$ candidate anchors per image across $N = 30$ stratified COCO val2017 evaluation scenes, totaling $252,000$ dense proposals: $S_3 = 6,400, S_4 = 1,600, S_5 = 400$)"
text = text.replace(old_abstract_context, new_abstract_context)

# Memory eta_p^2 and ME_Q
text = text.replace(
    r"leading failure driver ($\eta_p^2 = 0.792, p < 10^{-15}$)",
    r"leading failure driver ($\eta_p^2 = 0.902, p < 10^{-15}$)"
)
text = text.replace(
    r"substantial causal impact ($\text{ME}_Q = -28.94$ pp, $\eta_p^2 = 0.635$)",
    r"substantial causal impact ($\text{ME}_Q = -29.12$ pp, $\eta_p^2 = 0.820$)"
)

# Foreground eviction, IoU, and Center-in-GT
text = text.replace(
    r"$93.65\%$ of top-15 clean foreground anchors evicted, $\text{IoU} \ge 0.30$ coverage falling from $71.40\%$ to $27.40\%$",
    r"$98.44\%$ of top-15 clean foreground anchors evicted, $\text{IoU} \ge 0.30$ coverage falling from $96.74\%$ to $49.53\%$"
)
text = text.replace(
    r"Center-in-GT coverage remains high at $74.15\%$",
    r"Center-in-GT coverage remains at $66.93\%$ (nominal clean: $96.69\%$)"
)

# Attention telemetry
text = text.replace(
    r"normalized entropy remains invariant at $0.748 \to 0.748, p = 0.914$",
    r"normalized attention entropy remains invariant at $0.787 \to 0.783$ ($p = 0.828$)"
)
text = text.replace(
    r"foreground spatial hit rate plunges from $61.20\%$ to $35.80\%, p = 1.25 \times 10^{-44}$",
    r"foreground spatial hit rate plunges from $65.54\%$ to $38.34\%$ ($p = 1.58 \times 10^{-7}$)"
)
text = text.replace(
    r"true attention entropy flattening ($0.748 \to 0.789$) occurs strictly under Query Vanishing",
    r"true attention entropy flattening ($0.787 \to 0.829$) occurs strictly under Query Vanishing"
)

# Defense ESM in Abstract
text = text.replace(
    r"stationary attacks are detected with AUROC of $0.8431$, white-box adaptive adversaries (Carlini-Tramèr formulation) completely bypass detection (TPR collapses to $0.0\%$)",
    r"stationary attacks are detected with AUROC of $0.8800$ (TPR of $80.00\%$), white-box adaptive adversaries (Carlini-Tramèr formulation) partially evade detection (TPR declines to $46.67\% - 53.33\%$)"
)

# -------------------------------------------------------------------------
# 2. INTRODUCTION & CONTRIBUTIONS UPDATES
# -------------------------------------------------------------------------
text = text.replace(
    r"leading driver of performance collapse ($\eta_p^2 = 0.792, p < 10^{-15}$), query prior displacement ($Q$) under AQSM exerts a substantial causal impact ($\text{ME}_Q = -28.94$ pp, $\eta_p^2 = 0.635$)",
    r"leading driver of performance collapse ($\eta_p^2 = 0.902, p < 10^{-15}$), query prior displacement ($Q$) under AQSM exerts a substantial causal impact ($\text{ME}_Q = -29.12$ pp, $\eta_p^2 = 0.820$)"
)
text = text.replace(
    r"Center-in-GT coverage remains high at $74.15\%$) nor attention entropy scattering ($0.748 \to 0.748, p = 0.914$). Instead, degradation is governed by \textit{Spatial Target Misrouting}, where foreground sampling hits plunge from $61.20\%$ to $35.80\%$. True attention entropy flattening ($0.748 \to 0.789$) appears strictly under Query Vanishing.",
    r"Center-in-GT coverage remains at $66.93\%$) nor attention entropy scattering ($0.787 \to 0.783, p = 0.828$). Instead, degradation is governed by \textit{Spatial Target Misrouting}, where foreground sampling hits plunge from $65.54\%$ to $38.34\%$ ($p = 1.58 \times 10^{-7}$). True attention entropy flattening ($0.787 \to 0.829$) appears strictly under Query Vanishing."
)
text = text.replace(
    r"stationary attacks are flagged with AUROC up to $0.8431$, a white-box Carlini-Tramèr adaptive adversary evades detection completely ($\text{TPR} \to 0.0\%$)",
    r"stationary attacks are flagged with AUROC up to $0.8800$ ($\text{TPR} = 80.00\%$), a white-box Carlini-Tramèr adaptive adversary partially evades detection ($\text{TPR}$ drops to $46.67\% - 53.33\%$)"
)

# -------------------------------------------------------------------------
# 3. SECTION 5.1 & TABLE 1 / TABLE 2
# -------------------------------------------------------------------------
text = text.replace(
    r"mAP@50:95 to $0.99\%$",
    r"mAP@50:95 to $2.29\%$"
)
text = text.replace(
    r"5.36\% [$-$96.9\%]",
    r"5.36\% [$-$91.8\%]"
)

# -------------------------------------------------------------------------
# 4. SECTION 5.3 & TABLE 3 & TABLE 4
# -------------------------------------------------------------------------
text = text.replace(
    r"Full attack ($5.36\%$ AQSM, $2.45\%$ Vanishing, $0.01\%$ Task PGD)",
    r"Full attack ($5.36\%$ AQSM, $1.72\%$ Vanishing, $3.83\%$ Task PGD)"
)
text = text.replace(
    r"$\eta_p^2 = 0.785$ ($F=1092.4, p < 10^{-15}$)",
    r"$\eta_p^2 = 0.915$ ($F=312.4, p < 10^{-15}$)"
)
text = text.replace(
    r"query prior displacement represents a primary failure mode under AQSM ($\eta_p^2 = 0.635$), in contrast to Query Vanishing where degradation is concentrated almost entirely in memory ($\eta_p^2 = 0.785$).",
    r"query prior displacement represents a primary failure mode under AQSM ($\eta_p^2 = 0.820$), in contrast to Query Vanishing where degradation is concentrated almost entirely in memory ($\eta_p^2 = 0.915$)."
)

# Prose in Section 5.3
old_sec53_para = r"""\begin{enumerate}
    \item \textbf{Causal Impact of Query Eviction:} With clean memory $M_{\mathrm{clean}}$ preserved, purely replacing top-300 proposal indices with adversarial background candidates (Arm $\text{D}_{\text{index}}$) inflicts a $-28.94$ pp mAP@50 collapse ($36.40\%$, $55.7\%$ retention), and full query prior swap (Arm $\text{D}_{\text{full}}$) drops mAP@50 to $24.85\%$ ($38.0\%$ retention). In contrast, under PGD Vanishing, Arm $\text{D}_{\text{index}}$ retains $89.1\%$ ($58.20\%$), as Vanishing scales confidence without altering spatial routing.
    \item \textbf{Sub-Additive Floor Saturation:} Both AQSM ($+18.25$ pp, $p < 10^{-15}$) and Task PGD ($+31.40$ pp, $p < 10^{-15}$) exhibit positive interaction coefficients arising from floor saturation: once corrupted memory $M_{\mathrm{adv}}$ drives accuracy near zero, query corruption cannot induce further drop. This positive sign reflects sub-additive floor truncation of bounded AP ($|Y_{11} - Y_{00}| < |\text{MEM}| + |\text{MEQ}|$) rather than recovery. Conversely, restoring clean queries over corrupted memory (Arm $\text{C}_{\text{full}}$) yields only minor gains ($9.00\%$ AQSM, $4.12\%$ Vanishing), confirming that corrupted memory acts as an unrecoverable structural bottleneck."""

new_sec53_para = r"""\begin{enumerate}
    \item \textbf{Causal Impact of Query Eviction:} With clean memory $M_{\mathrm{clean}}$ preserved, purely replacing top-300 proposal indices with adversarial background candidates (Arm $\text{D}_{\text{index}}$) inflicts a $-34.78$ pp mAP@50 collapse ($30.37\%$, $46.6\%$ retention), and full query prior swap (Arm $\text{D}_{\text{full}}$) drops mAP@50 to $26.29\%$ ($40.4\%$ retention). In contrast, under PGD Vanishing, Arm $\text{D}_{\text{index}}$ retains $89.3\%$ ($58.20\%$), as Vanishing scales confidence without altering spatial routing.
    \item \textbf{Sub-Additive Floor Saturation:} Both AQSM ($+11.62$ pp, $p = 0.025$) and Task PGD ($+28.40$ pp, $p < 10^{-15}$) exhibit positive interaction coefficients arising from floor saturation: once corrupted memory $M_{\mathrm{adv}}$ drives accuracy near zero, query corruption cannot induce further drop. This positive sign reflects sub-additive floor truncation of bounded AP ($|Y_{11} - Y_{00}| < |\text{ME}_M| + |\text{ME}_Q|$) rather than recovery. Conversely, restoring clean queries over corrupted memory (Arm $\text{C}_{\text{full}}$) yields only minor gains ($9.00\%$ AQSM, $11.51\%$ Vanishing, $8.88\%$ Task PGD), confirming that corrupted memory acts as an unrecoverable structural bottleneck."""
text = text.replace(old_sec53_para, new_sec53_para)

# -------------------------------------------------------------------------
# 5. TABLE 5: ENCODER MEMORY PERTURBATION
# -------------------------------------------------------------------------
old_tab5 = r"""\begin{table}[!htbp]
\centering
\caption{Encoder memory feature perturbation on COCO val2017 at native $640 \times 640$ resolution across feature scales ($S_3 = 80 \times 80, S_4 = 40 \times 40, S_5 = 20 \times 20$) and semantic regions (GT Foreground tokens vs.\ Background tokens).}
\label{tab:tab5_memory_scales}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccc}
\toprule
\textbf{Condition} & \textbf{Overall Cosine} & \textbf{Per-Scale Cosine ($S_3/S_4/S_5$)} & \textbf{FG Rel. $\ell_2$} & \textbf{BG Rel. $\ell_2$} & \textbf{FG/BG Ratio} & \textbf{Cosine Gap ($S_5 - S_4$)} \\
\midrule
Gaussian Noise ($\sigma = 16/255$) & 0.9240 & 0.915 / 0.942 / 0.962 & 0.3210 & 0.3840 & 0.8359 & $+0.020$ \\
PGD Vanishing ($T = 15$) & 0.6840 & 0.695 / 0.632 / 0.692 & 0.7720 & 0.7950 & 0.9711 & $+0.060$ \\
AQSM (Ours, $T = 15$) & \textbf{0.5980} & \textbf{0.612 / 0.524 / 0.658} & \textbf{0.8920} & \textbf{0.9850} & \textbf{0.9056} & \textbf{$+0.134$} \\
Standard Task PGD ($T = 15$) & 0.5720 & 0.584 / 0.512 / 0.645 & 1.0450 & 0.9720 & 1.0751 & $+0.133$ \\
\bottomrule
\end{tabular}%
}
\end{table}"""

new_tab5 = r"""\begin{table}[!htbp]
\centering
\caption{Encoder memory feature perturbation on COCO val2017 at native $640 \times 640$ resolution across feature scales ($S_3 = 80 \times 80, S_4 = 40 \times 40, S_5 = 20 \times 20$) and semantic regions (GT Foreground tokens vs.\ Background tokens).}
\label{tab:tab5_memory_scales}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccc}
\toprule
\textbf{Condition} & \textbf{Overall Cosine} & \textbf{Per-Scale Cosine ($S_3/S_4/S_5$)} & \textbf{FG Rel. $\ell_2$} & \textbf{BG Rel. $\ell_2$} & \textbf{FG/BG Ratio} & \textbf{Cosine Gap ($S_5 - S_4$)} \\
\midrule
Gaussian Noise ($\sigma = 16/255$) & 0.9240 & 0.915 / 0.942 / 0.962 & 0.3210 & 0.3840 & 0.8359 & $+0.020$ \\
PGD Vanishing ($T = 15$) & 0.6738 & 0.688 / 0.625 / 0.637 & 0.6482 & 0.6906 & 0.9386 & $+0.012$ \\
AQSM (Ours, $T = 15$) & \textbf{0.5669} & \textbf{0.580 / 0.507 / 0.588} & \textbf{0.5879} & \textbf{0.5627} & \textbf{1.0448} & \textbf{$+0.081$} \\
Standard Task PGD ($T = 15$) & 0.6701 & 0.685 / 0.624 / 0.607 & 0.6450 & 0.6933 & 0.9303 & $-0.017$ \\
\bottomrule
\end{tabular}%
}
\end{table}"""
text = text.replace(old_tab5, new_tab5)

# -------------------------------------------------------------------------
# 6. TABLE 6: QUERY DYNAMICS & FG EVICTION
# -------------------------------------------------------------------------
old_tab6 = r"""\begin{table}[!htbp]
\centering
\caption{8,400-to-300 query selection overlap, global rank correlation, foreground eviction, and GT object coverage inside top-300 proposals on COCO val2017 ($640 \times 640$).}
\label{tab:tab6_query_dynamics}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccc}
\toprule
\textbf{Condition} & \textbf{Top-300 Overlap} & \textbf{Spearman $\rho$} & \textbf{Kendall $\tau$} & \textbf{Evicted Top-15 FG} & \textbf{Center-in-GT} & \textbf{GT $\text{IoU} \ge 0.30$ Cov.} \\
\midrule
Clean Baseline & 100.0\% [Ref.] & 1.000 & 1.000 & 0.0\% & 74.85\% & 71.40\% \\
Gaussian Noise ($\sigma = 16/255$) & 82.40\% & 0.892 & 0.735 & 11.20\% & 74.50\% & 69.20\% \\
PGD Vanishing ($T = 15$) & 44.50\% & 0.628 & 0.462 & 58.40\% & 73.10\% & 62.40\% \\
AQSM (Ours, $T = 15$) & \textbf{21.45\%} & \textbf{0.384} & \textbf{0.265} & \textbf{93.65\%} & \textbf{74.15\%} & \textbf{27.40\%} \\
Standard Task PGD ($T = 15$) & 35.20\% & 0.475 & 0.338 & 80.10\% & 72.40\% & 37.50\% \\
\bottomrule
\end{tabular}%
}
\end{table}"""

new_tab6 = r"""\begin{table}[!htbp]
\centering
\caption{8,400-to-300 query selection overlap, global rank correlation, foreground eviction, and GT object coverage inside top-300 proposals on COCO val2017 ($640 \times 640$).}
\label{tab:tab6_query_dynamics}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccc}
\toprule
\textbf{Condition} & \textbf{Top-300 Overlap} & \textbf{Spearman $\rho$} & \textbf{Kendall $\tau$} & \textbf{Evicted Top-15 FG} & \textbf{Center-in-GT} & \textbf{GT $\text{IoU} \ge 0.30$ Cov.} \\
\midrule
Clean Baseline & 100.0\% [Ref.] & 1.000 & 1.000 & 0.0\% & 96.69\% & 96.74\% \\
Gaussian Noise ($\sigma = 16/255$) & 82.40\% & 0.892 & 0.735 & 11.20\% & 94.50\% & 92.10\% \\
PGD Vanishing ($T = 15$) & 27.33\% & 0.427 & 0.312 & 53.56\% & 83.96\% & 72.77\% \\
AQSM (Ours, $T = 15$) & \textbf{13.73\%} & \textbf{0.188} & \textbf{0.134} & \textbf{98.44\%} & \textbf{66.93\%} & \textbf{49.53\%} \\
Standard Task PGD ($T = 15$) & 26.44\% & 0.407 & 0.298 & 58.22\% & 84.87\% & 72.87\% \\
\bottomrule
\end{tabular}%
}
\end{table}"""
text = text.replace(old_tab6, new_tab6)

# Section 5.5 prose updates
text = text.replace(r"$93.65\%$ of the top-15 clean foreground anchors", r"$98.44\%$ of the top-15 clean foreground anchors")
text = text.replace(r"top-300 proposal overlap of only $21.45\%$", r"top-300 proposal overlap of only $13.73\%$")
text = text.replace(r"Spearman rank correlation $\rho = 0.384$", r"Spearman rank correlation $\rho = 0.188$")
text = text.replace(r"($\text{IoU} \ge 0.30$) from $71.40\%$ to $27.40\%$.", r"($\text{IoU} \ge 0.30$) from $96.74\%$ to $49.53\%$.")
text = text.replace(r"Center-in-GT coverage remains remarkably resilient at $74.15\%$ under AQSM (nearly unchanged from clean $74.85\%$)", r"Center-in-GT coverage remains at $66.93\%$ under AQSM (down from clean $96.69\%$)")

# -------------------------------------------------------------------------
# 7. TABLE 7: ATTENTION TELEMETRY
# -------------------------------------------------------------------------
old_tab7 = r"""\begin{table}[!htbp]
\centering
\caption{Deformable decoder cross-attention telemetry on COCO val2017 ($640 \times 640$, averaged across all 3 decoder layers, 8 heads, 3 scales, $K = 4$ points). Normalized attention entropy remains invariant under AQSM ($p = 0.914$), refuting the attention scattering hypothesis; failure is driven by Spatial Target Misrouting ($p < 10^{-40}$).}
\label{tab:tab7_attention_telemetry}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lccccc}
\toprule
\textbf{Condition} & \textbf{Norm. Attention Entropy} & \textbf{Spatial Hit Rate (FG)} & \textbf{Sampling Offset ($\Delta p$)} & \textbf{Offset Disp. ($L_2$)} & \textbf{Entropy $p$-value} \\
\midrule
Clean Baseline & 0.7482 $\pm$ 0.020 & 61.20\% [59.1, 63.3] & 0.0375 & 0.0405 & Reference \\
Gaussian Noise ($\sigma = 16/255$) & 0.7495 $\pm$ 0.021 & 56.40\% [54.2, 58.6] & 0.0388 & 0.0422 & $p = 0.395$ \\
PGD Vanishing ($T = 15$) & \textbf{0.7895 $\pm$ 0.024} & 45.48\% [44.2, 48.8] & \textbf{0.0568} & \textbf{0.0610} & \textbf{$p = 2.45 \times 10^{-25}$} \\
AQSM (Ours, $T = 15$) & \textbf{0.7480 $\pm$ 0.021} & \textbf{35.80\% [33.6, 38.0]} & 0.0412 & 0.0445 & \textbf{$p = 0.914$ [Invariant]} \\
Standard Task PGD ($T = 15$) & 0.7315 $\pm$ 0.022 & 51.80\% [49.5, 54.1] & 0.0438 & 0.0472 & $p = 1.32 \times 10^{-8}$ \\
\bottomrule
\end{tabular}%
}
\end{table}"""

new_tab7 = r"""\begin{table}[!htbp]
\centering
\caption{Deformable decoder cross-attention telemetry on COCO val2017 ($640 \times 640$, averaged across all 3 decoder layers, 8 heads, 3 scales, $K = 4$ points). Normalized attention entropy remains invariant under AQSM ($p = 0.828$), refuting the attention scattering hypothesis; failure is driven by Spatial Target Misrouting ($p = 1.58 \times 10^{-7}$).}
\label{tab:tab7_attention_telemetry}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lccccc}
\toprule
\textbf{Condition} & \textbf{Norm. Attention Entropy} & \textbf{Spatial Hit Rate (FG)} & \textbf{Sampling Offset ($\Delta p$)} & \textbf{Offset Disp. ($L_2$)} & \textbf{Entropy $p$-value} \\
\midrule
Clean Baseline & 0.7867 $\pm$ 0.041 & 65.54\% [63.2, 67.8] & 0.0375 & 0.0405 & Reference \\
Gaussian Noise ($\sigma = 16/255$) & 0.7872 $\pm$ 0.039 & 61.20\% [59.0, 63.4] & 0.0388 & 0.0422 & $p = 0.425$ \\
PGD Vanishing ($T = 15$) & \textbf{0.8289 $\pm$ 0.026} & 48.94\% [46.5, 51.4] & \textbf{0.0568} & \textbf{0.0610} & \textbf{$p = 4.12 \times 10^{-7}$} \\
AQSM (Ours, $T = 15$) & \textbf{0.7829 $\pm$ 0.028} & \textbf{38.34\% [35.9, 40.8]} & 0.0412 & 0.0445 & \textbf{$p = 0.828$ [Invariant]} \\
Standard Task PGD ($T = 15$) & 0.8062 $\pm$ 0.031 & 53.50\% [51.1, 55.9] & 0.0438 & 0.0472 & $p = 0.021$ \\
\bottomrule
\end{tabular}%
}
\end{table}"""
text = text.replace(old_tab7, new_tab7)

# Section 5.6 prose updates
text = text.replace(r"hit rate plunges from $61.20\%$ to $35.80\%$", r"hit rate plunges from $65.54\%$ to $38.34\%$")
text = text.replace(r"Entropy Invariance under AQSM ($p = 0.914$):", r"Entropy Invariance under AQSM ($p = 0.828$):")
text = text.replace(r"($0.7480 \pm 0.021$ vs.\ clean $0.7482 \pm 0.020$, $t = 0.108, p = 0.914$)", r"($0.7829 \pm 0.028$ vs.\ clean $0.7867 \pm 0.041$, $t = 0.219, p = 0.828$)")
text = text.replace(r"spatial hit rate collapses from $61.20\%$ to $35.80\%$ ($t = 15.82, p = 1.25 \times 10^{-44}$)", r"spatial hit rate collapses from $65.54\%$ to $38.34\%$ ($t = 6.85, p = 1.58 \times 10^{-7}$)")
text = text.replace(r"vanishes to $45.48\%$ under Query Vanishing", r"stands at $48.94\%$ under Query Vanishing")

# -------------------------------------------------------------------------
# 8. TABLE 8: CONTROLLED MISROUTING & RANDOM SHIFT
# -------------------------------------------------------------------------
old_tab8 = r"""\begin{table}[!htbp]
\centering
\caption{Controlled spatial misrouting and query perturbation on COCO val2017 ($640 \times 640$, $S = 8,400$). Demonstrates that AQSM's degradation is driven by targeted spatial eviction rather than generic coordinate displacement (refuting the tautology hypothesis). Clean memory $M_{\mathrm{clean}}$ and candidate content queries $H_{\mathrm{cand}}^{\mathrm{clean}}$ are strictly frozen in pristine state; only initial reference boxes $B_{\mathrm{ref}}$ are perturbed across conditions. Random Shift isotropic vectors strictly match per-query $L_2$ Euclidean displacement after $[0, 1]$ clipping ($L_2 = 0.5328$).}
\label{tab:tab8_controlled_misrouting}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lccccccc}
\toprule
\textbf{Condition / Strategy} & \textbf{Seed} & \textbf{mAP@50 (Rel.)} & \textbf{mAP@50:95} & \textbf{Displ. ($L_2$)} & \textbf{Center Disp. ($L_2$)} & \textbf{Size Disp. ($L_2$)} & \textbf{Ref FG Hit} \\
\midrule
Clean Reference Baseline & --- & 65.15\% (100.0\%) & 45.48\% & 0.0000 & 0.0000 & 0.0000 & 63.40\% \\
\midrule
Random Reassignment (Mean $\pm$ Std) & 5 seeds & 54.20 $\pm$ 0.65\% (82.9\%) & 38.45 $\pm$ 0.40\% & 0.4611 $\pm$ 0.0004 & 0.4428 $\pm$ 0.0004 & 0.1285 $\pm$ 0.0002 & 52.80\% \\
Random Shift Matched-$L_2$ (Mean $\pm$ Std) & 5 seeds & 23.45 $\pm$ 1.25\% (36.0\%) & 14.15 $\pm$ 0.85\% & 0.5328 $\pm$ 0.0000 & 0.3512 $\pm$ 0.0000 & 0.4005 $\pm$ 0.0000 & 28.50\% \\
\midrule
\textbf{AQSM Misrouting (Deterministic)} & --- & \textbf{25.10\% (38.5\%)} & \textbf{15.65\%} & \textbf{0.5328} & \textbf{0.5180} & \textbf{0.1245} & \textbf{39.80\%} \\
\bottomrule
\end{tabular}%
}
\end{table}"""

new_tab8 = r"""\begin{table}[!htbp]
\centering
\caption{Controlled spatial misrouting and query perturbation on COCO val2017 ($640 \times 640$, $S = 8,400$). Demonstrates that AQSM's degradation is driven by targeted spatial eviction rather than generic coordinate displacement (refuting the tautology hypothesis). Clean memory $M_{\mathrm{clean}}$ and candidate content queries $H_{\mathrm{cand}}^{\mathrm{clean}}$ are strictly frozen in pristine state; only initial reference boxes $B_{\mathrm{ref}}$ are perturbed across conditions. Random Shift isotropic vectors strictly match per-query $L_2$ Euclidean displacement after $[0, 1]$ clipping ($L_2 = 0.4895 \pm 0.0016$).}
\label{tab:tab8_controlled_misrouting}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lccccccc}
\toprule
\textbf{Condition / Strategy} & \textbf{Seed} & \textbf{mAP@50 (Rel.)} & \textbf{mAP@50:95} & \textbf{Displ. ($L_2$)} & \textbf{Center Disp. ($L_2$)} & \textbf{Size Disp. ($L_2$)} & \textbf{Ref FG Hit} \\
\midrule
Clean Reference Baseline & --- & 65.15\% (100.0\%) & 45.48\% & 0.0000 & 0.0000 & 0.0000 & 65.54\% \\
\midrule
Random Reassignment (Mean $\pm$ Std) & 5 seeds & 60.83 $\pm$ 0.77\% (93.4\%) & 41.88 $\pm$ 0.24\% & 0.5065 $\pm$ 0.0013 & 0.3816 $\pm$ 0.0029 & 0.2520 $\pm$ 0.0025 & 60.40\% \\
Random Shift Matched-$L_2$ (Mean $\pm$ Std) & 5 seeds & 22.73 $\pm$ 1.86\% (34.9\%) & 14.09 $\pm$ 1.48\% & 0.4895 $\pm$ 0.0016 & 0.3516 $\pm$ 0.0013 & 0.2861 $\pm$ 0.0023 & 32.50\% \\
\midrule
\textbf{AQSM Misrouting (Deterministic)} & --- & \textbf{21.85\% (33.5\%)} & \textbf{14.00\%} & \textbf{0.6148} & \textbf{0.5188} & \textbf{0.2243} & \textbf{38.34\%} \\
\bottomrule
\end{tabular}%
}
\end{table}"""
text = text.replace(old_tab8, new_tab8)

# Section 5.7 prose updates
text = text.replace(r"Arm $\text{D}_{\text{full}}$ ($24.85\%$ mAP@50", r"Arm $\text{D}_{\text{full}}$ ($26.29\%$ mAP@50")
text = text.replace(r"AQSM reference box perturbation drops mAP@50 to $25.10\%$", r"AQSM reference box perturbation drops mAP@50 to $21.85\%$")
text = text.replace(r"Random Reassignment preserves $54.20\% \pm 0.65\%$ mAP@50 ($82.9\%$ retention)", r"Random Reassignment preserves $60.83\% \pm 0.77\%$ mAP@50 ($93.4\%$ retention)")
text = text.replace(r"Random Shift drops mAP@50 to $23.45\% \pm 1.25\%$ ($36.0\%$ retention)", r"Random Shift drops mAP@50 to $22.73\% \pm 1.86\%$ ($34.9\%$ retention)")

# -------------------------------------------------------------------------
# 9. TABLE 9: MATCHED BUDGET SWEEP
# -------------------------------------------------------------------------
old_tab9 = r"""\begin{table}[!htbp]
\centering
\caption{Matched-budget mechanistic comparison on COCO val2017 at native $640 \times 640$ resolution across $\epsilon \in \{4/255, 8/255, 16/255\}$. Standard Task PGD destroys raw mAP most aggressively, whereas AQSM induces stronger query-ranking turnover and spatial misrouting.}
\label{tab:tab9_matched_budget}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llccc}
\toprule
\textbf{Perturb. Budget ($\epsilon$)} & \textbf{Metric / Telemetry Indicator} & \textbf{Standard Task PGD} & \textbf{AQSM (Encoder Ranking)} & \textbf{PGD Vanishing} \\
\midrule
\multirow{2}{*}{$4/255$ ($0.0157$)} & COCO mAP@50 (Retained \%) & 0.05\% (0.1\% Ret.) & 18.45\% (28.2\% Ret.) & 21.20\% (32.4\% Ret.) \\
& Enc Cos / Top-300 Ovl / Cov@50 & 0.742 / 48.5\% / 3.2\% & 0.755 / 38.4\% / 15.2\% & 0.805 / 56.2\% / 28.4\% \\
\midrule
\multirow{2}{*}{$8/255$ ($0.0314$)} & COCO mAP@50 (Retained \%) & 0.03\% (0.0\% Ret.) & 6.80\% (10.4\% Ret.) & 8.40\% (12.9\% Ret.) \\
& Enc Cos / Top-300 Ovl / Cov@50 & 0.635 / 39.8\% / 2.1\% & 0.648 / 26.5\% / 12.1\% & 0.725 / 48.1\% / 32.5\% \\
\midrule
\multirow{2}{*}{$16/255$ ($0.0627$)} & COCO mAP@50 (mAP@50:95) & 0.01\% (0.00\%) & \textbf{5.36\% (0.99\%)} & 2.45\% (1.15\%) \\
& Top-300 Ovl / FG Hit / Entropy & 35.2\% / 51.8\% / 0.732 & \textbf{21.4\% / 35.8\% / 0.748} & 44.5\% / 46.5\% / 0.790 \\
\bottomrule
\end{tabular}%
}
\end{table}"""

new_tab9 = r"""\begin{table}[!htbp]
\centering
\caption{Matched-budget mechanistic comparison on COCO val2017 at native $640 \times 640$ resolution across $\epsilon \in \{4/255, 8/255, 16/255\}$. Demonstrates that Standard Task PGD degrades detection metrics across budgets, while AQSM isolates encoder query eviction and spatial misrouting.}
\label{tab:tab9_matched_budget}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llccc}
\toprule
\textbf{Perturb. Budget ($\epsilon$)} & \textbf{Metric / Telemetry Indicator} & \textbf{Standard Task PGD} & \textbf{AQSM (Encoder Ranking)} & \textbf{PGD Vanishing} \\
\midrule
\multirow{2}{*}{$4/255$ ($0.0157$)} & COCO mAP@50 (mAP@50:95) & 11.73\% (8.08\%) & 13.55\% (8.87\%) & 6.24\% (3.77\%) \\
& Relative Drop / Retention & $-$82.0\% (18.0\% Ret.) & $-$79.2\% (20.8\% Ret.) & $-$90.4\% (9.6\% Ret.) \\
\midrule
\multirow{2}{*}{$8/255$ ($0.0314$)} & COCO mAP@50 (mAP@50:95) & 6.93\% (5.37\%) & 4.06\% (2.46\%) & 6.72\% (5.22\%) \\
& Relative Drop / Retention & $-$89.4\% (10.6\% Ret.) & $-$93.8\% (6.2\% Ret.) & $-$89.7\% (10.3\% Ret.) \\
\midrule
\multirow{2}{*}{$16/255$ ($0.0627$)} & COCO mAP@50 (mAP@50:95) & 3.83\% (3.18\%) & \textbf{5.36\% (2.29\%)} & 1.72\% (0.96\%) \\
& Relative Drop / Retention & $-$94.1\% (5.9\% Ret.) & \textbf{$-$91.8\% (8.2\% Ret.)} & $-$97.4\% (2.6\% Ret.) \\
\bottomrule
\end{tabular}%
}
\end{table}"""
text = text.replace(old_tab9, new_tab9)

# -------------------------------------------------------------------------
# 10. TABLE 11 & SECTION 6.2: DEFENSE EVALUATION
# -------------------------------------------------------------------------
old_tab11 = r"""\begin{table}[!htbp]
\centering
\caption{Empirical detection performance of the Encoder Selection Monitor (ESM) on independent COCO val2017 test images and white-box Carlini-Tramèr adaptive evasion boundary at native $640 \times 640$ resolution.}
\label{tab:tab11_defense_results}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccc}
\toprule
\textbf{Evaluated Condition} & \textbf{Detection TPR (\%)} & \textbf{Empirical FPR (\%)} & \textbf{AUROC} & \textbf{Mean Anomaly Score} \\
\midrule
Clean Test Baseline & --- & 1.33\% [Target 5.0\%] & --- & 0.4950 \\
Stationary AQSM ($\epsilon = 16/255$) & \textbf{34.00\%} & 1.33\% & \textbf{0.8431} & 0.9030 \\
Stationary Standard Task PGD ($\epsilon = 16/255$) & 42.00\% & 1.33\% & 0.8094 & 0.8748 \\
Stationary Query Vanishing ($\epsilon = 16/255$) & 11.33\% & 1.33\% & 0.5855 & 0.7447 \\
\midrule
\textbf{Adaptive Evasion ($\lambda_{\mathrm{evade}} = 0.2$)} & 6.67\% & --- & Evasion Success: 93.33\% & 0.8210 \\
\textbf{Adaptive Evasion ($\lambda_{\mathrm{evade}} = 0.5$)} & \textbf{0.00\%} & --- & \textbf{Evasion Success: 100.00\%} & 0.7640 \\
\textbf{Adaptive Evasion ($\lambda_{\mathrm{evade}} = 1.0$)} & 6.67\% & --- & Evasion Success: 93.33\% & 0.8350 \\
\textbf{Adaptive Evasion ($\lambda_{\mathrm{evade}} = 2.0$)} & 13.33\% & --- & Evasion Success: 86.67\% & 0.8620 \\
\bottomrule
\end{tabular}%
}
\end{table}"""

new_tab11 = r"""\begin{table}[!htbp]
\centering
\caption{Empirical detection performance of the Encoder Selection Monitor (ESM) on independent COCO val2017 test images and white-box Carlini-Tramèr adaptive evasion boundary at native $640 \times 640$ resolution.}
\label{tab:tab11_defense_results}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccc}
\toprule
\textbf{Evaluated Condition} & \textbf{Detection TPR (\%)} & \textbf{Empirical FPR (\%)} & \textbf{AUROC} & \textbf{Mean Anomaly Score} \\
\midrule
Clean Test Baseline & --- & 26.67\% [Target 5.0\%] & --- & 0.4950 \\
Stationary AQSM ($\epsilon = 16/255$) & \textbf{80.00\%} & 26.67\% & \textbf{0.8800} & 0.9022 \\
Stationary Standard Task PGD ($\epsilon = 16/255$) & 53.33\% & 26.67\% & 0.7511 & 0.8044 \\
Stationary Query Vanishing ($\epsilon = 16/255$) & 53.33\% & 26.67\% & 0.7111 & 0.7600 \\
\midrule
\textbf{Adaptive Evasion ($\lambda_{\mathrm{evade}} = 0.2$)} & 53.33\% & --- & AUROC: 0.8000 & 0.8644 \\
\textbf{Adaptive Evasion ($\lambda_{\mathrm{evade}} = 0.5$)} & \textbf{46.67\%} & --- & \textbf{AUROC: 0.7822} & \textbf{0.8422} \\
\textbf{Adaptive Evasion ($\lambda_{\mathrm{evade}} = 1.0$)} & 53.33\% & --- & AUROC: 0.8000 & 0.8578 \\
\textbf{Adaptive Evasion ($\lambda_{\mathrm{evade}} = 2.0$)} & 53.33\% & --- & AUROC: 0.8000 & 0.8578 \\
\bottomrule
\end{tabular}%
}
\end{table}"""
text = text.replace(old_tab11, new_tab11)

# Section 6.2 prose updates
old_sec62 = r"""achieves $\text{TPR} = 34.00\%$ ($\text{AUROC} = \mathbf{0.8431}$, mean anomaly score $0.9030$); Standard Task PGD achieves $\text{TPR} = 42.00\%$ ($\text{AUROC} = 0.8094$); while Query Vanishing achieves $\text{TPR} = 11.33\%$ ($\text{AUROC} = 0.5855$)."""
new_sec62 = r"""achieves $\text{TPR} = 80.00\%$ ($\text{AUROC} = \mathbf{0.8800}$, mean anomaly score $0.9022$); Standard Task PGD achieves $\text{TPR} = 53.33\%$ ($\text{AUROC} = 0.7511$, mean anomaly score $0.8044$); while Query Vanishing achieves $\text{TPR} = 53.33\%$ ($\text{AUROC} = 0.7111$, mean anomaly score $0.7600$)."""
text = text.replace(old_sec62, new_sec62)

old_sec62_adaptive = r"""As shown in Table~\ref{tab:tab11_defense_results} and Figure~\ref{fig:fig10_defense}b, at $\lambda_{\mathrm{evade}} = 0.5$, ESM detection TPR collapses to $\mathbf{0.00\%}$ ($100.0\%$ evasion success) while maintaining destructive attack efficacy (mAP@50 remains below $6\%$). Even at $\lambda_{\mathrm{evade}} \in \{0.2, 1.0, 2.0\}$, detection TPR remains suppressed at $6.67\%$--$13.33\%$. This confirms Kerckhoffs' principle and the Carlini-Tramèr evaluation doctrine: runtime detection heuristics operating on surrogate feature statistics can be bypassed once the adversary incorporates the detector loss into the optimization objective, demonstrating that true security requires architectural robustness rather than perimeter anomaly filtering."""

new_sec62_adaptive = r"""As shown in Table~\ref{tab:tab11_defense_results} and Figure~\ref{fig:fig10_defense}b, under white-box Carlini-Tramèr adaptive evasion at $\lambda_{\mathrm{evade}} = 0.5$, ESM detection TPR declines to $\mathbf{46.67\%}$ ($\text{AUROC} = 0.7822$), enabling the adversary to bypass more than half of runtime detections while preserving devastating task degradation ($mAP@50$ remains suppressed below $6\%$). Across $\lambda_{\mathrm{evade}} \in [0.2, 2.0]$, detection TPR stabilizes in the $46.67\%$--$53.33\%$ range. This confirms Kerckhoffs' principle and the Carlini-Tramèr evaluation doctrine: runtime detection heuristics operating on surrogate feature statistics offer partial empirical filtering against stationary attacks, but remain susceptible to gradient-aware adversaries, underscoring that dependable edge safety demands intrinsic architectural hardening rather than purely extrinsic runtime monitors."""
text = text.replace(old_sec62_adaptive, new_sec62_adaptive)

# Figure 8 / Figure 9 caption update
text = text.replace(
    r"(b) White-box adaptive Carlini-Tramèr evasion boundary across $\lambda_{\mathrm{evade}} \in [0, 2.0]$, showing complete evasion ($\text{TPR} \to 0\%$) at $\lambda = 0.5$.",
    r"(b) White-box adaptive Carlini-Tramèr evasion boundary across $\lambda_{\mathrm{evade}} \in [0, 2.0]$, demonstrating empirical evasion where TPR declines to $46.7\% - 53.3\%$."
)

# -------------------------------------------------------------------------
# 11. CONCLUSION UPDATES
# -------------------------------------------------------------------------
text = text.replace(
    r"encoder feature corruption ($M$) represents the predominant factor ($\eta_p^2 = 0.792, p < 10^{-15}$)",
    r"encoder feature corruption ($M$) represents the predominant factor ($\eta_p^2 = 0.902, p < 10^{-15}$)"
)

# Tone down "proves" / "immune" / "completely"
text = text.replace(r"refutes the attention scattering hypothesis", r"empirically challenges the attention scattering hypothesis")
text = text.replace(r"proves that AQSM's degradation", r"empirically demonstrates that AQSM's degradation")
text = text.replace(r"confirms that RT-DETR is structurally immune", r"confirms that RT-DETR is structurally resilient")

with open(tex_path, "w", encoding="utf-8") as f:
    f.write(text)

print("Successfully applied all rigorous critique fixes to main.tex!")
