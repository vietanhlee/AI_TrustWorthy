#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: synchronize_main_tex.py
Mục đích: Đồng bộ hóa toàn diện 100% bản thảo main.tex với bộ số liệu thực nghiệm N=1,000 (results 4).
Thực hiện các chỉnh sửa chi tiết chuẩn xác từng phần:
  - Abstract & Introduction Key Contributions
  - Section 3 Threat Model
  - Table 1 & Component-wise Ablation
  - Section 5.1 & Table 2 Main Benchmark
  - Section 5.3, Table 3, Table 4, ANOVA & Non-parametric Wilcoxon / Bootstrap CIs
  - Section 5.5 & Table 6 Query Selection Dynamics
  - Section 5.6 & Table 7 Attention Telemetry (sửa lỗi Dropoff Ratio, Margin, FG Hit Rate, Drift, Layer 1)
  - Section 5.7 & Table 8 Controlled Spatial Misrouting & Tautology Refutation
  - Section 5.8 & Table 9 Matched-Budget Sweep
  - Section 5.9 & Tesla T4 Hardware Latency Breakdown
  - Section 7.2, Section 7.3, Table 11 ESM & Adaptive Evasion across N_test=700
  - Section 8.1, Section 8.2, Table 12 Hardened SQA
  - Section 9 Conclusion
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
TEX_FILE = BASE_DIR / "main.tex"


def main():
    if not TEX_FILE.exists():
        print(f"[LỖI] Không tìm thấy file: {TEX_FILE}")
        sys.exit(1)

    with open(TEX_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    replacements = []

    # -------------------------------------------------------------
    # 1. ABSTRACT
    # -------------------------------------------------------------
    old_abstract = r"""\begin{abstract}
Real-Time Detection Transformers (RT-DETR) achieve deterministic edge latency and state-of-the-art accuracy by eliminating Non-Maximum Suppression (NMS) post-processing in favor of bipartite Hungarian matching and dynamic query selection. In this architecture, multi-scale feature maps dynamically route candidate proposals to initialize decoder content queries and positional reference boxes. In this work, we conduct a systematic mechanistic evaluation of this dynamic query selection interface under gradient-based adversarial perturbations. Evaluated on the Microsoft COCO val2017 benchmark ($N = 1,000$ images, $S = 8,400$ spatial candidates at native $640 \times 640$ resolution), Adversarial Query Selection Manipulation (AQSM) collapses mAP@50 from $65.50\%$ to $2.99\%$ (a relative reduction of $-95.43\%$, and mAP@50:95 from $48.89\%$ to $1.53\%$). Through a $2 \times 2$ factorial counterfactual feature intervention via activation patching, we disentangle encoder memory representations from query prior selection. We show that while encoder memory corruption is the primary damage driver ($66.69\%$ of treatment factor sum-of-squares, $\eta_p^2 = 0.816$), query prior displacement acts as an essential secondary transmission channel ($\text{ME}_Q = -28.50$~pp, $\eta_p^2 = 0.732$), collapsing clean-memory mAP@50 to $32.47\%$ under pure index eviction and $26.51\%$ under full query prior replacement. Internal telemetry clarifies the nuanced proposal dynamics: under AQSM, $74.70\%$ of selected query anchors maintain center coordinates within true object boundaries (nominal clean: $98.98\%$), yet high-quality spatial coverage ($\text{IoU} \ge 0.30$) collapses from $98.46\%$ to $54.98\%$, refuting total candidate extinction while demonstrating partial spatial displacement. Furthermore, normalized attention entropy exhibits negligible practical change ($0.8010 \to 0.7910, \Delta = -0.0100$, Cohen's $d = -0.20$). Instead, failure is governed by \textit{Spatial Target Misrouting}, where initial reference coordinates suffer targeted center translation, reducing foreground cross-attention sampling hits from $63.51\%$ down to $38.29\%$ ($d = -1.08$, $p < 10^{-120}$) and reference point foreground containment from $68.93\%$ down to $39.33\%$. Controlled spatial interventions reveal acute geometric sensitivity of deformable attention to reference coordinates, where AQSM's directional center translation actively misroutes queries to adjacent background clutter (suppressing detection recall to $31.34\%$, compared to $40.75\%$ under isotropic random shift). Finally, evaluating an empirical Encoder Selection Monitor (ESM) demonstrates that while stationary attacks are flagged with AUROC up to $0.9122$ ($\text{TPR} = 63.14\%$, empirical clean FPR $6.57\%$, bounded by $10.0\%$ union limit), a white-box Carlini-Tram\`er adaptive adversary targeting joint distribution statistics bypasses detection, demonstrating that runtime filtering heuristics cannot substitute for intrinsic architectural hardening.
\end{abstract}"""

    new_abstract = r"""\begin{abstract}
Real-Time Detection Transformers (RT-DETR) achieve deterministic edge latency and state-of-the-art accuracy by eliminating Non-Maximum Suppression (NMS) post-processing in favor of bipartite Hungarian matching and dynamic query selection. In this architecture, multi-scale feature maps dynamically route candidate proposals to initialize decoder content queries and positional reference boxes. In this work, we conduct a systematic mechanistic evaluation of this dynamic query selection interface under gradient-based adversarial perturbations. Evaluated on the Microsoft COCO val2017 benchmark ($N = 1,000$ images, $S = 8,400$ spatial candidates at native $640 \times 640$ resolution), Adversarial Query Selection Manipulation (AQSM) collapses mAP@50 from $65.50\%$ to $2.50\%$ (a relative reduction of $-96.19\%$, and mAP@50:95 from $48.89\%$ to $1.37\%$, compared to $0.67\%$ under Standard Task PGD and $1.16\%$ under Query Vanishing). Through a $2 \times 2$ factorial counterfactual feature intervention via activation patching, we disentangle encoder memory representations from query prior selection. We show that while encoder memory corruption is the primary damage driver ($67.97\%$ of treatment factor sum-of-squares, $\eta_p^2 = 0.806$), query prior displacement acts as an essential secondary transmission channel ($\text{ME}_Q = -28.24$~pp, $26.81\%$ sum-of-squares, $\eta_p^2 = 0.735$), collapsing clean-memory mAP@50 to $33.04\%$ under pure index eviction (Arm $\text{D}_{\text{index}}$) and $25.11\%$ under full query prior replacement (Arm $\text{D}_{\text{full}}$). Internal telemetry clarifies the nuanced proposal dynamics: under AQSM, Top-300 proposal overlap with clean baseline drops to $12.21\%$ and top-15 foreground eviction reaches $98.50\%$ (while $74.70\%$ of selected query anchors maintain center coordinates within true object boundaries, nominal clean: $98.98\%$), refuting total candidate extinction while demonstrating targeted spatial displacement. Furthermore, normalized attention entropy exhibits negligible practical change ($0.8009 \to 0.7901, \Delta = -0.0108$, Cohen's $d = -0.21$; Layer 1: $0.8263 \to 0.8146$). Instead, failure is governed by \textit{Spatial Target Misrouting}, where initial reference coordinates suffer targeted center translation, reducing foreground cross-attention sampling hits from $63.51\%$ down to $38.77\%$ ($d = -1.03$, $p < 10^{-120}$) and reference point foreground containment from $68.93\%$ down to $39.86\%$. Controlled spatial interventions reveal acute geometric sensitivity of deformable attention to reference coordinates, where AQSM's directional center translation actively misroutes queries to adjacent background clutter (suppressing detection recall to $30.48\%$, compared to $40.87\%$ under isotropic random shift at $21.55\%$ mAP@50). Evaluating an empirical Encoder Selection Monitor (ESM) demonstrates that while stationary attacks are flagged with AUROC up to $0.9161$ ($\text{TPR} = 64.57\%$, empirical clean FPR $6.57\%$, bounded by $10.0\%$ union limit), a white-box Carlini-Tram\`er adaptive adversary targeting joint distribution statistics suppresses detection TPR down to $1.71\%$ ($98.29\%$ evasion success across $N_{\text{test}} = 700$ images), demonstrating that runtime filtering heuristics cannot substitute for intrinsic architectural hardening. Finally, an intrinsic architectural hardening prototype—\textit{Static Query Augmentation} (SQA)—recovers $+15.92$~pp mAP@50 under AQSM ($2.50\% \to 18.42\%$) with minimal clean penalty ($-0.67$~pp).
\end{abstract}"""
    replacements.append((old_abstract, new_abstract, "Abstract"))

    # -------------------------------------------------------------
    # 2. INTRO LINE 106 & KEY CONTRIBUTIONS (LINES 131-140)
    # -------------------------------------------------------------
    old_intro_p = r"AQSM collapses mAP@50 from $65.50\%$ to $2.99\%$ (a relative reduction of $-95.43\%$, and mAP@50:95 from $48.89\%$ to $1.53\%$)."
    new_intro_p = r"AQSM collapses mAP@50 from $65.50\%$ to $2.50\%$ (a relative reduction of $-96.19\%$, and mAP@50:95 from $48.89\%$ to $1.37\%$)."
    replacements.append((old_intro_p, new_intro_p, "Intro line 106"))

    old_contrib = r"""\paragraph{Key Contributions:}
The core contributions of this work are summarized as follows:
\begin{enumerate}
    \item \textbf{Causal Factorial Disentanglement on COCO val2017:} A $2 \times 2$ factorial feature-swap surgery on COCO val2017 ($640 \times 640$, $S=8,400$, $N=1,000$ images) demonstrates that while encoder memory corruption ($M$) is the leading driver of performance collapse ($66.69\%$ of treatment factor sum-of-squares, $\eta_p^2 = 0.816, p < 10^{-15}$), query prior displacement ($Q$) under AQSM exerts a substantial causal impact ($\text{ME}_Q = -28.50$ pp, $\eta_p^2 = 0.732$), driving clean-memory performance down to $32.47\%$ under pure index eviction (Arm $\text{D}_{\text{index}}$) and $26.51\%$ under full query prior replacement (Arm $\text{D}_{\text{full}}$).
    \item \textbf{Proposal Dynamics and Spatial Target Misrouting:} Internal telemetry reveals that failure stems neither from total proposal starvation (Center-in-GT coverage remains at $74.70\%$ (nominal clean: $98.98\%$)) nor attention entropy scattering ($0.8010 \to 0.7910, \Delta = -0.0100, d = -0.20$), but from \textit{Spatial Target Misrouting}, where foreground sampling hits plunge from $63.51\%$ to $38.29\%$ ($d = -1.08$) and reference point containment from $68.93\%$ to $39.33\%$. True attention entropy flattening ($0.8010 \to 0.8494$) appears strictly under Query Vanishing.
    \item \textbf{Geometric Sensitivity and Spatial Failure Characterization:} Controlled perturbation experiments demonstrate that while RT-DETR is robust to query slot permutation (Random Reassignment retains $60.87\%$ mAP@50, $92.9\%$ of clean baseline), deformable cross-attention exhibits acute geometric sensitivity: both calibrated isotropic random shift ($21.70\%$ mAP@50) and AQSM ($24.80\%$ mAP@50) trigger severe accuracy degradation. However, their mechanics diverge: isotropic random shift scatters anchors uniformly into empty canvas space (collapsing foreground hit rate to $32.87\%$), whereas AQSM actively steers query reference boxes toward misleading background clutter, collapsing detection recall to $31.34\%$ (vs.\ $40.75\%$ under random shift).
    \item \textbf{Multi-Tier Robustness Benchmark:} Standardized evaluation across encoder-boundary query manipulation, end-to-end task PGD, and environmental corruptions demonstrates that AQSM compresses the proposal spectrum by inflating background scores around rank \#300 by $3.42\times$.
    \item \textbf{Physical Edge Profiling and Sponge Immunity:} Benchmarking on a Raspberry Pi 5 ($418.20$ ms clean vs.\ $418.50$ ms adversarial) and an NVIDIA Jetson Orin Nano ($14.20$ ms clean vs.\ $14.22$ ms adversarial) confirms RT-DETR's structural immunity to NMS computational sponge attacks, while revealing that the dynamic selection interface constitutes a substantial integrity vulnerability.
    \item \textbf{Empirical Defense Evaluation and Adaptive Evasion Boundary:} Evaluating an empirical Encoder Selection Monitor (ESM) demonstrates that while stationary attacks are flagged with AUROC up to $0.9122$ ($\text{TPR} = 63.14\%$, empirical FPR $6.57\%$), a white-box Carlini-Tram\`er adaptive adversary bypasses detection (TPR declines to $0.00\%$ at $\lambda = 0.5$), demonstrating that runtime heuristics cannot substitute for intrinsic architectural hardening. In addition, we empirically validate an intrinsic architectural hardening prototype—\textit{Static Query Augmentation}—recovering $+15.43$ pp mAP@50 under AQSM ($2.99\% \to 18.42\%$) with negligible ($+0.38$ ms, $+2.68\%$) edge latency overhead.
\end{enumerate}"""

    new_contrib = r"""\paragraph{Key Contributions:}
The core contributions of this work are summarized as follows:
\begin{enumerate}
    \item \textbf{Causal Factorial Disentanglement on COCO val2017:} A $2 \times 2$ factorial feature-swap surgery on COCO val2017 ($640 \times 640$, $S=8,400$, $N=1,000$ images) demonstrates that while encoder memory corruption ($M$) is the leading driver of performance collapse ($67.97\%$ of treatment factor sum-of-squares, $\eta_p^2 = 0.806, p < 10^{-15}$), query prior displacement ($Q$) under AQSM exerts a substantial causal impact ($\text{ME}_Q = -28.24$ pp, $26.81\%$ sum-of-squares, $\eta_p^2 = 0.735$), driving clean-memory performance down to $33.04\%$ under pure index eviction (Arm $\text{D}_{\text{index}}$) and $25.11\%$ under full query prior replacement (Arm $\text{D}_{\text{full}}$).
    \item \textbf{Proposal Dynamics and Spatial Target Misrouting:} Internal telemetry reveals that failure stems neither from total proposal starvation (Center-in-GT coverage remains at $74.70\%$ (nominal clean: $98.98\%$)) nor attention entropy scattering ($0.8009 \to 0.7901, \Delta = -0.0108, d = -0.21$; Layer 1: $0.8263 \to 0.8146$), but from \textit{Spatial Target Misrouting}, where foreground sampling hits plunge from $63.51\%$ to $38.77\%$ ($d = -1.03$) and reference point containment from $68.93\%$ to $39.86\%$. Top-300 overlap collapses to $12.21\%$ and foreground eviction reaches $98.50\%$. True attention entropy flattening ($0.8009 \to 0.8495$; Layer 1: $0.8495$) appears strictly under Query Vanishing.
    \item \textbf{Geometric Sensitivity and Spatial Failure Characterization:} Controlled perturbation experiments demonstrate that while RT-DETR is robust to query slot permutation (Random Reassignment retains $60.87\%$ mAP@50, $92.9\%$ of clean baseline), deformable cross-attention exhibits acute geometric sensitivity: both calibrated isotropic random shift ($21.55\%$ mAP@50) and AQSM ($23.90\%$ mAP@50) trigger severe accuracy degradation. However, their mechanics diverge: isotropic random shift scatters anchors uniformly into empty canvas space (collapsing foreground hit rate to $32.90\%$), whereas AQSM actively steers query reference boxes toward misleading background clutter, collapsing detection recall to $30.48\%$ (vs.\ $40.87\%$ under random shift, while retaining $39.86\%$ foreground hit rate).
    \item \textbf{Multi-Tier Robustness Benchmark:} Standardized evaluation across encoder-boundary query manipulation, end-to-end task PGD, and environmental corruptions demonstrates that AQSM compresses the proposal spectrum by inflating background scores around rank \#300 by $3.42\times$.
    \item \textbf{Physical Edge Profiling and Sponge Immunity:} Benchmarking on a Raspberry Pi 5 ($418.20$ ms clean vs.\ $418.50$ ms adversarial) and an NVIDIA Jetson Orin Nano ($14.20$ ms clean vs.\ $14.22$ ms adversarial) confirms RT-DETR's structural immunity to NMS computational sponge attacks. Detailed stage profiling on NVIDIA Tesla T4 ($37.39$ ms total, $26.75$ FPS) isolates Backbone ($19.34$ ms, $51.73\%$), Hybrid Encoder ($10.10$ ms, $27.01\%$), Query Selection ($1.64$ ms, $4.38\%$), and Decoder ($6.04$ ms, $16.14\%$).
    \item \textbf{Empirical Defense Evaluation and Adaptive Evasion Boundary:} Evaluating an empirical Encoder Selection Monitor (ESM) demonstrates that while stationary attacks are flagged with AUROC up to $0.9161$ ($\text{TPR} = 64.57\%$, empirical FPR $6.57\%$), a white-box Carlini-Tram\`er adaptive adversary bypasses detection (TPR is suppressed to $1.71\%$ at $\lambda = 0.5$, achieving $98.29\%$ evasion success across $N_{\text{test}} = 700$ images), demonstrating that runtime heuristics cannot substitute for intrinsic architectural hardening. In addition, we empirically validate an intrinsic architectural hardening prototype—\textit{Static Query Augmentation}—recovering $+15.92$ pp mAP@50 under AQSM ($2.50\% \to 18.42\%$) with negligible clean penalty ($-0.67$ pp) and minimal edge latency overhead ($+0.38$ ms, $+2.68\%$).
\end{enumerate}"""
    replacements.append((old_contrib, new_contrib, "Key Contributions"))

    # -------------------------------------------------------------
    # 3. SECTION 3 (THREAT MODEL LINES 191, 219)
    # -------------------------------------------------------------
    old_sec3_191 = r"from $65.50\%$ to $2.99\%$ ($-95.43\%$)."
    new_sec3_191 = r"from $65.50\%$ to $2.50\%$ ($-96.19\%$, relative drop)."
    replacements.append((old_sec3_191, new_sec3_191, "Threat Model line 191"))

    old_sec3_219 = r"the oracle GT-matched attack yields $2.85\%$ mAP@50 (vs.\ $2.99\%$ for the clean proxy, $\Delta = 0.14$ pp)"
    new_sec3_219 = r"the oracle GT-matched attack yields $2.38\%$ mAP@50 (vs.\ $2.50\%$ for the clean proxy, $\Delta = 0.12$ pp)"
    replacements.append((old_sec3_219, new_sec3_219, "Threat Model line 219"))

    # -------------------------------------------------------------
    # 4. TABLE 1 (ABLATION)
    # -------------------------------------------------------------
    old_tab1_row = r"\textbf{Full Tripartite AQSM} & \textbf{2.99\% [$-$95.4\%]} & \textbf{12.4\% [10.5, 14.6]} & \textbf{98.3\% [97.1, 99.4]} & \textbf{Compound Eviction \& Misrouting} \\"
    new_tab1_row = r"\textbf{Full Tripartite AQSM} & \textbf{2.50\% [$-$96.2\%]} & \textbf{12.2\% [10.0, 14.5]} & \textbf{98.5\% [97.1, 99.5]} & \textbf{Compound Eviction \& Misrouting} \\"
    replacements.append((old_tab1_row, new_tab1_row, "Table 1 row"))

    # -------------------------------------------------------------
    # 5. SECTION 5.1 & TABLE 2 (MAIN BENCHMARK)
    # -------------------------------------------------------------
    old_sec51_text = r"Tier 1 AQSM collapses mAP@50 to $2.99\%$ (a $-95.43\%$ relative degradation) and mAP@50:95 to $1.53\%$. Tier 2 Query Vanishing drives mAP@50 to $0.90\%$ ($-98.63\%$), suppressing valid detections to $0.18$ boxes per scene. Standard Task PGD achieves mAP@50 of $0.83\%$ ($-98.73\%$)"
    new_sec51_text = r"Tier 1 AQSM collapses mAP@50 to $2.50\%$ (a $-96.19\%$ relative degradation) and mAP@50:95 to $1.37\%$. Tier 2 Query Vanishing drives mAP@50 to $1.16\%$ ($-98.22\%$, mAP@50:95 to $0.68\%$), suppressing valid detections to $0.18$ boxes per scene. Standard Task PGD achieves mAP@50 of $0.67\%$ ($-98.98\%$, mAP@50:95 to $0.41\%$)"
    replacements.append((old_sec51_text, new_sec51_text, "Section 5.1 text"))

    old_tab2_rows = r"""Tier 1: AQSM ($\epsilon = 16/255$) & \textbf{2.99\% [$-$95.4\%]} & \textbf{1.53\% [$-$96.9\%]} & $7.15 \pm 4.30$ & \textbf{98.3\% [97.1, 99.4]} & 12.8\% [9.5, 16.8] & \textbf{87.6\% [84.2, 91.0]} \\
Tier 2: Query Vanishing ($\epsilon = 16/255$) & 0.90\% [$-$98.6\%] & 0.48\% [$-$99.0\%] & $0.18 \pm 0.52$ & 97.8\% [95.5, 99.1] & \textbf{52.4\% [46.8, 58.1]} & 58.5\% [52.7, 64.2] \\
Baseline: Standard Task PGD ($\epsilon = 16/255$) & 0.83\% [$-$98.7\%] & 0.45\% [$-$99.1\%] & $29.40 \pm 8.70$ & 96.2\% [93.1, 98.4] & 0.0\% [0.0, 1.0] & 44.5\% [38.9, 50.3] \\"""

    new_tab2_rows = r"""Tier 1: AQSM ($\epsilon = 16/255$) & \textbf{2.50\% [$-$96.2\%]} & \textbf{1.37\% [$-$97.2\%]} & $7.15 \pm 4.30$ & \textbf{98.5\% [97.1, 99.5]} & 12.8\% [9.5, 16.8] & \textbf{87.8\% [84.2, 91.0]} \\
Tier 2: Query Vanishing ($\epsilon = 16/255$) & 1.16\% [$-$98.2\%] & 0.68\% [$-$98.6\%] & $0.18 \pm 0.52$ & 97.8\% [95.5, 99.1] & \textbf{52.4\% [46.8, 58.1]} & 69.6\% [65.2, 74.0] \\
Baseline: Standard Task PGD ($\epsilon = 16/255$) & 0.67\% [$-$99.0\%] & 0.41\% [$-$99.2\%] & $29.40 \pm 8.70$ & 96.2\% [93.1, 98.4] & 0.0\% [0.0, 1.0] & 73.6\% [68.9, 78.3] \\"""
    replacements.append((old_tab2_rows, new_tab2_rows, "Table 2 rows"))

    # -------------------------------------------------------------
    # 6. SECTION 5.3, TABLE 3 & TABLE 4 (CAUSAL SURGERY & ANOVA)
    # -------------------------------------------------------------
    old_arms_desc = r"""\begin{enumerate}
    \item \textbf{Arm A ($M_{\mathrm{clean}} + Q_{\mathrm{clean}}$):} Clean baseline ($65.50\%$ mAP@50, $48.89\%$ mAP@50:95). Clean content and clean reference boxes at indices $\mathcal{I}_{\mathrm{clean}}$.
    \item \textbf{Arm B ($M_{\mathrm{adv}} + Q_{\mathrm{adv}}[\mathcal{I}_{\mathrm{adv}}]$):} Full attack ($2.99\%$ AQSM, $0.90\%$ Vanishing, $0.83\%$ Task PGD).
    \item \textbf{Arm $\text{C}_{\text{full}}$ ($M_{\mathrm{adv}} + Q_{\mathrm{clean}}[\mathcal{I}_{\mathrm{clean}}]$):} Attacked memory $M_{\mathrm{adv}}$ paired with complete clean query priors (clean content + clean reference boxes at $\mathcal{I}_{\mathrm{clean}}$).
    \item \textbf{Arm $\text{D}_{\text{full}}$ ($M_{\mathrm{clean}} + Q_{\mathrm{adv}}[\mathcal{I}_{\mathrm{adv}}]$):} Clean memory $M_{\mathrm{clean}}$ initialized with complete attacked query priors ($H_{\mathrm{cand}}^{\mathrm{adv}}[\mathcal{I}_{\mathrm{adv}}], B_{\mathrm{cand}}^{\mathrm{adv}}[\mathcal{I}_{\mathrm{adv}}]$).
    \item \textbf{Arm $\text{D}_{\text{index}}$ ($M_{\mathrm{clean}} + Q_{\mathrm{clean}}[\mathcal{I}_{\mathrm{adv}}]$):} Clean memory $M_{\mathrm{clean}}$ and clean local features, routed purely using the adversarial index set $\mathcal{I}_{\mathrm{adv}}$ (87.6\% slots replaced under AQSM).
\end{enumerate}"""

    new_arms_desc = r"""\begin{enumerate}
    \item \textbf{Arm A ($M_{\mathrm{clean}} + Q_{\mathrm{clean}}$):} Clean baseline ($65.50\%$ mAP@50, $48.89\%$ mAP@50:95). Clean content and clean reference boxes at indices $\mathcal{I}_{\mathrm{clean}}$.
    \item \textbf{Arm B ($M_{\mathrm{adv}} + Q_{\mathrm{adv}}[\mathcal{I}_{\mathrm{adv}}]$):} Full attack ($2.50\%$ AQSM, $1.16\%$ Vanishing, $0.67\%$ Task PGD).
    \item \textbf{Arm $\text{C}_{\text{full}}$ ($M_{\mathrm{adv}} + Q_{\mathrm{clean}}[\mathcal{I}_{\mathrm{clean}}]$):} Attacked memory $M_{\mathrm{adv}}$ paired with complete clean query priors ($6.25\%$ AQSM, $3.79\%$ Vanishing, $1.40\%$ Task PGD).
    \item \textbf{Arm $\text{D}_{\text{full}}$ ($M_{\mathrm{clean}} + Q_{\mathrm{adv}}[\mathcal{I}_{\mathrm{adv}}]$):} Clean memory $M_{\mathrm{clean}}$ initialized with complete attacked query priors ($25.11\%$ AQSM, $38.71\%$ Vanishing, $39.50\%$ Task PGD).
    \item \textbf{Arm $\text{D}_{\text{index}}$ ($M_{\mathrm{clean}} + Q_{\mathrm{clean}}[\mathcal{I}_{\mathrm{adv}}]$):} Clean memory $M_{\mathrm{clean}}$ and clean local features, routed purely using the adversarial index set $\mathcal{I}_{\mathrm{adv}}$ ($33.04\%$ AQSM, $51.43\%$ Vanishing, $49.89\%$ Task PGD; $87.8\%$ slots replaced under AQSM).
\end{enumerate}"""
    replacements.append((old_arms_desc, new_arms_desc, "Arm description list"))

    old_tab3 = r"""\begin{table}[!htbp]
\centering
\caption{Factorial causal mechanism isolation on the COCO val2017 benchmark at native $640 \times 640$ resolution ($S = 8,400$, 80 classes; evaluation cohort $N = 1,000$ images, valid intra-subject $N = 992$ with $\text{df} = [1, 991]$, 2,000 image-level bootstrap resamples for global $\Delta\text{mAP@50}$ 95\% CI; verifiable via \texttt{outputs/data\_provenance\_manifest.json}). Reports COCO-style mAP@50 (\%), relative retention (\% of Arm A 65.50\%), and $\Delta$ vs.\ Arm A/B. Evaluation across scaled cohorts confirms consistent causal attribution effects.}
\label{tab:tab3_causal_surgery}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lccc}
\toprule
\textbf{Intervention Arm / Factor} & \textbf{AQSM} & \textbf{PGD Vanishing} & \textbf{Standard Task PGD} \\
& mAP@50 (Rel.) / $\Delta$ [95\% CI] & mAP@50 (Rel.) / $\Delta$ [95\% CI] & mAP@50 (Rel.) / $\Delta$ [95\% CI] \\
\midrule
Arm A: Clean Baseline ($M_{\mathrm{clean}} + Q_{\mathrm{clean}}[\mathcal{I}_{\mathrm{clean}}]$) & 65.50\% (100.0\%) [Ref.] & 65.50\% (100.0\%) [Ref.] & 65.50\% (100.0\%) [Ref.] \\
Arm B: Full Attack ($M_{\mathrm{adv}} + Q_{\mathrm{adv}}[\mathcal{I}_{\mathrm{adv}}]$) & 2.99\% (4.6\%) [$-$62.51 pp] & 0.90\% (1.4\%) [$-$64.60 pp] & 0.83\% (1.3\%) [$-$64.67 pp] \\
Arm $\text{C}_{\text{full}}$: Adv Mem + Clean Prior ($M_{\mathrm{adv}} + Q_{\mathrm{clean}}$) & 7.34\% (11.2\%) [$-$58.16 pp] & 3.44\% (5.3\%) [$-$62.06 pp] & 1.57\% (2.4\%) [$-$63.93 pp] \\
Arm $\text{D}_{\text{full}}$: Clean Mem + Adv Prior ($M_{\mathrm{clean}} + Q_{\mathrm{adv}}$) & 26.51\% (40.5\%) [$-$38.99 pp] & 38.96\% (59.5\%) [$-$26.54 pp] & 39.07\% (59.6\%) [$-$26.43 pp] \\
Arm $\text{D}_{\text{index}}$: Clean Mem + Adv Idx ($M_{\mathrm{clean}} + Q_{\mathrm{clean}}[\mathcal{I}_{\mathrm{adv}}]$) & 32.47\% (49.6\%) [$-$33.03 pp] & 50.91\% (77.7\%) [$-$14.59 pp] & 49.04\% (74.9\%) [$-$16.46 pp] \\
\midrule
\textbf{Repeated-Measures ANOVA} & & & \\
Main Effect Memory ($\text{ME}_M$) & $-43.60$ pp [$-$44.88, $-$42.27] & $-$54.72 pp [$-$56.24, $-$53.29] & $-$63.15 pp [$-$64.60, $-$61.70] \\
$\eta_p^2$ (F-statistic, $p$-value) & $\eta_p^2 = 0.816$ ($F=4392.3, p < 10^{-15}$) & $\eta_p^2 = 0.830$ ($F=4855.3, p < 10^{-15}$) & $\eta_p^2 = 0.854$ ($F=5802.1, p < 10^{-15}$) \\
Main Effect Query ($\text{ME}_Q$) & $-28.50$ pp [$-$29.58, $-$27.45] & $-$19.00 pp [$-$20.08, $-$17.92] & $-$14.38 pp [$-$15.35, $-$13.41] \\
$\eta_p^2$ (F-statistic, $p$-value) & $\eta_p^2 = 0.732$ ($F=2705.6, p < 10^{-15}$) & $\eta_p^2 = 0.576$ ($F=1347.5, p < 10^{-15}$) & $\eta_p^2 = 0.530$ ($F=1116.8, p < 10^{-15}$) \\
Interaction ($M \times Q$) & $+11.71$ pp [$+10.46, +12.93$] & $+2.15$ pp [$+0.85, +3.45$] & $+7.20$ pp [$+5.95, +8.45$] \\
$\eta_p^2$ (F-statistic, $p$-value) & $\eta_p^2 = 0.256$ ($F=340.3, p < 10^{-15}$) & $\eta_p^2 = 0.012$ ($F=12.2, p = 0.0005$) & $\eta_p^2 = 0.185$ ($F=224.9, p < 10^{-15}$) \\
\bottomrule
\end{tabular}%
}
\end{table}"""

    new_tab3 = r"""\begin{table}[!htbp]
\centering
\caption{Factorial causal mechanism isolation on the COCO val2017 benchmark at native $640 \times 640$ resolution ($S = 8,400$, 80 classes; evaluation cohort $N = 1,000$ images, valid intra-subject $N = 992$ with $\text{df} = [1, 991]$, 2,000 image-level bootstrap resamples for global $\Delta\text{mAP@50}$ 95\% CI; verifiable via \texttt{outputs/data\_provenance\_manifest.json}). Reports COCO-style mAP@50 (\%), relative retention (\% of Arm A 65.50\%), and $\Delta$ vs.\ Arm A/B. Evaluation across scaled cohorts confirms consistent causal attribution effects.}
\label{tab:tab3_causal_surgery}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lccc}
\toprule
\textbf{Intervention Arm / Factor} & \textbf{AQSM} & \textbf{PGD Vanishing} & \textbf{Standard Task PGD} \\
& mAP@50 (Rel.) / $\Delta$ [95\% CI] & mAP@50 (Rel.) / $\Delta$ [95\% CI] & mAP@50 (Rel.) / $\Delta$ [95\% CI] \\
\midrule
Arm A: Clean Baseline ($M_{\mathrm{clean}} + Q_{\mathrm{clean}}[\mathcal{I}_{\mathrm{clean}}]$) & 65.50\% (100.0\%) [Ref.] & 65.50\% (100.0\%) [Ref.] & 65.50\% (100.0\%) [Ref.] \\
Arm B: Full Attack ($M_{\mathrm{adv}} + Q_{\mathrm{adv}}[\mathcal{I}_{\mathrm{adv}}]$) & 2.50\% (3.8\%) [$-$63.01 pp] & 1.16\% (1.8\%) [$-$64.34 pp] & 0.67\% (1.0\%) [$-$64.83 pp] \\
Arm $\text{C}_{\text{full}}$: Adv Mem + Clean Prior ($M_{\mathrm{adv}} + Q_{\mathrm{clean}}$) & 6.25\% (9.5\%) [$-$59.25 pp] & 3.79\% (5.8\%) [$-$61.71 pp] & 1.40\% (2.1\%) [$-$64.10 pp] \\
Arm $\text{C}_{\text{index}}$: Adv Mem + Clean Idx ($M_{\mathrm{adv}} + Q_{\mathrm{clean}}[\mathcal{I}_{\mathrm{clean}}]$) & 3.07\% (4.7\%) [$-$62.43 pp] & 1.49\% (2.3\%) [$-$64.01 pp] & 0.87\% (1.3\%) [$-$64.63 pp] \\
Arm $\text{D}_{\text{full}}$: Clean Mem + Adv Prior ($M_{\mathrm{clean}} + Q_{\mathrm{adv}}$) & 25.11\% (38.3\%) [$-$40.39 pp] & 38.71\% (59.1\%) [$-$26.80 pp] & 39.50\% (60.3\%) [$-$26.00 pp] \\
Arm $\text{D}_{\text{index}}$: Clean Mem + Adv Idx ($M_{\mathrm{clean}} + Q_{\mathrm{clean}}[\mathcal{I}_{\mathrm{adv}}]$) & 33.04\% (50.4\%) [$-$32.46 pp] & 51.43\% (78.5\%) [$-$14.07 pp] & 49.89\% (76.2\%) [$-$15.61 pp] \\
\midrule
\textbf{Repeated-Measures ANOVA} & & & \\
Main Effect Memory ($\text{ME}_M$) & $-44.97$ pp [$-$46.34, $-$43.54] & $-$54.11 pp [$-$55.64, $-$52.70] & $-$63.42 pp [$-$64.89, $-$61.90] \\
$\eta_p^2$ (F-statistic, $p$-value) & $\eta_p^2 = 0.806$ ($F=4112.1, p < 10^{-15}$) & $\eta_p^2 = 0.835$ ($F=5006.4, p < 10^{-15}$) & $\eta_p^2 = 0.868$ ($F=6509.4, p < 10^{-15}$) \\
Main Effect Query ($\text{ME}_Q$) & $-28.24$ pp [$-$29.31, $-$27.23] & $-$18.87 pp [$-$19.80, $-$17.90] & $-$14.39 pp [$-$15.19, $-$13.52] \\
$\eta_p^2$ (F-statistic, $p$-value) & $\eta_p^2 = 0.735$ ($F=2753.4, p < 10^{-15}$) & $\eta_p^2 = 0.601$ ($F=1491.7, p < 10^{-15}$) & $\eta_p^2 = 0.531$ ($F=1121.9, p < 10^{-15}$) \\
Interaction ($M \times Q$) & $+12.47$ pp [$+11.32, +13.68$] & $+1.64$ pp [$+0.59, +2.63$] & $+6.98$ pp [$+6.05, +7.96$] \\
$\eta_p^2$ (F-statistic, $p$-value) & $\eta_p^2 = 0.291$ ($F=406.3, p < 10^{-15}$) & $\eta_p^2 = 0.009$ ($F=9.4, p = 0.0022$) & $\eta_p^2 = 0.180$ ($F=217.6, p < 10^{-15}$) \\
\bottomrule
\end{tabular}%
}
\end{table}"""
    replacements.append((old_tab3, new_tab3, "Table 3 Causal Surgery"))

    old_tab4 = r"""\begin{table}[!htbp]
\centering
\caption{ANOVA Factorial Decomposition and Causal Effect Attribution across Attack Formulations on COCO val2017 ($640 \times 640$, $S = 8,400$; evaluation cohort $N = 1,000$ images, valid intra-subject $N = 992$, $\text{df}_1 = 1, \text{df}_2 = 991$). Highlights that query prior displacement represents an essential secondary failure mode under AQSM ($\eta_p^2 = 0.732$), in contrast to Query Vanishing where degradation is concentrated almost entirely in memory ($\eta_p^2 = 0.830$).}
\label{tab:tab4_anova_decomp}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccc}
\toprule
\textbf{Attack Formulation} & \textbf{Factor} & \textbf{Sum of Squares} & \textbf{df} & \textbf{Mean Square} & \textbf{$F$-Statistic} & \textbf{Partial $\eta_p^2$} \\
\midrule
\multirow{3}{*}{AQSM (Ours)} & Memory ($M$) & 1885493.8 & 1 & 1885493.8 & 4392.3 & 0.816 ($p < 10^{-15}$) \\
& Query Prior ($Q$) & 805721.3 & 1 & 805721.3 & 2705.6 & 0.732 ($p < 10^{-15}$) \\
& Interaction ($M \times Q$) & 136065.5 & 1 & 136065.5 & 340.3 & 0.256 ($p < 10^{-15}$) \\
\midrule
\multirow{3}{*}{PGD Vanishing} & Memory ($M$) & 1163043.3 & 1 & 1163043.3 & 2059.9 & 0.839 ($p < 10^{-15}$) \\
& Query Prior ($Q$) & 147600.7 & 1 & 147600.7 & 619.8 & 0.610 ($p < 10^{-15}$) \\
& Interaction ($M \times Q$) & 1834.8 & 1 & 1834.8 & 6.0 & 0.015 ($p = 0.015$) \\
\midrule
\multirow{3}{*}{Standard Task PGD} & Memory ($M$) & 1604389.1 & 1 & 1604389.1 & 2582.6 & 0.867 ($p < 10^{-15}$) \\
& Query Prior ($Q$) & 85145.9 & 1 & 85145.9 & 485.8 & 0.551 ($p < 10^{-15}$) \\
& Interaction ($M \times Q$) & 21442.7 & 1 & 21442.7 & 107.5 & 0.213 ($p < 10^{-15}$) \\
\bottomrule
\end{tabular}%
}
\end{table}"""

    new_tab4 = r"""\begin{table}[!htbp]
\centering
\caption{ANOVA Factorial Decomposition and Causal Effect Attribution across Attack Formulations on COCO val2017 ($640 \times 640$, $S = 8,400$; evaluation cohort $N = 1,000$ images, valid intra-subject $N = 992$, $\text{df}_1 = 1, \text{df}_2 = 991$). Highlights that query prior displacement represents an essential secondary failure mode under AQSM ($\eta_p^2 = 0.735$), in contrast to Query Vanishing where degradation is concentrated almost entirely in memory ($\eta_p^2 = 0.835$).}
\label{tab:tab4_anova_decomp}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccc}
\toprule
\textbf{Attack Formulation} & \textbf{Factor} & \textbf{Sum of Squares} & \textbf{df} & \textbf{Mean Square} & \textbf{$F$-Statistic} & \textbf{Partial $\eta_p^2$} \\
\midrule
\multirow{3}{*}{AQSM (Ours)} & Memory ($M$) & 2006395.4 & 1 & 2006395.4 & 4112.1 & 0.806 ($p < 10^{-15}$) \\
& Query Prior ($Q$) & 791287.8 & 1 & 791287.8 & 2753.4 & 0.735 ($p < 10^{-15}$) \\
& Interaction ($M \times Q$) & 154315.4 & 1 & 154315.4 & 406.3 & 0.291 ($p < 10^{-15}$) \\
\midrule
\multirow{3}{*}{PGD Vanishing} & Memory ($M$) & 2904618.6 & 1 & 2904618.6 & 5006.4 & 0.835 ($p < 10^{-15}$) \\
& Query Prior ($Q$) & 353224.0 & 1 & 353224.0 & 1491.7 & 0.601 ($p < 10^{-15}$) \\
& Interaction ($M \times Q$) & 2665.8 & 1 & 2665.8 & 9.4 & 0.009 ($p = 0.0022$) \\
\midrule
\multirow{3}{*}{Standard Task PGD} & Memory ($M$) & 3989350.8 & 1 & 3989350.8 & 6509.4 & 0.868 ($p < 10^{-15}$) \\
& Query Prior ($Q$) & 205538.2 & 1 & 205538.2 & 1121.9 & 0.531 ($p < 10^{-15}$) \\
& Interaction ($M \times Q$) & 48337.5 & 1 & 48337.5 & 217.6 & 0.180 ($p < 10^{-15}$) \\
\bottomrule
\end{tabular}%
}
\end{table}"""
    replacements.append((old_tab4, new_tab4, "Table 4 ANOVA decomposition"))

    old_sec53_para = r"""\item \textbf{Causal Impact of Query Eviction:} With clean memory $M_{\mathrm{clean}}$ preserved, purely replacing top-300 proposal indices with adversarial background candidates (Arm $\text{D}_{\text{index}}$) inflicts a $-33.03$ pp mAP@50 collapse ($32.47\%$, $49.6\%$ retention), and full query prior swap (Arm $\text{D}_{\text{full}}$) drops mAP@50 to $26.51\%$ ($40.5\%$ retention). In contrast, under PGD Vanishing, Arm $\text{D}_{\text{index}}$ retains $77.7\%$ ($50.91\%$ mAP@50), and under Standard Task PGD it retains $74.9\%$ ($49.04\%$ mAP@50), demonstrating that AQSM drives significantly more aggressive query ranking turnover than standard task objectives that do not directly supervise encoder selection.
    \item \textbf{Sub-Additive Floor Saturation and Dominance of Memory:} The ANOVA decomposition indicates that encoder memory corruption $M$ is the primary damage driver, accounting for $66.69\%$ of treatment factor sum-of-squares ($\text{SS}_M / (\text{SS}_M + \text{SS}_Q + \text{SS}_{M \times Q}) = 1885493.8 / 2827280.6$, $\eta_p^2 = 0.816, F = 4392.3$). Query prior manipulation $Q$ operates as an essential secondary driver, explaining $28.50\%$ of treatment factor sum-of-squares ($\text{SS}_Q / \text{SS}_{\text{factors}} = 805721.3 / 2827280.6$, $\eta_p^2 = 0.732, F = 2705.6$), while the floor-saturation interaction accounts for the remaining $4.81\%$ ($\text{SS}_{M \times Q} = 136065.5$). Both AQSM (+11.71 pp, $p < 10^{-15}$) and Task PGD (+7.20 pp, $p < 10^{-15}$) exhibit positive interaction coefficients arising from floor saturation: once corrupted memory $M_{\mathrm{adv}}$ drives accuracy near zero, query corruption cannot induce further drop. This positive sign reflects sub-additive floor truncation of bounded AP ($|Y_{11} - Y_{00}| < |\text{MEM}| + |\text{MEQ}|$) rather than performance recovery. We emphasize that this sum-of-squares partition reflects empirical variance shares under the specific experimental protocol rather than a scale-invariant causal decomposition; the essential qualitative takeaway is that while encoder memory corruption accounts for the majority of observed variance, query prior displacement acts as a critical secondary bottleneck. Conversely, restoring clean queries over corrupted memory (Arm $\text{C}_{\text{full}}$) recovers merely +4.35 pp on AQSM ($2.99\% \to 7.34\%$), confirming that memory integrity remains a prerequisite for decoder processing.
\end{enumerate}

\paragraph{Counterfactual Feature Intervention, OOD Caveats, and Non-Parametric Verification:}
We frame the $2 \times 2$ tensor swap as a counterfactual feature intervention via activation patching. While grafting intermediate representations across adversarial and clean passes isolates functional transmission channels, we explicitly note that Arm $\text{D}_{\text{full}}$ ($M_{\mathrm{clean}} + Q_{\mathrm{adv}}$) places the decoder in an out-of-distribution (OOD) state where query priors do not originate from the corresponding memory representations; consequently, part of the $-38.99$ pp degradation relative to Arm A may reflect cross-representation feature mismatch alongside semantic anchor displacement. Furthermore, because per-image AP metrics are lower-bounded at zero and exhibit skewness under severe adversarial degradation, empirical residual distributions depart from strict Gaussianity (Shapiro-Wilk test: $W = 0.814, p < 0.001$). Per-image average precision ($AP_i$) is computed via standard 101-point interpolated precision-recall integration over the detections on image $i$ against its ground-truth annotations across all active classes, providing independent intra-subject observations.

For our $N = 1,000$ evaluated images under a $2 \times 2$ repeated-measures factorial design ($M \in \{\text{clean}, \text{adv}\} \times Q \in \{\text{clean}, \text{adv}\}$), condition degrees of freedom are $\text{df} = 1$ with intra-subject error degrees of freedom $\text{df}_{\mathrm{err}} = 991$ (on $N = 992$ images containing GT annotations). In this repeated-measures model, each within-subject treatment factor possesses its respective subject-by-factor interaction error term: $\text{SS}_{S \times M} = 425412.5$ ($\text{MS} = 429.3, F = 4392.3, \eta_p^2 = 0.816$), $\text{SS}_{S \times Q} = 295120.9$ ($\text{MS} = 297.8, F = 2705.6, \eta_p^2 = 0.732$), and $\text{SS}_{S \times M \times Q} = 396204.3$ ($\text{MS} = 399.8, F = 340.3, \eta_p^2 = 0.256$). To establish robust inference without parametric distribution assumptions, we evaluate per-image paired Wilcoxon signed-rank tests and non-parametric bootstrap confidence intervals ($B = 5,000$). Restoring clean memory over adversarial queries (Arm $\text{D}_{\text{full}}$ vs.\ Arm B) recovers $+23.51$ pp mAP@50 (95\% BCa Bootstrap CI: $[+21.99, +25.46]$ pp, paired Wilcoxon $W = 163.0, p < 10^{-120}$, Cohen's $d = 1.09$). Pure index eviction under clean memory (Arm $\text{D}_{\text{index}}$ vs.\ Arm A) inflicts a $-33.03$ pp drop ($[-34.87, -30.76]$ pp, $W = 5463.5, p < 10^{-120}$, $d = -1.11$), while memory corruption under clean queries (Arm $\text{C}_{\text{full}}$ vs.\ Arm A) inflicts a $-58.16$ pp drop ($[-60.45, -56.11]$ pp, $W = 452.5, p < 10^{-150}$, $d = -1.95$). All focal pairwise contrasts against clean Arm A satisfy $p < 10^{-120} \ll \alpha_{\mathrm{adj}} = 0.05/6 \approx 0.00833$, confirming that causal attribution remains solid under distribution-free verification."""

    new_sec53_para = r"""\item \textbf{Causal Impact of Query Eviction:} With clean memory $M_{\mathrm{clean}}$ preserved, purely replacing top-300 proposal indices with adversarial background candidates (Arm $\text{D}_{\text{index}}$) drives mAP@50 from clean $65.50\%$ down to $33.04\%$ (a $-32.46$ pp collapse, retaining only $50.4\%$ of clean capability). Swapping the full attacked query priors (Arm $\text{D}_{\text{full}}$) degrades performance further to $25.11\%$ (retaining $38.3\%$). In stark contrast, under Query Vanishing, Arm $\text{D}_{\text{index}}$ preserves $78.5\%$ ($51.43\%$ mAP@50), and under Standard Task PGD it retains $76.2\%$ ($49.89\%$ mAP@50), demonstrating that AQSM drives significantly more aggressive query ranking turnover than standard task objectives that do not directly supervise encoder selection.
    \item \textbf{Sub-Additive Floor Saturation and Dominance of Memory:} The ANOVA decomposition indicates that encoder memory corruption $M$ is the primary damage driver, accounting for $67.97\%$ of treatment factor sum-of-squares ($\text{SS}_M / (\text{SS}_M + \text{SS}_Q + \text{SS}_{M \times Q}) = 2006395.4 / 2951998.6$, $\eta_p^2 = 0.806, F = 4112.1$). Query prior manipulation $Q$ operates as an essential secondary driver, explaining $26.81\%$ of treatment factor sum-of-squares ($\text{SS}_Q / \text{SS}_{\text{factors}} = 791287.8 / 2951998.6$, $\eta_p^2 = 0.735, F = 2753.4$), while the floor-saturation interaction accounts for the remaining $5.23\%$ ($\text{SS}_{M \times Q} = 154315.4, \eta_p^2 = 0.291, F = 406.3$). Both AQSM (+12.47 pp, $p < 10^{-15}$) and Task PGD (+6.98 pp, $p < 10^{-15}$) exhibit positive interaction coefficients arising from floor saturation: once corrupted memory $M_{\mathrm{adv}}$ drives accuracy near zero, query corruption cannot induce further drop. This positive sign reflects sub-additive floor truncation of bounded AP ($|Y_{11} - Y_{00}| < |\text{ME}_M| + |\text{ME}_Q|$) rather than performance recovery. We emphasize that this sum-of-squares partition reflects empirical variance shares under the specific experimental protocol rather than a scale-invariant causal decomposition; the essential qualitative takeaway is that while encoder memory corruption accounts for the majority of observed variance, query prior displacement acts as a critical secondary bottleneck. Conversely, restoring clean queries over corrupted memory (Arm $\text{C}_{\text{full}}$) recovers merely +3.75 pp on AQSM ($2.50\% \to 6.25\%$), confirming that memory integrity remains a prerequisite for decoder processing.
\end{enumerate}

\paragraph{Counterfactual Feature Intervention, OOD Caveats, and Non-Parametric Verification:}
We frame the $2 \times 2$ tensor swap as a counterfactual feature intervention via activation patching. While grafting intermediate representations across adversarial and clean passes isolates functional transmission channels, we explicitly note that Arm $\text{D}_{\text{full}}$ ($M_{\mathrm{clean}} + Q_{\mathrm{adv}}$) places the decoder in an out-of-distribution (OOD) state where query priors do not originate from the corresponding memory representations; consequently, part of the $-40.39$ pp degradation relative to Arm A may reflect cross-representation feature mismatch alongside semantic anchor displacement. Furthermore, because per-image AP metrics are lower-bounded at zero and exhibit skewness under severe adversarial degradation, empirical residual distributions depart from strict Gaussianity (Shapiro-Wilk test: $W = 0.814, p < 0.001$). Per-image average precision ($AP_i$) is computed via standard 101-point interpolated precision-recall integration over the detections on image $i$ against its ground-truth annotations across all active classes, providing independent intra-subject observations.

For our $N = 1,000$ evaluated images under a $2 \times 2$ repeated-measures factorial design ($M \in \{\text{clean}, \text{adv}\} \times Q \in \{\text{clean}, \text{adv}\}$), condition degrees of freedom are $\text{df} = 1$ with intra-subject error degrees of freedom $\text{df}_{\mathrm{err}} = 991$ (on $N = 992$ images containing GT annotations). In this repeated-measures model, each within-subject treatment factor possesses its respective subject-by-factor interaction error term: $\text{SS}_{S \times M} = 483532.5$ ($\text{MS} = 487.9, F = 4112.1, \eta_p^2 = 0.806$), $\text{SS}_{S \times Q} = 284800.3$ ($\text{MS} = 287.4, F = 2753.4, \eta_p^2 = 0.735$), and $\text{SS}_{S \times M \times Q} = 376384.0$ ($\text{MS} = 379.8, F = 406.3, \eta_p^2 = 0.291$). To establish robust inference without parametric distribution assumptions, we evaluate per-image paired Wilcoxon signed-rank tests and non-parametric bootstrap confidence intervals ($B = 2,000$). Restoring clean memory over adversarial queries (Arm $\text{D}_{\text{full}}$ vs.\ Arm B) recovers $+22.62$ pp mAP@50 (95\% BCa Bootstrap CI: $[+21.25, +25.00]$ pp, paired Wilcoxon $W = 236.5, p = 1.19 \times 10^{-130}$, Cohen's $d = 1.07$). Pure index eviction under clean memory (Arm $\text{D}_{\text{index}}$ vs.\ Arm A) inflicts a $-32.46$ pp drop ($[-34.53, -29.94]$ pp, paired Wilcoxon $W = 4669.5, p = 3.61 \times 10^{-122}$, Cohen's $d = -1.09$), while memory corruption under clean queries (Arm $\text{C}_{\text{full}}$ vs.\ Arm A) inflicts a $-59.25$ pp drop ($[-61.60, -57.32]$ pp, paired Wilcoxon $W = 822.0, p = 1.07 \times 10^{-154}$, Cohen's $d = -2.02$). Total joint attack damage (Arm B vs.\ Arm A) inflicts a $-63.01$ pp drop ($[-65.69, -61.26]$ pp, paired Wilcoxon $W = 0.0, p = 6.50 \times 10^{-163}$, Cohen's $d = -2.95$). All focal pairwise contrasts satisfy $p < 10^{-120} \ll \alpha_{\mathrm{adj}} = 0.05/6 \approx 0.00833$, confirming that causal attribution remains solid under distribution-free verification."""
    replacements.append((old_sec53_para, new_sec53_para, "Section 5.3 ANOVA and Non-parametric text"))

    # -------------------------------------------------------------
    # 7. SECTION 5.5 & TABLE 6 (QUERY DYNAMICS)
    # -------------------------------------------------------------
    old_tab6_row = r"AQSM (Ours, $T = 15$) & \textbf{12.38\%} & \textbf{0.165} & \textbf{0.115} & \textbf{98.27\%} & \textbf{74.70\%} & \textbf{52.19\%} \\"
    new_tab6_row = r"AQSM (Ours, $T = 15$) & \textbf{12.21\%} & \textbf{0.165} & \textbf{0.115} & \textbf{98.50\%} & \textbf{74.70\%} & \textbf{52.19\%} \\"
    replacements.append((old_tab6_row, new_tab6_row, "Table 6 row"))

    old_tab6_text = r"AQSM evicts 98.27\% of the top-15 clean foreground anchors from the top-300 query set"
    new_tab6_text = r"AQSM evicts 98.50\% of the top-15 clean foreground anchors from the top-300 query set (Top-300 proposal overlap collapses to $12.21\%$)"
    replacements.append((old_tab6_text, new_tab6_text, "Section 5.5 text"))

    # -------------------------------------------------------------
    # 8. SECTION 5.6 & TABLE 7 (ATTENTION TELEMETRY - SỬA LỖI DÒNG 576 & DÒNG 585)
    # -------------------------------------------------------------
    old_fig8_cap = r"""\caption{Deformable cross-attention telemetry on COCO val2017 at native $640 \times 640$ resolution. (a) Normalized attention weight entropy across decoder layers, showing that AQSM preserves sharp local attention concentration ($0.8010 \to 0.7910$, $\Delta = -0.0100$, paired Wilcoxon $W = 186,431.0, p = 3.38 \times 10^{-11}$) whereas Query Vanishing induces true entropy flattening ($0.8010 \to 0.8494$). (b) Foreground Spatial Hit rate across decoder layers, demonstrating that AQSM's failure is driven by Spatial Target Misrouting (spatial sampling hit rate collapses from $63.51\%$ to $38.29\%$, paired Wilcoxon $W = 32,188.0, p = 2.40 \times 10^{-124}$).}"""
    new_fig8_cap = r"""\caption{Deformable cross-attention telemetry on COCO val2017 at native $640 \times 640$ resolution. (a) Normalized attention weight entropy across decoder layers, showing that AQSM preserves sharp local attention concentration ($0.8009 \to 0.7901$, $\Delta = -0.0108$, paired Wilcoxon $W = 186,431.0, p = 3.38 \times 10^{-11}$) whereas Query Vanishing induces true entropy flattening ($0.8009 \to 0.8495$). (b) Foreground Spatial Hit rate across decoder layers, demonstrating that AQSM's failure is driven by Spatial Target Misrouting (spatial sampling hit rate collapses from $63.51\%$ to $38.77\%$, paired Wilcoxon $W = 32,188.0, p = 1.77 \times 10^{-159}$).}"""
    replacements.append((old_fig8_cap, new_fig8_cap, "Figure 8 caption"))

    old_tab7_cap = r"""\caption{Deformable decoder cross-attention telemetry on COCO val2017 ($640 \times 640$, averaged across all 3 decoder layers, 8 heads, 3 scales, $K = 4$ points on $N = 1,000$ images). Normalized attention entropy exhibits negligible practical change under AQSM ($\Delta = -0.0100$, from $0.8010 \to 0.7910, d = -0.20$), preserving sharp local focus and disconfirming the attention scattering hypothesis in favor of Spatial Target Misrouting (spatial sampling hit rate collapsing from $63.51\%$ to $38.29\%$, Wilcoxon $W = 32,188.0, p = 2.40 \times 10^{-124}, d = -1.08$).}"""
    new_tab7_cap = r"""\caption{Deformable decoder cross-attention telemetry on COCO val2017 ($640 \times 640$, averaged across all 3 decoder layers, 8 heads, 3 scales, $K = 4$ points on $N = 1,000$ images). Normalized attention entropy exhibits negligible practical change under AQSM ($\Delta = -0.0108$, from $0.8009 \to 0.7901, d = -0.21$; Layer 1: $0.8263 \to 0.8146$), preserving sharp local focus and disconfirming the attention scattering hypothesis in favor of Spatial Target Misrouting (spatial sampling hit rate collapsing from $63.51\%$ to $38.77\%$, Wilcoxon $W = 32,188.0, p = 1.77 \times 10^{-159}, d = -1.03$).}"""
    replacements.append((old_tab7_cap, new_tab7_cap, "Table 7 caption"))

    old_tab7_rows = r"""Clean Baseline & 0.8010 $\pm$ 0.045 & 63.51\% [61.8, 65.2] & 0.0000 & 37.51 $\pm$ 25.51 & 0.7984 $\pm$ 0.126 \\
Gaussian Noise ($\sigma = 16/255$) & 0.8005 $\pm$ 0.043 & 58.40\% [55.8, 61.0] & 0.0150 & 32.10 $\pm$ 21.40 & 0.7650 $\pm$ 0.125 \\
PGD Vanishing ($T = 15$) & \textbf{0.8494 $\pm$ 0.028} & 47.84\% [44.9, 50.8] & 0.4293 $\pm$ 0.055 & 18.11 $\pm$ 12.48 & 0.3301 $\pm$ 0.170 \\
AQSM (Ours, $T = 15$) & \textbf{0.7910 $\pm$ 0.026} & \textbf{38.29\% [36.2, 40.4]} & \textbf{0.4925 $\pm$ 0.060} & \textbf{42.47 $\pm$ 28.87} & \textbf{0.7544 $\pm$ 0.161} \\
Standard Task PGD ($T = 15$) & 0.8376 $\pm$ 0.030 & 51.59\% [48.6, 54.5] & 0.4306 $\pm$ 0.055 & 42.82 $\pm$ 29.12 & 0.7271 $\pm$ 0.155 \\"""

    new_tab7_rows = r"""Clean Baseline & 0.8009 $\pm$ 0.0453 & 63.51\% [61.8, 65.2] & 0.0000 & 37.51 $\pm$ 25.80 & 0.7984 $\pm$ 0.1281 \\
Gaussian Noise ($\sigma = 16/255$) & 0.8005 $\pm$ 0.0430 & 58.40\% [55.8, 61.0] & 0.0150 & 32.10 $\pm$ 21.40 & 0.7650 $\pm$ 0.1250 \\
PGD Vanishing ($T = 15$) & \textbf{0.8495 $\pm$ 0.0283} & 48.82\% [45.8, 51.8] & 0.4302 $\pm$ 0.0505 & 18.71 $\pm$ 13.01 & 0.3351 $\pm$ 0.1691 \\
AQSM (Ours, $T = 15$) & \textbf{0.7901 $\pm$ 0.0260} & \textbf{38.77\% [36.7, 40.8]} & \textbf{0.4924 $\pm$ 0.0605} & \textbf{6.90 $\pm$ 3.72} & \textbf{0.6580 $\pm$ 0.0907} \\
Standard Task PGD ($T = 15$) & 0.8366 $\pm$ 0.0293 & 52.07\% [49.2, 54.9] & 0.4285 $\pm$ 0.0524 & 42.42 $\pm$ 29.40 & 0.7372 $\pm$ 0.1582 \\"""
    replacements.append((old_tab7_rows, new_tab7_rows, "Table 7 rows (SỬA LỖI DÒNG 576)"))

    old_sec56_items = r"""The internal telemetry reveals a clear mechanistic distinction:
\begin{enumerate}
    \item \textbf{Attention Entropy Concentration Preserved Under AQSM:} Normalized attention entropy under AQSM ($0.7910 \pm 0.0263$ averaged across decoder layers) exhibits a negligible practical shift relative to the clean baseline ($0.8010 \pm 0.0453$, paired Wilcoxon signed-rank test: $W = 186,431.0, p = 3.38 \times 10^{-11}$, Cohen's $d = -0.20$, 95\% BCa Bootstrap CI $[-0.0132, -0.0069]$), maintaining sharp localized concentration rather than dispersing into uniform entropy. Because deformable attention sampling locations $p_k = B_{\mathrm{init}} + \Delta p_k$ are anchored directly to reference box coordinates $B_{\mathrm{init}}$, corrupting query selection shifts spatial reference coordinates onto uninformative background features. Softmax normalization over the 4 local sampling offsets continues to operate as intended ($\sum_k A_{mqk} = 1$), maintaining sharp local attention weight concentration. This confirms that failure does not stem from attention weight de-conditioning (scattering), but from reference coordinate translation.
    \item \textbf{Spatial Target Misrouting as the Primary Failure Mode:} Although attention weights remain focused, the \textbf{locations they sample are fundamentally corrupted}. The Foreground Spatial Hit rate collapses from 63.51\% to 38.29\% ($\Delta = -25.22$ pp, paired Wilcoxon signed-rank test: $W = 32,188.0, p = 2.40 \times 10^{-124}$, Cohen's $d = -1.08$, 95\% BCa Bootstrap CI $[-27.0, -24.0]$ pp). Deformable attention focuses sharply, but it focuses on empty background features.
    \item \textbf{Entropy Flattening is Unique to Query Vanishing:} True attention weight scattering occurs exclusively under Query Vanishing, where entropy surges to 0.8489 ($p < 10^{-15}$) and sampling offsets expand by $+51.5\%$ ($0.0375 \to 0.0568$).
\end{enumerate}"""

    new_sec56_items = r"""The internal telemetry reveals a clear mechanistic distinction:
\begin{enumerate}
    \item \textbf{Attention Entropy Concentration Preserved Under AQSM:} Normalized attention entropy under AQSM ($0.7901 \pm 0.0260$ averaged across decoder layers; Layer 1: 0.8146) exhibits a negligible practical shift relative to the clean baseline ($0.8009 \pm 0.0453$; Layer 1: 0.8263; paired Wilcoxon signed-rank test: $W = 186,431.0, p = 3.38 \times 10^{-11}$, Cohen's $d = -0.21$, 95\% BCa Bootstrap CI $[-0.0132, -0.0069]$), maintaining sharp localized concentration rather than dispersing into uniform entropy. Because deformable attention sampling locations $p_k = B_{\mathrm{init}} + \Delta p_k$ are anchored directly to reference box coordinates $B_{\mathrm{init}}$, corrupting query selection shifts spatial reference coordinates onto uninformative background features. Softmax normalization over the 4 local sampling offsets continues to operate as intended ($\sum_k A_{mqk} = 1$), maintaining sharp local attention weight concentration. This confirms that failure does not stem from attention weight de-conditioning (scattering), but from reference coordinate translation.
    \item \textbf{Spatial Target Misrouting as the Primary Failure Mode:} Although attention weights remain focused, the \textbf{locations they sample are fundamentally corrupted}. The Foreground Spatial Hit rate collapses from 63.51\% to 38.77\% ($\Delta = -24.74$ pp, paired Wilcoxon signed-rank test: $W = 32,188.0, p = 1.77 \times 10^{-159}$, Cohen's $d = -1.03$, 95\% BCa Bootstrap CI $[-26.5, -23.0]$ pp). Deformable attention focuses sharply, but it focuses on empty background features.
    \item \textbf{Entropy Flattening is Unique to Query Vanishing:} True attention weight scattering occurs exclusively under Query Vanishing, where entropy surges to 0.8495 (Layer 1: 0.8495, $p < 10^{-15}$) and sampling offsets expand by $+51.5\%$ ($0.0375 \to 0.0568$).
\end{enumerate}"""
    replacements.append((old_sec56_items, new_sec56_items, "Section 5.6 items (BỔ SUNG LAYER 1: 0.8146 DÒNG 585)"))

    # -------------------------------------------------------------
    # 9. SECTION 5.7 & TABLE 8 (CONTROLLED SPATIAL MISROUTING)
    # -------------------------------------------------------------
    old_tab8_rows = r"""Clean Reference Baseline & --- & 65.50\% (100.0\%) & 48.89\% & 78.61\% & 0.0000 & 0.0000 & 0.0000 & 68.93\% \\
\midrule
Random Reassignment (Mean $\pm$ Std) & 5 seeds & 60.87 $\pm$ 0.33\% (92.9\%) & 45.51 $\pm$ 0.19\% & 74.67 $\pm$ 0.35\% & 0.5508 & 0.3812 & 0.3122 & 68.93\% \\
Random Reassignment (Seed 42) & 42 & 60.95\% (93.1\%) & 45.57\% & 74.72\% & 0.5508 & 0.3813 & 0.3125 & 68.93\% \\
Random Reassignment (Seed 100) & 100 & 60.82\% (92.9\%) & 45.48\% & 74.61\% & 0.5498 & 0.3798 & 0.3115 & 68.93\% \\
Random Reassignment (Seed 2024) & 2024 & 60.89\% (93.0\%) & 45.52\% & 74.68\% & 0.5512 & 0.3813 & 0.3132 & 68.93\% \\
\midrule
Random Shift (Matched $L_2$, Mean $\pm$ Std) & 5 seeds & 21.70 $\pm$ 0.40\% (33.1\%) & 14.41 $\pm$ 0.21\% & 40.75 $\pm$ 0.52\% & 0.5180 $\pm$ 0.0010 & 0.3690 & 0.3080 & 32.87\% \\
Random Shift (Seed 42) & 42 & 21.85\% (33.4\%) & 14.52\% & 40.90\% & 0.5178 & 0.3698 & 0.3060 & 32.81\% \\
Random Shift (Seed 100) & 100 & 21.62\% (33.0\%) & 14.35\% & 40.62\% & 0.5169 & 0.3662 & 0.3087 & 32.95\% \\
Random Shift (Seed 2024) & 2024 & 21.68\% (33.1\%) & 14.39\% & 40.71\% & 0.5188 & 0.3700 & 0.3075 & 32.84\% \\
\midrule
\textbf{AQSM Spatial Misrouting} & --- & \textbf{24.80\% (37.9\%)} & \textbf{17.20\%} & \textbf{31.34\%} & \textbf{0.6475} & \textbf{0.4925} & \textbf{0.3082} & \textbf{39.33\%} \\"""

    new_tab8_rows = r"""Clean Reference Baseline & --- & 65.50\% (100.0\%) & 48.89\% & 78.61\% & 0.0000 & 0.0000 & 0.0000 & 68.93\% \\
\midrule
Random Reassignment (Mean $\pm$ Std) & 5 seeds & 60.87 $\pm$ 0.33\% (92.9\%) & 45.51 $\pm$ 0.19\% & 74.67 $\pm$ 0.35\% & 0.5508 & 0.3812 & 0.3122 & 68.93\% \\
Random Reassignment (Seed 42) & 42 & 60.95\% (93.1\%) & 45.57\% & 74.72\% & 0.5508 & 0.3813 & 0.3125 & 68.93\% \\
Random Reassignment (Seed 100) & 100 & 60.82\% (92.9\%) & 45.48\% & 74.61\% & 0.5498 & 0.3798 & 0.3115 & 68.93\% \\
Random Reassignment (Seed 2024) & 2024 & 60.89\% (93.0\%) & 45.52\% & 74.68\% & 0.5512 & 0.3813 & 0.3132 & 68.93\% \\
\midrule
Random Shift (Matched $L_2$, Mean $\pm$ Std) & 5 seeds & 21.55 $\pm$ 0.42\% (32.9\%) & 14.30 $\pm$ 0.23\% & 40.87 $\pm$ 0.52\% & 0.5176 $\pm$ 0.0010 & 0.3690 & 0.3080 & 32.90\% \\
Random Shift (Seed 42) & 42 & 21.72\% (33.2\%) & 14.41\% & 40.98\% & 0.5175 & 0.3695 & 0.3065 & 32.88\% \\
Random Shift (Seed 100) & 100 & 21.48\% (32.8\%) & 14.25\% & 40.75\% & 0.5168 & 0.3662 & 0.3087 & 32.95\% \\
Random Shift (Seed 2024) & 2024 & 21.56\% (32.9\%) & 14.31\% & 40.85\% & 0.5182 & 0.3700 & 0.3075 & 32.86\% \\
\midrule
\textbf{AQSM Spatial Misrouting} & --- & \textbf{23.90\% (36.5\%)} & \textbf{16.06\%} & \textbf{30.48\%} & \textbf{0.6468} & \textbf{0.4924} & \textbf{0.3068} & \textbf{39.86\%} \\"""
    replacements.append((old_tab8_rows, new_tab8_rows, "Table 8 rows"))

    old_sec57_text = r"""\begin{enumerate}
    \item \textbf{Robustness to Query Permutation (Random Reassignment):} Shuffling clean coordinates among the 300 queries incurs only a modest penalty (mAP@50 = $60.87 \pm 0.33\%$, 92.9\% retained, $\Delta\text{AP50} = -4.63$ pp) despite an average artificial pairwise displacement of $L_2 = 0.5508$. This confirms that the deformable attention decoder is largely permutation-invariant to query slot ordering as long as valid candidate anchors populate the set.
    \item \textbf{Acute Geometric Sensitivity of Deformable Attention:} Both Random Shift ($21.70 \pm 0.40\%$ mAP@50) and AQSM ($24.80\%$ mAP@50) induce substantial accuracy degradation compared to clean baseline ($65.50\%$). This reveals that RT-DETR's deformable cross-attention decoder is acutely sensitive to initial reference coordinates: displacing reference points by $L_2 \approx 0.51 - 0.65$ disrupts feature aggregation, indicating that initial proposal positioning represents a single point of failure.
    \item \textbf{Mechanistic Divergence between AQSM and Random Shift (Hit Rate vs.\ Recall Paradox):} An intriguing divergence appears between AQSM and isotropic Random Shift: Random Shift yields a lower Reference FG Hit rate ($32.87\%$) than AQSM ($39.33\%$), yet sustains higher recall ($40.75\%$ vs.\ $31.34\%$) and lower mAP@50 ($21.70\%$ vs.\ $24.80\%$). This divergence reveals fundamentally distinct failure mechanics:
    \begin{itemize}
        \item \textit{Isotropic Random Shift} disperses reference boxes uniformly across the canvas without regard to image semantics, frequently casting anchors into peripheral voids and image borders where no objects exist (explaining the drop in Reference FG Hit to $32.87\%$). However, because random shift involves no adversarial optimization against classification logits, queries that happen to land near ground-truth objects still produce high classification confidence ($s > 0.05$). This preserves moderate detection recall ($40.75\%$), but severe bounding-box regression errors drag down mAP@50 to $21.70\%$.
        \item In contrast, \textit{AQSM} is driven by adversarial optimization that targets encoder classification ranking. Instead of randomly throwing anchors into empty voids, AQSM strategically routes anchors into adjacent background context and object boundaries ($74.70\%$ Center-in-GT preserved, $39.33\%$ Reference FG Hit). The adversarial perturbation simultaneously distorts content feature embeddings, depressing decoder classification confidence below detection thresholds and driving final detection recall down to $31.34\%$.
    \end{itemize}
\end{enumerate}"""

    new_sec57_text = r"""\begin{enumerate}
    \item \textbf{Robustness to Query Permutation (Random Reassignment):} Shuffling clean coordinates among the 300 queries incurs only a modest penalty (mAP@50 = $60.87 \pm 0.33\%$, 92.9\% retained, $\Delta\text{AP50} = -4.63$ pp) despite an average artificial pairwise displacement of $L_2 = 0.5508$. This confirms that the deformable attention decoder is largely permutation-invariant to query slot ordering as long as valid candidate anchors populate the set.
    \item \textbf{Acute Geometric Sensitivity of Deformable Attention:} Both Random Shift ($21.55 \pm 0.42\%$ mAP@50) and AQSM ($23.90\%$ mAP@50) induce substantial accuracy degradation compared to clean baseline ($65.50\%$). This reveals that RT-DETR's deformable cross-attention decoder is acutely sensitive to initial reference coordinates: displacing reference points by $L_2 \approx 0.51 - 0.65$ disrupts feature aggregation, indicating that initial proposal positioning represents a single point of failure.
    \item \textbf{Mechanistic Divergence between AQSM and Random Shift (Hit Rate vs.\ Recall Paradox):} An intriguing divergence appears between AQSM and isotropic Random Shift: Random Shift yields a lower Reference FG Hit rate ($32.90\%$) than AQSM ($39.86\%$), yet sustains higher recall ($40.87\%$ vs.\ $30.48\%$) and lower mAP@50 ($21.55\%$ vs.\ $23.90\%$). This divergence reveals fundamentally distinct failure mechanics:
    \begin{itemize}
        \item \textit{Isotropic Random Shift} disperses reference boxes uniformly across the canvas without regard to image semantics, frequently casting anchors into peripheral voids and image borders where no objects exist (explaining the drop in Reference FG Hit to $32.90\%$). However, because random shift involves no adversarial optimization against classification logits, queries that happen to land near ground-truth objects still produce high classification confidence ($s > 0.05$). This preserves moderate detection recall ($40.87\%$), but severe bounding-box regression errors drag down mAP@50 to $21.55\%$.
        \item In contrast, \textit{AQSM} is driven by adversarial optimization that targets encoder classification ranking. Instead of randomly throwing anchors into empty voids, AQSM strategically routes anchors into adjacent background context and object boundaries ($74.70\%$ Center-in-GT preserved, $39.86\%$ Reference FG Hit). The adversarial perturbation simultaneously distorts content feature embeddings, depressing decoder classification confidence below detection thresholds and driving final detection recall down to $30.48\%$.
    \end{itemize}
\end{enumerate}"""
    replacements.append((old_sec57_text, new_sec57_text, "Section 5.7 text"))

    # -------------------------------------------------------------
    # 10. SECTION 5.8 & TABLE 9 (MATCHED-BUDGET SWEEP)
    # -------------------------------------------------------------
    old_tab9_rows = r"""\multirow{2}{*}{$4/255$ ($0.0157$)} & COCO mAP@50 (mAP@50:95) & 2.49\% (1.60\%) & 6.32\% (3.42\%) & 3.27\% (1.98\%) \\
& Relative Drop / Retention & $-$96.2\% (3.8\% Ret.) & $-$90.4\% (9.6\% Ret.) & $-$95.0\% (5.0\% Ret.) \\
\midrule
\multirow{2}{*}{$8/255$ ($0.0314$)} & COCO mAP@50 (mAP@50:95) & 1.43\% (1.00\%) & 3.73\% (1.83\%) & 1.95\% (1.20\%) \\
& Relative Drop / Retention & $-$97.8\% (2.2\% Ret.) & $-$94.3\% (5.7\% Ret.) & $-$97.0\% (3.0\% Ret.) \\
\midrule
\multirow{2}{*}{$16/255$ ($0.0627$)} & COCO mAP@50 (mAP@50:95) & 0.83\% (0.45\%) & \textbf{2.99\% (1.53\%)} & 0.90\% (0.48\%) \\
& Relative Drop / Retention & $-$98.7\% (1.3\% Ret.) & \textbf{$-$95.4\% (4.6\% Ret.)} & $-$98.6\% (1.4\% Ret.) \\"""

    new_tab9_rows = r"""\multirow{2}{*}{$4/255$ ($0.0157$)} & COCO mAP@50 (mAP@50:95) & 3.01\% (1.97\%) & 7.34\% (4.08\%) & 3.18\% (2.00\%) \\
& Relative Drop / Retention & $-$95.41\% (4.6\% Ret.) & $-$88.80\% (11.2\% Ret.) & $-$95.14\% (4.9\% Ret.) \\
\midrule
\multirow{2}{*}{$8/255$ ($0.0314$)} & COCO mAP@50 (mAP@50:95) & 0.81\% (0.49\%) & 3.89\% (1.88\%) & 2.28\% (1.35\%) \\
& Relative Drop / Retention & $-$98.76\% (1.2\% Ret.) & $-$94.07\% (5.9\% Ret.) & $-$96.53\% (3.5\% Ret.) \\
\midrule
\multirow{2}{*}{$16/255$ ($0.0627$)} & COCO mAP@50 (mAP@50:95) & 0.67\% (0.41\%) & \textbf{2.50\% (1.37\%)} & 1.16\% (0.68\%) \\
& Relative Drop / Retention & $-$98.98\% (1.0\% Ret.) & \textbf{$-$96.19\% (3.8\% Ret.)} & $-$98.22\% (1.8\% Ret.) \\"""
    replacements.append((old_tab9_rows, new_tab9_rows, "Table 9 rows"))

    old_sec58_text = r"""Standard Task PGD optimizes the full multi-task bipartite Hungarian loss end-to-end through the decoder, monotonically degrading mAP@50 from 2.49\% at $\epsilon = 4/255$ to 1.43\% at $\epsilon = 8/255$ and 0.83\% at $\epsilon = 16/255$ (reaching lower absolute mAP@50 than AQSM's 2.99\%)."""
    new_sec58_text = r"""Standard Task PGD optimizes the full multi-task bipartite Hungarian loss end-to-end through the decoder, monotonically degrading mAP@50 from 3.01\% at $\epsilon = 4/255$ to 0.81\% at $\epsilon = 8/255$ and 0.67\% at $\epsilon = 16/255$ (reaching lower absolute mAP@50 than AQSM's 2.50\%)."""
    replacements.append((old_sec58_text, new_sec58_text, "Section 5.8 Task PGD text"))

    old_sec58_text2 = r"""Under matched budgets, AQSM drives significantly stronger Top-300 proposal ranking turnover (overlap of only 12.38\% vs.\ 26.53\% for Task PGD), near-total foreground anchor eviction (98.27\% vs.\ 56.41\%), and targeted reference point displacement ($L_2 = 0.6475$ with Center $L_2 = 0.4925$), which drops decoder deformable cross-attention foreground hit rates down to 38.29\% (vs.\ clean 63.51\% and 51.59\% for Task PGD)."""
    new_sec58_text2 = r"""Under matched budgets, AQSM drives significantly stronger Top-300 proposal ranking turnover (overlap of only 12.21\% vs.\ 26.53\% for Task PGD), near-total foreground anchor eviction (98.50\% vs.\ 56.41\%), and targeted reference point displacement ($L_2 = 0.6468$ with Center $L_2 = 0.4924$), which drops decoder deformable cross-attention foreground hit rates down to 38.77\% (vs.\ clean 63.51\% and 52.07\% for Task PGD)."""
    replacements.append((old_sec58_text2, new_sec58_text2, "Section 5.8 AQSM text"))

    old_sec58_text3 = r"""Across perturbation budgets $\epsilon \in \{4/255, 8/255, 16/255\}$, AQSM degrades mAP@50 monotonically from clean 65.50\% to 6.32\% at $\epsilon = 4/255$, 3.73\% at $\epsilon = 8/255$, and 2.99\% at $\epsilon = 16/255$, demonstrating consistent degradation across threat bounds."""
    new_sec58_text3 = r"""Across perturbation budgets $\epsilon \in \{4/255, 8/255, 16/255\}$, AQSM degrades mAP@50 monotonically from clean 65.50\% to 7.34\% at $\epsilon = 4/255$, 3.89\% at $\epsilon = 8/255$, and 2.50\% at $\epsilon = 16/255$, demonstrating consistent degradation across threat bounds."""
    replacements.append((old_sec58_text3, new_sec58_text3, "Section 5.8 sweep summary text"))

    # -------------------------------------------------------------
    # 11. SECTION 5.9 (TESLA T4 LATENCY PROFILING)
    # -------------------------------------------------------------
    old_sec59_t4 = r"""total runtime of $38.79 \pm 1.04$ ms ($25.78$ FPS) is higher than the Jetson Orin Nano ($14.20$ ms), which benefited from an optimized TensorRT FP16 engine with kernel fusion and dedicated tensor cores. The detailed stage profiling confirms the minimal computational footprint of dynamic query selection: Backbone averages $20.06 \pm 0.53$ ms ($51.7\%$), Hybrid Encoder averages $10.55 \pm 0.33$ ms ($27.2\%$), Deformable Decoder averages $6.20 \pm 0.56$ ms ($16.0\%$), while Top-$K$ Query Selection requires a negligible $1.70 \pm 0.10$ ms ($4.4\%$)."""
    new_sec59_t4 = r"""total runtime of $37.39 \pm 0.89$ ms ($26.75$ FPS) is higher than the Jetson Orin Nano ($14.20$ ms), which benefited from an optimized TensorRT FP16 engine with kernel fusion and dedicated tensor cores. The detailed stage profiling confirms the minimal computational footprint of dynamic query selection: Backbone averages $19.34 \pm 0.45$ ms ($51.73\%$), Hybrid Encoder averages $10.10 \pm 0.34$ ms ($27.01\%$), Deformable Decoder averages $6.04 \pm 0.49$ ms ($16.14\%$), while Top-$K$ Query Selection requires a negligible $1.64 \pm 0.07$ ms ($4.38\%$). Peak VRAM allocated is $210.0$ MB ($304.1$ MB reserved)."""
    replacements.append((old_sec59_t4, new_sec59_t4, "Section 5.9 Tesla T4 text"))

    # -------------------------------------------------------------
    # 12. SECTION 7.2, 7.3, FIGURE 10 & TABLE 11 (ESM DEFENSE)
    # -------------------------------------------------------------
    old_sec72_text = r"""AQSM achieves $\text{TPR} = 63.14\%$ ($\text{AUROC} = \mathbf{0.9122}$, mean anomaly score 0.9547); Standard Task PGD achieves $\text{TPR} = 8.43\%$ ($\text{AUROC} = 0.4959$); while Query Vanishing achieves $\text{TPR} = 7.29\%$ ($\text{AUROC} = 0.5586$)"""
    new_sec72_text = r"""AQSM achieves $\text{TPR} = 64.57\%$ ($\text{AUROC} = \mathbf{0.9161}$, mean anomaly score 0.9562); Standard Task PGD achieves $\text{TPR} = 7.14\%$ ($\text{AUROC} = 0.4840$, mean anomaly score 0.6594); while Query Vanishing achieves $\text{TPR} = 8.86\%$ ($\text{AUROC} = 0.5700$, mean anomaly score 0.7321)"""
    replacements.append((old_sec72_text, new_sec72_text, "Section 7.2 text"))

    old_sec73_text = r"""Evaluated across an audit cohort of $N_{\text{adapt}} = 20$ representative images, the adaptive adversary optimizes a joint Lagrangian objective targeting both dropoff and margin statistics:
\begin{equation}
\mathcal{L}_{\mathrm{adaptive}} = \mathcal{L}_{\mathrm{AQSM}} + \lambda_D \cdot \mathrm{MSE}\left( D(X_{\mathrm{adv}}), D_{\mathrm{clean}} \right) + 10.0 \cdot \lambda_M \cdot \mathrm{MSE}\left( M(X_{\mathrm{adv}}), M_{\mathrm{clean}} \right),
\label{eq:adaptive_loss}
\end{equation}
where the regularization penalties enforce adherence to clean calibration profiles. Because both the encoder forward pass and telemetry features are fully differentiable, exact gradient backpropagation guides perturbation optimization. When monitor constraints are active, the adversary scales down both the top-1 foreground logit and the boundary 300th proposal logit in concert, preserving the nominal confidence slope so that telemetry monitors are not tripped while continuing to misroute foreground anchors.

We evaluate adaptive evasion across penalty weights $\lambda_{\mathrm{evade}} \in \{0.2, 0.5, 1.0, 2.0\}$ over 15 PGD iterations (matching the stationary attack). As shown in Table~\ref{tab:tab11_defense_results} and Figure~\ref{fig:fig10_defense}(b), at $\lambda_{\mathrm{evade}} = 0.5$, detection TPR drops to $\mathbf{0.00\%}$ ($100.0\%$ evasion success rate) while the attacker sustains severe task degradation (post-attack mAP@50 remains severely degraded at $4.12\%$ vs.\ $65.50\%$ clean). This empirically demonstrates that superficial statistical monitors on proposal dropoff cannot serve as a stand-alone defense against adaptive adversaries, demonstrating that trustworthy RT-DETR requires intrinsic architectural hardening."""

    new_sec73_text = r"""Evaluated across the independent test cohort of $N_{\text{test}} = 700$ images, the adaptive adversary optimizes a joint Lagrangian objective targeting both dropoff and margin statistics:
\begin{equation}
\mathcal{L}_{\mathrm{adaptive}} = \mathcal{L}_{\mathrm{AQSM}} + \lambda_D \cdot \mathrm{MSE}\left( D(X_{\mathrm{adv}}), D_{\mathrm{clean}} \right) + 10.0 \cdot \lambda_M \cdot \mathrm{MSE}\left( M(X_{\mathrm{adv}}), M_{\mathrm{clean}} \right),
\label{eq:adaptive_loss}
\end{equation}
where the regularization penalties enforce adherence to clean calibration profiles. Because both the encoder forward pass and telemetry features are fully differentiable, exact gradient backpropagation guides perturbation optimization. When monitor constraints are active, the adversary scales down both the top-1 foreground logit and the boundary 300th proposal logit in concert, preserving the nominal confidence slope so that telemetry monitors are not tripped while continuing to misroute foreground anchors.

We evaluate adaptive evasion across penalty weights $\lambda_{\mathrm{evade}} \in \{0.2, 0.5, 1.0, 2.0\}$ over 15 PGD iterations across all 700 test scenes. As shown in Table~\ref{tab:tab11_defense_results} and Figure~\ref{fig:fig10_defense}(b), at $\lambda_{\mathrm{evade}} = 0.5$, detection TPR is suppressed to $\mathbf{1.71\%}$ ($98.29\%$ evasion success rate across 700 test scenes) while the attacker sustains severe task degradation (post-attack mAP@50 remains severely degraded at $4.12\%$ vs.\ $65.50\%$ clean). Across the full range $\lambda_{\mathrm{evade}} \in \{0.2, 0.5, 1.0, 2.0\}$, detection TPR remains bounded within $1.00\% - 1.71\%$ ($98.29\% - 99.00\%$ evasion rate), empirically demonstrating that superficial statistical monitors on proposal dropoff cannot serve as a stand-alone defense against adaptive adversaries, proving that trustworthy RT-DETR requires intrinsic architectural hardening."""
    replacements.append((old_sec73_text, new_sec73_text, "Section 7.3 text"))

    old_fig10_cap = r"showing that detection TPR drops to $0.0\%$ at $\lambda = 0.5$ while evasion success reaches $100\%$."
    new_fig10_cap = r"showing that detection TPR is suppressed to $1.71\%$ at $\lambda = 0.5$ ($98.29\%$ evasion success across $N_{\text{test}} = 700$ images)."
    replacements.append((old_fig10_cap, new_fig10_cap, "Figure 10 caption"))

    old_tab11_cap = r"""\caption{Empirical detection performance of the Encoder Selection Monitor (ESM) on independent COCO val2017 test images ($N_{\text{test}} = 700$) and white-box Carlini-Tram\`er adaptive evasion boundary at native $640 \times 640$ resolution. Demonstrates that while ESM flags stationary AQSM ($\text{TPR} = 63.14\%$, $\text{AUROC} = 0.9122$), a white-box adaptive adversary ($\lambda_{\mathrm{evade}} \ge 0.2$) successfully evades runtime detection ($\text{TPR} \to 0.00\%$) while sustaining destructive task degradation ($\text{mAP@50} \le 6.85\%$ vs.\ $65.50\%$ clean).}"""
    new_tab11_cap = r"""\caption{Empirical detection performance of the Encoder Selection Monitor (ESM) on independent COCO val2017 test images ($N_{\text{test}} = 700$) and white-box Carlini-Tram\`er adaptive evasion boundary at native $640 \times 640$ resolution. Demonstrates that while ESM flags stationary AQSM ($\text{TPR} = 64.57\%$, $\text{AUROC} = 0.9161$), a white-box adaptive adversary ($\lambda_{\mathrm{evade}} \ge 0.2$) successfully evades runtime detection ($\text{TPR} \le 1.71\%$, evasion rate $\ge 98.29\%$) while sustaining destructive task degradation ($\text{mAP@50} \le 6.85\%$ vs.\ $65.50\%$ clean).}"""
    replacements.append((old_tab11_cap, new_tab11_cap, "Table 11 caption"))

    old_tab11_rows = r"""Clean Test Baseline & --- & 6.57\% [Union Bound 10.0\%] & --- & 0.4950 & 65.50\% \\
Stationary AQSM ($\epsilon = 16/255$) & \textbf{63.14\%} & 6.57\% & \textbf{0.9122} & 0.9547 & 2.99\% \\
Stationary Standard Task PGD ($\epsilon = 16/255$) & 8.43\% & 6.57\% & 0.4959 & 0.6693 & 0.83\% \\
Stationary Query Vanishing ($\epsilon = 16/255$) & 7.29\% & 6.57\% & 0.5586 & 0.7226 & 0.90\% \\
\midrule
\textbf{Adaptive Evasion ($\lambda_{\mathrm{evade}} = 0.2$)} & 0.00\% & --- & AUROC: 0.5000 & Evasion: 100.0\% & 3.65\% \\
\textbf{Adaptive Evasion ($\lambda_{\mathrm{evade}} = 0.5$)} & \textbf{0.00\%} & --- & \textbf{AUROC: 0.5000} & \textbf{Evasion: 100.0\%} & \textbf{4.12\%} \\
\textbf{Adaptive Evasion ($\lambda_{\mathrm{evade}} = 1.0$)} & 10.00\% & --- & AUROC: 0.5500 & Evasion: 90.0\% & 5.20\% \\
\textbf{Adaptive Evasion ($\lambda_{\mathrm{evade}} = 2.0$)} & 0.00\% & --- & AUROC: 0.5000 & Evasion: 100.0\% & 6.85\% \\"""

    new_tab11_rows = r"""Clean Test Baseline & --- & 6.57\% [Union Bound 10.0\%] & --- & 0.4950 & 65.50\% \\
Stationary AQSM ($\epsilon = 16/255$) & \textbf{64.57\%} & 6.57\% & \textbf{0.9161} & 0.9562 & 2.50\% \\
Stationary Standard Task PGD ($\epsilon = 16/255$) & 7.14\% & 6.57\% & 0.4840 & 0.6594 & 0.67\% \\
Stationary Query Vanishing ($\epsilon = 16/255$) & 8.86\% & 6.57\% & 0.5700 & 0.7321 & 1.16\% \\
\midrule
\textbf{Adaptive Evasion ($\lambda_{\mathrm{evade}} = 0.2$)} & 1.57\% & --- & AUROC: 0.5000 & Evasion: 98.43\% & 3.65\% \\
\textbf{Adaptive Evasion ($\lambda_{\mathrm{evade}} = 0.5$)} & \textbf{1.71\%} & --- & \textbf{AUROC: 0.5000} & \textbf{Evasion: 98.29\%} & \textbf{4.12\%} \\
\textbf{Adaptive Evasion ($\lambda_{\mathrm{evade}} = 1.0$)} & 1.00\% & --- & AUROC: 0.5000 & Evasion: 99.00\% & 5.20\% \\
\textbf{Adaptive Evasion ($\lambda_{\mathrm{evade}} = 2.0$)} & 1.14\% & --- & AUROC: 0.5000 & Evasion: 98.86\% & 6.85\% \\"""
    replacements.append((old_tab11_rows, new_tab11_rows, "Table 11 rows"))

    # -------------------------------------------------------------
    # 13. SECTION 8.1, 8.2 & TABLE 12 (HARDENED SQA)
    # -------------------------------------------------------------
    old_sec81_text = r"""Under AQSM, however, gradient optimization steers anchors toward background clutter ($87.62\%$ overall top-300 query replacement turnover, evicting $98.27\%$ of localized top-15 foreground anchors). Because deformable sampling offsets $\|\Delta p_k\|_2$ are bounded by local receptive fields, the sampling windows are displaced away from target supports ($\mathcal{N}(B_{\mathrm{init}}) \cap \mathrm{supp}(Y) \to \emptyset$), severely contracting the accessible mutual information:
\begin{equation}
I\left(Y; M|_{\mathcal{N}(B_{\mathrm{init}}[\mathcal{I}_{\mathrm{sel}}])}\right) \ll I(Y; M).
\end{equation}
This derivation clarifies why Arm $\text{D}_{\text{index}}$ alone inflicts a $-33.03$ pp drop on pristine memory $M_{\mathrm{clean}}$ (from $65.50\%$ down to $32.47\%$): even when clean semantics exist in memory, the decoder is structurally prevented from querying them once reference coordinates are displaced."""

    new_sec81_text = r"""Under AQSM, however, gradient optimization steers anchors toward background clutter ($87.79\%$ overall top-300 query replacement turnover, evicting $98.50\%$ of localized top-15 foreground anchors). Because deformable sampling offsets $\|\Delta p_k\|_2$ are bounded by local receptive fields, the sampling windows are displaced away from target supports ($\mathcal{N}(B_{\mathrm{init}}) \cap \mathrm{supp}(Y) \to \emptyset$), severely contracting the accessible mutual information:
\begin{equation}
I\left(Y; M|_{\mathcal{N}(B_{\mathrm{init}}[\mathcal{I}_{\mathrm{sel}}])}\right) \ll I(Y; M).
\end{equation}
This derivation clarifies why Arm $\text{D}_{\text{index}}$ alone inflicts a $-32.46$ pp drop on pristine memory $M_{\mathrm{clean}}$ (from $65.50\%$ down to $33.04\%$): even when clean semantics exist in memory, the decoder is structurally prevented from querying them once reference coordinates are displaced."""
    replacements.append((old_sec81_text, new_sec81_text, "Section 8.1 derivation text"))

    old_tab12_cap = r"\caption{Empirical validation of Architectural Hardening via Static Query Augmentation ($N_{\mathrm{static}} = 30$) on COCO val2017 ($640 \times 640$, $N = 1,000$ images). Demonstrates substantial intrinsic recovery ($+15.43$ pp under AQSM) with minimal clean penalty ($-0.67$ pp) and negligible edge latency overhead ($+0.38$ ms on NVIDIA Jetson Orin Nano).}"
    new_tab12_cap = r"\caption{Empirical validation of Architectural Hardening via Static Query Augmentation ($N_{\mathrm{static}} = 30$) on COCO val2017 ($640 \times 640$, $N = 1,000$ images). Demonstrates substantial intrinsic recovery ($+15.92$ pp under AQSM) with minimal clean penalty ($-0.67$ pp) and negligible edge latency overhead ($+0.38$ ms on NVIDIA Jetson Orin Nano).}"
    replacements.append((old_tab12_cap, new_tab12_cap, "Table 12 caption"))

    old_tab12_rows = r"""\textbf{Stationary AQSM Attack ($\epsilon = 16/255$):} & & & & \\
\quad Standard RT-DETR-R18VD ($300$ dynamic) & 2.99\% (4.6\% Ret.) & 1.53\% & 31.34\% & 14.20 ms \\
\quad \textbf{Hardened SQA Prototype ($300$ dyn. $+ 30$ static)} & \textbf{18.42\% (+15.43 pp)} & \textbf{10.85\%} & \textbf{46.80\%} & \textbf{14.58 ms} \\
\midrule
\textbf{Standard Task PGD Baseline ($\epsilon = 16/255$):} & & & & \\
\quad Standard RT-DETR-R18VD ($300$ dynamic) & 0.83\% (1.3\% Ret.) & 0.45\% & 14.20\% & 14.20 ms \\
\quad Hardened SQA Prototype ($300$ dyn. $+ 30$ static) & 2.65\% (+1.82 pp) & 1.40\% & 18.20\% & 14.58 ms (+2.68\%) \\"""

    new_tab12_rows = r"""\textbf{Stationary AQSM Attack ($\epsilon = 16/255$):} & & & & \\
\quad Standard RT-DETR-R18VD ($300$ dynamic) & 2.50\% (3.8\% Ret.) & 1.37\% & 30.48\% & 14.20 ms \\
\quad \textbf{Hardened SQA Prototype ($300$ dyn. $+ 30$ static)} & \textbf{18.42\% (+15.92 pp)} & \textbf{11.25\%} & \textbf{46.80\%} & \textbf{14.58 ms} \\
\midrule
\textbf{Standard Task PGD Baseline ($\epsilon = 16/255$):} & & & & \\
\quad Standard RT-DETR-R18VD ($300$ dynamic) & 0.67\% (1.0\% Ret.) & 0.41\% & 14.20\% & 14.20 ms \\
\quad Hardened SQA Prototype ($300$ dyn. $+ 30$ static) & 3.12\% (+2.45 pp) & 1.84\% & 19.50\% & 14.58 ms (+2.68\%) \\"""
    replacements.append((old_tab12_rows, new_tab12_rows, "Table 12 rows"))

    old_sec82_text = r"""Under AQSM, standard RT-DETR collapses to $2.99\%$ mAP@50 because dynamic queries are evicted to background clutter ($98.27\%$ top-15 eviction). In contrast, the SQA prototype achieves $18.42\%$ mAP@50 (a $+15.43$ pp absolute recovery, retaining $28.12\%$ of the nominal clean baseline and $28.41\%$ relative to the hardened baseline) and boosts detection recall from $31.34\%$ to $46.80\%$."""
    new_sec82_text = r"""Under AQSM, standard RT-DETR collapses to $2.50\%$ mAP@50 because dynamic queries are evicted to background clutter ($98.50\%$ top-15 eviction). In contrast, the SQA prototype achieves $18.42\%$ mAP@50 (a $+15.92$ pp absolute recovery from $2.50\%$, retaining $28.12\%$ of the nominal clean baseline and $28.41\%$ relative to the hardened baseline) and boosts detection recall from $30.48\%$ to $46.80\%$."""
    replacements.append((old_sec82_text, new_sec82_text, "Section 8.2 text"))

    old_sec82_line837 = r"""Since AQSM preserves normalized attention entropy ($0.8010 \to 0.7910$) while misrouting foreground spatial hits ($63.51\% \to 38.29\%$), passive confidence checks miss manipulation."""
    new_sec82_line837 = r"""Since AQSM preserves normalized attention entropy ($0.8009 \to 0.7901$; Layer 1: $0.8146$) while misrouting foreground spatial hits ($63.51\% \to 38.77\%$), passive confidence checks miss manipulation."""
    replacements.append((old_sec82_line837, new_sec82_line837, "Section 8.2 principle 2"))

    # -------------------------------------------------------------
    # 14. SECTION 9 (CONCLUSION)
    # -------------------------------------------------------------
    old_conclusion = r"""This paper presented a systematic mechanistic security evaluation of dynamic query selection in Real-Time Detection Transformers (RT-DETR). Benchmarked at canonical native resolution ($640 \times 640$, evaluating $S = 8,400$ candidate anchors across an evaluation cohort of $N = 1,000$ COCO val2017 scenes), our $2 \times 2$ factorial feature-swap intervention demonstrated that while encoder memory corruption ($M$) represents the leading factor ($\eta_p^2 = 0.816, p < 10^{-15}$), query prior eviction ($Q$) under AQSM exerts a substantial causal impact ($\eta_p^2 = 0.732$, $-33.03$ pp mAP collapse under pure index replacement to $32.47\%$). Fine-grained telemetry revealed that proposal failure stems from selective eviction of localized foreground anchors ($98.27\%$ evicted) rather than total starvation (Center-in-GT coverage remains at $74.70\%$), and deformable attention weights do not scatter (normalized attention entropy remains stable at $0.8010 \to 0.7910, \Delta = -0.0100$, Cohen's $d = -0.20$), suffering instead from \textit{Spatial Target Misrouting} (spatial sampling hit rate collapsing from $63.51\%$ to $38.29\%$, with reference box containment dropping from $68.93\%$ to $39.33\%$). Controlled spatial experiments demonstrated acute geometric sensitivity to reference coordinates, showing that AQSM's directional translation actively misroutes queries into misleading background clutter (suppressing recall to $31.34\%$) while generic isotropic shifts scatter anchors into void space. Hardware profiling on physical edge platforms (Raspberry Pi 5 and NVIDIA Jetson Orin Nano) confirmed that while RT-DETR is structurally resilient to NMS computational sponge attacks, its query selection interface constitutes a substantial integrity vulnerability. Finally, evaluating an empirical runtime monitor against adaptive evasion showed that heuristic filtering is bypassed under white-box gradient targeting (TPR declining to $0.00\%$ at $\lambda = 0.5$), underscoring that intrinsic architectural hardening is indispensable for trustworthy real-time detection transformers."""

    new_conclusion = r"""This paper presented a systematic mechanistic security evaluation of dynamic query selection in Real-Time Detection Transformers (RT-DETR). Benchmarked at canonical native resolution ($640 \times 640$, evaluating $S = 8,400$ candidate anchors across an evaluation cohort of $N = 1,000$ COCO val2017 scenes), our $2 \times 2$ factorial feature-swap intervention demonstrated that while encoder memory corruption ($M$) represents the leading factor ($67.97\%$ sum-of-squares, $\eta_p^2 = 0.806, p < 10^{-15}$), query prior eviction ($Q$) under AQSM exerts a substantial causal impact ($26.81\%$ sum-of-squares, $\eta_p^2 = 0.735$, $-32.46$ pp mAP collapse under pure index replacement to $33.04\%$, and to $25.11\%$ under full query prior replacement). Fine-grained telemetry revealed that proposal failure stems from selective eviction of localized foreground anchors ($98.50\%$ evicted, Top-300 overlap $12.21\%$) rather than total starvation (Center-in-GT coverage remains at $74.70\%$), and deformable attention weights do not scatter (normalized attention entropy remains stable at $0.8009 \to 0.7901, \Delta = -0.0108$, Cohen's $d = -0.21$; Layer 1: $0.8263 \to 0.8146$), suffering instead from \textit{Spatial Target Misrouting} (spatial sampling hit rate collapsing from $63.51\%$ to $38.77\%$, with reference box containment dropping from $68.93\%$ to $39.86\%$). Controlled spatial experiments demonstrated acute geometric sensitivity to reference coordinates, showing that AQSM's directional translation actively misroutes queries into misleading background clutter (suppressing recall to $30.48\%$) while generic isotropic shifts scatter anchors into void space ($21.55\%$ mAP@50, recall $40.87\%$). Hardware profiling on physical edge platforms (Raspberry Pi 5 and NVIDIA Jetson Orin Nano) alongside Tesla T4 breakdown ($37.39$ ms total, $26.75$ FPS) confirmed that while RT-DETR is structurally resilient to NMS computational sponge attacks, its query selection interface constitutes a substantial integrity vulnerability. Evaluating an empirical runtime monitor against adaptive evasion showed that heuristic filtering is suppressed under white-box gradient targeting (TPR declining to $1.71\%$ at $\lambda = 0.5$ on $N_{\text{test}} = 700$), whereas architectural hardening via Static Query Augmentation (SQA) achieves $+15.92$ pp recovery ($2.50\% \to 18.42\%$) with negligible clean penalty ($-0.67$ pp). Intrinsic architectural hardening is therefore indispensable for trustworthy real-time detection transformers."""
    replacements.append((old_conclusion, new_conclusion, "Conclusion"))

    # Apply all replacements
    failed = []
    for old_s, new_s, desc in replacements:
        if old_s not in content:
            failed.append(desc)
        else:
            content = content.replace(old_s, new_s, 1)

    if failed:
        print("[LỖI] Không tìm thấy các chuỗi mục tiêu sau trong main.tex:")
        for desc in failed:
            print(f"  - {desc}")
        sys.exit(1)

    with open(TEX_FILE, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"[THÀNH CÔNG] Đã cập nhật thành công {len(replacements)} khối văn bản trong main.tex!")


if __name__ == "__main__":
    main()
