# -*- coding: utf-8 -*-
"""
Script to apply nuanced academic framing, rigorous information theory derivation,
title enhancement, and edge platform clarification to paper_1_aqsm_mechanism/main.tex.
"""

from pathlib import Path

TEX_PATH = Path("paper_1_aqsm_mechanism/main.tex")

def apply_improvements():
    with open(TEX_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Title Enhancement: Replace "Runtime Intrusion Defense" with "Runtime Guardrail Assessment"
    old_title = "\\title{Towards Reliable Real-Time DETR: Query Selection Vulnerability Analysis, Mechanistic Failure Modes, and Runtime Intrusion Defense on Edge Computing Platforms}"
    new_title = "\\title{Towards Reliable Real-Time DETR: Query Selection Vulnerability Analysis, Mechanistic Failure Modes, and Runtime Guardrail Assessment on Edge Platforms}"
    assert old_title in content, "Title not found"
    content = content.replace(old_title, new_title)

    # 2. Section 7.1: Information-Theoretic Derivation
    old_infotheory = """\\subsection{Information-Theoretic Modeling of Dynamic Query Selection}
The structural vulnerability of RT-DETR can be understood qualitatively through the lens of data processing constraints. In standard DETR \\citep{carion2020end}, decoder queries are global learnable embeddings that attend across all spatial feature tokens $M$. In RT-DETR, dynamic query selection performs an aggressive dimensionality reduction: selecting $N_q = 300$ candidate anchors from $S = 8,400$ possibilities (selection ratio $\\kappa = 300 / 8,400 \\approx 3.57\\%$). Because deformable cross-attention samples multi-scale memory tokens locally around the initial reference points ($p_k = B_{\\mathrm{init}} + \\Delta p_k$), the effective spatial transmission of target information to predictions $\\hat{Y}$ is heavily constrained by the spatial relevance of the admitted query set $\\mathcal{I}_{\\mathrm{sel}}$:
\\begin{equation}
I(Y; \\hat{Y}) \\le I(Y; M \\mid \\mathcal{I}_{\\mathrm{sel}}, B_{\\mathrm{init}}[\\mathcal{I}_{\\mathrm{sel}}]).
\\end{equation}
When an adversary manipulates selection order to evict true foreground anchors into background regions, the deformable attention mechanism is forced to anchor its local sampling offsets to distal background coordinates. Although memory $M$ remains accessible, the constrained local sampling offset range prevents the decoder from recovering distant target semantics, explaining why Arm $\\text{D}_{\\text{index}}$ alone induces a major drop on pristine feature memory and establishing query selection as a critical vulnerability surface."""

    new_infotheory = """\\subsection{Information-Theoretic Modeling of Dynamic Query Selection}
The structural vulnerability of RT-DETR can be formalized through the lens of spatially constrained information transmission. Let $Y$ denote ground-truth object states, $M$ the multi-scale encoder feature memory, and $\\hat{Y}$ the decoder predictions. In standard DETR \\citep{carion2020end}, decoder queries are global learnable embeddings that compute dense cross-attention across all spatial tokens $M$, allowing global semantic aggregation: $I(Y; \\hat{Y}) \\le I(Y; M)$. In RT-DETR, dynamic query selection performs aggressive dimensionality reduction: admitting $N_q = 300$ candidate anchors from $S = 8,400$ possibilities (selection ratio $\\kappa = 300 / 8,400 \\approx 3.57\\%$).

Crucially, deformable cross-attention restricts memory sampling to local spatial receptive windows $\\mathcal{N}(B_{\\mathrm{init}})$ centered around the initial reference coordinates $B_{\\mathrm{init}}[\\mathcal{I}_{\\mathrm{sel}}]$ ($p_k = B_{\\mathrm{init}} + \\Delta p_k$). Consequently, the effective feature representation accessible to the decoder is the spatial restriction $M|_{\\mathcal{N}(B_{\\mathrm{init}})}$. By the Data Processing Inequality on the Markov chain $Y \\to M \\to M|_{\\mathcal{N}(B_{\\mathrm{init}})} \\to \\hat{Y}$, the mutual information conveyed to predictions is bounded by:
\\begin{equation}
I(Y; \\hat{Y}) \\le I\\left(Y; M|_{\\mathcal{N}(B_{\\mathrm{init}}[\\mathcal{I}_{\\mathrm{sel}}])}\\right).
\\label{eq:info_bound}
\\end{equation}
Under nominal clean inference, top-$K$ selection routes anchors centered on genuine objects ($99.02\\%$ Center-in-GT), maximizing the overlap $\\mathcal{N}(B_{\\mathrm{init}}) \\cap \\mathrm{supp}(Y)$. Under AQSM, however, gradient optimization steers anchors toward background clutter ($87.87\\%$ eviction). Because deformable sampling offsets $\\|\\Delta p_k\\|_2$ are bounded by local receptive fields, the sampling windows are displaced away from target supports ($\\mathcal{N}(B_{\\mathrm{init}}) \\cap \\mathrm{supp}(Y) \\to \\emptyset$), contracting the mutual information bound:
\\begin{equation}
I\\left(Y; M|_{\\mathcal{N}(B_{\\mathrm{init}}[\\mathcal{I}_{\\mathrm{sel}}])}\\right) \\ll I(Y; M).
\\end{equation}
This derivation clarifies why Arm $\\text{D}_{\\text{index}}$ alone inflicts a $-32.49$ pp drop on pristine memory $M_{\\mathrm{clean}}$: even when clean semantics exist in memory, the decoder is structurally prevented from querying them once reference coordinates are displaced."""
    assert old_infotheory in content, "Info theory not found"
    content = content.replace(old_infotheory, new_infotheory)

    # 3. Section 7.3: Edge Latency Clarification (GPU vs CPU)
    old_edge_text = """Real-time robotic systems (e.g., warehouse AGVs, autonomous drones, and mobile inspection robots) depend on deterministic latency bounds ($\\le 20$ ms) to execute obstacle avoidance loops. While convolutional models like YOLO offer slightly faster nominal execution on clean inputs ($11.50$ ms vs.\\ $14.20$ ms on Jetson Orin Nano), our hardware profiling proves that an adversary can inject computational sponges that trigger a $+232\\%$ to $+242\\%$ latency surge ($39.40$ ms on Jetson, $815.40$ ms on Raspberry Pi 5), directly causing deadline misses. RT-DETR eliminates this failure mode entirely through its $\\mathcal{O}(1)$ static compute graph. However, systems engineers must recognize that in trading NMS for dynamic query selection, the operational risk shifts from \\textit{availability attacks} (sponge latency degradation) to \\textit{integrity attacks} (silent spatial misrouting). Hardening the encoder selection interface is therefore the indispensable next step for robotic vision safety."""

    new_edge_text = """Real-time robotic payloads depend on predictable execution latency to maintain control loop stability. On embedded GPU platforms (e.g., NVIDIA Jetson Orin Nano at $14.20$ ms, $\\approx 70.4$ FPS), RT-DETR comfortably satisfies hard real-time robotic deadlines ($\\le 20$ ms), whereas CPU-only edge boards (e.g., Raspberry Pi 5 at $418.20$ ms, $\\approx 2.39$ FPS) serve near-real-time surveillance monitoring. Crucially, our hardware profiling demonstrates that convolutional architectures like YOLOv8-S are vulnerable to computational sponge attacks: input-dependent NMS clustering triggers a $+232\\%$ to $+242\\%$ latency surge ($11.50 \\to 39.40$ ms on Jetson, violating the $20$ ms deadline; $245.20 \\to 815.40$ ms on Pi 5). RT-DETR eliminates this availability risk entirely through its $\\mathcal{O}(1)$ static compute graph ($14.20 \\to 14.22$ ms on Jetson; $418.20 \\to 418.50$ ms on Pi 5). However, systems engineers must recognize that in eliminating NMS, the threat surface shifts from \\textit{availability attacks} (latency exhaustion) to \\textit{integrity attacks} (silent spatial misrouting). Hardening the encoder selection interface is therefore the indispensable next step for robotic vision safety."""
    assert old_edge_text in content, "Edge text not found"
    content = content.replace(old_edge_text, new_edge_text)

    # 4. Abstract: Memory Dominance Framing
    old_abs_mem = "Through a $2 \\times 2$ factorial causal feature-swap intervention, we disentangle encoder memory representations from query prior selection, demonstrating that while memory corruption is the primary driver of performance degradation ($\\eta_p^2 = 0.820$), query prior manipulation contributes a substantial causal effect ($\\text{ME}_Q = -28.89$~pp, $\\eta_p^2 = 0.731$)"
    new_abs_mem = "Through a $2 \\times 2$ factorial counterfactual feature intervention via activation patching, we disentangle encoder memory representations from query prior selection. We show that while encoder memory corruption is the primary damage driver ($66.2\\%$ of ANOVA sum-of-squares, $\\eta_p^2 = 0.820$), query prior displacement acts as an essential secondary transmission channel ($\\text{ME}_Q = -28.89$~pp, $\\eta_p^2 = 0.731$)"
    assert old_abs_mem in content, "Abs mem not found"
    content = content.replace(old_abs_mem, new_abs_mem)

    with open(TEX_PATH, "w", encoding="utf-8") as f:
        f.write(content)

    print(">>> Đã áp dụng các cải tiến học thuật bổ sung vào main.tex thành công!")

if __name__ == "__main__":
    apply_improvements()
