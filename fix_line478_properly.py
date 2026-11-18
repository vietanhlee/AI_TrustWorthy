"""
Fix line 478 in main.tex with proper LaTeX math escaping.
"""

with open(r"G:\Edge_Sec\paper_1_aqsm_mechanism\main.tex", "r", encoding="utf-8") as f:
    text = f.read()

pos = text.find("Because per-image AP metrics are lower-bounded at zero and exhibit skewness")
end = text.find(r"\subsection{Encoder Memory Perturba", pos)

clean_para = (
    "Because per-image AP metrics are lower-bounded at zero and exhibit skewness under severe adversarial degradation, "
    "empirical residual distributions depart from strict Gaussianity (Shapiro-Wilk test: $W = 0.814, p < 0.001$). "
    "Per-image average precision ($AP_i$) is computed via standard 101-point interpolated precision-recall integration over "
    "the detections on image $i$ against its ground-truth annotations across all active classes, providing independent intra-subject observations. "
    "For our $N = 30$ evaluated images under a $2 \\times 2$ repeated-measures factorial design ($M \\in \\{\\text{clean}, \\text{adv}\\} \\times Q \\in \\{\\text{clean}, \\text{adv}\\}$), "
    "condition degrees of freedom are $\\text{df} = 1$ with model error degrees of freedom $\\text{df}_{\\mathrm{err}} = N - 1 = 29$. "
    "Both parametric $F$-tests ($F_{1, 29} = 267.13, p = 3.33 \\times 10^{-16}$ for Memory; $F_{1, 29} = 132.55, p = 2.46 \\times 10^{-12}$ for Query) "
    "and non-parametric Wilcoxon signed-rank tests corroborate decisive statistical significance. Beyond asymptotic $p$-values, effect sizes are massive: "
    "partial eta-squared $\\eta_p^2 = \\text{SS}_{\\mathrm{factor}} / (\\text{SS}_{\\mathrm{factor}} + \\text{SS}_{\\mathrm{error}}) = 0.902$ for Memory and $\\eta_p^2 = 0.820$ for Query. "
    "To rigorously evaluate pairwise intervention contrasts without parametric assumptions, we evaluate the $K = 6$ focal pairwise hypotheses between primary arms "
    "(Arm A vs.\\ B, Arm A vs.\\ $\\text{C}_{\\text{full}}$, Arm A vs.\\ $\\text{D}_{\\text{full}}$, Arm A vs.\\ $\\text{D}_{\\text{index}}$, Arm B vs.\\ $\\text{C}_{\\text{full}}$, and Arm B vs.\\ $\\text{D}_{\\text{full}}$) "
    "with Bonferroni critical threshold $\\alpha_{\\mathrm{adj}} = 0.05 / 6 \\approx 0.00833$. "
    "Under two-sided Wilcoxon signed-rank tests, all key contrasts against clean Arm A (Arm B vs.\\ A, Arm $\\text{C}_{\\text{full}}$ vs.\\ A, Arm $\\text{D}_{\\text{full}}$ vs.\\ A, Arm $\\text{D}_{\\text{index}}$ vs.\\ A) "
    "satisfy $p < 10^{-6} \\ll \\alpha_{\\mathrm{adj}}$, proving that causal attribution does not rely on distributional normality.\n\n"
)

text = text[:pos] + clean_para + text[end:]

with open(r"G:\Edge_Sec\paper_1_aqsm_mechanism\main.tex", "w", encoding="utf-8") as f:
    f.write(text)

print("Line 478 properly updated!")
