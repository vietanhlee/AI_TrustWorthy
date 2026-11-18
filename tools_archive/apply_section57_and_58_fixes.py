"""
Script to update Section 5.7 and Section 5.8 to perfectly align with ground truth
and address Claude's critique points 3.4, 3.5, and 3.6.
"""

with open(r"G:\Edge_Sec\paper_1_aqsm_mechanism\main.tex", "r", encoding="utf-8") as f:
    text = f.read()

# -------------------------------------------------------------------------
# Fix Section 5.7 paragraph on Mechanistic Asymmetry and Jensen's inequality
# -------------------------------------------------------------------------
old_sec57_para = r"""\item \textbf{Mechanistic Asymmetry in Component-Wise Displacements:} Decomposing displacement vectors into Center component $\Delta_{\text{center}} = (\Delta c_x, \Delta c_y)$ and Size component $\Delta_{\text{size}} = (\Delta w, \Delta h)$ exposes the structural directional bias: AQSM concentrates $87.9\%$ of its displacement energy into center translation ($\|\Delta_{\text{center}}\|^2 / \|\Delta_{\text{total}}\|^2 = 0.5180^2 / 0.5328^2 = 87.9\%$, or $93.8\%$ of linear magnitude $0.5180 / 0.5328$), while box dimensions undergo minimal adjustment (Size $L_2 = 0.1245$). In contrast, isotropic Random Shift distributes displacement across center ($0.3510$) and size ($0.3812$). Due to Jensen's inequality across candidate queries and image cohorts ($\mathbb{E}[\sqrt{X+Y}] \le \sqrt{\mathbb{E}[X]+\mathbb{E}[Y]}$), the expected total norm $\mathbb{E}[\|\Delta_{\text{total}}\|_2] = 0.5328$ strictly bounds the Pythagorean sum of expected marginals $\sqrt{0.5180^2 + 0.1245^2} = 0.5327$, explaining the non-linear difference. This directional translation specifically steers initial reference points away from true foreground centers into background regions, dropping Reference Center FG Hit from $63.40\%$ to $39.80\%$.
\end{enumerate}"""

new_sec57_para = r"""\item \textbf{Mechanistic Asymmetry in Component-Wise Displacements:} Decomposing reference box displacement vectors into center translation $\Delta_{\text{center}} = (\Delta c_x, \Delta c_y)$ and bounding-box dilation $\Delta_{\text{size}} = (\Delta w, \Delta h)$ resolves the geometric mechanism underlying spatial failure. Isotropic Random Shift disperses displacement uniformly across coordinates, yielding an equal energy partition ($51.6\%$ on center, $48.4\%$ on dimensions; Center $L_2 = 0.3516 \pm 0.0013$, Size $L_2 = 0.2861 \pm 0.0023$, total $L_2 = 0.4895 \pm 0.0016$). In contrast, AQSM concentrates $71.2\%$ of its total squared Euclidean displacement energy into reference center translation (Center $L_2 = 0.5188$ vs.\ Size $L_2 = 0.2243$, total $L_2 = 0.6148$). This directional translation specifically steers initial decoder sampling coordinates off true foreground centroids onto adjacent uninformative background regions, directly reducing the decoder's spatial cross-attention foreground hit rate from nominal $65.54\%$ down to $38.34\%$.
\end{enumerate}"""

assert old_sec57_para in text, "old_sec57_para not found in text"
text = text.replace(old_sec57_para, new_sec57_para)

# -------------------------------------------------------------------------
# Fix Section 5.8 paragraph on Divergent Failure Modes
# -------------------------------------------------------------------------
old_sec58_para = r"""\paragraph{Divergent Failure Modes Under Matched Budgets:}
Rather than asserting that AQSM reduces raw mAP@50 more than Standard Task PGD, our controlled comparison reveals a nuanced mechanistic distinction:
\begin{enumerate}
    \item \textbf{Task PGD Destroys Detection via Foreground \& Box Corruption:} Across every budget ($\epsilon = 4/255$ to $16/255$), Standard Task PGD collapses mAP@50 virtually to $0.01\%$--$0.05\%$ (vs.\ $18.45\% \to 5.36\%$ for AQSM) because end-to-end Hungarian supervision concentrates perturbation energy onto foreground objects ($\text{FG}/\text{BG} = 1.075$ vs.\ $0.906$), corrupts initial bounding-box regression, and floods output logits ($29.40$ active boxes).
    \item \textbf{AQSM Exploits Encoder-Only Routing to Induce Spatial Misrouting:} Using only encoder-head gradients without decoder supervision, AQSM inflicts significantly stronger Top-300 ranking turnover ($21.4\%$ vs.\ $35.2\%$ overlap, $p = 2.15 \times 10^{-38}$), drives far worse Spatial Target Misrouting in deformable attention (FG Hit = $35.8\%$ vs.\ $51.8\%, \Delta = -16.0$ pp, $p = 1.10 \times 10^{-40}$), and induces greater initial reference point misrouting ($0.518$ vs.\ $0.462, p < 10^{-36}$). Crucially, even at perturbation budget $\epsilon = 16/255$ (where AQSM reduces mAP@50 to $5.36\%$), output active boxes remain moderate ($7.15$ boxes) relative to Standard Task PGD's excessive prediction flooding ($29.40$ active boxes, a $+550\%$ surge over clean $4.52$), demonstrating that adversarial query ranking manipulation operates primarily via targeted anchor eviction and reference displacement rather than brute-force output flooding.
\end{enumerate}"""

new_sec58_para = r"""\paragraph{Divergent Failure Modes Under Matched Budgets:}
Rather than claiming that AQSM establishes higher raw mAP destruction than Standard Task PGD, our controlled comparison reveals a nuanced mechanistic distinction:
\begin{enumerate}
    \item \textbf{Task PGD Achieves Deeper End-to-End Task Collapse:} Standard Task PGD optimizes the full multi-task bipartite Hungarian loss end-to-end through the decoder, monotonically degrading mAP@50 from $11.73\%$ at $\epsilon = 4/255$ to $6.93\%$ at $\epsilon = 8/255$ and $3.83\%$ at $\epsilon = 16/255$ (reaching lower absolute mAP@50 than AQSM's $5.36\%$). This end-to-end gradient supervision concentrates perturbation energy across all transformer layers indiscriminately, corrupts bounding-box coordinate heads, and floods the final prediction set ($29.40 \pm 8.70$ active boxes vs.\ clean $4.52 \pm 3.10$).
    \item \textbf{AQSM Selectively Manipulates Dynamic Query Selection:} In contrast, AQSM is formulated strictly as an encoder-selection diagnostic probe without requiring decoder backpropagation during attack generation. Under matched budgets, AQSM drives significantly stronger Top-300 proposal ranking turnover (overlap of only $13.73\%$ vs.\ $26.44\%$ for Task PGD), near-total foreground anchor eviction ($98.44\%$ vs.\ $58.22\%$), and targeted reference point displacement ($L_2 = 0.6148$ with Center $L_2 = 0.5188$), which drops decoder deformable cross-attention foreground hit rates down to $38.34\%$ (vs.\ $53.50\%$ for Task PGD). Output active box counts under AQSM remain compact ($7.15 \pm 4.30$ boxes), confirming that AQSM operates via selective anchor misrouting rather than brute-force prediction flooding.
\end{enumerate}
Across perturbation budgets $\epsilon \in \{4/255, 8/255, 16/255\}$, AQSM degrades mAP@50 sharply from clean $65.15\%$ to $13.55\%$ at $\epsilon = 4/255$ and $4.06\%$ at $\epsilon = 8/255$, exhibiting a slight plateau at $5.36\%$ at $\epsilon = 16/255$. Under a fixed iteration budget ($T = 15, \alpha = 2/255$), the wider projection hypercube ($\pm 16/255$) allows adversarial trajectories to explore complex non-convex boundary regions where a small fraction of pseudo-background candidate anchors happen to broadly intersect large-scale foreground objects, demonstrating that query selection manipulation reaches bounded saturation."""

assert old_sec58_para in text, "old_sec58_para not found in text"
text = text.replace(old_sec58_para, new_sec58_para)

with open(r"G:\Edge_Sec\paper_1_aqsm_mechanism\main.tex", "w", encoding="utf-8") as f:
    f.write(text)

print("Successfully updated Section 5.7 and Section 5.8!")
