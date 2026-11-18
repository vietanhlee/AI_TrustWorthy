"""
Targeted precise string replacement for the remaining occurrences using exact text.replace().
"""

with open(r"G:\Edge_Sec\paper_1_aqsm_mechanism\main.tex", "r", encoding="utf-8") as f:
    text = f.read()

replacements = [
    # 1. Section 5.3 numbers
    (
        r"inflicts a $-28.94$ pp mAP@50 collapse ($36.40\%$, $55.7\%$ retention), and full query prior swap (Arm $\text{D}_{\text{full}}$) drops mAP@50 to $24.85\%$ ($38.0\%$ retention)",
        r"inflicts a $-34.78$ pp mAP@50 collapse ($30.37\%$, $46.6\%$ retention), and full query prior swap (Arm $\text{D}_{\text{full}}$) drops mAP@50 to $26.29\%$ ($40.4\%$ retention)"
    ),
    (
        r"Both AQSM ($+18.25$ pp, $p < 10^{-15}$) and Task PGD ($+31.40$ pp, $p < 10^{-15}$)",
        r"Both AQSM ($+11.62$ pp, $p = 0.025$) and Task PGD ($+28.40$ pp, $p < 10^{-15}$)"
    ),
    # 2. Section 1 / Intro numbers
    (
        r"($\eta_p^2 = 0.635$, $-28.94$ pp mAP drop under pure index replacement)",
        r"($\eta_p^2 = 0.820$, $-29.12$ pp mAP drop under pure index replacement)"
    ),
    (
        r"leading driver of performance collapse ($\eta_p^2 = 0.792, p < 10^{-15}$)",
        r"leading driver of performance collapse ($\eta_p^2 = 0.902, p < 10^{-15}$)"
    ),
    (
        r"represents the predominant factor ($\eta_p^2 = 0.792, p < 10^{-15}$)",
        r"represents the predominant factor ($\eta_p^2 = 0.902, p < 10^{-15}$)"
    ),
    # 3. Defense ESM numbers
    (
        r"stationary attacks are flagged with AUROC up to $0.8431$, a white-box Carlini-Tramèr adaptive adversary evades detection completely ($\text{TPR} \to 0.0\%$)",
        r"stationary attacks are flagged with AUROC up to $0.8800$ ($\text{TPR} = 80.00\%$), a white-box Carlini-Tramèr adaptive adversary partially evades detection ($\text{TPR}$ drops to $46.67\% - 53.33\%$)"
    ),
    (
        r"achieves $\text{TPR} = 34.00\%$ ($\text{AUROC} = \mathbf{0.8431}$, mean anomaly score $0.9030$)",
        r"achieves $\text{TPR} = 80.00\%$ ($\text{AUROC} = \mathbf{0.8800}$, mean anomaly score $0.9022$)"
    )
]

for old, new in replacements:
    count = text.count(old)
    print(f"Replacing '{old[:40]}...': {count} occurrences found")
    if count > 0:
        text = text.replace(old, new)

with open(r"G:\Edge_Sec\paper_1_aqsm_mechanism\main.tex", "w", encoding="utf-8") as f:
    f.write(text)

print("Targeted text replacements completed successfully!")
