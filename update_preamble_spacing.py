"""
Update microtype to full expansion and set bibsep to 0.5pt
"""

with open(r"G:\Edge_Sec\paper_1_aqsm_mechanism\main.tex", "r", encoding="utf-8") as f:
    text = f.read()

text = text.replace(
    r"\usepackage[protrusion=true,expansion=false]{microtype}",
    r"\usepackage[protrusion=true,expansion=true]{microtype}"
)
text = text.replace(r"\setlength{\bibsep}{2.0pt}", r"\setlength{\bibsep}{0.5pt}")
text = text.replace(r"\setlength{\bibsep}{1.0pt}", r"\setlength{\bibsep}{0.5pt}")

with open(r"G:\Edge_Sec\paper_1_aqsm_mechanism\main.tex", "w", encoding="utf-8") as f:
    f.write(text)

print("Preamble microtype & bibsep updated!")
