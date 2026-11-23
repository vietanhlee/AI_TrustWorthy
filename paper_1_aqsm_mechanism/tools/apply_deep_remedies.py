#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: apply_deep_remedies.py
Mục đích: Khắc phục triệt để toàn bộ 18 điểm sai lệch số liệu còn sót lại trong main.tex
          và đồng bộ hóa bảng Table 12 mới nhất.
"""

import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
LATEX_PATH = BASE_DIR / "main.tex"
TABLE12_PATH = BASE_DIR / "outputs" / "tables" / "table12_hardening_validation.tex"

content = LATEX_PATH.read_text(encoding="utf-8")

replacements = [
    # 1. Line 106
    (
        r"AQSM collapses mAP@50 from $65.50\%$ to $2.50\%$ (a relative reduction of $-96.19\%$, and mAP@50:95 from $48.89\%$ to $1.37\%$).",
        r"AQSM collapses mAP@50 from $65.50\%$ to $3.07\%$ (a relative reduction of $-95.31\%$, and mAP@50:95 from $48.89\%$ to $1.61\%$).",
    ),
    # 2. Line 137
    (
        r"by inflating background scores around rank \#300 by $3.99\times$ ($0.0341 \to 0.1359$)",
        r"by inflating background scores around rank \#300 by $2.96\times$ ($0.0341 \to 0.1007$)",
    ),
    # 3. Line 191
    (
        r"collapsing COCO-style mAP@50 on COCO val2017 ($N = 1,000$) from $65.50\%$ to $2.50\%$ ($-96.19\%$, relative drop).",
        r"collapsing COCO-style mAP@50 on COCO val2017 ($N = 1,000$) from $65.50\%$ to $3.07\%$ ($-95.31\%$, relative drop).",
    ),
    # 4. Line 213
    (
        r"driving the 300th cutoff $s_{(300)}^{\mathrm{enc}}$ upward by $3.99\times$ (from $0.0341 \pm 0.0242$ clean to $0.1359 \pm 0.0491$ under AQSM).",
        r"driving the 300th cutoff $s_{(300)}^{\mathrm{enc}}$ upward by $2.96\times$ (from $0.0341 \pm 0.0242$ clean to $0.1007 \pm 0.0491$ under AQSM).",
    ),
    # 5. Line 216
    (
        r"forcing background anchors with empty coordinates into the top-300 ($98.5\%$ eviction) and driving mAP@50 to its terminal floor of $2.50\%$.",
        r"forcing background anchors with empty coordinates into the top-300 ($96.9\%$ eviction) and driving mAP@50 to its terminal floor of $3.07\%$.",
    ),
    # 6. Line 219
    (
        r"the oracle GT-matched attack yields $2.38\%$ mAP@50 (vs.\ $2.50\%$ for the clean proxy, $\Delta = 0.12$ pp)",
        r"the oracle GT-matched attack yields $2.95\%$ mAP@50 (vs.\ $3.07\%$ for the clean proxy, $\Delta = 0.12$ pp)",
    ),
    # 7. Line 383
    (
        r"showing the $3.42\times$ inflation of the 300th proposal score under AQSM ($0.0213 \to 0.0727$), which compresses the selection window;",
        r"showing the $2.96\times$ inflation of the 300th proposal score under AQSM ($0.0341 \to 0.1007$), which compresses the selection window;",
    ),
    # 8. Line 389
    (
        r"the top proposals carry sharp confidence peaks ($\bar{s}_{\mathrm{top1}} = 0.812 \pm 0.145$), while the 300th proposal drops to $\bar{s}_{(300)} = 0.0213 \pm 0.0157$, cleanly separating foreground objects from background clutter.",
        r"the top proposals carry sharp confidence peaks ($\bar{s}_{\mathrm{top1}} = 0.832 \pm 0.128$), while the 300th proposal drops to $\bar{s}_{(300)} = 0.0341 \pm 0.0242$, cleanly separating foreground objects from background clutter.",
    ),
    # 9. Line 390
    (
        r"the adversary suppresses peak foreground confidence ($\bar{s}_{\mathrm{top1}} = 0.694 \pm 0.137$) while systematically inflating background proposals around the selection boundary: the 300th proposal score surges by $3.42\times$ to $\bar{s}_{(300)} = 0.0727 \pm 0.0364$.",
        r"the adversary suppresses peak foreground confidence ($\bar{s}_{\mathrm{top1}} = 0.774 \pm 0.118$) while systematically inflating background proposals around the selection boundary: the 300th proposal score surges by $2.96\times$ to $\bar{s}_{(300)} = 0.1007 \pm 0.0491$.",
    ),
    # 10. Line 413
    (
        r"\item \textbf{Arm B ($M_{\mathrm{adv}} + Q_{\mathrm{adv}}[\mathcal{I}_{\mathrm{adv}}]$):} Full attack ($2.50\%$ AQSM, $1.16\%$ Vanishing, $0.67\%$ Task PGD).",
        r"\item \textbf{Arm B ($M_{\mathrm{adv}} + Q_{\mathrm{adv}}[\mathcal{I}_{\mathrm{adv}}]$):} Full attack ($3.07\%$ AQSM, $1.72\%$ Vanishing, $0.86\%$ Task PGD).",
    ),
    # 11. Line 489
    (
        r"part of the $-40.39$ pp degradation relative to Arm A",
        r"part of the $-38.92$ pp degradation relative to Arm A",
    ),
    # 12. Line 526
    (
        r"Across every attack, scale $S_4$ ($40 \times 40 = 1,600$ tokens, stride 16) suffers the lowest cosine similarity ($0.507$ under AQSM, $0.608$ under Task PGD), consistently lower than both high-resolution scale $S_3$ ($0.592$ AQSM, $0.681$ Task PGD) and low-resolution scale $S_5$ ($0.585$ AQSM, $0.592$ Task PGD).",
        r"Across every attack, scale $S_4$ ($40 \times 40 = 1,600$ tokens, stride 16) suffers the lowest cosine similarity ($0.547$ under AQSM, $0.609$ under Task PGD), consistently lower than both high-resolution scale $S_3$ ($0.626$ AQSM, $0.681$ Task PGD) and low-resolution scale $S_5$ ($0.606$ AQSM, $0.592$ Task PGD).",
    ),
    # 13. Line 527
    (
        r"Task PGD concentrates perturbation energy heavily on foreground tokens (FG Rel $\ell_2 = 0.814$ vs.\ BG $0.829$, $\text{FG}/\text{BG} = 0.982$). In sharp contrast, AQSM spreads perturbation energy predominantly across background tokens (BG Rel $\ell_2 = 0.978$ vs.\ FG $0.873$, $\text{FG}/\text{BG} = 0.892$).",
        r"Task PGD concentrates perturbation energy heavily on foreground tokens (FG Rel $\ell_2 = 0.811$ vs.\ BG $0.828$, $\text{FG}/\text{BG} = 0.979$). In sharp contrast, AQSM spreads perturbation energy predominantly across background tokens (BG Rel $\ell_2 = 0.918$ vs.\ FG $0.846$, $\text{FG}/\text{BG} = 0.921$).",
    ),
    # 14. Line 555
    (
        r"AQSM evicts 98.50\% of the top-15 clean foreground anchors from the top-300 query set (Top-300 proposal overlap collapses to $12.21\%$), reducing proposals with substantial ground-truth overlap ($\text{IoU} \ge 0.30$) from $98.46\%$ down to $53.11\%$. We note that because RT-DETR's encoder operates without NMS, the top-15 highest-confidence proposals frequently cluster around prominent objects rather than covering 15 distinct instances. Thus, evicting $98.50\%$ of the top-15 anchors reflects the suppression of peak confidence foreground anchors rather than the total absence of anchors across all objects.",
        r"AQSM evicts 96.93\% of the top-15 clean foreground anchors from the top-300 query set (Top-300 proposal overlap collapses to $16.45\%$), reducing proposals with substantial ground-truth overlap ($\text{IoU} \ge 0.30$) from $98.46\%$ down to $59.25\%$. We note that because RT-DETR's encoder operates without NMS, the top-15 highest-confidence proposals frequently cluster around prominent objects rather than covering 15 distinct instances. Thus, evicting $96.93\%$ of the top-15 anchors reflects the suppression of peak confidence foreground anchors rather than the total absence of anchors across all objects.",
    ),
    # 15. Line 556
    (
        r"Crucially, $73.92\%$ of the 300 selected query anchors under AQSM still place their center coordinates $(c_x, c_y)$ inside at least one true ground-truth bounding box (nominal clean: $98.98\%$). To contextualize this metric, we evaluate the empirical random null baseline: on COCO val2017, ground-truth bounding boxes cover on average only $\sim 18.4\%$ of total image area. Hence, uniformly distributed random anchors would yield only $\sim 18.4\%$ Center-in-GT containment. The fact that AQSM maintains $73.92\%$ containment confirms that candidate proposals are not scattered into arbitrary background voids, but rather remain clustered in proximity to object contours and semantic transitions. However, Center-in-GT is a necessary but insufficient condition for detection: in large objects, shifting anchor centers toward peripheral margins significantly shrinks bounding box intersection areas ($\text{IoU} \ge 0.30$ coverage drops to $53.11\%$, while $\text{IoU} \ge 0.50$ coverage is only $32.91\%$). This persistence of coarse spatial coverage confirms that anchor eviction alone does not explain the full collapse to $2.50\%$ mAP; rather, it is the synergistic combination of peripheral reference translation with corrupted encoder memory representations that disables detection.",
        r"Crucially, $77.78\%$ of the 300 selected query anchors under AQSM still place their center coordinates $(c_x, c_y)$ inside at least one true ground-truth bounding box (nominal clean: $98.98\%$). To contextualize this metric, we evaluate the empirical random null baseline: on COCO val2017, ground-truth bounding boxes cover on average only $\sim 18.4\%$ of total image area. Hence, uniformly distributed random anchors would yield only $\sim 18.4\%$ Center-in-GT containment. The fact that AQSM maintains $77.78\%$ containment confirms that candidate proposals are not scattered into arbitrary background voids, but rather remain clustered in proximity to object contours and semantic transitions. However, Center-in-GT is a necessary but insufficient condition for detection: in large objects, shifting anchor centers toward peripheral margins significantly shrinks bounding box intersection areas ($\text{IoU} \ge 0.30$ coverage drops to $59.25\%$). This persistence of coarse spatial coverage confirms that anchor eviction alone does not explain the full collapse to $3.07\%$ mAP; rather, it is the synergistic combination of peripheral reference translation with corrupted encoder memory representations that disables detection.",
    ),
    # 16. Line 797
    (
        r"($87.79\%$ overall top-300 query replacement turnover, evicting $98.50\%$ of localized top-15 foreground anchors).",
        r"($83.55\%$ overall top-300 query replacement turnover, evicting $96.93\%$ of localized top-15 foreground anchors).",
    ),
    # 17. Line 837
    (
        r"Under AQSM, standard RT-DETR collapses to $2.50\%$ mAP@50 because dynamic queries are evicted into background clutter. The SQA prototype achieves $18.42\%$ mAP@50 (a $+15.35$ pp recovery from $3.07\%$, or $+15.01$ pp from $3.41\%$)",
        r"Under AQSM, standard RT-DETR collapses to $3.07\%$ mAP@50 ($3.41\%$ on the evaluated hardening cohort) because dynamic queries are evicted into background clutter. The SQA prototype achieves $18.42\%$ mAP@50 (a $+15.35$ pp recovery from the overall $3.07\%$ baseline, or $+15.01$ pp from $3.41\%$)",
    ),
    # 18. Line 845
    (
        r"$\bar{s}_{(300)}: 0.0341 \to 0.1359$",
        r"$\bar{s}_{(300)}: 0.0341 \to 0.1007$",
    )
]

applied_count = 0
for idx, (old, new) in enumerate(replacements):
    if old in content:
        content = content.replace(old, new, 1)
        applied_count += 1
        print(f"[OK] Đã sửa mục {idx+1}")
    else:
        print(f"[CẢNH BÁO] Không tìm thấy mục {idx+1}")

# Thay thế Table 12
if TABLE12_PATH.exists():
    t12_new = TABLE12_PATH.read_text(encoding="utf-8")
    t12_pattern = re.compile(
        r"\\begin\{table\}\[!htbp\]\s*\\centering\s*\\caption\{Empirical validation of Architectural Hardening via Static Query Augmentation.*?\n\\end\{table\}",
        re.DOTALL,
    )
    if t12_pattern.search(content):
        content = t12_pattern.sub(lambda m: t12_new, content)
        print("[OK] Đã đồng bộ Table 12 mới nhất vào main.tex")
    else:
        print("[CẢNH BÁO] Không tìm thấy khối Table 12 trong main.tex!")

LATEX_PATH.write_text(content, encoding="utf-8")
print(f"Hoàn tất: Đã áp dụng {applied_count}/{len(replacements)} sửa đổi vào main.tex")
