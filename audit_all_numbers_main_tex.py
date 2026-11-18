"""
Audit script to search for any remaining inconsistent numbers in main.tex
"""

import re

with open(r"G:\Edge_Sec\paper_1_aqsm_mechanism\main.tex", "r", encoding="utf-8") as f:
    text = f.read()

stale_patterns = [
    r"24\.85",
    r"36\.40",
    r"28\.94",
    r"18\.25",
    r"0\.99%",
    r"2\.45%",
    r"93\.65%",
    r"21\.45%",
    r"74\.15%",
    r"27\.40%",
    r"61\.20%",
    r"35\.80%",
    r"0\.792",
    r"0\.635",
    r"0\.785",
    r"0\.8431",
    r"0\.00%",
    r"0\.0%",
]

print("=== CHECKING FOR STALE NUMBERS ===")
found_any = False
for pat in stale_patterns:
    matches = list(re.finditer(pat, text))
    if matches:
        found_any = True
        print(f"FOUND {len(matches)} occurrences of pattern: {pat}")
        for m in matches:
            start = max(0, m.start() - 50)
            end = min(len(text), m.end() + 50)
            print("   Context:", text[start:end].replace("\n", " "))

if not found_any:
    print("ALL STALE PATTERNS CLEAR! Zero discrepancies found!")
