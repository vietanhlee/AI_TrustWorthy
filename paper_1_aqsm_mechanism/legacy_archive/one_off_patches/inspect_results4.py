# -*- coding: utf-8 -*-
import json
from pathlib import Path

base = Path(r"C:\Users\levie\Downloads\results (4)\AI_TrustWorthy\paper_1_aqsm_mechanism\outputs\summary_json")

def print_dict(d, prefix="  "):
    for k, v in d.items():
        if isinstance(v, dict):
            print(f"{prefix}{k}:")
            print_dict(v, prefix + "  ")
        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
            print(f"{prefix}{k}: [list of {len(v)} dicts]")
            for item in v[:3]:
                print(f"{prefix}  {item}")
        else:
            print(f"{prefix}{k}: {v}")

for jf in ["deformable_attention_telemetry.json", "tautology_control_results.json", "matched_budget_sweep.json", "defense_esm_evaluation.json", "latency_profile.json"]:
    p = base / jf
    if p.exists():
        print(f"=== {jf} ===")
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        print_dict(d)

