# -*- coding: utf-8 -*-
import json
import os
from pathlib import Path

def inspect_summary():
    s_dir = Path("paper_1_aqsm_mechanism/outputs/summary_json")
    print("==================================================================")
    print(" GROUND TRUTH EXTRACTION FROM SUMMARY JSON ARTIFACTS")
    print("==================================================================")
    
    for p in sorted(s_dir.glob("*.json")):
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
            print(f"\n>>> [FILE] {p.name}")
            
            # Print top level metadata
            for meta_k in ["experiment", "n_images", "n_evaluated", "clean_baseline", "clean_metrics"]:
                if meta_k in d:
                    print(f"    {meta_k}: {d[meta_k]}")
                    
            if p.name == "defense_evaluation_summary.json" or p.name == "defense_esm_evaluation.json":
                if "attacks" in d:
                    for atk, val in d["attacks"].items():
                        print(f"    Attack: {atk} -> TPR={val.get('tpr_detection_rate_pct')}%, FPR={val.get('fpr_false_alarm_pct')}%, AUROC={val.get('auroc')}")
                if "calibration_test_split" in d:
                    print(f"    Split: {d['calibration_test_split']}")
                if "adaptive_evasion" in d:
                    for lk, lval in d["adaptive_evasion"].items():
                        print(f"    Adaptive {lk} -> TPR={lval.get('detected_tpr_pct')}%, Evasion={lval.get('evasion_success_pct')}%")

            elif p.name == "hardening_evaluation_results.json":
                for k, v in d.items():
                    if isinstance(v, dict):
                        print(f"    {k}: {v}")

            elif "causal_swap_results" in p.name:
                arms = d.get("arms", {})
                for arm_k, arm_v in arms.items():
                    print(f"    {arm_k}: mAP@50={arm_v.get('mAP@50')}, mAP@50:95={arm_v.get('mAP@50:95')}")
                if "two_way_anova" in d:
                    print("    ANOVA:", d["two_way_anova"])
                if "paired_bootstrap_delta_d_vs_b" in d:
                    print("    Bootstrap D vs B:", d["paired_bootstrap_delta_d_vs_b"])

            elif p.name == "controlled_misrouting_results.json":
                for k, v in d.items():
                    if k in ["interventions", "results", "empirical_evaluation"]:
                        print(f"    {k} keys: {list(v.keys())}")
                        for sub_k, sub_v in v.items():
                            if isinstance(sub_v, dict):
                                print(f"      {sub_k}: mAP@50={sub_v.get('mAP@50')}, recall={sub_v.get('recall')}, fg_hit={sub_v.get('fg_hit')}")

            elif p.name == "deformable_attention_telemetry.json" or p.name == "mechanism_telemetry_results.json":
                for k in ["entropy_analysis", "spatial_misrouting_telemetry", "attention_entropy", "sampling_locations"]:
                    if k in d:
                        print(f"    {k}: {d[k]}")

            elif p.name == "matched_budget_sweep.json":
                sw = d.get("sweep_results", {})
                for atk, ev in sw.items():
                    for ek, eval_data in ev.items():
                        print(f"    {atk} {ek}: mAP@50={eval_data.get('mAP@50')}, mAP@50:95={eval_data.get('mAP@50:95')}, evicted={eval_data.get('mean_evicted_ratio')}")

        except Exception as e:
            print(f"    ERROR reading {p.name}: {e}")

if __name__ == "__main__":
    inspect_summary()
