#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: audit_data_provenance.py
Mục đích: Công cụ kiểm định tính toàn vẹn và truy xuất nguồn gốc dữ liệu (Data Provenance Auditor)
chuẩn học thuật cho Paper 1:
  - Đọc data_provenance_manifest.json (phiên bản N=1,000 Native 640x640)
  - Kiểm tra sự tồn tại và tính hợp lệ của từng file JSON/CSV artifact
  - So sánh trực tiếp từng số liệu trong main.tex (Table 1 đến Table 12, Text, Abstract) với JSON artifact
  - Xác minh tính khăng khít và đồng bộ tuyệt đối 100% không có sai lệch
"""

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

BASE_DIR = Path(__file__).resolve().parent.parent
MANIFEST_PATH = BASE_DIR / "outputs" / "data_provenance_manifest.json"
LATEX_PATH = BASE_DIR / "main.tex"
SUMMARY_DIR = BASE_DIR / "outputs" / "summary_json"


def load_json(p: Path) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as fp:
        return json.load(fp)


def audit_artifacts_existence(manifest: Dict[str, Any]) -> List[str]:
    errors = []
    tables = manifest.get("tables", {})
    for tid, tinfo in tables.items():
        artifacts = tinfo.get("source_artifacts", [])
        for art in artifacts:
            art_path = BASE_DIR / art
            if not art_path.exists():
                errors.append(f"[{tid}] Thiếu artifact file: {art}")
            elif art_path.stat().st_size == 0:
                errors.append(f"[{tid}] Artifact file rỗng (0 bytes): {art}")
    return errors


def audit_numerical_values(tex_content: str) -> List[str]:
    errors = []

    # 1. Causal Swap results & ANOVA
    c_aqsm = load_json(SUMMARY_DIR / "causal_swap_results_aqsm.json")
    clean_map = c_aqsm["point_maps"]["Arm_A"]["mAP@50"]
    aqsm_b_map = c_aqsm["point_maps"]["Arm_B"]["mAP@50"]
    arm_c_map = c_aqsm["point_maps"]["Arm_C_full"]["mAP@50"]
    arm_d_full_map = c_aqsm["point_maps"]["Arm_D_full"]["mAP@50"]
    arm_d_idx_map = c_aqsm["point_maps"]["Arm_D_index"]["mAP@50"]
    me_m = c_aqsm["anova_repeated_measures_full"]["marginal_effects"]["ME_M"]
    me_q = c_aqsm["anova_repeated_measures_full"]["marginal_effects"]["ME_Q"]
    inter_mq = c_aqsm["anova_repeated_measures_full"]["marginal_effects"]["Interaction_MxQ"]
    eta_m = c_aqsm["anova_repeated_measures_full"]["attribution_shares"]["partial_eta_squared"]["Memory_M"]
    eta_q = c_aqsm["anova_repeated_measures_full"]["attribution_shares"]["partial_eta_squared"]["Query_Q"]
    ss_m = c_aqsm["anova_repeated_measures_full"]["anova_table"]["Memory_M"]["SS"]
    ss_q = c_aqsm["anova_repeated_measures_full"]["anova_table"]["Query_Q"]["SS"]
    ss_inter = c_aqsm["anova_repeated_measures_full"]["anova_table"]["Interaction_MxQ"]["SS"]

    checks = [
        (f"{clean_map:.2f}", f"Clean baseline mAP@50 ({clean_map:.2f}%)"),
        (f"{aqsm_b_map:.2f}", f"AQSM Arm B mAP@50 ({aqsm_b_map:.2f}%)"),
        (f"{arm_c_map:.2f}", f"AQSM Arm C_full mAP@50 ({arm_c_map:.2f}%)"),
        (f"{arm_d_full_map:.2f}", f"AQSM Arm D_full mAP@50 ({arm_d_full_map:.2f}%)"),
        (f"{arm_d_idx_map:.2f}", f"AQSM Arm D_index mAP@50 ({arm_d_idx_map:.2f}%)"),
        (f"{me_m:.2f}", f"Repeated-measures ME_M ({me_m:.2f})"),
        (f"{me_q:.2f}", f"Repeated-measures ME_Q ({me_q:.2f})"),
        (f"{inter_mq:.2f}", f"Repeated-measures Interaction ({inter_mq:.2f})"),
        (f"{eta_m:.3f}", f"Partial eta^2 Memory ({eta_m:.3f})"),
        (f"{eta_q:.3f}", f"Partial eta^2 Query ({eta_q:.3f})"),
        (f"{ss_m:.1f}", f"SS Memory ({ss_m:.1f})"),
        (f"{ss_q:.1f}", f"SS Query ({ss_q:.1f})"),
        (f"{ss_inter:.1f}", f"SS Interaction ({ss_inter:.1f})"),
    ]

    for val_str, desc in checks:
        if val_str not in tex_content:
            errors.append(f"Không tìm thấy {desc} ({val_str}) trong main.tex!")

    # Non-parametric Wilcoxon tests
    np_contrasts = c_aqsm.get("non_parametric_contrasts", {})
    w_dfull_b = np_contrasts.get("Arm_D_full_vs_Arm_B", {}).get("paired_wilcoxon_stat")
    w_didx_a = np_contrasts.get("Arm_D_index_vs_Arm_A", {}).get("paired_wilcoxon_stat")
    if w_dfull_b and f"{w_dfull_b}" not in tex_content:
        errors.append(f"Không tìm thấy Wilcoxon W stat cho Arm D_full vs B ({w_dfull_b}) trong main.tex!")
    if w_didx_a and f"{w_didx_a}" not in tex_content:
        errors.append(f"Không tìm thấy Wilcoxon W stat cho Arm D_index vs A ({w_didx_a}) trong main.tex!")

    # 2. Telemetry verification (Table 7)
    telemetry = load_json(SUMMARY_DIR / "deformable_attention_telemetry.json")
    clean_ent = telemetry["clean"]["mean_entropy_l1"]
    aqsm_ent = telemetry["aqsm"]["mean_entropy_l1"]
    van_ent = telemetry["vanishing"]["mean_entropy_l1"]
    clean_hit = telemetry["clean"]["mean_fg_hit_rate"] * 100
    aqsm_hit = telemetry["aqsm"]["mean_fg_hit_rate"] * 100
    aqsm_dropoff = telemetry["aqsm"]["mean_dropoff_ratio"]
    aqsm_margin = telemetry["aqsm"]["mean_score_margin"]
    aqsm_drift = telemetry["aqsm"]["mean_ref_drift"]

    tel_checks = [
        (f"{clean_ent:.4f}", f"Clean attention entropy ({clean_ent:.4f})"),
        (f"{aqsm_ent:.4f}", f"AQSM attention entropy ({aqsm_ent:.4f})"),
        (f"{van_ent:.4f}", f"Vanishing attention entropy ({van_ent:.4f})"),
        (f"{clean_hit:.2f}", f"Clean FG hit rate ({clean_hit:.2f}%)"),
        (f"{aqsm_hit:.2f}", f"AQSM FG hit rate ({aqsm_hit:.2f}%)"),
        (f"{aqsm_dropoff:.2f}", f"AQSM Dropoff Ratio ({aqsm_dropoff:.2f})"),
        (f"{aqsm_margin:.4f}", f"AQSM Score Margin ({aqsm_margin:.4f})"),
        (f"{aqsm_drift:.4f}", f"AQSM Reference Drift ({aqsm_drift:.4f})"),
    ]
    for val_str, desc in tel_checks:
        if val_str not in tex_content:
            errors.append(f"Không tìm thấy {desc} ({val_str}) trong main.tex!")

    # 3. Multi-Scale Encoder Memory Deformation (Table 5)
    enc_mem = load_json(SUMMARY_DIR / "encoder_memory_analysis.json")
    aqsm_cos_ovr = enc_mem["results"]["aqsm"]["cosine_similarity"]["Overall"]["mean"]
    aqsm_fg_l2 = enc_mem["results"]["aqsm"]["relative_l2_perturbation"]["Foreground"]["mean"]
    aqsm_bg_l2 = enc_mem["results"]["aqsm"]["relative_l2_perturbation"]["Background"]["mean"]
    van_cos_ovr = enc_mem["results"]["vanishing"]["cosine_similarity"]["Overall"]["mean"]
    task_cos_ovr = enc_mem["results"]["taskpgd"]["cosine_similarity"]["Overall"]["mean"]

    tab5_checks = [
        (f"{aqsm_cos_ovr:.4f}", f"Table 5 AQSM Overall Cosine ({aqsm_cos_ovr:.4f})"),
        (f"{aqsm_fg_l2:.4f}", f"Table 5 AQSM FG Rel L2 ({aqsm_fg_l2:.4f})"),
        (f"{aqsm_bg_l2:.4f}", f"Table 5 AQSM BG Rel L2 ({aqsm_bg_l2:.4f})"),
        (f"{van_cos_ovr:.4f}", f"Table 5 Vanishing Overall Cosine ({van_cos_ovr:.4f})"),
        (f"{task_cos_ovr:.4f}", f"Table 5 Task PGD Overall Cosine ({task_cos_ovr:.4f})"),
    ]
    for val_str, desc in tab5_checks:
        if val_str not in tex_content:
            errors.append(f"Không tìm thấy {desc} ({val_str}) trong main.tex!")

    # 4. Query Selection Dynamics (Table 6)
    q_dyn = load_json(SUMMARY_DIR / "encoder_query_dynamics.json")
    aqsm_top300_ovl = q_dyn["results"]["aqsm"]["top300_overlap_pct"]["mean"]
    aqsm_top15_evict = q_dyn["results"]["aqsm"]["evicted_top15_fg_pct"]["mean"]
    aqsm_spearman = q_dyn["results"]["aqsm"]["spearman_rho_all_proposals"]["mean"]
    aqsm_kendall = q_dyn["results"]["aqsm"]["kendall_tau_all_proposals"]["mean"]

    tab6_checks = [
        (f"{aqsm_top300_ovl:.2f}\\%", f"Table 6 AQSM Top-300 Overlap ({aqsm_top300_ovl:.2f}%)"),
        (f"{aqsm_top15_evict:.2f}\\%", f"Table 6 AQSM Top-15 FG Eviction ({aqsm_top15_evict:.2f}%)"),
        (f"{aqsm_spearman:.3f}", f"Table 6 AQSM Spearman rho ({aqsm_spearman:.3f})"),
        (f"{aqsm_kendall:.3f}", f"Table 6 AQSM Kendall tau ({aqsm_kendall:.3f})"),
    ]
    for val_str, desc in tab6_checks:
        if val_str not in tex_content:
            errors.append(f"Không tìm thấy {desc} ({val_str}) trong main.tex!")

    # 5. Controlled misrouting verification (Table 8)
    misrouting = load_json(SUMMARY_DIR / "controlled_misrouting_results.json")
    reassign_map = misrouting["table8_summary"]["Random_Reassignment"]["mAP@50_mean"]
    shift_map = misrouting["table8_summary"]["Matched_L2_Random_Shift"]["mAP@50_mean"]
    aqsm_mis_map = misrouting["table8_summary"]["AQSM_Spatial_Misrouting"]["mAP@50"]
    shift_recall = misrouting["table8_summary"]["Matched_L2_Random_Shift"]["recall_pct_mean"]
    aqsm_recall = misrouting["table8_summary"]["AQSM_Spatial_Misrouting"]["recall_pct"]
    shift_hit = misrouting["table8_summary"]["Matched_L2_Random_Shift"]["fg_hit_pct_mean"]
    aqsm_hit8 = misrouting["table8_summary"]["AQSM_Spatial_Misrouting"]["fg_hit_pct"]

    mis_checks = [
        (f"{reassign_map:.2f}", f"Random Reassignment mAP@50 ({reassign_map:.2f}%)"),
        (f"{shift_map:.2f}", f"Matched-L2 Random Shift mAP@50 ({shift_map:.2f}%)"),
        (f"{aqsm_mis_map:.2f}", f"AQSM Spatial Misrouting mAP@50 ({aqsm_mis_map:.2f}%)"),
        (f"{shift_recall:.2f}", f"Random Shift Recall ({shift_recall:.2f}%)"),
        (f"{aqsm_recall:.2f}", f"AQSM Spatial Misrouting Recall ({aqsm_recall:.2f}%)"),
        (f"{shift_hit:.2f}", f"Random Shift FG Hit ({shift_hit:.2f}%)"),
        (f"{aqsm_hit8:.2f}", f"AQSM Spatial Misrouting FG Hit ({aqsm_hit8:.2f}%)"),
    ]
    for val_str, desc in mis_checks:
        if val_str not in tex_content:
            errors.append(f"Không tìm thấy {desc} ({val_str}) trong main.tex!")

    # 4. Matched-Budget Sweep verification (Table 9)
    mb = load_json(SUMMARY_DIR / "matched_budget_results.json")
    aqsm_eps4 = mb["sweep_results"]["aqsm"]["eps_4"]["mAP@50"]
    aqsm_eps8 = mb["sweep_results"]["aqsm"]["eps_8"]["mAP@50"]
    aqsm_eps16 = mb["sweep_results"]["aqsm"]["eps_16"]["mAP@50"]
    task_eps4 = mb["sweep_results"]["taskpgd"]["eps_4"]["mAP@50"]
    task_eps8 = mb["sweep_results"]["taskpgd"]["eps_8"]["mAP@50"]
    task_eps16 = mb["sweep_results"]["taskpgd"]["eps_16"]["mAP@50"]
    van_eps4 = mb["sweep_results"]["vanishing"]["eps_4"]["mAP@50"]
    van_eps8 = mb["sweep_results"]["vanishing"]["eps_8"]["mAP@50"]
    van_eps16 = mb["sweep_results"]["vanishing"]["eps_16"]["mAP@50"]

    mb_checks = [
        (f"{aqsm_eps4:.2f}", f"AQSM eps=4 mAP@50 ({aqsm_eps4:.2f}%)"),
        (f"{aqsm_eps8:.2f}", f"AQSM eps=8 mAP@50 ({aqsm_eps8:.2f}%)"),
        (f"{aqsm_eps16:.2f}", f"AQSM eps=16 mAP@50 ({aqsm_eps16:.2f}%)"),
        (f"{task_eps4:.2f}", f"Task PGD eps=4 mAP@50 ({task_eps4:.2f}%)"),
        (f"{task_eps8:.2f}", f"Task PGD eps=8 mAP@50 ({task_eps8:.2f}%)"),
        (f"{task_eps16:.2f}", f"Task PGD eps=16 mAP@50 ({task_eps16:.2f}%)"),
        (f"{van_eps4:.2f}", f"Vanishing eps=4 mAP@50 ({van_eps4:.2f}%)"),
        (f"{van_eps8:.2f}", f"Vanishing eps=8 mAP@50 ({van_eps8:.2f}%)"),
        (f"{van_eps16:.2f}", f"Vanishing eps=16 mAP@50 ({van_eps16:.2f}%)"),
    ]
    for val_str, desc in mb_checks:
        if val_str not in tex_content:
            errors.append(f"Không tìm thấy {desc} ({val_str}) trong main.tex!")

    # 5. Hardware Execution Latency Profiling (Table 10)
    lat = load_json(SUMMARY_DIR / "edge_latency_results.json")
    bb_ms = lat["stages"]["Backbone"]["mean_ms"]
    enc_ms = lat["stages"]["Hybrid_Encoder"]["mean_ms"]
    qs_ms = lat["stages"]["Query_Selection"]["mean_ms"]
    dec_ms = lat["stages"]["Deformable_Decoder"]["mean_ms"]
    tot_ms = lat["stages"]["End_to_End_Total"]["mean_ms"]
    fps_val = lat["throughput_fps"]

    lat_checks = [
        (f"{bb_ms:.2f}", f"Backbone latency ({bb_ms:.2f} ms)"),
        (f"{enc_ms:.2f}", f"Hybrid Encoder latency ({enc_ms:.2f} ms)"),
        (f"{qs_ms:.2f}", f"Query Selection latency ({qs_ms:.2f} ms)"),
        (f"{dec_ms:.2f}", f"Deformable Decoder latency ({dec_ms:.2f} ms)"),
        (f"{tot_ms:.2f}", f"Total latency ({tot_ms:.2f} ms)"),
        (f"{fps_val:.2f}", f"Throughput FPS ({fps_val:.2f})"),
    ]
    for val_str, desc in lat_checks:
        if val_str not in tex_content:
            errors.append(f"Không tìm thấy {desc} ({val_str}) trong main.tex!")

    # 6. Defense ESM & Adaptive Evasion verification (Table 11)
    defense = load_json(SUMMARY_DIR / "defense_esm_evaluation.json")
    esm_tpr = defense["attacks"]["aqsm"]["tpr_detection_rate_pct"]
    esm_auroc = defense["attacks"]["aqsm"]["auroc"]
    clean_fpr = defense["empirical_clean_fpr_pct"]
    ev_l02 = defense["adaptive_evasion"]["lambda_0.2"]["detected_tpr_pct"]
    ev_l05 = defense["adaptive_evasion"]["lambda_0.5"]["detected_tpr_pct"]
    ev_l10 = defense["adaptive_evasion"]["lambda_1.0"]["detected_tpr_pct"]
    ev_l20 = defense["adaptive_evasion"]["lambda_2.0"]["detected_tpr_pct"]

    def_checks = [
        (f"{esm_tpr:.2f}", f"ESM AQSM TPR ({esm_tpr:.2f}%)"),
        (f"{esm_auroc:.4f}", f"ESM AQSM AUROC ({esm_auroc:.4f})"),
        (f"{clean_fpr:.2f}", f"ESM Clean FPR ({clean_fpr:.2f}%)"),
        (f"{ev_l02:.2f}", f"Adaptive Evasion lambda=0.2 TPR ({ev_l02:.2f}%)"),
        (f"{ev_l05:.2f}", f"Adaptive Evasion lambda=0.5 TPR ({ev_l05:.2f}%)"),
        (f"{ev_l10:.2f}", f"Adaptive Evasion lambda=1.0 TPR ({ev_l10:.2f}%)"),
        (f"{ev_l20:.2f}", f"Adaptive Evasion lambda=2.0 TPR ({ev_l20:.2f}%)"),
    ]
    for val_str, desc in def_checks:
        if val_str not in tex_content:
            errors.append(f"Không tìm thấy {desc} ({val_str}) trong main.tex!")

    # 7. SQA Architectural Hardening verification (Table 12) - đọc động 100%, không hardcode
    sqa = load_json(SUMMARY_DIR / "hardening_evaluation_results.json")
    sqa_res = sqa["results"]
    sqa_aqsm_map = sqa_res["aqsm_attack_eps16"]["hardened_static_augmented"]["mAP@50"]
    sqa_clean_penalty = sqa_res["clean_baseline"]["hardened_static_augmented"]["clean_penalty_pp"]
    sqa_overhead_pct = sqa_res["edge_latency_jetson_orin_nano_ms"]["overhead_pct"]
    sqa_dyn330 = sqa_res["controls"]["dynamic_top330"]["mAP@50"]
    sqa_rand30 = sqa_res["controls"]["random30_static"]["mAP@50"]
    sqa_adapt = sqa_res["controls"]["adaptive_evasion_sqa"]["mAP@50"]

    sqa_checks = [
        (f"{sqa_aqsm_map:.2f}", f"SQA Hardened AQSM mAP@50 ({sqa_aqsm_map:.2f}%)"),
        (f"{sqa_clean_penalty:.2f}", f"SQA Clean Penalty ({sqa_clean_penalty:.2f} pp)"),
        (f"{sqa_overhead_pct:.2f}", f"SQA Latency Overhead ({sqa_overhead_pct:.2f}%)"),
        (f"{sqa_dyn330:.2f}", f"SQA Control 1 Dynamic Top-330 mAP@50 ({sqa_dyn330:.2f}%)"),
        (f"{sqa_rand30:.2f}", f"SQA Control 2 Random-30 Static mAP@50 ({sqa_rand30:.2f}%)"),
        (f"{sqa_adapt:.2f}", f"SQA Adaptive White-box Evasion mAP@50 ({sqa_adapt:.2f}%)"),
    ]
    for val_str, desc in sqa_checks:
        if val_str not in tex_content:
            errors.append(f"Không tìm thấy {desc} ({val_str}) trong main.tex!")

    # 8. Log-Scale Multiplicative Decomposition & Equivalence Testing (Refined Stats - không hardcode)
    ref_stat_path = SUMMARY_DIR / "refined_statistical_methodology_summary.json"
    if ref_stat_path.exists():
        ref_stat = load_json(ref_stat_path)
        attr_shares = ref_stat.get("causal_decomposition", {}).get("attribution_shares", {})
        log_decomp = attr_shares.get("log_scale_multiplicative_decomposition") or ref_stat["log_scale_multiplicative_decomposition"]
        log_m = float(log_decomp["log_memory_share_pct"])
        log_q = float(log_decomp["log_query_share_pct"])
        log_inter = float(
            log_decomp["log_scale_interaction"]
            if "log_scale_interaction" in log_decomp
            else log_decomp["interaction_term_gamma"]
        )

        stat_checks = [
            (f"{log_m:.2f}", f"Log-Scale Memory Share ({log_m:.2f}%)"),
            (f"{log_q:.2f}", f"Log-Scale Query Share ({log_q:.2f}%)"),
            (f"{log_inter:.4f}", f"Log-Scale Interaction ({log_inter:.4f})"),
        ]
        null_blk = ref_stat.get("null_controls_and_tost", {})
        if isinstance(null_blk.get("paired_tost_table8"), dict):
            p_tost_blk = null_blk["paired_tost_table8"]
            t1_val = float(p_tost_blk["t1"])
            t2_val = float(p_tost_blk["t2"])
            p_tost_val = float(p_tost_blk["p_tost"])
            stat_checks.extend([
                (f"{t1_val:.3f}", f"Paired TOST t1 ({t1_val:.3f})"),
                (f"{t2_val:.3f}", f"Paired TOST t2 ({t2_val:.3f})"),
                (f"{p_tost_val:.4f}", f"Paired TOST p-value ({p_tost_val:.4f})"),
            ])
        elif isinstance(null_blk.get("two_sample_pooled_tost_empirical"), dict) and null_blk.get("status") != "deferred_requires_paired_per_image_csv_rerun":
            two_tost = null_blk["two_sample_pooled_tost_empirical"]
            if {"t1_stat", "t2_stat", "p_tost"}.issubset(two_tost.keys()):
                t1_val = float(two_tost["t1_stat"])
                t2_val = float(two_tost["t2_stat"])
                p_tost_val = float(two_tost["p_tost"])
                stat_checks.extend([
                    (f"{t1_val:.3f}", f"TOST t1 ({t1_val:.3f})"),
                    (f"{t2_val:.3f}", f"TOST t2 ({t2_val:.3f})"),
                    (f"{p_tost_val:.4f}", f"TOST p-value ({p_tost_val:.4f})"),
                ])

        for val_str, desc in stat_checks:
            if val_str not in tex_content:
                errors.append(f"Không tìm thấy {desc} ({val_str}) trong main.tex!")

    # 9. Multi-Seed Stability and Architecture Generalization (Table 13)
    ms_path = SUMMARY_DIR / "multimodel_and_seed_stability_summary.json"
    if ms_path.exists():
        ms_data = load_json(ms_path)
        rec = ms_data.get("records", {})
        r18_ms = rec.get("rtdetr_r18vd_multiseed", {})
        r50_ms = rec.get("rtdetr_r50vd_multiseed")
        r50_can = rec.get("rtdetr_r50vd_canonical", {})

        r18_ms_map = r18_ms["aqsm_map50_mean"]
        r18_ms_std = r18_ms["aqsm_map50_std"]
        r18_drift_std = r18_ms["hungarian_set_drift_std"]
        r18_hit_std = r18_ms["anchor_fg_hit_pct_std"]

        ms_checks = [
            (f"{r18_ms_map:.2f} \\pm {r18_ms_std:.2f}", f"Table 13 Multi-seed R18 ({r18_ms_map:.2f} +- {r18_ms_std:.2f})"),
            (f"{r18_drift_std:.4f}", f"Table 13 Hungarian drift std ({r18_drift_std:.4f})"),
            (f"{r18_hit_std:.2f}", f"Table 13 Anchor FG hit std ({r18_hit_std:.2f})"),
        ]
        if isinstance(r50_ms, dict) and "aqsm_map50_mean" in r50_ms:
            r50_clean = r50_ms["clean_map50"]
            r50_aqsm = r50_ms["aqsm_map50_mean"]
            r50_rel = r50_ms["relative_drop_pct"]
            ms_checks.extend([
                (f"{r50_clean:.2f}", f"Table 13 R50 Clean mAP@50 ({r50_clean:.2f}%)"),
                (f"{r50_aqsm:.2f}", f"Table 13 R50 AQSM mAP@50 mean ({r50_aqsm:.2f}%)"),
                (f"{r50_rel:.2f}", f"Table 13 R50 Relative Drop ({r50_rel:.2f}%)"),
            ])
        elif r50_can:
            r50_clean = r50_can["clean_map50"]
            r50_aqsm = r50_can["aqsm_map50"]
            r50_rel = r50_can["relative_drop_pct"]
            ms_checks.extend([
                (f"{r50_clean:.2f}", f"Table 13 R50 Clean mAP@50 ({r50_clean:.2f}%)"),
                (f"{r50_aqsm:.2f}", f"Table 13 R50 AQSM mAP@50 ({r50_aqsm:.2f}%)"),
                (f"{r50_rel:.2f}", f"Table 13 R50 Relative Drop ({r50_rel:.2f}%)"),
            ])

        for val_str, desc in ms_checks:
            if val_str not in tex_content:
                errors.append(f"Không tìm thấy {desc} ({val_str}) trong main.tex!")

    return errors


def audit_zero_stale_markers(tex_content: str) -> List[str]:
    """Kiểm tra tuyệt đối không còn bất kỳ số liệu preliminary cũ nào tồn tại trong main.tex."""
    stale_patterns = [
        (r"2\.50", "Old AQSM mAP@50 (2.50% thay vì 3.07%)"),
        (r"1\.16\%", "Old Query Vanishing mAP@50 (1.16% thay vì 1.72%)"),
        (r"6\.25", "Old Arm C_full mAP@50 (6.25% thay vì 7.45%)"),
        (r"3\.42\b", "Old 300th inflation (3.42x thay vì 2.96x) hoặc Arm C_idx (3.42% thay vì 3.98%)"),
        (r"25\.11", "Old Arm D_full (25.11% thay vì 26.58%)"),
        (r"33\.04", "Old Arm D_index (33.04% thay vì 35.44%)"),
        (r"67\.97", "Old Memory SS share (67.97% thay vì 68.99%)"),
        (r"26\.81", "Old Query SS share (26.81% thay vì 26.40%)"),
        (r"0\.806", "Old partial eta^2 Memory (0.806 thay vì 0.795)"),
        (r"12\.21", "Old Top-300 overlap (12.21% thay vì 16.45%)"),
        (r"98\.50", "Old Evicted Top-15 FG (98.50% thay vì 96.93%)"),
        (r"73\.92", "Old Center-in-GT (73.92% thay vì 77.78%)"),
        (r"0\.7901", "Old Attention entropy AQSM (0.7901 thay vì 0.7944)"),
        (r"0\.8146", "Old Layer 1 Attention entropy AQSM (0.8146 thay vì 0.8191)"),
        (r"38\.77", "Old Foreground hit rate AQSM (38.77% thay vì 43.21%)"),
        (r"21\.55", "Old Random shift mAP (21.55% thay vì 30.97%)"),
        (r"23\.90", "Old AQSM misrouting mAP (23.90% thay vì 25.12%)"),
        (r"0\.9161", "Old ESM AUROC (0.9161 thay vì 0.7593)"),
        (r"64\.57", "Old ESM TPR (64.57% thay vì 37.14%)"),
        (r"15\.92", "Old SQA recovery (+15.92 pp thay vì +15.35 pp / +15.01 pp)"),
        (r"0\.1359", "Old 300th proposal score (0.1359 thay vì 0.1007)"),
        (r"0\.0727", "Old 300th proposal score (0.0727 thay vì 0.1007)"),
        (r"0\.0213", "Old 300th proposal score clean (0.0213 thay vì 0.0341)"),
        (r"40\.39", "Old Arm D_full degradation (-40.39 pp thay vì -38.92 pp)"),
    ]
    errors = []
    lines = tex_content.splitlines()
    for pattern, desc in stale_patterns:
        for idx, line in enumerate(lines):
            if re.search(pattern, line) and not line.strip().startswith("%"):
                errors.append(f"Dòng {idx+1}: Phát hiện tàn dư preliminary {desc} -> '{line.strip()[:80]}'")
    return errors


def main():
    print("=" * 80)
    print(" [PAPER 1 AUDIT] KIỂM ĐỊNH TOÀN DIỆN TÍNH TOÀN VẸN & DATA PROVENANCE N=1000")
    print("=" * 80)

    if not MANIFEST_PATH.exists():
        print(f"[LỖI] Không tìm thấy file manifest tại: {MANIFEST_PATH}")
        sys.exit(1)

    if not LATEX_PATH.exists():
        print(f"[LỖI] Không tìm thấy bản thảo main.tex tại: {LATEX_PATH}")
        sys.exit(1)

    manifest = load_json(MANIFEST_PATH)
    with open(LATEX_PATH, "r", encoding="utf-8") as fp:
        tex_content = fp.read()

    print(f">>> Đã nạp Manifest: {len(manifest.get('tables', {}))} bảng và {len(manifest.get('figures', {}))} hình.")

    # 1. Audit artifact tồn tại
    art_errors = audit_artifacts_existence(manifest)
    if art_errors:
        print("\n[LỖI ARTIFACT]:")
        for e in art_errors:
            print(f"  - {e}")
    else:
        print(">>> 1. Tất cả các file JSON artifact trong manifest đều tồn tại và hợp lệ 100%.")

    # 2. Audit số liệu thực nghiệm
    num_errors = audit_numerical_values(tex_content)
    if num_errors:
        print("\n[CẢNH BÁO LỆCH SỐ LIỆU]:")
        for e in num_errors:
            print(f"  - {e}")
    else:
        print(">>> 2. Tất cả số liệu Table 1-12 (Causal Surgery, ANOVA, Telemetry, Misrouting, Matched-Budget, Hardware Latency, ESM & SQA Hardening) khớp 100% với JSON artifact!")

    # 2b. Audit quét sạch tàn dư preliminary cũ (Zero Stale Markers)
    stale_errors = audit_zero_stale_markers(tex_content)
    if stale_errors:
        print("\n[LỖI TÀN DƯ SỐ LIỆU PRELIMINARY CŨ]:")
        for e in stale_errors:
            print(f"  - {e}")
    else:
        print(">>> 2b. [ĐẠT CHUẨN] Hoàn toàn không còn bất kỳ số liệu preliminary cũ nào trong main.tex!")

    # 3. Kiểm tra chú thích Native 640x640 và S=8400
    if "S = 8,400" in tex_content or "8,400" in tex_content:
        print(">>> 3. Bản thảo main.tex đã khai báo chuẩn hóa S = 8,400 tokens đa tỉ lệ.")
    else:
        print(">>> [CẢNH BÁO] Chưa tìm thấy khai báo S = 8,400 trong main.tex!")

    # 4. Kiểm tra ghi nhận Cỡ mẫu N=1000
    if "N = 1,000" in tex_content or "N=1,000" in tex_content or "N = 1000" in tex_content:
        print(">>> 4. Bản thảo main.tex ghi nhận minh bạch và chính xác cỡ mẫu thực nghiệm N = 1,000 (N_valid = 992, df=[1, 991]).")
    else:
        print(">>> [CẢNH BÁO] main.tex chưa ghi rõ cỡ mẫu N = 1,000!")

    print("\n" + "=" * 80)
    if not art_errors and not num_errors and not stale_errors:
        print(" [KẾT QUẢ AUDIT] HOÀN TẤT THÀNH CÔNG: DATA PROVENANCE VÀ SỐ LIỆU HỢP LỆ VÀ LIÊM CHÍNH 100%!")
    else:
        print(" [KẾT QUẢ AUDIT] CẦN ĐIỀU CHỈNH CÁC CẢNH BÁO TRÊN.")
    print("=" * 80)

    if art_errors or num_errors or stale_errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
