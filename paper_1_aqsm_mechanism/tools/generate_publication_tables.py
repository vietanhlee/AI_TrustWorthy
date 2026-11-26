#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: paper_1_aqsm_mechanism.tools.generate_publication_tables
Mục đích: Tự động kết xuất toàn bộ 13 Bảng biểu (Table 1 đến Table 13) chuẩn xuất bản Elsevier EAAI
từ 100% dữ liệu thực nghiệm thật trong summary_json/:
  - Table 1: Component-wise ablation of the Tripartite AQSM Objective
  - Table 2: COCO val2017 Main Benchmark across Controls & Adversarial Threat Tiers
  - Table 3: 2x2 Factorial Causal Mechanism Isolation (Activation Patching)
  - Table 4: Repeated-Measures ANOVA Attribution Table (SS, df, MS, F, p, partial eta^2)
  - Table 5: Encoder Memory Feature Perturbation across Multi-Scale Tokens (S3, S4, S5)
  - Table 6: 8,400-to-300 Query Selection Dynamics & Spatial Coverage Verification
  - Table 7: Deformable Attention Telemetry (Normalized Entropy vs Spatial FG Hit Rate)
  - Table 8: Controlled Spatial Misrouting & Tautology Refutation (Random vs Matched vs AQSM)
  - Table 9: Matched-Budget Robustness Comparison across Epsilon (eps 4, 8, 16)
  - Table 10: Hardware Execution Latency & Computational Sponge Evaluation (Edge & Server)
  - Table 11: Empirical Defense Evaluation (ESM Stationary vs Adaptive Carlini-Tramer)
  - Table 12: Empirical Validation of Architectural Hardening (Static Query Augmentation - SQA)
  - Table 13: Multi-Seed Optimization Stability and Cross-Variant Architectural Generalization

Đặc tính chuẩn công nghiệp (Production Standards):
  - Tuyệt đối không hardcode, đọc động 100% từ JSON artifacts.
  - Kháng lỗi đa tầng (Defensive Schema Handling): Hỗ trợ đầy đủ các alias khóa (evasion_success_pct,
    conditioned_evasion_success_pct, raw_monitor_bypass_pct) đảm bảo không bao giờ phát sinh KeyError.
  - Cách ly ngoại lệ độc lập từng bảng biểu (Fault Isolation), báo cáo trạng thái chi tiết.
  - Tự động nhận diện đường dẫn linh hoạt trên Linux / Kaggle / Colab / Local.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from paper_1_aqsm_mechanism.experiments.config import (
    OUTPUTS_DIR,
    PAPER_DIR,
    SUMMARY_JSON_DIR,
    get_img_size,
    get_outputs_dir,
    get_pgd_num_steps,
    get_summary_json_dir,
    update_experiment_paths,
)


def _fmt_p(p_val: Any) -> str:
    """Định dạng p-value chuẩn học thuật từ dữ liệu thực tế."""
    if p_val is None:
        return "N/A"
    try:
        p = float(p_val)
    except (TypeError, ValueError):
        return "N/A"
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.4f}"


def get_stat(d: Any, default: float = 0.0) -> float:
    """Trích xuất giá trị mean an toàn từ dict {'mean': ...} hoặc số nguyên / số thực."""
    if isinstance(d, dict):
        return float(d.get("mean", default))
    elif isinstance(d, (int, float)):
        return float(d)
    return default


def get_val(d: Any, path: List[str], default: Any = 0.0) -> Any:
    """Truy xuất sâu an toàn theo danh sách khóa không bao giờ ném KeyError."""
    curr = d
    for k in path:
        if isinstance(curr, dict) and k in curr:
            curr = curr[k]
        else:
            return default
    return curr if curr is not None else default


def require_val(d: Any, path: List[str], label: str = "") -> Any:
    """Truy xuất bắt buộc từ JSON summary; báo lỗi rõ ràng nếu thiếu khóa (không fallback số tĩnh)."""
    curr = d
    for k in path:
        if isinstance(curr, dict) and k in curr and curr[k] is not None:
            curr = curr[k]
        else:
            tag = label or ".".join(path)
            raise KeyError(f"[STRICT PROVENANCE] Thiếu trường dữ liệu bắt buộc '{tag}' trong JSON (path={path}).")
    return curr


def require_stat(d: Any, label: str = "") -> float:
    """Trích xuất giá trị float hoặc dict['mean'] bắt buộc (không dùng default ẩn)."""
    if isinstance(d, dict):
        if "mean" in d and d["mean"] is not None:
            return float(d["mean"])
        raise KeyError(f"[STRICT PROVENANCE] Thiếu khóa 'mean' trong stat dict cho '{label}': {list(d.keys())}")
    if isinstance(d, (int, float)):
        return float(d)
    raise KeyError(f"[STRICT PROVENANCE] Giá trị không hợp lệ cho '{label}': {d!r}")


def require_any_key(d: Dict[str, Any], aliases: List[str], label: str = "") -> Any:
    """Tìm một trong các khóa alias bắt buộc trong dict (không hardcode fallback số)."""
    for k in aliases:
        if isinstance(d, dict) and k in d and d[k] is not None:
            return d[k]
    raise KeyError(f"[STRICT PROVENANCE] Thiếu toàn bộ các alias {aliases} cho '{label}'.")


def load_json(p: Path) -> Dict[str, Any]:
    """Nạp file JSON với thông báo lỗi rõ ràng."""
    if not p.exists():
        raise FileNotFoundError(f"[ERROR] Không tìm thấy file JSON artifact: {p}")
    with open(p, "r", encoding="utf-8") as fp:
        return json.load(fp)


def safe_load_json(candidate_paths: List[Path], default_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Dò tìm và nạp file JSON đầu tiên tồn tại trong danh sách ứng viên."""
    for p in candidate_paths:
        if p and p.exists() and p.is_file():
            try:
                with open(p, "r", encoding="utf-8") as fp:
                    return json.load(fp)
            except Exception:
                continue
    if default_dict is not None:
        return default_dict
    checked = [str(p) for p in candidate_paths]
    raise FileNotFoundError(f"[ERROR] Không tìm thấy file JSON artifact nào trong: {checked}")


def save_text(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fp:
        fp.write(text)


def save_csv_rows(p: Path, headers: List[str], rows: List[List[Any]]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(headers)
        writer.writerows(rows)


# ==============================================================================
# TABLE 1: Component-wise ablation of the Tripartite AQSM Objective
# ==============================================================================
def generate_table1(summary_dir: Path) -> Tuple[str, str, List[str], List[List[Any]]]:
    matched = safe_load_json([
        summary_dir / "matched_budget_results.json",
        summary_dir / "matched_budget_sweep.json",
        OUTPUTS_DIR / "summary_json" / "matched_budget_results.json",
    ])
    dynamics = safe_load_json([
        summary_dir / "query_selection_dynamics.json",
        summary_dir / "encoder_query_dynamics.json",
        OUTPUTS_DIR / "summary_json" / "query_selection_dynamics.json",
    ])

    clean_map = float(require_val(matched, ["clean_baseline", "mAP@50"], "matched.clean_baseline.mAP@50"))
    aqsm_res = require_val(dynamics, ["results", "aqsm"], "dynamics.results.aqsm")
    aqsm_ov = require_stat(require_val(aqsm_res, ["top300_overlap_pct"], "aqsm.top300_overlap_pct"), "aqsm.top300_overlap_pct")
    aqsm_ev = require_stat(require_val(aqsm_res, ["evicted_top15_fg_pct"], "aqsm.evicted_top15_fg_pct"), "aqsm.evicted_top15_fg_pct")
    aqsm_sweep = require_val(matched, ["sweep_results", "aqsm", "eps_16"], "matched.sweep_results.aqsm.eps_16")
    aqsm_map = float(require_val(aqsm_sweep, ["mAP@50"], "aqsm_sweep.mAP@50"))
    aqsm_drop = float(require_val(aqsm_sweep, ["relative_drop_pct"], "aqsm_sweep.relative_drop_pct"))
    n_eval_t1 = int(matched.get("n_images", dynamics.get("n_images", 0)))
    n_str_t1 = f"$N = {n_eval_t1:,}$ images, " if n_eval_t1 > 0 else ""

    ablation_json = safe_load_json([
        summary_dir / "tripartite_ablation_results.json",
        summary_dir / "aqsm_ablation_results.json",
        get_summary_json_dir() / "tripartite_ablation_results.json",
    ], default_dict={})
    variants = ablation_json.get("variants", {})

    headers = ["AQSM Formulation / Loss Variant", "mAP@50 [Rel. Drop]", "Top-300 Overlap", "Evicted Top-15 FG", "Dominant Failure Mode"]
    rows = [
        ["Clean Baseline (No Attack)", f"{clean_map:.2f}% [---]", "100.0%", "0.0%", "Nominal inference"],
    ]
    tex_mid_rows = []
    ablation_specs = [
        ("L_rank", r"Margin-Ranking Loss only ($\mathcal{L}_{\mathrm{rank}}$)", "Margin-Ranking Loss only (L_rank)", "Partial proposal reordering"),
        ("L_suppr", r"Foreground Suppression Loss only ($\mathcal{L}_{\mathrm{suppr}}$)", "Foreground Suppression Loss only (L_suppr)", "Foreground confidence suppression"),
        ("L_elev", r"Distal Background Elevation only ($\mathcal{L}_{\mathrm{elev}}$)", "Distal Background Elevation only (L_elev)", "Background cutoff elevation"),
        ("L_suppr+L_rank", r"$\mathcal{L}_{\mathrm{suppr}} + \mathcal{L}_{\mathrm{rank}}$", "L_suppr + L_rank", "Compound foreground suppression"),
        ("L_suppr+L_elev", r"$\mathcal{L}_{\mathrm{suppr}} + \mathcal{L}_{\mathrm{elev}}$", "L_suppr + L_elev", "Suppression-elevation dual pressure"),
        ("L_rank+L_elev", r"$\mathcal{L}_{\mathrm{rank}} + \mathcal{L}_{\mathrm{elev}}$", "L_rank + L_elev", "Dual-boundary selection compression"),
    ]
    for vkey, tex_lbl, md_lbl, mode_note in ablation_specs:
        v = variants.get(vkey)
        if isinstance(v, dict) and "mAP@50" in v:
            vm = float(v["mAP@50"])
            vd = float(v.get("relative_drop_pct", (clean_map - vm) / max(clean_map, 1e-6) * 100.0))
            vo = float(v.get("top300_overlap_pct", 0.0))
            ve = float(v.get("evicted_top15_fg_pct", 0.0))
            rows.append([md_lbl, f"{vm:.2f}% [-{vd:.1f}%]", f"{vo:.1f}%", f"{ve:.1f}%", str(v.get("failure_mode", mode_note))])
            tex_mid_rows.append(f"{tex_lbl} & {vm:.2f}\\% [$-${vd:.1f}\\%] & {vo:.1f}\\% & {ve:.1f}\\% & {v.get('failure_mode', mode_note)} \\\\")

    rows.append(["Full Tripartite AQSM", f"{aqsm_map:.2f}% [-{aqsm_drop:.1f}%]", f"{aqsm_ov:.1f}%", f"{aqsm_ev:.1f}%", "Compound Eviction & Misrouting"])
    tex_mid_block = ("\n\\midrule\n" + "\n".join(tex_mid_rows)) if tex_mid_rows else ""

    eff_res_t1 = int(get_img_size())
    tex = r"""\begin{table}[!htbp]
\centering
\caption{Component-wise ablation of the Tripartite AQSM Objective on COCO val2017 (""" + n_str_t1 + rf"""${eff_res_t1} \times {eff_res_t1}$). Evaluates mAP@50, Top-300 proposal overlap with clean baseline, and Ground-Truth foreground eviction rate.""" + r"""}
\label{tab:tab1_ablation}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccc}
\toprule
\textbf{AQSM Formulation / Loss Variant} & \textbf{mAP@50 [Rel. Drop]} & \textbf{Top-300 Overlap} & \textbf{Evicted Top-15 FG} & \textbf{Dominant Failure Mode} \\
\midrule
Clean Baseline (No Attack) & """ + f"{clean_map:.2f}\\% [---]" + r""" & 100.0\% & 0.0\% & Nominal inference \\""" + tex_mid_block + r"""
\midrule
\textbf{Full Tripartite AQSM} & \textbf{""" + f"{aqsm_map:.2f}\\% [$-${aqsm_drop:.1f}\\%]" + r"""} & \textbf{""" + f"{aqsm_ov:.1f}\\%" + r"""} & \textbf{""" + f"{aqsm_ev:.1f}\\%" + r"""} & \textbf{Compound Eviction \& Misrouting} \\
\bottomrule
\end{tabular}%
}
\end{table}"""

    md = "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] + ["| " + " | ".join(r) + " |" for r in rows])
    return tex, md, headers, rows


# ==============================================================================
# TABLE 2: Main Benchmark across Controls & Adversarial Threat Tiers
# ==============================================================================
def generate_table2(summary_dir: Path) -> Tuple[str, str, List[str], List[List[Any]]]:
    matched = safe_load_json([
        summary_dir / "matched_budget_results.json",
        summary_dir / "matched_budget_sweep.json",
        OUTPUTS_DIR / "summary_json" / "matched_budget_results.json",
    ])
    clean = require_val(matched, ["clean_baseline"], "matched.clean_baseline")
    clean_m50 = float(require_val(clean, ["mAP@50"], "clean.mAP@50"))
    clean_m95 = float(require_val(clean, ["mAP@50:95"], "clean.mAP@50:95"))

    sw = require_val(matched, ["sweep_results"], "matched.sweep_results")
    aq_16 = require_val(sw, ["aqsm", "eps_16"], "sw.aqsm.eps_16")
    van_16 = require_val(sw, ["vanishing", "eps_16"], "sw.vanishing.eps_16")
    pgd_16 = require_val(sw, ["taskpgd", "eps_16"], "sw.taskpgd.eps_16")

    aq_m50 = float(require_val(aq_16, ["mAP@50"], "aq_16.mAP@50"))
    aq_m50_drop = float(require_val(aq_16, ["relative_drop_pct"], "aq_16.relative_drop_pct"))
    aq_m95 = float(require_val(aq_16, ["mAP@50:95"], "aq_16.mAP@50:95"))
    aq_m95_drop = (clean_m95 - aq_m95) / max(clean_m95, 1e-6) * 100.0 if clean_m95 > 0.0 else 0.0

    van_m50 = float(require_val(van_16, ["mAP@50"], "van_16.mAP@50"))
    van_m50_drop = float(require_val(van_16, ["relative_drop_pct"], "van_16.relative_drop_pct"))
    van_m95 = float(require_val(van_16, ["mAP@50:95"], "van_16.mAP@50:95"))
    van_m95_drop = (clean_m95 - van_m95) / max(clean_m95, 1e-6) * 100.0 if clean_m95 > 0.0 else 0.0

    pgd_m50 = float(require_val(pgd_16, ["mAP@50"], "pgd_16.mAP@50"))
    pgd_m50_drop = float(require_val(pgd_16, ["relative_drop_pct"], "pgd_16.relative_drop_pct"))
    pgd_m95 = float(require_val(pgd_16, ["mAP@50:95"], "pgd_16.mAP@50:95"))
    pgd_m95_drop = (clean_m95 - pgd_m95) / max(clean_m95, 1e-6) * 100.0 if clean_m95 > 0.0 else 0.0

    n_eval_t2 = int(matched.get("n_images", 0))
    model_t2 = str(matched.get("model", "RT-DETR"))
    esm_ref = safe_load_json([summary_dir / "defense_esm_evaluation.json", get_summary_json_dir() / "defense_esm_evaluation.json"], default_dict={})
    box_mon = esm_ref.get("box_count_monitor", {})
    clean_box_s = box_mon.get("clean_box_stats", {})
    box_atks = box_mon.get("attacks", {})

    def _fmt_boxes(stats_dict: Dict[str, Any]) -> str:
        if isinstance(stats_dict, dict) and "mean" in stats_dict:
            return f"{float(stats_dict['mean']):.2f} ± {float(stats_dict.get('std', 0.0)):.2f}"
        return "---"

    def _fmt_boxes_tex(stats_dict: Dict[str, Any]) -> str:
        if isinstance(stats_dict, dict) and "mean" in stats_dict:
            return f"${float(stats_dict['mean']):.2f} \\pm {float(stats_dict.get('std', 0.0)):.2f}$"
        return "---"

    def _fmt_asr(blk: Dict[str, Any], field: str, default: str = "---") -> str:
        if isinstance(blk, dict) and field in blk and blk[field] is not None:
            v = blk[field]
            if isinstance(v, dict) and "rate" in v:
                ci = v.get("ci95")
                if ci and len(ci) == 2:
                    return f"{float(v['rate']):.1f}% [{float(ci[0]):.1f}, {float(ci[1]):.1f}]"
                return f"{float(v['rate']):.1f}%"
            return f"{float(v):.1f}%"
        return default

    def _fmt_asr_tex(blk: Dict[str, Any], field: str, default: str = "---") -> str:
        return _fmt_asr(blk, field, default).replace("%", "\\%")

    clean_boxes_md = _fmt_boxes(clean_box_s)
    aq_boxes_md = _fmt_boxes(box_atks.get("aqsm", {}).get("box_stats", {}))
    van_boxes_md = _fmt_boxes(box_atks.get("vanishing", {}).get("box_stats", {}))
    pgd_boxes_md = _fmt_boxes(box_atks.get("taskpgd", {}).get("box_stats", {}))

    clean_boxes_tex = _fmt_boxes_tex(clean_box_s)
    aq_boxes_tex = _fmt_boxes_tex(box_atks.get("aqsm", {}).get("box_stats", {}))
    van_boxes_tex = _fmt_boxes_tex(box_atks.get("vanishing", {}).get("box_stats", {}))
    pgd_boxes_tex = _fmt_boxes_tex(box_atks.get("taskpgd", {}).get("box_stats", {}))

    headers = ["Threat Tier & Condition", "mAP@50 [Rel. Drop]", "mAP@50:95 [Rel.]", "Active Boxes", "ASR_obj [CI 95%]", "ASR_silence [CI 95%]", "ASR_starve [CI 95%]"]
    rows = [
        ["Nominal Clean Baseline", f"{clean_m50:.2f}% [Ref.]", f"{clean_m95:.2f}% [Ref.]", clean_boxes_md, "0.0% [Reference]", "0.0% [Reference]", _fmt_asr(clean, "asr_starve")],
        ["Tier C: AQSM (eps = 16/255)", f"{aq_m50:.2f}% [-{aq_m50_drop:.1f}%]", f"{aq_m95:.2f}% [-{aq_m95_drop:.1f}%]", aq_boxes_md, _fmt_asr(aq_16, "asr_obj"), _fmt_asr(aq_16, "asr_silence"), _fmt_asr(aq_16, "asr_starve")],
        ["Tier B: Query Vanishing (eps = 16/255)", f"{van_m50:.2f}% [-{van_m50_drop:.1f}%]", f"{van_m95:.2f}% [-{van_m95_drop:.1f}%]", van_boxes_md, _fmt_asr(van_16, "asr_obj"), _fmt_asr(van_16, "asr_silence"), _fmt_asr(van_16, "asr_starve")],
        ["Tier B: Standard Task PGD (eps = 16/255)", f"{pgd_m50:.2f}% [-{pgd_m50_drop:.1f}%]", f"{pgd_m95:.2f}% [-{pgd_m95_drop:.1f}%]", pgd_boxes_md, _fmt_asr(pgd_16, "asr_obj"), _fmt_asr(pgd_16, "asr_silence"), _fmt_asr(pgd_16, "asr_starve")],
    ]
    for ctrl_key, ctrl_label in [("gaussian_noise", "Gaussian Noise (sigma = 16/255)"), ("uniform_noise", "Uniform Noise (± 16/255)")]:
        cb = matched.get("controls", {}).get(ctrl_key) or sw.get(ctrl_key, {}).get("eps_16")
        if isinstance(cb, dict) and "mAP@50" in cb:
            cm50 = float(cb["mAP@50"])
            cm95 = float(cb.get("mAP@50:95", 0.0))
            cd50 = float(cb.get("relative_drop_pct", (clean_m50 - cm50) / max(clean_m50, 1e-6) * 100.0))
            cd95 = (clean_m95 - cm95) / max(clean_m95, 1e-6) * 100.0 if clean_m95 > 0.0 else 0.0
            rows.insert(1, [ctrl_label, f"{cm50:.2f}% [-{cd50:.1f}%]", f"{cm95:.2f}% [-{cd95:.1f}%]", _fmt_boxes(cb.get("box_stats", {})), _fmt_asr(cb, "asr_obj"), _fmt_asr(cb, "asr_silence"), _fmt_asr(cb, "asr_starve")])

    n_str_t2 = f"$N = {n_eval_t2:,}$ images, " if n_eval_t2 > 0 else ""
    eff_res_t2 = int(get_img_size())
    tex_noise_rows_t2: List[str] = []
    for ctrl_key, ctrl_tex_label in [("gaussian_noise", "Gaussian Noise ($\\sigma = 16/255$)"), ("uniform_noise", "Uniform Noise ($\\pm 16/255$)")]:
        cb = matched.get("controls", {}).get(ctrl_key) or sw.get(ctrl_key, {}).get("eps_16")
        if isinstance(cb, dict) and "mAP@50" in cb:
            cm50 = float(cb["mAP@50"])
            cm95 = float(cb.get("mAP@50:95", 0.0))
            cd50 = float(cb.get("relative_drop_pct", (clean_m50 - cm50) / max(clean_m50, 1e-6) * 100.0))
            cd95 = (clean_m95 - cm95) / max(clean_m95, 1e-6) * 100.0 if clean_m95 > 0.0 else 0.0
            cbox_tex = _fmt_boxes_tex(cb.get("box_stats", {}))
            tex_noise_rows_t2.append(
                f"{ctrl_tex_label} & {cm50:.2f}\\% [$-${cd50:.1f}\\%] & {cm95:.2f}\\% [$-${cd95:.1f}\\%] & {cbox_tex} & {_fmt_asr_tex(cb, 'asr_obj')} & {_fmt_asr_tex(cb, 'asr_silence')} & {_fmt_asr_tex(cb, 'asr_starve')} \\\\"
            )

    tex_rows_t2 = [
        f"Nominal Clean Baseline & {clean_m50:.2f}\\% [Ref.] & {clean_m95:.2f}\\% [Ref.] & {clean_boxes_tex} & 0.0\\% [Reference] & 0.0\\% [Reference] & {_fmt_asr_tex(clean, 'asr_starve')} \\\\",
        *tex_noise_rows_t2,
        "\\midrule",
        f"Tier C: AQSM ($\\epsilon = 16/255$) & \\textbf{{{aq_m50:.2f}\\% [$-${aq_m50_drop:.1f}\\%]}} & \\textbf{{{aq_m95:.2f}\\% [$-${aq_m95_drop:.1f}\\%]}} & {aq_boxes_tex} & {_fmt_asr_tex(aq_16, 'asr_obj')} & {_fmt_asr_tex(aq_16, 'asr_silence')} & {_fmt_asr_tex(aq_16, 'asr_starve')} \\\\",
        f"Tier B: Query Vanishing ($\\epsilon = 16/255$) & {van_m50:.2f}\\% [$-${van_m50_drop:.1f}\\%] & {van_m95:.2f}\\% [$-${van_m95_drop:.1f}\\%] & {van_boxes_tex} & {_fmt_asr_tex(van_16, 'asr_obj')} & {_fmt_asr_tex(van_16, 'asr_silence')} & {_fmt_asr_tex(van_16, 'asr_starve')} \\\\",
        f"Tier B: Standard Task PGD ($\\epsilon = 16/255$) & {pgd_m50:.2f}\\% [$-${pgd_m50_drop:.1f}\\%] & {pgd_m95:.2f}\\% [$-${pgd_m95_drop:.1f}\\%] & {pgd_boxes_tex} & {_fmt_asr_tex(pgd_16, 'asr_obj')} & {_fmt_asr_tex(pgd_16, 'asr_silence')} & {_fmt_asr_tex(pgd_16, 'asr_starve')} \\\\",
    ]

    tex = r"""\begin{table}[!htbp]
\centering
\caption{COCO val2017 object detection performance (mAP@50 and mAP@50:95, """ + n_str_t2 + r"""80 semantic categories) on \texttt{""" + model_t2.replace("_", r"\_") + r"""} at native $""" + f"{eff_res_t2} \\times {eff_res_t2}" + r"""$ resolution across controls and adversarial threat tiers.}
\label{tab:tab2_main_benchmark}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccc}
\toprule
\textbf{Threat Tier \& Condition} & \textbf{mAP@50 [Rel. Drop]} & \textbf{mAP@50:95 [Rel.]} & \textbf{Active Boxes} & \textbf{$\text{ASR}_{\text{obj}}$ [CI 95\%]} & \textbf{$\text{ASR}_{\text{silence}}$ [CI 95\%]} & \textbf{$\text{ASR}_{\text{starve}}$ [CI 95\%]} \\
\midrule
""" + "\n".join(tex_rows_t2) + r"""
\bottomrule
\end{tabular}%
}
\end{table}"""

    md = "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] + ["| " + " | ".join(r) + " |" for r in rows])
    return tex, md, headers, rows


# ==============================================================================
# TABLE 3: 2x2 Factorial Causal Mechanism Isolation (Activation Patching)
# ==============================================================================
def generate_table3(summary_dir: Path) -> Tuple[str, str, List[str], List[List[Any]]]:
    c_aqsm = safe_load_json([summary_dir / "causal_swap_results_aqsm.json", OUTPUTS_DIR / "summary_json" / "causal_swap_results_aqsm.json"])
    c_van = safe_load_json([summary_dir / "causal_swap_results_vanishing.json", OUTPUTS_DIR / "summary_json" / "causal_swap_results_vanishing.json"])
    c_pgd = safe_load_json([summary_dir / "causal_swap_results_taskpgd.json", OUTPUTS_DIR / "summary_json" / "causal_swap_results_taskpgd.json"])

    p_aqsm = require_val(c_aqsm, ["point_maps"], "aqsm.point_maps")
    p_van = require_val(c_van, ["point_maps"], "van.point_maps")
    p_pgd = require_val(c_pgd, ["point_maps"], "pgd.point_maps")

    clean_map = float(require_val(p_aqsm, ["Arm_A", "mAP@50"], "aqsm.Arm_A.mAP@50"))
    b_aqsm = float(require_val(p_aqsm, ["Arm_B", "mAP@50"], "aqsm.Arm_B.mAP@50"))
    b_van = float(require_val(p_van, ["Arm_B", "mAP@50"], "van.Arm_B.mAP@50"))
    b_pgd = float(require_val(p_pgd, ["Arm_B", "mAP@50"], "pgd.Arm_B.mAP@50"))

    cfull_aqsm = float(require_val(p_aqsm, ["Arm_C_full", "mAP@50"], "aqsm.Arm_C_full.mAP@50"))
    cfull_van = float(require_val(p_van, ["Arm_C_full", "mAP@50"], "van.Arm_C_full.mAP@50"))
    cfull_pgd = float(require_val(p_pgd, ["Arm_C_full", "mAP@50"], "pgd.Arm_C_full.mAP@50"))

    cidx_aqsm = float(require_val(p_aqsm, ["Arm_C_index", "mAP@50"], "aqsm.Arm_C_index.mAP@50"))
    cidx_van = float(require_val(p_van, ["Arm_C_index", "mAP@50"], "van.Arm_C_index.mAP@50"))
    cidx_pgd = float(require_val(p_pgd, ["Arm_C_index", "mAP@50"], "pgd.Arm_C_index.mAP@50"))

    dfull_aqsm = float(require_val(p_aqsm, ["Arm_D_full", "mAP@50"], "aqsm.Arm_D_full.mAP@50"))
    dfull_van = float(require_val(p_van, ["Arm_D_full", "mAP@50"], "van.Arm_D_full.mAP@50"))
    dfull_pgd = float(require_val(p_pgd, ["Arm_D_full", "mAP@50"], "pgd.Arm_D_full.mAP@50"))

    didx_aqsm = float(require_val(p_aqsm, ["Arm_D_index", "mAP@50"], "aqsm.Arm_D_index.mAP@50"))
    didx_van = float(require_val(p_van, ["Arm_D_index", "mAP@50"], "van.Arm_D_index.mAP@50"))
    didx_pgd = float(require_val(p_pgd, ["Arm_D_index", "mAP@50"], "pgd.Arm_D_index.mAP@50"))

    # ANOVA AQSM
    an_aqsm = require_val(c_aqsm, ["anova_repeated_measures_full"], "aqsm.anova_repeated_measures_full")
    me_m_aqsm = float(require_val(an_aqsm, ["marginal_effects", "ME_M"], "aqsm.ME_M"))
    me_q_aqsm = float(require_val(an_aqsm, ["marginal_effects", "ME_Q"], "aqsm.ME_Q"))
    inter_aqsm = float(require_val(an_aqsm, ["marginal_effects", "Interaction_MxQ"], "aqsm.Interaction_MxQ"))
    ci_m_aqsm = require_val(an_aqsm, ["marginal_effects", "ME_M_ci95"], "aqsm.ME_M_ci95")
    ci_q_aqsm = require_val(an_aqsm, ["marginal_effects", "ME_Q_ci95"], "aqsm.ME_Q_ci95")
    ci_inter_aqsm = require_val(an_aqsm, ["marginal_effects", "Interaction_ci95"], "aqsm.Interaction_ci95")
    eta_m_aqsm = float(require_val(an_aqsm, ["attribution_shares", "partial_eta_squared", "Memory_M"], "aqsm.eta_m"))
    eta_q_aqsm = float(require_val(an_aqsm, ["attribution_shares", "partial_eta_squared", "Query_Q"], "aqsm.eta_q"))
    eta_inter_aqsm = float(require_val(an_aqsm, ["attribution_shares", "partial_eta_squared", "Interaction_MxQ"], "aqsm.eta_inter"))
    f_m_aqsm = float(require_val(an_aqsm, ["anova_table", "Memory_M", "F"], "aqsm.F_m"))
    f_q_aqsm = float(require_val(an_aqsm, ["anova_table", "Query_Q", "F"], "aqsm.F_q"))
    f_inter_aqsm = float(require_val(an_aqsm, ["anova_table", "Interaction_MxQ", "F"], "aqsm.F_inter"))
    p_m_aqsm = _fmt_p(an_aqsm["anova_table"]["Memory_M"].get("p_value"))
    p_q_aqsm = _fmt_p(an_aqsm["anova_table"]["Query_Q"].get("p_value"))
    p_inter_aqsm = _fmt_p(an_aqsm["anova_table"]["Interaction_MxQ"].get("p_value"))

    # ANOVA Vanishing
    an_van = require_val(c_van, ["anova_repeated_measures_full"], "van.anova_repeated_measures_full")
    me_m_van = float(require_val(an_van, ["marginal_effects", "ME_M"], "van.ME_M"))
    me_q_van = float(require_val(an_van, ["marginal_effects", "ME_Q"], "van.ME_Q"))
    inter_van = float(require_val(an_van, ["marginal_effects", "Interaction_MxQ"], "van.Interaction_MxQ"))
    ci_m_van = require_val(an_van, ["marginal_effects", "ME_M_ci95"], "van.ME_M_ci95")
    ci_q_van = require_val(an_van, ["marginal_effects", "ME_Q_ci95"], "van.ME_Q_ci95")
    ci_inter_van = require_val(an_van, ["marginal_effects", "Interaction_ci95"], "van.Interaction_ci95")
    eta_m_van = float(require_val(an_van, ["attribution_shares", "partial_eta_squared", "Memory_M"], "van.eta_m"))
    eta_q_van = float(require_val(an_van, ["attribution_shares", "partial_eta_squared", "Query_Q"], "van.eta_q"))
    eta_inter_van = float(require_val(an_van, ["attribution_shares", "partial_eta_squared", "Interaction_MxQ"], "van.eta_inter"))
    f_m_van = float(require_val(an_van, ["anova_table", "Memory_M", "F"], "van.F_m"))
    f_q_van = float(require_val(an_van, ["anova_table", "Query_Q", "F"], "van.F_q"))
    f_inter_van = float(require_val(an_van, ["anova_table", "Interaction_MxQ", "F"], "van.F_inter"))
    p_m_van = _fmt_p(an_van["anova_table"]["Memory_M"].get("p_value"))
    p_q_van = _fmt_p(an_van["anova_table"]["Query_Q"].get("p_value"))
    p_inter_van = _fmt_p(an_van["anova_table"]["Interaction_MxQ"].get("p_value"))

    # ANOVA TaskPGD
    an_pgd = require_val(c_pgd, ["anova_repeated_measures_full"], "pgd.anova_repeated_measures_full")
    me_m_pgd = float(require_val(an_pgd, ["marginal_effects", "ME_M"], "pgd.ME_M"))
    me_q_pgd = float(require_val(an_pgd, ["marginal_effects", "ME_Q"], "pgd.ME_Q"))
    inter_pgd = float(require_val(an_pgd, ["marginal_effects", "Interaction_MxQ"], "pgd.Interaction_MxQ"))
    ci_m_pgd = require_val(an_pgd, ["marginal_effects", "ME_M_ci95"], "pgd.ME_M_ci95")
    ci_q_pgd = require_val(an_pgd, ["marginal_effects", "ME_Q_ci95"], "pgd.ME_Q_ci95")
    ci_inter_pgd = require_val(an_pgd, ["marginal_effects", "Interaction_ci95"], "pgd.Interaction_ci95")
    eta_m_pgd = float(require_val(an_pgd, ["attribution_shares", "partial_eta_squared", "Memory_M"], "pgd.eta_m"))
    eta_q_pgd = float(require_val(an_pgd, ["attribution_shares", "partial_eta_squared", "Query_Q"], "pgd.eta_q"))
    eta_inter_pgd = float(require_val(an_pgd, ["attribution_shares", "partial_eta_squared", "Interaction_MxQ"], "pgd.eta_inter"))
    f_m_pgd = float(require_val(an_pgd, ["anova_table", "Memory_M", "F"], "pgd.F_m"))
    f_q_pgd = float(require_val(an_pgd, ["anova_table", "Query_Q", "F"], "pgd.F_q"))
    f_inter_pgd = float(require_val(an_pgd, ["anova_table", "Interaction_MxQ", "F"], "pgd.F_inter"))
    p_m_pgd = _fmt_p(an_pgd["anova_table"]["Memory_M"].get("p_value"))
    p_q_pgd = _fmt_p(an_pgd["anova_table"]["Query_Q"].get("p_value"))
    p_inter_pgd = _fmt_p(an_pgd["anova_table"]["Interaction_MxQ"].get("p_value"))

    n_eval_t3 = int(c_aqsm.get("n_evaluated_images", 0))
    n_valid_t3 = int(c_aqsm.get("n_valid_intra_subject_samples", an_aqsm.get("N", 0)))
    df2_t3 = max(1, n_valid_t3 - 1)

    _d_cm = max(clean_map, 1e-6)
    headers = ["Intervention Arm / Factor", "AQSM mAP@50", "PGD Vanishing mAP@50", "Standard Task PGD mAP@50"]
    rows = [
        ["Arm A: Clean Baseline", f"{clean_map:.2f}% (100.0%)", f"{clean_map:.2f}% (100.0%)", f"{clean_map:.2f}% (100.0%)"],
        ["Arm B: Full Attack", f"{b_aqsm:.2f}% ({b_aqsm/_d_cm*100:.1f}%) [-{clean_map-b_aqsm:.2f} pp]", f"{b_van:.2f}% ({b_van/_d_cm*100:.1f}%) [-{clean_map-b_van:.2f} pp]", f"{b_pgd:.2f}% ({b_pgd/_d_cm*100:.1f}%) [-{clean_map-b_pgd:.2f} pp]"],
        ["Arm C_full: Adv Mem + Clean Prior", f"{cfull_aqsm:.2f}% ({cfull_aqsm/_d_cm*100:.1f}%) [-{clean_map-cfull_aqsm:.2f} pp]", f"{cfull_van:.2f}% ({cfull_van/_d_cm*100:.1f}%) [-{clean_map-cfull_van:.2f} pp]", f"{cfull_pgd:.2f}% ({cfull_pgd/_d_cm*100:.1f}%) [-{clean_map-cfull_pgd:.2f} pp]"],
        ["Arm C_index: Adv Mem + Clean Idx", f"{cidx_aqsm:.2f}% ({cidx_aqsm/_d_cm*100:.1f}%) [-{clean_map-cidx_aqsm:.2f} pp]", f"{cidx_van:.2f}% ({cidx_van/_d_cm*100:.1f}%) [-{clean_map-cidx_van:.2f} pp]", f"{cidx_pgd:.2f}% ({cidx_pgd/_d_cm*100:.1f}%) [-{clean_map-cidx_pgd:.2f} pp]"],
        ["Arm D_full: Clean Mem + Adv Prior", f"{dfull_aqsm:.2f}% ({dfull_aqsm/_d_cm*100:.1f}%) [-{clean_map-dfull_aqsm:.2f} pp]", f"{dfull_van:.2f}% ({dfull_van/_d_cm*100:.1f}%) [-{clean_map-dfull_van:.2f} pp]", f"{dfull_pgd:.2f}% ({dfull_pgd/_d_cm*100:.1f}%) [-{clean_map-dfull_pgd:.2f} pp]"],
        ["Arm D_index: Clean Mem + Adv Idx", f"{didx_aqsm:.2f}% ({didx_aqsm/_d_cm*100:.1f}%) [-{clean_map-didx_aqsm:.2f} pp]", f"{didx_van:.2f}% ({didx_van/_d_cm*100:.1f}%) [-{clean_map-didx_van:.2f} pp]", f"{didx_pgd:.2f}% ({didx_pgd/_d_cm*100:.1f}%) [-{clean_map-didx_pgd:.2f} pp]"],
        ["ME_M (Memory Main Effect)", f"{me_m_aqsm:.2f} pp (eta_p^2 = {eta_m_aqsm:.3f})", f"{me_m_van:.2f} pp (eta_p^2 = {eta_m_van:.3f})", f"{me_m_pgd:.2f} pp (eta_p^2 = {eta_m_pgd:.3f})"],
        ["ME_Q (Query Main Effect)", f"{me_q_aqsm:.2f} pp (eta_p^2 = {eta_q_aqsm:.3f})", f"{me_q_van:.2f} pp (eta_p^2 = {eta_q_van:.3f})", f"{me_q_pgd:.2f} pp (eta_p^2 = {eta_q_pgd:.3f})"],
        ["Interaction M x Q", f"{inter_aqsm:+.2f} pp", f"{inter_van:+.2f} pp", f"{inter_pgd:+.2f} pp"],
    ]

    tex = r"""\begin{table}[!htbp]
\centering
\caption{Factorial interventional failure decomposition on the COCO val2017 benchmark at native $640 \times 640$ resolution (evaluation cohort $N = """ + f"{n_eval_t3:,}" + r"""$ images, valid intra-subject $N = """ + f"{n_valid_t3:,}" + r"""$ with $\text{df} = [1, """ + f"{df2_t3}" + r"""]$; verifiable via \texttt{outputs/data\_provenance\_manifest.json}). Reports COCO-style mAP@50 (\%), relative retention (\% of Arm A """ + f"{clean_map:.2f}\\%" + r"""), and $\Delta$ vs.\ Arm A/B.}
\label{tab:tab3_causal_surgery}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lccc}
\toprule
\textbf{Intervention Arm / Factor} & \textbf{AQSM} & \textbf{PGD Vanishing} & \textbf{Standard Task PGD} \\
& mAP@50 (Rel.) / $\Delta$ [95\% CI] & mAP@50 (Rel.) / $\Delta$ [95\% CI] & mAP@50 (Rel.) / $\Delta$ [95\% CI] \\
\midrule
Arm A: Clean Baseline ($M_{\mathrm{clean}} + Q_{\mathrm{clean}}[\mathcal{I}_{\mathrm{clean}}]$) & """ + f"{clean_map:.2f}\\% (100.0\\%) [Ref.]" + r""" & """ + f"{clean_map:.2f}\\% (100.0\\%) [Ref.]" + r""" & """ + f"{clean_map:.2f}\\% (100.0\\%) [Ref.]" + r""" \\
Arm B: Full Attack ($M_{\mathrm{adv}} + Q_{\mathrm{adv}}[\mathcal{I}_{\mathrm{adv}}]$) & """ + f"{b_aqsm:.2f}\\% ({b_aqsm/_d_cm*100:.1f}\\%) [$-${clean_map-b_aqsm:.2f} pp]" + r""" & """ + f"{b_van:.2f}\\% ({b_van/_d_cm*100:.1f}\\%) [$-${clean_map-b_van:.2f} pp]" + r""" & """ + f"{b_pgd:.2f}\\% ({b_pgd/_d_cm*100:.1f}\\%) [$-${clean_map-b_pgd:.2f} pp]" + r""" \\
Arm $\text{C}_{\text{full}}$: Adv Mem + Clean Prior ($M_{\mathrm{adv}} + Q_{\mathrm{clean}}$) & """ + f"{cfull_aqsm:.2f}\\% ({cfull_aqsm/_d_cm*100:.1f}\\%) [$-${clean_map-cfull_aqsm:.2f} pp]" + r""" & """ + f"{cfull_van:.2f}\\% ({cfull_van/_d_cm*100:.1f}\\%) [$-${clean_map-cfull_van:.2f} pp]" + r""" & """ + f"{cfull_pgd:.2f}\\% ({cfull_pgd/_d_cm*100:.1f}\\%) [$-${clean_map-cfull_pgd:.2f} pp]" + r""" \\
Arm $\text{C}_{\text{index}}$: Adv Mem + Adv Feat at Clean Idx ($M_{\mathrm{adv}} + Q_{\mathrm{adv}}[\mathcal{I}_{\mathrm{clean}}]$) & """ + f"{cidx_aqsm:.2f}\\% ({cidx_aqsm/_d_cm*100:.1f}\\%) [$-${clean_map-cidx_aqsm:.2f} pp]" + r""" & """ + f"{cidx_van:.2f}\\% ({cidx_van/_d_cm*100:.1f}\\%) [$-${clean_map-cidx_van:.2f} pp]" + r""" & """ + f"{cidx_pgd:.2f}\\% ({cidx_pgd/_d_cm*100:.1f}\\%) [$-${clean_map-cidx_pgd:.2f} pp]" + r""" \\
Arm $\text{D}_{\text{full}}$: Clean Mem + Adv Prior ($M_{\mathrm{clean}} + Q_{\mathrm{adv}}$) & """ + f"{dfull_aqsm:.2f}\\% ({dfull_aqsm/_d_cm*100:.1f}\\%) [$-${clean_map-dfull_aqsm:.2f} pp]" + r""" & """ + f"{dfull_van:.2f}\\% ({dfull_van/_d_cm*100:.1f}\\%) [$-${clean_map-dfull_van:.2f} pp]" + r""" & """ + f"{dfull_pgd:.2f}\\% ({dfull_pgd/_d_cm*100:.1f}\\%) [$-${clean_map-dfull_pgd:.2f} pp]" + r""" \\
Arm $\text{D}_{\text{index}}$: Clean Mem + Adv Idx ($M_{\mathrm{clean}} + Q_{\mathrm{clean}}[\mathcal{I}_{\mathrm{adv}}]$) & """ + f"{didx_aqsm:.2f}\\% ({didx_aqsm/_d_cm*100:.1f}\\%) [$-${clean_map-didx_aqsm:.2f} pp]" + r""" & """ + f"{didx_van:.2f}\\% ({didx_van/_d_cm*100:.1f}\\%) [$-${clean_map-didx_van:.2f} pp]" + r""" & """ + f"{didx_pgd:.2f}\\% ({didx_pgd/_d_cm*100:.1f}\\%) [$-${clean_map-didx_pgd:.2f} pp]" + r""" \\
\midrule
\textbf{Repeated-Measures ANOVA} & & & \\
Main Effect Memory ($\text{ME}_M$) & """ + f"${me_m_aqsm:.2f}$ pp [${ci_m_aqsm[0]:.2f}, {ci_m_aqsm[1]:.2f}$]" + r""" & """ + f"${me_m_van:.2f}$ pp [${ci_m_van[0]:.2f}, {ci_m_van[1]:.2f}$]" + r""" & """ + f"${me_m_pgd:.2f}$ pp [${ci_m_pgd[0]:.2f}, {ci_m_pgd[1]:.2f}$]" + r""" \\
$\eta_p^2$ (F-statistic, $p$-value) & $\eta_p^2 = """ + f"{eta_m_aqsm:.3f}" + r"""$ ($F=""" + f"{f_m_aqsm:.1f}" + r""", """ + p_m_aqsm + r"""$) & $\eta_p^2 = """ + f"{eta_m_van:.3f}" + r"""$ ($F=""" + f"{f_m_van:.1f}" + r""", """ + p_m_van + r"""$) & $\eta_p^2 = """ + f"{eta_m_pgd:.3f}" + r"""$ ($F=""" + f"{f_m_pgd:.1f}" + r""", """ + p_m_pgd + r"""$) \\
Main Effect Query ($\text{ME}_Q$) & """ + f"${me_q_aqsm:.2f}$ pp [${ci_q_aqsm[0]:.2f}, {ci_q_aqsm[1]:.2f}$]" + r""" & """ + f"${me_q_van:.2f}$ pp [${ci_q_van[0]:.2f}, {ci_q_van[1]:.2f}$]" + r""" & """ + f"${me_q_pgd:.2f}$ pp [${ci_q_pgd[0]:.2f}, {ci_q_pgd[1]:.2f}$]" + r""" \\
$\eta_p^2$ (F-statistic, $p$-value) & $\eta_p^2 = """ + f"{eta_q_aqsm:.3f}" + r"""$ ($F=""" + f"{f_q_aqsm:.1f}" + r""", """ + p_q_aqsm + r"""$) & $\eta_p^2 = """ + f"{eta_q_van:.3f}" + r"""$ ($F=""" + f"{f_q_van:.1f}" + r""", """ + p_q_van + r"""$) & $\eta_p^2 = """ + f"{eta_q_pgd:.3f}" + r"""$ ($F=""" + f"{f_q_pgd:.1f}" + r""", """ + p_q_pgd + r"""$) \\
Interaction ($M \times Q$) & """ + f"${inter_aqsm:+.2f}$ pp [${ci_inter_aqsm[0]:+.2f}, {ci_inter_aqsm[1]:+.2f}$]" + r""" & """ + f"${inter_van:+.2f}$ pp [${ci_inter_van[0]:+.2f}, {ci_inter_van[1]:+.2f}$]" + r""" & """ + f"${inter_pgd:+.2f}$ pp [${ci_inter_pgd[0]:+.2f}, {ci_inter_pgd[1]:+.2f}$]" + r""" \\
$\eta_p^2$ (F-statistic, $p$-value) & $\eta_p^2 = """ + f"{eta_inter_aqsm:.3f}" + r"""$ ($F=""" + f"{f_inter_aqsm:.1f}" + r""", """ + p_inter_aqsm + r"""$) & $\eta_p^2 = """ + f"{eta_inter_van:.3f}" + r"""$ ($F=""" + f"{f_inter_van:.1f}" + r""", """ + p_inter_van + r"""$) & $\eta_p^2 = """ + f"{eta_inter_pgd:.3f}" + r"""$ ($F=""" + f"{f_inter_pgd:.1f}" + r""", """ + p_inter_pgd + r"""$) \\
\bottomrule
\end{tabular}%
}
\end{table}"""

    md = "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] + ["| " + " | ".join(r) + " |" for r in rows])
    return tex, md, headers, rows


# ==============================================================================
# TABLE 4: Repeated-Measures ANOVA Attribution Table
# ==============================================================================
def generate_table4(summary_dir: Path) -> Tuple[str, str, List[str], List[List[Any]]]:
    c_aqsm = safe_load_json([summary_dir / "causal_swap_results_aqsm.json", OUTPUTS_DIR / "summary_json" / "causal_swap_results_aqsm.json"])
    c_van = safe_load_json([summary_dir / "causal_swap_results_vanishing.json", OUTPUTS_DIR / "summary_json" / "causal_swap_results_vanishing.json"])
    c_pgd = safe_load_json([summary_dir / "causal_swap_results_taskpgd.json", OUTPUTS_DIR / "summary_json" / "causal_swap_results_taskpgd.json"])

    an_full = require_val(c_aqsm, ["anova_repeated_measures_full"], "aqsm.anova_repeated_measures_full")
    an = require_val(an_full, ["anova_table"], "aqsm.anova_table")
    sh = require_val(an_full, ["attribution_shares", "partial_eta_squared"], "aqsm.partial_eta_squared")

    ss_m = float(require_val(an, ["Memory_M", "SS"], "aqsm.Memory_M.SS"))
    ss_q = float(require_val(an, ["Query_Q", "SS"], "aqsm.Query_Q.SS"))
    ss_inter = float(require_val(an, ["Interaction_MxQ", "SS"], "aqsm.Interaction_MxQ.SS"))

    f_m = float(require_val(an, ["Memory_M", "F"], "aqsm.Memory_M.F"))
    f_q = float(require_val(an, ["Query_Q", "F"], "aqsm.Query_Q.F"))
    f_inter = float(require_val(an, ["Interaction_MxQ", "F"], "aqsm.Interaction_MxQ.F"))

    eta_m = float(require_val(sh, ["Memory_M"], "aqsm.eta_m"))
    eta_q = float(require_val(sh, ["Query_Q"], "aqsm.eta_q"))
    eta_inter = float(require_val(sh, ["Interaction_MxQ"], "aqsm.eta_inter"))

    # Vanishing
    an_van = require_val(c_van, ["anova_repeated_measures_full", "anova_table"], "van.anova_table")
    sh_van = require_val(c_van, ["anova_repeated_measures_full", "attribution_shares", "partial_eta_squared"], "van.partial_eta_squared")
    ss_m_van = float(require_val(an_van, ["Memory_M", "SS"], "van.Memory_M.SS"))
    ss_q_van = float(require_val(an_van, ["Query_Q", "SS"], "van.Query_Q.SS"))
    ss_inter_van = float(require_val(an_van, ["Interaction_MxQ", "SS"], "van.Interaction_MxQ.SS"))
    f_m_van = float(require_val(an_van, ["Memory_M", "F"], "van.Memory_M.F"))
    f_q_van = float(require_val(an_van, ["Query_Q", "F"], "van.Query_Q.F"))
    f_inter_van = float(require_val(an_van, ["Interaction_MxQ", "F"], "van.Interaction_MxQ.F"))
    eta_m_van = float(require_val(sh_van, ["Memory_M"], "van.eta_m"))
    eta_q_van = float(require_val(sh_van, ["Query_Q"], "van.eta_q"))
    eta_inter_van = float(require_val(sh_van, ["Interaction_MxQ"], "van.eta_inter"))

    # TaskPGD
    an_pgd = require_val(c_pgd, ["anova_repeated_measures_full", "anova_table"], "pgd.anova_table")
    sh_pgd = require_val(c_pgd, ["anova_repeated_measures_full", "attribution_shares", "partial_eta_squared"], "pgd.partial_eta_squared")
    ss_m_pgd = float(require_val(an_pgd, ["Memory_M", "SS"], "pgd.Memory_M.SS"))
    ss_q_pgd = float(require_val(an_pgd, ["Query_Q", "SS"], "pgd.Query_Q.SS"))
    ss_inter_pgd = float(require_val(an_pgd, ["Interaction_MxQ", "SS"], "pgd.Interaction_MxQ.SS"))
    f_m_pgd = float(require_val(an_pgd, ["Memory_M", "F"], "pgd.Memory_M.F"))
    f_q_pgd = float(require_val(an_pgd, ["Query_Q", "F"], "pgd.Query_Q.F"))
    f_inter_pgd = float(require_val(an_pgd, ["Interaction_MxQ", "F"], "pgd.Interaction_MxQ.F"))
    eta_m_pgd = float(require_val(sh_pgd, ["Memory_M"], "pgd.eta_m"))
    eta_q_pgd = float(require_val(sh_pgd, ["Query_Q"], "pgd.eta_q"))
    eta_inter_pgd = float(require_val(sh_pgd, ["Interaction_MxQ"], "pgd.eta_inter"))

    p_m = _fmt_p(an["Memory_M"].get("p_value"))
    p_q = _fmt_p(an["Query_Q"].get("p_value"))
    p_i = _fmt_p(an["Interaction_MxQ"].get("p_value"))
    p_m_v = _fmt_p(an_van["Memory_M"].get("p_value"))
    p_q_v = _fmt_p(an_van["Query_Q"].get("p_value"))
    p_i_v = _fmt_p(an_van["Interaction_MxQ"].get("p_value"))
    p_m_p = _fmt_p(an_pgd["Memory_M"].get("p_value"))
    p_q_p = _fmt_p(an_pgd["Query_Q"].get("p_value"))
    p_i_p = _fmt_p(an_pgd["Interaction_MxQ"].get("p_value"))

    n_eval_t4 = int(c_aqsm.get("n_evaluated_images", 0))
    n_valid_t4 = int(c_aqsm.get("n_valid_intra_subject_samples", an_full.get("N", 0)))
    df2_t4 = max(1, n_valid_t4 - 1)

    headers = ["Attack Formulation", "Factor", "Sum of Squares", "df", "Mean Square", "F-Statistic", "Partial eta_p^2"]
    rows = [
        ["AQSM (Probe)", "Memory (M)", f"{ss_m:.1f}", "1", f"{ss_m:.1f}", f"{f_m:.1f}", f"{eta_m:.3f} ({p_m})"],
        ["AQSM (Probe)", "Query Prior (Q)", f"{ss_q:.1f}", "1", f"{ss_q:.1f}", f"{f_q:.1f}", f"{eta_q:.3f} ({p_q})"],
        ["AQSM (Probe)", "Interaction (M x Q)", f"{ss_inter:.1f}", "1", f"{ss_inter:.1f}", f"{f_inter:.1f}", f"{eta_inter:.3f} ({p_i})"],
        ["PGD Vanishing", "Memory (M)", f"{ss_m_van:.1f}", "1", f"{ss_m_van:.1f}", f"{f_m_van:.1f}", f"{eta_m_van:.3f} ({p_m_v})"],
        ["PGD Vanishing", "Query Prior (Q)", f"{ss_q_van:.1f}", "1", f"{ss_q_van:.1f}", f"{f_q_van:.1f}", f"{eta_q_van:.3f} ({p_q_v})"],
        ["PGD Vanishing", "Interaction (M x Q)", f"{ss_inter_van:.1f}", "1", f"{ss_inter_van:.1f}", f"{f_inter_van:.1f}", f"{eta_inter_van:.3f} ({p_i_v})"],
        ["Standard Task PGD", "Memory (M)", f"{ss_m_pgd:.1f}", "1", f"{ss_m_pgd:.1f}", f"{f_m_pgd:.1f}", f"{eta_m_pgd:.3f} ({p_m_p})"],
        ["Standard Task PGD", "Query Prior (Q)", f"{ss_q_pgd:.1f}", "1", f"{ss_q_pgd:.1f}", f"{f_q_pgd:.1f}", f"{eta_q_pgd:.3f} ({p_q_p})"],
        ["Standard Task PGD", "Interaction (M x Q)", f"{ss_inter_pgd:.1f}", "1", f"{ss_inter_pgd:.1f}", f"{f_inter_pgd:.1f}", f"{eta_inter_pgd:.3f} ({p_i_p})"],
    ]

    tex = r"""\begin{table}[!htbp]
\centering
\caption{ANOVA Factorial Decomposition and Interventional-Protocol Attribution across Attack Formulations on COCO val2017 ($640 \times 640$; evaluation cohort $N = """ + f"{n_eval_t4:,}" + r"""$ images, valid intra-subject $N = """ + f"{n_valid_t4:,}" + r"""$, $\text{df}_1 = 1, \text{df}_2 = """ + f"{df2_t4}" + r"""$). Highlights that query prior displacement represents an essential secondary protocol attribution under AQSM ($\eta_p^2 = """ + f"{eta_q:.3f}" + r"""$), in contrast to Query Vanishing where degradation concentrates in memory ($\eta_p^2 = """ + f"{eta_m_van:.3f}" + r"""$).}
\label{tab:tab4_anova_decomp}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccc}
\toprule
\textbf{Attack Formulation} & \textbf{Factor} & \textbf{Sum of Squares} & \textbf{df} & \textbf{Mean Square} & \textbf{$F$-Statistic} & \textbf{Partial $\eta_p^2$} \\
\midrule
\multirow{3}{*}{AQSM (Probe)} & Memory ($M$) & """ + f"{ss_m:.1f}" + r""" & 1 & """ + f"{ss_m:.1f}" + r""" & """ + f"{f_m:.1f}" + r""" & """ + f"{eta_m:.3f} (${p_m}$)" + r""" \\
& Query Prior ($Q$) & """ + f"{ss_q:.1f}" + r""" & 1 & """ + f"{ss_q:.1f}" + r""" & """ + f"{f_q:.1f}" + r""" & """ + f"{eta_q:.3f} (${p_q}$)" + r""" \\
& Interaction ($M \times Q$) & """ + f"{ss_inter:.1f}" + r""" & 1 & """ + f"{ss_inter:.1f}" + r""" & """ + f"{f_inter:.1f}" + r""" & """ + f"{eta_inter:.3f} (${p_i}$)" + r""" \\
\midrule
\multirow{3}{*}{PGD Vanishing} & Memory ($M$) & """ + f"{ss_m_van:.1f}" + r""" & 1 & """ + f"{ss_m_van:.1f}" + r""" & """ + f"{f_m_van:.1f}" + r""" & """ + f"{eta_m_van:.3f} (${p_m_v}$)" + r""" \\
& Query Prior ($Q$) & """ + f"{ss_q_van:.1f}" + r""" & 1 & """ + f"{ss_q_van:.1f}" + r""" & """ + f"{f_q_van:.1f}" + r""" & """ + f"{eta_q_van:.3f} (${p_q_v}$)" + r""" \\
& Interaction ($M \times Q$) & """ + f"{ss_inter_van:.1f}" + r""" & 1 & """ + f"{ss_inter_van:.1f}" + r""" & """ + f"{f_inter_van:.1f}" + r""" & """ + f"{eta_inter_van:.3f} (${p_i_v}$)" + r""" \\
\midrule
\multirow{3}{*}{Standard Task PGD} & Memory ($M$) & """ + f"{ss_m_pgd:.1f}" + r""" & 1 & """ + f"{ss_m_pgd:.1f}" + r""" & """ + f"{f_m_pgd:.1f}" + r""" & """ + f"{eta_m_pgd:.3f} (${p_m_p}$)" + r""" \\
& Query Prior ($Q$) & """ + f"{ss_q_pgd:.1f}" + r""" & 1 & """ + f"{ss_q_pgd:.1f}" + r""" & """ + f"{f_q_pgd:.1f}" + r""" & """ + f"{eta_q_pgd:.3f} (${p_q_p}$)" + r""" \\
& Interaction ($M \times Q$) & """ + f"{ss_inter_pgd:.1f}" + r""" & 1 & """ + f"{ss_inter_pgd:.1f}" + r""" & """ + f"{f_inter_pgd:.1f}" + r""" & """ + f"{eta_inter_pgd:.3f} (${p_i_p}$)" + r""" \\
\bottomrule
\end{tabular}%
}
\end{table}"""

    md = "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] + ["| " + " | ".join(r) + " |" for r in rows])
    return tex, md, headers, rows


# ==============================================================================
# TABLE 5: Encoder Memory Feature Perturbation across Multi-Scale Tokens
# ==============================================================================
def generate_table5(summary_dir: Path) -> Tuple[str, str, List[str], List[List[Any]]]:
    mem_data = safe_load_json([summary_dir / "encoder_memory_analysis.json", OUTPUTS_DIR / "summary_json" / "encoder_memory_analysis.json"])
    mem = mem_data.get("results", mem_data)

    def _req_cos(atk: str, scale: str) -> float:
        return require_stat(require_val(mem, [atk, "cosine_similarity", scale], f"{atk}.cosine.{scale}"), f"{atk}.cosine.{scale}")

    def _req_l2(atk: str, region: str) -> float:
        return require_stat(require_val(mem, [atk, "relative_l2_perturbation", region], f"{atk}.l2.{region}"), f"{atk}.l2.{region}")

    van_cos_ov = _req_cos("vanishing", "Overall")
    van_cos_s3 = _req_cos("vanishing", "S3")
    van_cos_s4 = _req_cos("vanishing", "S4")
    van_cos_s5 = _req_cos("vanishing", "S5")
    van_l2_fg = _req_l2("vanishing", "Foreground")
    van_l2_bg = _req_l2("vanishing", "Background")

    aq_cos_ov = _req_cos("aqsm", "Overall")
    aq_cos_s3 = _req_cos("aqsm", "S3")
    aq_cos_s4 = _req_cos("aqsm", "S4")
    aq_cos_s5 = _req_cos("aqsm", "S5")
    aq_l2_fg = _req_l2("aqsm", "Foreground")
    aq_l2_bg = _req_l2("aqsm", "Background")

    pgd_cos_ov = _req_cos("taskpgd", "Overall")
    pgd_cos_s3 = _req_cos("taskpgd", "S3")
    pgd_cos_s4 = _req_cos("taskpgd", "S4")
    pgd_cos_s5 = _req_cos("taskpgd", "S5")
    pgd_l2_fg = _req_l2("taskpgd", "Foreground")
    pgd_l2_bg = _req_l2("taskpgd", "Background")

    n_eval_t5 = int(mem_data.get("n_images", 0))
    n_str_t5 = f"Evaluated on $N = {n_eval_t5:,}$ cohort." if n_eval_t5 > 0 else ""
    t_steps_t5 = int(mem_data.get("pgd_steps", get_pgd_num_steps()))
    eff_res_t5 = int(get_img_size())

    headers = ["Condition", "Overall Cosine", "Per-Scale Cosine (S3/S4/S5)", "FG Rel. L2", "BG Rel. L2", "FG/BG Ratio", "Cosine Gap (S5 - S4)"]
    rows = [
        [f"PGD Vanishing (T = {t_steps_t5})", f"{van_cos_ov:.4f}", f"{van_cos_s3:.3f} / {van_cos_s4:.3f} / {van_cos_s5:.3f}", f"{van_l2_fg:.4f}", f"{van_l2_bg:.4f}", f"{van_l2_fg/van_l2_bg if van_l2_bg > 0 else 1.0:.4f}", f"{van_cos_s5 - van_cos_s4:+.3f}"],
        [f"AQSM Probe (T = {t_steps_t5})", f"{aq_cos_ov:.4f}", f"{aq_cos_s3:.3f} / {aq_cos_s4:.3f} / {aq_cos_s5:.3f}", f"{aq_l2_fg:.4f}", f"{aq_l2_bg:.4f}", f"{aq_l2_fg/aq_l2_bg if aq_l2_bg > 0 else 1.0:.4f}", f"{aq_cos_s5 - aq_cos_s4:+.3f}"],
        [f"Standard Task PGD (T = {t_steps_t5})", f"{pgd_cos_ov:.4f}", f"{pgd_cos_s3:.3f} / {pgd_cos_s4:.3f} / {pgd_cos_s5:.3f}", f"{pgd_l2_fg:.4f}", f"{pgd_l2_bg:.4f}", f"{pgd_l2_fg/pgd_l2_bg if pgd_l2_bg > 0 else 1.0:.4f}", f"{pgd_cos_s5 - pgd_cos_s4:+.3f}"],
    ]
    noise_tex_row = ""
    if "noise" in mem or "gaussian_noise" in mem:
        nk = "gaussian_noise" if "gaussian_noise" in mem else "noise"
        nc_ov = _req_cos(nk, "Overall")
        nc_s3, nc_s4, nc_s5 = _req_cos(nk, "S3"), _req_cos(nk, "S4"), _req_cos(nk, "S5")
        nl_fg, nl_bg = _req_l2(nk, "Foreground"), _req_l2(nk, "Background")
        rows.insert(0, ["Gaussian Noise (sigma = 16/255)", f"{nc_ov:.4f}", f"{nc_s3:.3f} / {nc_s4:.3f} / {nc_s5:.3f}", f"{nl_fg:.4f}", f"{nl_bg:.4f}", f"{nl_fg/nl_bg if nl_bg > 0 else 1.0:.4f}", f"{nc_s5 - nc_s4:+.3f}"])
        noise_tex_row = f"Gaussian Noise ($\\sigma = 16/255$) & {nc_ov:.4f} & {nc_s3:.3f} / {nc_s4:.3f} / {nc_s5:.3f} & {nl_fg:.4f} & {nl_bg:.4f} & {nl_fg/nl_bg if nl_bg > 0 else 1.0:.4f} & {nc_s5 - nc_s4:+.3f} \\\\\n"

    tex = r"""\begin{table}[!htbp]
\centering
\caption{Encoder memory feature perturbation on COCO val2017 at native $""" + f"{eff_res_t5} \\times {eff_res_t5}" + r"""$ resolution across feature scales ($S_3, S_4, S_5$) and semantic regions (GT Foreground tokens vs.\ Background tokens). """ + n_str_t5 + r"""}
\label{tab:tab5_memory_scales}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccc}
\toprule
\textbf{Condition} & \textbf{Overall Cosine} & \textbf{Per-Scale Cosine ($S_3/S_4/S_5$)} & \textbf{FG Rel. $\ell_2$} & \textbf{BG Rel. $\ell_2$} & \textbf{FG/BG Ratio} & \textbf{Cosine Gap ($S_5 - S_4$)} \\
\midrule
""" + noise_tex_row + f"PGD Vanishing ($T = {t_steps_t5}$) & " + f"{van_cos_ov:.4f}" + r""" & """ + f"{van_cos_s3:.3f} / {van_cos_s4:.3f} / {van_cos_s5:.3f}" + r""" & """ + f"{van_l2_fg:.4f}" + r""" & """ + f"{van_l2_bg:.4f}" + r""" & """ + f"{van_l2_fg/van_l2_bg if van_l2_bg > 0 else 1.0:.4f}" + r""" & """ + f"{van_cos_s5 - van_cos_s4:+.3f}" + r""" \\
""" + f"AQSM Probe ($T = {t_steps_t5}$) & " + r"""\textbf{""" + f"{aq_cos_ov:.4f}" + r"""} & \textbf{""" + f"{aq_cos_s3:.3f} / {aq_cos_s4:.3f} / {aq_cos_s5:.3f}" + r"""} & \textbf{""" + f"{aq_l2_fg:.4f}" + r"""} & \textbf{""" + f"{aq_l2_bg:.4f}" + r"""} & \textbf{""" + f"{aq_l2_fg/aq_l2_bg if aq_l2_bg > 0 else 1.0:.4f}" + r"""} & \textbf{""" + f"{aq_cos_s5 - aq_cos_s4:+.3f}" + r"""} \\
""" + f"Standard Task PGD ($T = {t_steps_t5}$) & " + f"{pgd_cos_ov:.4f}" + r""" & """ + f"{pgd_cos_s3:.3f} / {pgd_cos_s4:.3f} / {pgd_cos_s5:.3f}" + r""" & """ + f"{pgd_l2_fg:.4f}" + r""" & """ + f"{pgd_l2_bg:.4f}" + r""" & """ + f"{pgd_l2_fg/pgd_l2_bg if pgd_l2_bg > 0 else 1.0:.4f}" + r""" & """ + f"{pgd_cos_s5 - pgd_cos_s4:+.3f}" + r""" \\
\bottomrule
\end{tabular}%
}
\end{table}"""

    md = "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] + ["| " + " | ".join(r) + " |" for r in rows])
    return tex, md, headers, rows


# ==============================================================================
# TABLE 6: 8,400-to-300 Query Selection Dynamics & Spatial Coverage Verification
# ==============================================================================
def generate_table6(summary_dir: Path) -> Tuple[str, str, List[str], List[List[Any]]]:
    dyn_data = safe_load_json([
        summary_dir / "query_selection_dynamics.json",
        summary_dir / "encoder_query_dynamics.json",
        OUTPUTS_DIR / "summary_json" / "query_selection_dynamics.json",
    ])
    dyn = dyn_data.get("results", dyn_data)

    aq = require_val(dyn, ["aqsm"], "dyn.aqsm")
    van = require_val(dyn, ["vanishing"], "dyn.vanishing")
    pgd = require_val(dyn, ["taskpgd"], "dyn.taskpgd")

    clean_cg = require_stat(require_val(aq, ["object_coverage", "center_in_gt_clean_pct"], "aq.cov.center_clean"))
    clean_iou30 = require_stat(require_val(aq, ["object_coverage", "iou30_clean_pct"], "aq.cov.iou30_clean"))
    aq_cg = require_stat(require_val(aq, ["object_coverage", "center_in_gt_adv_pct"], "aq.cov.center_adv"))
    aq_iou30 = require_stat(require_val(aq, ["object_coverage", "iou30_adv_pct"], "aq.cov.iou30_adv"))

    van_cg = require_stat(require_val(van, ["object_coverage", "center_in_gt_adv_pct"], "van.cov.center_adv"))
    van_iou30 = require_stat(require_val(van, ["object_coverage", "iou30_adv_pct"], "van.cov.iou30_adv"))

    pgd_cg = require_stat(require_val(pgd, ["object_coverage", "center_in_gt_adv_pct"], "pgd.cov.center_adv"))
    pgd_iou30 = require_stat(require_val(pgd, ["object_coverage", "iou30_adv_pct"], "pgd.cov.iou30_adv"))

    aq_ov = require_stat(require_val(aq, ["top300_overlap_pct"], "aq.top300_overlap_pct"))
    aq_rho = require_stat(require_val(aq, ["spearman_rho_all_proposals"], "aq.spearman"))
    aq_tau = require_stat(require_val(aq, ["kendall_tau_all_proposals"], "aq.kendall"))
    aq_ev = require_stat(require_val(aq, ["evicted_top15_fg_pct"], "aq.evicted_top15"))

    van_ov = require_stat(require_val(van, ["top300_overlap_pct"], "van.top300_overlap_pct"))
    van_rho = require_stat(require_val(van, ["spearman_rho_all_proposals"], "van.spearman"))
    van_tau = require_stat(require_val(van, ["kendall_tau_all_proposals"], "van.kendall"))
    van_ev = require_stat(require_val(van, ["evicted_top15_fg_pct"], "van.evicted_top15"))

    pgd_ov = require_stat(require_val(pgd, ["top300_overlap_pct"], "pgd.top300_overlap_pct"))
    pgd_rho = require_stat(require_val(pgd, ["spearman_rho_all_proposals"], "pgd.spearman"))
    pgd_tau = require_stat(require_val(pgd, ["kendall_tau_all_proposals"], "pgd.kendall"))
    pgd_ev = require_stat(require_val(pgd, ["evicted_top15_fg_pct"], "pgd.evicted_top15"))

    mis_ref = safe_load_json([
        summary_dir / "controlled_misrouting_results.json",
        summary_dir / "tautology_control_results.json",
    ], default_dict={})
    aq_fghit_raw = mis_ref.get("table8_summary", {}).get("AQSM_Spatial_Misrouting", {}).get("fg_hit_pct")
    fghit_note = f", while anchor foreground hit rate drops to ${float(aq_fghit_raw):.2f}\\%$ under AQSM" if aq_fghit_raw is not None else ""

    n_eval_t6 = int(dyn_data.get("n_images", 0))
    n_str_t6 = f", $N = {n_eval_t6:,}$ images" if n_eval_t6 > 0 else ""
    t_steps_t6 = int(dyn_data.get("pgd_steps", get_pgd_num_steps()))
    eff_res_t6 = int(get_img_size())

    headers = ["Condition", "Top-300 Overlap", "Spearman rho", "Kendall tau", "Evicted Top-15 FG", "GT Coverage (Center)", "GT Coverage (IoU >= 0.30)"]
    rows = [
        ["Clean Baseline", "100.0% [Ref.]", "1.000", "1.000", "0.0%", f"{clean_cg:.2f}%", f"{clean_iou30:.2f}%"],
        [f"PGD Vanishing (T = {t_steps_t6})", f"{van_ov:.2f}%", f"{van_rho:.3f}", f"{van_tau:.3f}", f"{van_ev:.2f}%", f"{van_cg:.2f}%", f"{van_iou30:.2f}%"],
        [f"AQSM Probe (T = {t_steps_t6})", f"{aq_ov:.2f}%", f"{aq_rho:.3f}", f"{aq_tau:.3f}", f"{aq_ev:.2f}%", f"{aq_cg:.2f}%", f"{aq_iou30:.2f}%"],
        [f"Standard Task PGD (T = {t_steps_t6})", f"{pgd_ov:.2f}%", f"{pgd_rho:.3f}", f"{pgd_tau:.3f}", f"{pgd_ev:.2f}%", f"{pgd_cg:.2f}%", f"{pgd_iou30:.2f}%"],
    ]
    noise_tex_t6 = ""
    if "noise" in dyn or "gaussian_noise" in dyn:
        nk = "gaussian_noise" if "gaussian_noise" in dyn else "noise"
        nb = dyn[nk]
        n_ov = require_stat(nb.get("top300_overlap_pct"), f"{nk}.ov")
        n_rho = require_stat(nb.get("spearman_rho_all_proposals"), f"{nk}.rho")
        n_tau = require_stat(nb.get("kendall_tau_all_proposals"), f"{nk}.tau")
        n_ev = require_stat(nb.get("evicted_top15_fg_pct"), f"{nk}.ev")
        n_cg = require_stat(nb.get("object_coverage", {}).get("center_in_gt_adv_pct"), f"{nk}.cg")
        n_iou30 = require_stat(nb.get("object_coverage", {}).get("iou30_adv_pct"), f"{nk}.iou30")
        rows.insert(1, ["Gaussian Noise (sigma = 16/255)", f"{n_ov:.2f}%", f"{n_rho:.3f}", f"{n_tau:.3f}", f"{n_ev:.2f}%", f"{n_cg:.2f}%", f"{n_iou30:.2f}%"])
        noise_tex_t6 = f"Gaussian Noise ($\\sigma = 16/255$) & {n_ov:.2f}\\% & {n_rho:.3f} & {n_tau:.3f} & {n_ev:.2f}\\% & {n_cg:.2f}\\% & {n_iou30:.2f}\\% \\\\\n"

    tex = r"""\begin{table}[!htbp]
\centering
\caption{Query selection overlap, global rank correlation, foreground eviction, and Ground-Truth object coverage inside selected proposals on COCO val2017 ($""" + f"{eff_res_t6} \\times {eff_res_t6}" + r"""$""" + n_str_t6 + r"""). GT Coverage denotes the percentage of target objects covered by at least one selected query anchor""" + fghit_note + r""".}
\label{tab:tab6_query_dynamics}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccc}
\toprule
\textbf{Condition} & \textbf{Top-300 Overlap} & \textbf{Spearman $\rho$} & \textbf{Kendall $\tau$} & \textbf{Evicted Top-15 FG} & \textbf{GT Coverage (Center)} & \textbf{GT Coverage ($\text{IoU} \ge 0.30$)} \\
\midrule
Clean Baseline & 100.0\% [Ref.] & 1.000 & 1.000 & 0.0\% & """ + f"{clean_cg:.2f}\\%" + r""" & """ + f"{clean_iou30:.2f}\\%" + r""" \\
""" + noise_tex_t6 + f"PGD Vanishing ($T = {t_steps_t6}$) & " + f"{van_ov:.2f}\\%" + r""" & """ + f"{van_rho:.3f}" + r""" & """ + f"{van_tau:.3f}" + r""" & """ + f"{van_ev:.2f}\\%" + r""" & """ + f"{van_cg:.2f}\\%" + r""" & """ + f"{van_iou30:.2f}\\%" + r""" \\
""" + f"AQSM Probe ($T = {t_steps_t6}$) & " + r"""\textbf{""" + f"{aq_ov:.2f}\\%" + r"""} & \textbf{""" + f"{aq_rho:.3f}" + r"""} & \textbf{""" + f"{aq_tau:.3f}" + r"""} & \textbf{""" + f"{aq_ev:.2f}\\%" + r"""} & \textbf{""" + f"{aq_cg:.2f}\\%" + r"""} & \textbf{""" + f"{aq_iou30:.2f}\\%" + r"""} \\
""" + f"Standard Task PGD ($T = {t_steps_t6}$) & " + f"{pgd_ov:.2f}\\%" + r""" & """ + f"{pgd_rho:.3f}" + r""" & """ + f"{pgd_tau:.3f}" + r""" & """ + f"{pgd_ev:.2f}\\%" + r""" & """ + f"{pgd_cg:.2f}\\%" + r""" & """ + f"{pgd_iou30:.2f}\\%" + r""" \\
\bottomrule
\end{tabular}%
}
\end{table}"""

    md = "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] + ["| " + " | ".join(r) + " |" for r in rows])
    return tex, md, headers, rows


# ==============================================================================
# TABLE 7: Deformable Attention Telemetry
# ==============================================================================
def generate_table7(summary_dir: Path) -> Tuple[str, str, List[str], List[List[Any]]]:
    tel = safe_load_json([
        summary_dir / "deformable_attention_telemetry.json",
        summary_dir / "mechanism_telemetry_results.json",
        OUTPUTS_DIR / "summary_json" / "deformable_attention_telemetry.json",
    ])

    cl = require_val(tel, ["clean"], "tel.clean")
    aq = require_val(tel, ["aqsm"], "tel.aqsm")
    van = require_val(tel, ["vanishing"], "tel.vanishing")
    pgd = require_val(tel, ["taskpgd"], "tel.taskpgd")

    cl_l1 = float(require_val(cl, ["mean_entropy_l1"], "tel.cl.ent_l1"))
    aq_l1 = float(require_val(aq, ["mean_entropy_l1"], "tel.aq.ent_l1"))

    def _ent_mean_std(blk: Dict[str, Any], tag: str) -> Tuple[float, float]:
        if "mean_entropy_all_layers" in blk and "std_entropy_all_layers" in blk:
            return float(blk["mean_entropy_all_layers"]), float(blk["std_entropy_all_layers"])
        m = (float(require_val(blk, ["mean_entropy_l1"], f"{tag}.l1")) + float(require_val(blk, ["mean_entropy_l2"], f"{tag}.l2")) + float(require_val(blk, ["mean_entropy_l3"], f"{tag}.l3"))) / 3.0
        s = float(require_val(blk, ["std_entropy_l1"], f"{tag}.std_l1"))
        return m, s

    cl_ent, cl_ent_std = _ent_mean_std(cl, "tel.cl")
    aq_ent, aq_ent_std = _ent_mean_std(aq, "tel.aq")
    van_ent, van_ent_std = _ent_mean_std(van, "tel.van")
    pgd_ent, pgd_ent_std = _ent_mean_std(pgd, "tel.pgd")

    cl_hit = float(require_val(cl, ["mean_fg_hit_rate"], "tel.cl.fg_hit")) * 100.0
    cl_drop = float(require_val(cl, ["mean_dropoff_ratio"], "tel.cl.dropoff"))
    cl_drop_std = float(require_val(cl, ["std_dropoff_ratio"], "tel.cl.std_dropoff"))
    cl_marg = float(require_val(cl, ["mean_score_margin"], "tel.cl.margin"))
    cl_marg_std = float(require_val(cl, ["std_score_margin"], "tel.cl.std_margin"))
    cl_drift = float(cl.get("mean_ref_drift", 0.0))

    van_hit = float(require_val(van, ["mean_fg_hit_rate"], "tel.van.fg_hit")) * 100.0
    van_drop = float(require_val(van, ["mean_dropoff_ratio"], "tel.van.dropoff"))
    van_drop_std = float(require_val(van, ["std_dropoff_ratio"], "tel.van.std_dropoff"))
    van_marg = float(require_val(van, ["mean_score_margin"], "tel.van.margin"))
    van_marg_std = float(require_val(van, ["std_score_margin"], "tel.van.std_margin"))
    van_drift = float(require_val(van, ["mean_ref_drift"], "tel.van.ref_drift"))
    van_drift_std = float(require_val(van, ["std_ref_drift"], "tel.van.std_ref_drift"))

    aq_hit = float(require_val(aq, ["mean_fg_hit_rate"], "tel.aq.fg_hit")) * 100.0
    aq_drop = float(require_val(aq, ["mean_dropoff_ratio"], "tel.aq.dropoff"))
    aq_drop_std = float(require_val(aq, ["std_dropoff_ratio"], "tel.aq.std_dropoff"))
    aq_marg = float(require_val(aq, ["mean_score_margin"], "tel.aq.margin"))
    aq_marg_std = float(require_val(aq, ["std_score_margin"], "tel.aq.std_margin"))
    aq_drift = float(require_val(aq, ["mean_ref_drift"], "tel.aq.ref_drift"))
    aq_drift_std = float(require_val(aq, ["std_ref_drift"], "tel.aq.std_ref_drift"))

    pgd_hit = float(require_val(pgd, ["mean_fg_hit_rate"], "tel.pgd.fg_hit")) * 100.0
    pgd_drop = float(require_val(pgd, ["mean_dropoff_ratio"], "tel.pgd.dropoff"))
    pgd_drop_std = float(require_val(pgd, ["std_dropoff_ratio"], "tel.pgd.std_dropoff"))
    pgd_marg = float(require_val(pgd, ["mean_score_margin"], "tel.pgd.margin"))
    pgd_marg_std = float(require_val(pgd, ["std_score_margin"], "tel.pgd.std_margin"))
    pgd_drift = float(require_val(pgd, ["mean_ref_drift"], "tel.pgd.ref_drift"))
    pgd_drift_std = float(require_val(pgd, ["std_ref_drift"], "tel.pgd.std_ref_drift"))

    mis_ref = safe_load_json([
        summary_dir / "controlled_misrouting_results.json",
        summary_dir / "tautology_control_results.json",
    ], default_dict={})
    aq_hung_raw = mis_ref.get("table8_summary", {}).get("AQSM_Spatial_Misrouting", {}).get("hungarian_dist")
    hung_note = rf" (corresponding permutation-invariant Hungarian set distance is $\Delta_{{\mathrm{{Hungarian}}}} = {float(aq_hung_raw):.4f}$)" if aq_hung_raw is not None else ""
    ent_delta = aq_ent - cl_ent

    n_eval_t7 = int(tel.get("n_images", cl.get("n_images", 0)))
    n_str_t7 = f" on $N = {n_eval_t7:,}$ images" if n_eval_t7 > 0 else ""
    t_steps_t7 = int(tel.get("pgd_steps", get_pgd_num_steps()))
    eff_res_t7 = int(get_img_size())

    headers = ["Condition", "Norm. Attention Entropy", "Spatial Sampling Hit Rate (FG)", "Reference Drift (Delta p)", "Dropoff Ratio", "Score Margin"]
    rows = [
        ["Clean Baseline", f"{cl_ent:.4f} ± {cl_ent_std:.4f}", f"{cl_hit:.2f}%", f"{cl_drift:.4f}", f"{cl_drop:.2f} ± {cl_drop_std:.2f}", f"{cl_marg:.4f} ± {cl_marg_std:.4f}"],
        [f"PGD Vanishing (T = {t_steps_t7})", f"{van_ent:.4f} ± {van_ent_std:.4f}", f"{van_hit:.2f}%", f"{van_drift:.4f} ± {van_drift_std:.4f}", f"{van_drop:.2f} ± {van_drop_std:.2f}", f"{van_marg:.4f} ± {van_marg_std:.4f}"],
        [f"AQSM Probe (T = {t_steps_t7})", f"{aq_ent:.4f} ± {aq_ent_std:.4f}", f"{aq_hit:.2f}%", f"{aq_drift:.4f} ± {aq_drift_std:.4f}", f"{aq_drop:.2f} ± {aq_drop_std:.2f}", f"{aq_marg:.4f} ± {aq_marg_std:.4f}"],
        [f"Standard Task PGD (T = {t_steps_t7})", f"{pgd_ent:.4f} ± {pgd_ent_std:.4f}", f"{pgd_hit:.2f}%", f"{pgd_drift:.4f} ± {pgd_drift_std:.4f}", f"{pgd_drop:.2f} ± {pgd_drop_std:.2f}", f"{pgd_marg:.4f} ± {pgd_marg_std:.4f}"],
    ]

    tex = r"""\begin{table}[!htbp]
\centering
\caption{Deformable decoder cross-attention telemetry on COCO val2017 ($""" + f"{eff_res_t7} \\times {eff_res_t7}" + r"""$, averaged across decoder layers, 8 heads, multi-scale tokens""" + n_str_t7 + r"""). Normalized attention entropy exhibits negligible practical change under AQSM ($\Delta = """ + f"{ent_delta:+.4f}" + r"""$, from $""" + f"{cl_ent:.4f} \\to {aq_ent:.4f}" + r"""$; Layer 1: $""" + f"{cl_l1:.4f} \\to {aq_l1:.4f}" + r"""$), showing that attention weights remain localized rather than dispersing, while Spatial Target Misrouting reduces foreground sampling hit rate from $""" + f"{cl_hit:.2f}\\%" + r"""$ to $""" + f"{aq_hit:.2f}\\%" + r"""$. Reference Drift ($\Delta_{\mathrm{slot}}$) denotes rank-paired slot coordinate displacement""" + hung_note + r""".}
\label{tab:tab7_attention_telemetry}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lccccc}
\toprule
\textbf{Condition} & \textbf{Norm. Attention Entropy} & \textbf{Spatial Sampling Hit Rate (FG)} & \textbf{Reference Drift ($\Delta p$)} & \textbf{Dropoff Ratio} & \textbf{Score Margin} \\
\midrule
Clean Baseline & """ + f"{cl_ent:.4f} $\\pm$ {cl_ent_std:.4f}" + r""" & """ + f"{cl_hit:.2f}\\%" + r""" & """ + f"{cl_drift:.4f}" + r""" & """ + f"{cl_drop:.2f} $\\pm$ {cl_drop_std:.2f}" + r""" & """ + f"{cl_marg:.4f} $\\pm$ {cl_marg_std:.4f}" + r""" \\
PGD Vanishing ($T = 15$) & \textbf{""" + f"{van_ent:.4f} $\\pm$ {van_ent_std:.4f}" + r"""} & """ + f"{van_hit:.2f}\\%" + r""" & """ + f"{van_drift:.4f} $\\pm$ {van_drift_std:.4f}" + r""" & """ + f"{van_drop:.2f} $\\pm$ {van_drop_std:.2f}" + r""" & """ + f"{van_marg:.4f} $\\pm$ {van_marg_std:.4f}" + r""" \\
AQSM Probe ($T = 15$) & \textbf{""" + f"{aq_ent:.4f} $\\pm$ {aq_ent_std:.4f}" + r"""} & \textbf{""" + f"{aq_hit:.2f}\\%" + r"""} & \textbf{""" + f"{aq_drift:.4f} $\\pm$ {aq_drift_std:.4f}" + r"""} & \textbf{""" + f"{aq_drop:.2f} $\\pm$ {aq_drop_std:.2f}" + r"""} & \textbf{""" + f"{aq_marg:.4f} $\\pm$ {aq_marg_std:.4f}" + r"""} \\
Standard Task PGD ($T = 15$) & """ + f"{pgd_ent:.4f} $\\pm$ {pgd_ent_std:.4f}" + r""" & """ + f"{pgd_hit:.2f}\\%" + r""" & """ + f"{pgd_drift:.4f} $\\pm$ {pgd_drift_std:.4f}" + r""" & """ + f"{pgd_drop:.2f} $\\pm$ {pgd_drop_std:.2f}" + r""" & """ + f"{pgd_marg:.4f} $\\pm$ {pgd_marg_std:.4f}" + r""" \\
\bottomrule
\end{tabular}%
}
\end{table}"""

    md = "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] + ["| " + " | ".join(r) + " |" for r in rows])
    return tex, md, headers, rows


# ==============================================================================
# TABLE 8: Controlled Spatial Misrouting & Tautology Refutation
# ==============================================================================
def generate_table8(summary_dir: Path) -> Tuple[str, str, List[str], List[List[Any]]]:
    mis_data = safe_load_json([
        summary_dir / "controlled_misrouting_results.json",
        summary_dir / "tautology_control_results.json",
        OUTPUTS_DIR / "summary_json" / "controlled_misrouting_results.json",
    ])
    mis = require_val(mis_data, ["table8_summary"], "mis.table8_summary")
    detailed = require_val(mis_data, ["detailed_by_condition"], "mis.detailed_by_condition")

    cl = require_val(mis, ["Clean_Baseline"], "t8.Clean_Baseline")
    rr = require_val(mis, ["Random_Reassignment"], "t8.Random_Reassignment")
    rs = require_val(mis, ["Matched_L2_Random_Shift"], "t8.Matched_L2_Random_Shift")
    drs = require_val(mis, ["Decoupled_Random_Shift"], "t8.Decoupled_Random_Shift")
    aq = require_val(mis, ["AQSM_Spatial_Misrouting"], "t8.AQSM_Spatial_Misrouting")

    cl_map50 = float(require_val(cl, ["mAP@50"], "t8.cl.mAP@50"))
    cl_map95 = float(require_val(cl, ["mAP@50:95"], "t8.cl.mAP@50:95"))
    cl_rec = float(require_val(cl, ["recall_pct"], "t8.cl.recall"))
    cl_fghit = float(require_val(cl, ["fg_hit_pct"], "t8.cl.fg_hit"))
    cl_hung = float(require_val(cl, ["hungarian_dist"], "t8.cl.hung"))
    cl_cov = float(require_val(cl, ["gt_object_cov_pct"], "t8.cl.cov"))

    rr_m50 = float(require_val(rr, ["mAP@50_mean"], "t8.rr.m50"))
    rr_m50_std = float(require_val(rr, ["mAP@50_std"], "t8.rr.m50_std"))
    rr_m95 = float(require_val(rr, ["mAP@50:95_mean"], "t8.rr.m95"))
    rr_m95_std = float(require_val(rr, ["mAP@50:95_std"], "t8.rr.m95_std"))
    rr_rec = float(require_val(rr, ["recall_pct_mean"], "t8.rr.rec"))
    rr_l2 = float(require_val(rr, ["delta_l2"], "t8.rr.l2"))
    rr_fghit = float(require_val(rr, ["fg_hit_pct_mean"], "t8.rr.fg_hit"))
    rr_hung = float(require_val(rr, ["hungarian_dist"], "t8.rr.hung"))
    rr_cov = float(require_val(rr, ["gt_object_cov_pct"], "t8.rr.cov"))

    rs_m50 = float(require_val(rs, ["mAP@50_mean"], "t8.rs.m50"))
    rs_m50_std = float(require_val(rs, ["mAP@50_std"], "t8.rs.m50_std"))
    rs_m95 = float(require_val(rs, ["mAP@50:95_mean"], "t8.rs.m95"))
    rs_m95_std = float(require_val(rs, ["mAP@50:95_std"], "t8.rs.m95_std"))
    rs_rec = float(require_val(rs, ["recall_pct_mean"], "t8.rs.rec"))
    rs_l2 = float(require_val(rs, ["delta_l2"], "t8.rs.l2"))
    rs_fghit = float(require_val(rs, ["fg_hit_pct_mean"], "t8.rs.fg_hit"))
    rs_hung = float(require_val(rs, ["hungarian_dist"], "t8.rs.hung"))
    rs_cov = float(require_val(rs, ["gt_object_cov_pct"], "t8.rs.cov"))

    aq_m50 = float(require_val(aq, ["mAP@50"], "t8.aq.m50"))
    aq_m95 = float(require_val(aq, ["mAP@50:95"], "t8.aq.m95"))
    aq_rec = float(require_val(aq, ["recall_pct"], "t8.aq.rec"))
    aq_l2 = float(require_val(aq, ["delta_l2"], "t8.aq.l2"))
    aq_fghit = float(require_val(aq, ["fg_hit_pct"], "t8.aq.fg_hit"))
    aq_hung = float(require_val(aq, ["hungarian_dist"], "t8.aq.hung"))
    aq_cov = float(require_val(aq, ["gt_object_cov_pct"], "t8.aq.cov"))

    def _std_vals(vals: List[float]) -> float:
        if len(vals) <= 1:
            return 0.0
        m = sum(vals) / len(vals)
        return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))

    def _extract_sorted_seeds(prefix: str) -> List[int]:
        found_seeds: List[int] = []
        prefix_tag = f"{prefix}_seed"
        for k in detailed.keys():
            if k.startswith(prefix_tag):
                tail = k[len(prefix_tag):]
                if tail.isdigit():
                    found_seeds.append(int(tail))
        return sorted(found_seeds)

    rr_seed_ids = _extract_sorted_seeds("Random_Reassign")
    rs_seed_ids = _extract_sorted_seeds("Random_Shift")
    rr_seeds = [detailed[f"Random_Reassign_seed{s}"] for s in rr_seed_ids]
    rs_seeds = [detailed[f"Random_Shift_seed{s}"] for s in rs_seed_ids]
    rr_rec_std = _std_vals([float(v["recall"]) for v in rr_seeds]) if rr_seeds else 0.0
    rs_rec_std = _std_vals([float(v["recall"]) for v in rs_seeds]) if rs_seeds else 0.0
    rs_l2_std = float(rs["delta_l2_std"]) if "delta_l2_std" in rs else (_std_vals([float(v["delta_l2"]) for v in rs_seeds]) if rs_seeds else 0.0)
    n_seed_cnt = max(len(rr_seeds), len(rs_seeds), 1)

    def _get_seed_row(cond_prefix: str, seed: int):
        entry = require_val(detailed, [f"{cond_prefix}_seed{seed}"], f"detailed.{cond_prefix}_seed{seed}")
        return (
            float(entry["mAP@50"]),
            float(entry["mAP@50:95"]),
            float(entry["recall"]),
            float(entry["delta_l2"]),
            float(entry["delta_center_l2"]),
            float(entry["delta_size_l2"]),
            float(entry["fg_hit"]),
        )

    _d_cl50 = max(cl_map50, 1e-6)
    rr_seed_tex_lines: List[str] = []
    for s in rr_seed_ids:
        r_vals = _get_seed_row("Random_Reassign", s)
        rr_seed_tex_lines.append(
            f"Random Reassignment (Seed {s}) & {s} & {r_vals[0]:.2f}\\% ({r_vals[0]/_d_cl50*100:.1f}\\%) & {r_vals[1]:.2f}\\% & {r_vals[2]:.2f}\\% & {r_vals[3]:.4f} & {rr_hung:.4f} & {r_vals[6]:.2f}\\% & {rr_cov:.2f}\\% \\\\"
        )

    rs_seed_tex_lines: List[str] = []
    for s in rs_seed_ids:
        r_vals = _get_seed_row("Random_Shift", s)
        rs_seed_tex_lines.append(
            f"Random Shift (Seed {s}) & {s} & {r_vals[0]:.2f}\\% ({r_vals[0]/_d_cl50*100:.1f}\\%) & {r_vals[1]:.2f}\\% & {r_vals[2]:.2f}\\% & {r_vals[3]:.4f} & {rs_hung:.4f} & {r_vals[6]:.2f}\\% & {rs_cov:.2f}\\% \\\\"
        )

    drs_m50 = float(require_val(drs, ["mAP@50_mean"], "t8.drs.m50"))
    drs_m50_std = float(require_val(drs, ["mAP@50_std"], "t8.drs.m50_std"))
    drs_m95 = float(require_val(drs, ["mAP@50:95_mean"], "t8.drs.m95"))
    drs_m95_std = float(require_val(drs, ["mAP@50:95_std"], "t8.drs.m95_std"))
    drs_rec = float(require_val(drs, ["recall_pct_mean"], "t8.drs.rec"))
    drs_l2 = float(require_val(drs, ["delta_l2"], "t8.drs.l2"))
    drs_fghit = float(require_val(drs, ["fg_hit_pct_mean"], "t8.drs.fg_hit"))
    drs_hung = float(require_val(drs, ["hungarian_dist"], "t8.drs.hung"))
    drs_cov = float(require_val(drs, ["gt_object_cov_pct"], "t8.drs.cov"))

    headers = ["Condition / Strategy", "Seed", "mAP@50 (Rel.)", "mAP@50:95", "Recall", "Slot Disp. (L2)", "Hungarian Dist.", "Ref FG Hit", "GT Obj. Cov."]
    rows = [
        ["Clean Reference Baseline", "---", f"{cl_map50:.2f}% (100.0%)", f"{cl_map95:.2f}%", f"{cl_rec:.2f}%", "0.0000", f"{cl_hung:.4f}", f"{cl_fghit:.2f}%", f"{cl_cov:.2f}%"],
        ["Random Reassignment (Mean ± Std)", f"{n_seed_cnt} seeds", f"{rr_m50:.2f} ± {rr_m50_std:.2f}% ({rr_m50/_d_cl50*100:.1f}%)", f"{rr_m95:.2f} ± {rr_m95_std:.2f}%", f"{rr_rec:.2f} ± {rr_rec_std:.2f}%", f"{rr_l2:.4f}", f"{rr_hung:.4f}", f"{rr_fghit:.2f}%", f"{rr_cov:.2f}%"],
        ["Random Shift (Matched L2, Mean ± Std)", f"{n_seed_cnt} seeds", f"{rs_m50:.2f} ± {rs_m50_std:.2f}% ({rs_m50/_d_cl50*100:.1f}%)", f"{rs_m95:.2f} ± {rs_m95_std:.2f}%", f"{rs_rec:.2f} ± {rs_rec_std:.2f}%", f"{rs_l2:.4f} ± {rs_l2_std:.4f}", f"{rs_hung:.4f}", f"{rs_fghit:.2f}%", f"{rs_cov:.2f}%"],
        ["Decoupled Random Shift (Shift + Perm)", f"{n_seed_cnt} seeds", f"{drs_m50:.2f} ± {drs_m50_std:.2f}% ({drs_m50/_d_cl50*100:.1f}%)", f"{drs_m95:.2f} ± {drs_m95_std:.2f}%", f"{drs_rec:.2f}%", f"{drs_l2:.4f}", f"{drs_hung:.4f}", f"{drs_fghit:.2f}%", f"{drs_cov:.2f}%"],
        ["AQSM Spatial Misrouting", "---", f"{aq_m50:.2f}% ({aq_m50/_d_cl50*100:.1f}%)", f"{aq_m95:.2f}%", f"{aq_rec:.2f}%", f"{aq_l2:.4f}", f"{aq_hung:.4f}", f"{aq_fghit:.2f}%", f"{aq_cov:.2f}%"],
    ]

    n_eval_t8 = int(mis.get("n_images", 0))
    n_str_t8 = f", $N = {n_eval_t8:,}$ images" if n_eval_t8 > 0 else ""
    eff_res_t8 = int(get_img_size())
    tex = r"""\begin{table}[!htbp]
\centering
\caption{Controlled spatial misrouting and query perturbation on COCO val2017 ($""" + f"{eff_res_t8} \\times {eff_res_t8}" + r"""$""" + n_str_t8 + r"""). Evaluates decoder sensitivity to reference coordinates versus query assignment under pristine clean memory $M_{\mathrm{clean}}$ and content embeddings $H_{\mathrm{cand}}^{\mathrm{clean}}$. Distinguishes rank-paired slot displacement ($\text{Slot Disp.}$, $L_2 = """ + f"{rr_l2:.4f}" + r"""$ under pure slot permutation) from permutation-invariant Hungarian matching distance ($\text{Hungarian Dist.}$, $""" + f"{rr_hung:.4f}" + r"""$ for Random Reassignment). Evaluates empirical Ground-Truth Object Coverage ($\text{GT Obj. Cov.}$, $\text{Cov}_{\text{center}}$) alongside Anchor Foreground Hit rate ($\text{Ref FG Hit}$).}
\label{tab:tab8_controlled_misrouting}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccccc}
\toprule
\textbf{Condition / Strategy} & \textbf{Seed} & \textbf{mAP@50 (Rel.)} & \textbf{mAP@50:95} & \textbf{Recall} & \textbf{Slot Disp. ($L_2$)} & \textbf{Hungarian Dist.} & \textbf{Ref FG Hit} & \textbf{GT Obj. Cov.} \\
\midrule
Clean Reference Baseline & --- & """ + f"{cl_map50:.2f}\\% (100.0\\%)" + r""" & """ + f"{cl_map95:.2f}\\%" + r""" & """ + f"{cl_rec:.2f}\\%" + r""" & 0.0000 & """ + f"{cl_hung:.4f}" + r""" & """ + f"{cl_fghit:.2f}\\%" + r""" & """ + f"{cl_cov:.2f}\\%" + r""" \\
\midrule
Random Reassignment (Mean $\pm$ Std) & """ + f"{n_seed_cnt}" + r""" seeds & """ + f"{rr_m50:.2f} $\\pm$ {rr_m50_std:.2f}\\% ({rr_m50/_d_cl50*100:.1f}\\%)" + r""" & """ + f"{rr_m95:.2f} $\\pm$ {rr_m95_std:.2f}\\%" + r""" & """ + f"{rr_rec:.2f} $\\pm$ {rr_rec_std:.2f}\\%" + r""" & """ + f"{rr_l2:.4f}" + r""" & """ + f"{rr_hung:.4f}" + r""" & """ + f"{rr_fghit:.2f}\\%" + r""" & """ + f"{rr_cov:.2f}\\%" + r""" \\
""" + "\n".join(rr_seed_tex_lines) + r"""
\midrule
Random Shift (Matched $L_2$, Mean $\pm$ Std) & """ + f"{n_seed_cnt}" + r""" seeds & """ + f"{rs_m50:.2f} $\\pm$ {rs_m50_std:.2f}\\% ({rs_m50/_d_cl50*100:.1f}\\%)" + r""" & """ + f"{rs_m95:.2f} $\\pm$ {rs_m95_std:.2f}\\%" + r""" & """ + f"{rs_rec:.2f} $\\pm$ {rs_rec_std:.2f}\\%" + r""" & """ + f"{rs_l2:.4f} $\\pm$ {rs_l2_std:.4f}" + r""" & """ + f"{rs_hung:.4f}" + r""" & """ + f"{rs_fghit:.2f}\\%" + r""" & """ + f"{rs_cov:.2f}\\%" + r""" \\
""" + "\n".join(rs_seed_tex_lines) + r"""
\midrule
\textit{Decoupled Random Shift (Shift + Perm)} & """ + f"{n_seed_cnt}" + r""" seeds & """ + f"{drs_m50:.2f} $\\pm$ {drs_m50_std:.2f}\\% ({drs_m50/_d_cl50*100:.1f}\\%)" + r""" & """ + f"{drs_m95:.2f} $\\pm$ {drs_m95_std:.2f}\\%" + r""" & """ + f"{drs_rec:.2f}\\%" + r""" & """ + f"{drs_l2:.4f}" + r""" & """ + f"{drs_hung:.4f}" + r""" & """ + f"{drs_fghit:.2f}\\%" + r""" & """ + f"{drs_cov:.2f}\\%" + r""" \\
\midrule
\textbf{AQSM Spatial Misrouting} & --- & \textbf{""" + f"{aq_m50:.2f}\\% ({aq_m50/_d_cl50*100:.1f}\\%)" + r"""} & \textbf{""" + f"{aq_m95:.2f}\\%" + r"""} & \textbf{""" + f"{aq_rec:.2f}\\%" + r"""} & \textbf{""" + f"{aq_l2:.4f}" + r"""} & \textbf{""" + f"{aq_hung:.4f}" + r"""} & \textbf{""" + f"{aq_fghit:.2f}\\%" + r"""} & \textbf{""" + f"{aq_cov:.2f}\\%" + r"""} \\
\bottomrule
\end{tabular}%
}
\end{table}"""

    md = "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] + ["| " + " | ".join(r) + " |" for r in rows])
    return tex, md, headers, rows


# ==============================================================================
# TABLE 9: Matched-Budget Robustness Comparison across Epsilon
# ==============================================================================
def generate_table9(summary_dir: Path) -> Tuple[str, str, List[str], List[List[Any]]]:
    matched_data = safe_load_json([
        summary_dir / "matched_budget_sweep.json",
        summary_dir / "matched_budget_results.json",
        OUTPUTS_DIR / "summary_json" / "matched_budget_sweep.json",
    ])
    matched = require_val(matched_data, ["sweep_results"], "matched_budget.sweep_results")

    aq_sw = require_val(matched, ["aqsm"], "matched.aqsm")
    pgd_sw = require_val(matched, ["taskpgd"], "matched.taskpgd")
    van_sw = require_val(matched, ["vanishing"], "matched.vanishing")

    headers = ["Perturb. Budget (eps)", "Metric / Telemetry Indicator", "Standard Task PGD", "AQSM (Encoder Ranking)", "PGD Vanishing"]
    rows = []
    available_eps_keys = sorted(
        [k for k in aq_sw.keys() if k.startswith("eps_") and k in pgd_sw and k in van_sw],
        key=lambda x: int(x.split("_")[1]) if x.split("_")[1].isdigit() else 0,
    )
    if not available_eps_keys:
        raise KeyError("Không tìm thấy khóa eps_* chung nào giữa aqsm/taskpgd/vanishing trong matched_budget!")

    eps_tex_blocks: List[str] = []
    eps_caption_vals: List[str] = []
    for idx_e, eps_key in enumerate(available_eps_keys):
        eps_num_str = eps_key.split("_")[1]
        eps_num_i = int(eps_num_str) if eps_num_str.isdigit() else 16
        eps_f = eps_num_i / 255.0
        eps_label = f"{eps_num_i}/255 ({eps_f:.4f})"
        eps_caption_vals.append(f"{eps_num_i}/255")

        aq = require_val(aq_sw, [eps_key], f"matched.aqsm.{eps_key}")
        pgd = require_val(pgd_sw, [eps_key], f"matched.taskpgd.{eps_key}")
        van = require_val(van_sw, [eps_key], f"matched.vanishing.{eps_key}")

        rows.append([eps_label, "COCO mAP@50 (mAP@50:95)", f"{float(pgd['mAP@50']):.2f}% ({float(pgd['mAP@50:95']):.2f}%)", f"{float(aq['mAP@50']):.2f}% ({float(aq['mAP@50:95']):.2f}%)", f"{float(van['mAP@50']):.2f}% ({float(van['mAP@50:95']):.2f}%)"])
        rows.append(["", "Relative Drop / Retention", f"-{float(pgd['relative_drop_pct']):.2f}% ({100-float(pgd['relative_drop_pct']):.1f}% Ret.)", f"-{float(aq['relative_drop_pct']):.2f}% ({100-float(aq['relative_drop_pct']):.1f}% Ret.)", f"-{float(van['relative_drop_pct']):.2f}% ({100-float(van['relative_drop_pct']):.1f}% Ret.)"])

        is_last = idx_e == (len(available_eps_keys) - 1)
        if is_last:
            row1 = (
                rf"\multirow{{2}}{{*}}{{${eps_num_i}/255$ (${eps_f:.4f}$)}} & COCO mAP@50 (mAP@50:95) & "
                f"{float(pgd['mAP@50']):.2f}\\% ({float(pgd['mAP@50:95']):.2f}\\%) & "
                rf"\textbf{{{float(aq['mAP@50']):.2f}\% ({float(aq['mAP@50:95']):.2f}\%)}} & "
                f"{float(van['mAP@50']):.2f}\\% ({float(van['mAP@50:95']):.2f}\\%) \\\\"
            )
            row2 = (
                f"& Relative Drop / Retention & $-${float(pgd['relative_drop_pct']):.2f}\\% ({100-float(pgd['relative_drop_pct']):.1f}\\% Ret.) & "
                rf"\textbf{{$-${float(aq['relative_drop_pct']):.2f}\% ({100-float(aq['relative_drop_pct']):.1f}\% Ret.)}} & "
                f"$-${float(van['relative_drop_pct']):.2f}\\% ({100-float(van['relative_drop_pct']):.1f}\\% Ret.) \\\\"
            )
        else:
            row1 = (
                rf"\multirow{{2}}{{*}}{{${eps_num_i}/255$ (${eps_f:.4f}$)}} & COCO mAP@50 (mAP@50:95) & "
                f"{float(pgd['mAP@50']):.2f}\\% ({float(pgd['mAP@50:95']):.2f}\\%) & "
                f"{float(aq['mAP@50']):.2f}\\% ({float(aq['mAP@50:95']):.2f}\\%) & "
                f"{float(van['mAP@50']):.2f}\\% ({float(van['mAP@50:95']):.2f}\\%) \\\\"
            )
            row2 = (
                f"& Relative Drop / Retention & $-${float(pgd['relative_drop_pct']):.2f}\\% ({100-float(pgd['relative_drop_pct']):.1f}\\% Ret.) & "
                f"$-${float(aq['relative_drop_pct']):.2f}\\% ({100-float(aq['relative_drop_pct']):.1f}\\% Ret.) & "
                f"$-${float(van['relative_drop_pct']):.2f}\\% ({100-float(van['relative_drop_pct']):.1f}\\% Ret.) \\\\"
            )
        eps_tex_blocks.append(row1 + "\n" + row2)

    n_eval_t9 = int(matched_data.get("n_images", 0))
    n_str_t9 = f" on $N = {n_eval_t9:,}$ images" if n_eval_t9 > 0 else ""
    eff_res_t9 = int(get_img_size())
    eps_cap_str = ", ".join(eps_caption_vals)

    tex = r"""\begin{table}[!htbp]
\centering
\caption{Matched-budget mechanistic comparison on COCO val2017 at native $""" + f"{eff_res_t9} \\times {eff_res_t9}" + r"""$ resolution across $\epsilon \in \{""" + eps_cap_str + r"""\}$""" + n_str_t9 + r""". Demonstrates that Standard Task PGD degrades detection metrics across budgets, while AQSM accentuates encoder query eviction and spatial misrouting as an essential secondary transmission channel alongside encoder memory corruption.}
\label{tab:tab9_matched_budget}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llccc}
\toprule
\textbf{Perturb. Budget ($\epsilon$)} & \textbf{Metric / Telemetry Indicator} & \textbf{Standard Task PGD} & \textbf{AQSM (Encoder Ranking)} & \textbf{PGD Vanishing} \\
\midrule
""" + "\n\\midrule\n".join(eps_tex_blocks) + r"""
\bottomrule
\end{tabular}%
}
\end{table}"""

    md = "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] + ["| " + " | ".join(r) + " |" for r in rows])
    return tex, md, headers, rows


# ==============================================================================
# TABLE 10: Hardware Execution Latency & Measured Stability under Proposal Flooding
# ==============================================================================
def generate_table10(summary_dir: Path) -> Tuple[str, str, List[str], List[List[Any]]]:
    lat = safe_load_json([
        summary_dir / "edge_latency_results.json",
        summary_dir / "latency_profile.json",
        get_summary_json_dir() / "edge_latency_results.json",
    ])

    stg = require_val(lat, ["stages"], "lat.stages")
    bb = float(require_val(stg, ["Backbone", "mean_ms"], "lat.Backbone"))
    enc = float(require_val(stg, ["Hybrid_Encoder", "mean_ms"], "lat.Hybrid_Encoder"))
    qs = float(require_val(stg, ["Query_Selection", "mean_ms"], "lat.Query_Selection"))
    dec = float(require_val(stg, ["Deformable_Decoder", "mean_ms"], "lat.Deformable_Decoder"))
    tot = float(require_val(stg, ["End_to_End_Total", "mean_ms"], "lat.End_to_End_Total"))
    pct_blk = lat.get("percentage_breakdown", {})
    bb_pct = float(pct_blk.get("Backbone_pct", bb / max(tot, 1e-6) * 100.0))
    enc_pct = float(pct_blk.get("Hybrid_Encoder_pct", enc / max(tot, 1e-6) * 100.0))
    qs_pct = float(pct_blk.get("Query_Selection_pct", qs / max(tot, 1e-6) * 100.0))
    dec_pct = float(pct_blk.get("Deformable_Decoder_pct", dec / max(tot, 1e-6) * 100.0))

    model_disp = str(lat.get("model", "RT-DETR")).split("/")[-1]
    dev_type = str(lat.get("device_type", "unknown")).upper()
    hw_meta = lat.get("hardware_metadata", {})
    hw_desc = str(hw_meta.get("gpu_name") or hw_meta.get("processor") or dev_type)
    n_runs = int(lat.get("n_runs", 0))

    adv_comp = lat.get("adversarial_latency_comparison", {})
    clean_adv_blk = adv_comp.get("Clean_Baseline", {})
    clean_ref_ms = float(clean_adv_blk.get("mean_ms", tot))

    headers = [
        "Stage / Operational Condition",
        "Mean ± Std (ms)",
        "P95 (ms)",
        "Throughput (FPS)",
        "Active Boxes (tau >= 0.35)",
        "Share / Latency Delta",
    ]
    rows: List[List[Any]] = [
        ["Stage 1: Backbone", f"{bb:.3f} ± {float(stg['Backbone'].get('std_ms', 0.0)):.3f}", f"{float(stg['Backbone'].get('p95_ms', bb)):.3f}", "---", "---", f"{bb_pct:.1f}% of E2E"],
        ["Stage 2: Hybrid Encoder", f"{enc:.3f} ± {float(stg['Hybrid_Encoder'].get('std_ms', 0.0)):.3f}", f"{float(stg['Hybrid_Encoder'].get('p95_ms', enc)):.3f}", "---", "---", f"{enc_pct:.1f}% of E2E"],
        ["Stage 3: Query Selection", f"{qs:.3f} ± {float(stg['Query_Selection'].get('std_ms', 0.0)):.3f}", f"{float(stg['Query_Selection'].get('p95_ms', qs)):.3f}", "---", "---", f"{qs_pct:.1f}% of E2E"],
        ["Stage 4: Deformable Decoder", f"{dec:.3f} ± {float(stg['Deformable_Decoder'].get('std_ms', 0.0)):.3f}", f"{float(stg['Deformable_Decoder'].get('p95_ms', dec)):.3f}", "---", "---", f"{dec_pct:.1f}% of E2E"],
        ["End-to-End Clean Stage Sum", f"{tot:.3f} ± {float(stg['End_to_End_Total'].get('std_ms', 0.0)):.3f}", f"{float(stg['End_to_End_Total'].get('p95_ms', tot)):.3f}", f"{float(lat.get('throughput_fps', 1000.0 / max(tot, 1e-4))):.2f}", "---", "100.0% Reference"],
    ]

    cond_labels = [
        ("Clean_Baseline", "Condition: Clean Baseline"),
        ("AQSM_Attack", "Condition: AQSM Attack (eps=16/255)"),
        ("Task_PGD", "Condition: Standard Task PGD (eps=16/255)"),
        ("RT_DETR_Flooding_Sponge", "Condition: Proposal-Flooding Sponge"),
    ]
    tex_cond_rows: List[str] = []
    for c_key, c_disp in cond_labels:
        cb = adv_comp.get(c_key)
        if isinstance(cb, dict) and "mean_ms" in cb:
            c_mean = float(cb["mean_ms"])
            c_std = float(cb.get("std_ms", 0.0))
            c_p95 = float(cb.get("p95_ms", c_mean))
            c_fps = float(cb.get("throughput_fps", 1000.0 / max(c_mean, 1e-4)))
            c_act = cb.get("active_boxes_tau35", "---")
            c_act_str = f"{float(c_act):.2f}" if isinstance(c_act, (int, float)) else str(c_act)
            dt_ms = c_mean - clean_ref_ms
            dt_pct = (dt_ms / max(clean_ref_ms, 1e-6)) * 100.0
            note_str = "Baseline" if c_key == "Clean_Baseline" else f"Delta t = {dt_ms:+.3f} ms ({dt_pct:+.2f}%)"
            rows.append([c_disp, f"{c_mean:.3f} ± {c_std:.3f}", f"{c_p95:.3f}", f"{c_fps:.2f}", c_act_str, note_str])
            tex_note = "Baseline" if c_key == "Clean_Baseline" else f"$\\Delta t = {dt_ms:+.3f}$ ms ({dt_pct:+.2f}\\%)"
            tex_cond_rows.append(
                f"{c_disp} & ${c_mean:.3f} \\pm {c_std:.3f}$ & {c_p95:.3f} & {c_fps:.2f} & {c_act_str} & {tex_note} \\\\"
            )

    cond_tex_block = ("\n\\midrule\n" + "\n".join(tex_cond_rows)) if tex_cond_rows else ""

    tex = r"""\begin{table}[!htbp]
\centering
\caption{Hardware execution latency stage breakdown and proposal-flooding stability on \texttt{""" + model_disp.replace("_", r"\_") + r"""} (Device: \texttt{""" + hw_desc.replace("_", r"\_") + r"""}, """ + f"$N_{{\\mathrm{{runs}}}} = {n_runs}$" + r""", $640 \times 640$). Reports measured inference latency across pipeline stages and adversarial attack conditions.}
\label{tab:tab10_latency_sponge}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lccccc}
\toprule
\textbf{Stage / Operational Condition} & \textbf{Mean $\pm$ Std (ms)} & \textbf{P95 (ms)} & \textbf{Throughput (FPS)} & \textbf{Active Boxes ($\tau \ge 0.35$)} & \textbf{Share / Latency Delta} \\
\midrule
Stage 1: Backbone & """ + f"${bb:.3f} \\pm {float(stg['Backbone'].get('std_ms', 0.0)):.3f}$ & {float(stg['Backbone'].get('p95_ms', bb)):.3f}" + r""" & --- & --- & """ + f"{bb_pct:.1f}\\% of E2E" + r""" \\
Stage 2: Hybrid Encoder & """ + f"${enc:.3f} \\pm {float(stg['Hybrid_Encoder'].get('std_ms', 0.0)):.3f}$ & {float(stg['Hybrid_Encoder'].get('p95_ms', enc)):.3f}" + r""" & --- & --- & """ + f"{enc_pct:.1f}\\% of E2E" + r""" \\
Stage 3: Query Selection & """ + f"${qs:.3f} \\pm {float(stg['Query_Selection'].get('std_ms', 0.0)):.3f}$ & {float(stg['Query_Selection'].get('p95_ms', qs)):.3f}" + r""" & --- & --- & """ + f"{qs_pct:.1f}\\% of E2E" + r""" \\
Stage 4: Deformable Decoder & """ + f"${dec:.3f} \\pm {float(stg['Deformable_Decoder'].get('std_ms', 0.0)):.3f}$ & {float(stg['Deformable_Decoder'].get('p95_ms', dec)):.3f}" + r""" & --- & --- & """ + f"{dec_pct:.1f}\\% of E2E" + r""" \\
End-to-End Clean Stage Sum & """ + f"${tot:.3f} \\pm {float(stg['End_to_End_Total'].get('std_ms', 0.0)):.3f}$ & {float(stg['End_to_End_Total'].get('p95_ms', tot)):.3f} & {float(lat.get('throughput_fps', 1000.0 / max(tot, 1e-4))):.2f}" + r""" & --- & 100.0\% Reference \\""" + cond_tex_block + r"""
\bottomrule
\end{tabular}%
}
\end{table}"""

    md = "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] + ["| " + " | ".join(r) + " |" for r in rows])
    return tex, md, headers, rows


# ==============================================================================
# TABLE 11: Empirical Defense Evaluation (ESM & Adaptive Evasion)
# ==============================================================================
def generate_table11(summary_dir: Path) -> Tuple[str, str, List[str], List[List[Any]]]:
    esm = safe_load_json([
        summary_dir / "defense_esm_evaluation.json",
        summary_dir / "defense_evaluation_summary.json",
        OUTPUTS_DIR / "summary_json" / "defense_esm_evaluation.json",
    ])
    atks = require_val(esm, ["attacks"], "esm.attacks")
    adapt = require_val(esm, ["adaptive_evasion"], "esm.adaptive_evasion")
    matched_ref = safe_load_json([
        summary_dir / "matched_budget_sweep.json",
        summary_dir / "matched_budget_results.json",
    ], default_dict={})
    cl_raw = matched_ref.get("clean_baseline", {}).get("mAP@50")
    sw_ref = matched_ref.get("sweep_results", {})
    aq_raw = sw_ref.get("aqsm", {}).get("eps_16", {}).get("mAP@50")
    pgd_raw = sw_ref.get("taskpgd", {}).get("eps_16", {}).get("mAP@50")
    van_raw = sw_ref.get("vanishing", {}).get("eps_16", {}).get("mAP@50")
    cl_m50_str = f"{float(cl_raw):.2f}%" if cl_raw is not None else "---"
    aq_m50_str = f"{float(aq_raw):.2f}%" if aq_raw is not None else "---"
    pgd_m50_str = f"{float(pgd_raw):.2f}%" if pgd_raw is not None else "---"
    van_m50_str = f"{float(van_raw):.2f}%" if van_raw is not None else "---"

    aq_tpr = float(require_val(atks, ["aqsm", "tpr_detection_rate_pct"], "esm.aqsm.tpr"))
    aq_auc = float(require_val(atks, ["aqsm", "auroc"], "esm.aqsm.auroc"))
    aq_score = float(require_val(atks, ["aqsm", "mean_anomaly_score"], "esm.aqsm.score"))

    pgd_tpr = float(require_val(atks, ["taskpgd", "tpr_detection_rate_pct"], "esm.pgd.tpr"))
    pgd_auc = float(require_val(atks, ["taskpgd", "auroc"], "esm.pgd.auroc"))
    pgd_score = float(require_val(atks, ["taskpgd", "mean_anomaly_score"], "esm.pgd.score"))

    van_tpr = float(require_val(atks, ["vanishing", "tpr_detection_rate_pct"], "esm.van.tpr"))
    van_auc = float(require_val(atks, ["vanishing", "auroc"], "esm.van.auroc"))
    van_score = float(require_val(atks, ["vanishing", "mean_anomaly_score"], "esm.van.score"))

    clean_fpr = float(require_val(esm, ["empirical_clean_fpr_pct"], "esm.clean_fpr"))
    target_fpr = float(require_val(esm, ["target_fpr_pct"], "esm.target_fpr"))
    union_bound = float(esm.get("union_bound_pct", target_fpr * 2.0))
    clean_anom_raw = esm.get("clean_mean_anomaly_score")
    clean_anom_str = f"{float(clean_anom_raw):.4f}" if clean_anom_raw is not None else "---"
    n_test_t11 = int(esm.get("n_test", 0))
    n_cohort_t11 = int(matched_ref.get("n_images", n_test_t11 + int(esm.get("n_cal", 0))))

    def _extract_bypass(lam_key: str) -> float:
        """Trích xuất tỷ lệ bypass vượt qua monitor (raw_monitor_bypass_pct hoặc 100 - TPR)."""
        lam_data = require_val(adapt, [lam_key], f"adapt.{lam_key}")
        if "raw_monitor_bypass_pct" in lam_data and lam_data["raw_monitor_bypass_pct"] is not None:
            return float(lam_data["raw_monitor_bypass_pct"])
        if "detected_tpr_pct" in lam_data and lam_data["detected_tpr_pct"] is not None:
            return 100.0 - float(lam_data["detected_tpr_pct"])
        raise KeyError(f"[STRICT PROVENANCE] Thiếu trường bypass/TPR trong {lam_key}")

    def _extract_tpr(lam_key: str) -> float:
        """Trích xuất tỷ lệ bị phát hiện TPR."""
        lam_data = require_val(adapt, [lam_key], f"adapt.{lam_key}")
        return float(require_val(lam_data, ["detected_tpr_pct"], f"adapt.{lam_key}.detected_tpr_pct"))

    def _extract_post_map(lam_key: str) -> str:
        lam_data = adapt.get(lam_key, {})
        val = lam_data.get("post_attack_map50_pct")
        if val is not None:
            return f"{float(val):.2f}%"
        return "N/A"

    tpr_02, bypass_02, pmap_02 = _extract_tpr("lambda_0.2"), _extract_bypass("lambda_0.2"), _extract_post_map("lambda_0.2")
    tpr_05, bypass_05, pmap_05 = _extract_tpr("lambda_0.5"), _extract_bypass("lambda_0.5"), _extract_post_map("lambda_0.5")
    tpr_10, bypass_10, pmap_10 = _extract_tpr("lambda_1.0"), _extract_bypass("lambda_1.0"), _extract_post_map("lambda_1.0")
    tpr_20, bypass_20, pmap_20 = _extract_tpr("lambda_2.0"), _extract_bypass("lambda_2.0"), _extract_post_map("lambda_2.0")

    headers = ["Evaluated Condition", "Detection TPR (%)", "Empirical FPR (%)", "AUROC", "Mean Anomaly Score", "Post-Attack mAP@50 (%)"]
    rows = [
        ["Clean Test Baseline", "---", f"{clean_fpr:.2f}% [Union Bound {union_bound:.1f}%]", "---", clean_anom_str, cl_m50_str],
        ["Stationary AQSM (eps = 16/255)", f"{aq_tpr:.2f}%", f"{clean_fpr:.2f}%", f"{aq_auc:.4f}", f"{aq_score:.4f}", aq_m50_str],
        ["Stationary Standard Task PGD (eps = 16/255)", f"{pgd_tpr:.2f}%", f"{clean_fpr:.2f}%", f"{pgd_auc:.4f}", f"{pgd_score:.4f}", pgd_m50_str],
        ["Stationary Query Vanishing (eps = 16/255)", f"{van_tpr:.2f}%", f"{clean_fpr:.2f}%", f"{van_auc:.4f}", f"{van_score:.4f}", van_m50_str],
        ["Adaptive Evasion (lambda = 0.2)", f"{tpr_02:.2f}%", "---", "Degenerate*", f"Bypass: {bypass_02:.2f}%", pmap_02],
        ["Adaptive Evasion (lambda = 0.5)", f"{tpr_05:.2f}%", "---", "Degenerate*", f"Bypass: {bypass_05:.2f}%", pmap_05],
        ["Adaptive Evasion (lambda = 1.0)", f"{tpr_10:.2f}%", "---", "Degenerate*", f"Bypass: {bypass_10:.2f}%", pmap_10],
        ["Adaptive Evasion (lambda = 2.0)", f"{tpr_20:.2f}%", "---", "Degenerate*", f"Bypass: {bypass_20:.2f}%", pmap_20],
    ]

    pmap_vals = [float(adapt[lk]["post_attack_map50_pct"]) for lk in ("lambda_0.2", "lambda_0.5", "lambda_1.0", "lambda_2.0") if adapt.get(lk, {}).get("post_attack_map50_pct") is not None]
    adapt_cap_note = f"$\\text{{mAP@50}} \\le {max(pmap_vals):.2f}\\%$" if pmap_vals else "severe task degradation"

    clean_vs_note = f" vs.\\ ${float(cl_raw):.2f}\\%$ clean" if cl_raw is not None else ""
    eff_res_t11 = int(get_img_size())

    tex = r"""\begin{table}[!htbp]
\centering
\caption{Empirical detection performance of the Encoder Selection Monitor (ESM) on independent COCO val2017 test images ($N_{\text{test}} = """ + f"{n_test_t11:,}" + r"""$) and white-box Carlini-Tram\`er adaptive evasion boundary at native $""" + f"{eff_res_t11} \\times {eff_res_t11}" + r"""$ resolution. Demonstrates that while ESM flags stationary AQSM ($\text{TPR} = """ + f"{aq_tpr:.2f}\\%" + r""", \text{AUROC} = """ + f"{aq_auc:.4f}" + r"""$), a white-box adaptive adversary ($\lambda_{\mathrm{evade}} \ge 0.2$) successfully evades runtime detection ($\text{TPR} \le """ + f"{max(tpr_02, tpr_05, tpr_10, tpr_20):.2f}\\%" + r"""$, bypass rate $\ge """ + f"{min(bypass_02, bypass_05, bypass_10, bypass_20):.2f}\\%" + r"""$) while sustaining """ + adapt_cap_note + clean_vs_note + r""".}
\label{tab:tab11_defense_results}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lccccc}
\toprule
\textbf{Evaluated Condition} & \textbf{Detection TPR (\%)} & \textbf{Empirical FPR (\%)} & \textbf{AUROC} & \textbf{Mean Anomaly Score} & \textbf{Post-Attack mAP@50 (\%)\textsuperscript{\textdagger}} \\
\midrule
Clean Test Baseline & --- & """ + f"{clean_fpr:.2f}\\% [Union Bound {union_bound:.1f}\\%]" + r""" & --- & """ + clean_anom_str + r""" & """ + cl_m50_str.replace("%", r"\%") + r""" \\
Stationary AQSM ($\epsilon = 16/255$) & \textbf{""" + f"{aq_tpr:.2f}\\%" + r"""} & """ + f"{clean_fpr:.2f}\\%" + r""" & \textbf{""" + f"{aq_auc:.4f}" + r"""} & """ + f"{aq_score:.4f}" + r""" & """ + aq_m50_str.replace("%", r"\%") + r""" \\
Stationary Standard Task PGD ($\epsilon = 16/255$) & """ + f"{pgd_tpr:.2f}\\%" + r""" & """ + f"{clean_fpr:.2f}\\%" + r""" & """ + f"{pgd_auc:.4f}" + r""" & """ + f"{pgd_score:.4f}" + r""" & """ + pgd_m50_str.replace("%", r"\%") + r""" \\
Stationary Query Vanishing ($\epsilon = 16/255$) & """ + f"{van_tpr:.2f}\\%" + r""" & """ + f"{clean_fpr:.2f}\\%" + r""" & """ + f"{van_auc:.4f}" + r""" & """ + f"{van_score:.4f}" + r""" & """ + van_m50_str.replace("%", r"\%") + r""" \\
\midrule
\textbf{Adaptive Evasion ($\lambda_{\mathrm{evade}} = 0.2$)} & """ + f"{tpr_02:.2f}\\%" + r""" & --- & Degenerate\textsuperscript{*} & """ + f"Bypass: {bypass_02:.2f}\\%" + r""" & """ + pmap_02.replace("%", r"\%") + r""" \\
\textbf{Adaptive Evasion ($\lambda_{\mathrm{evade}} = 0.5$)} & \textbf{""" + f"{tpr_05:.2f}\\%" + r"""} & --- & Degenerate\textsuperscript{*} & \textbf{""" + f"Bypass: {bypass_05:.2f}\\%" + r"""} & \textbf{""" + pmap_05.replace("%", r"\%") + r"""} \\
\textbf{Adaptive Evasion ($\lambda_{\mathrm{evade}} = 1.0$)} & """ + f"{tpr_10:.2f}\\%" + r""" & --- & Degenerate\textsuperscript{*} & """ + f"Bypass: {bypass_10:.2f}\\%" + r""" & """ + pmap_10.replace("%", r"\%") + r""" \\
\textbf{Adaptive Evasion ($\lambda_{\mathrm{evade}} = 2.0$)} & """ + f"{tpr_20:.2f}\\%" + r""" & --- & Degenerate\textsuperscript{*} & """ + f"Bypass: {bypass_20:.2f}\\%" + r""" & """ + pmap_20.replace("%", r"\%") + r""" \\
\bottomrule
\multicolumn{6}{l}{\footnotesize{\textdagger Stationary baseline mAP values reference the evaluation cohort ($N = """ + f"{n_cohort_t11:,}" + r"""$); adaptive evasion evaluated on the independent test split ($N_{\text{test}} = """ + f"{n_test_t11:,}" + r"""$).}} \\
\multicolumn{6}{l}{\footnotesize{*At the fixed deployment threshold $\tau_{\mathrm{fused}}$, adaptive evasion suppresses detection TPR below clean FPR ($""" + f"{clean_fpr:.2f}\\%" + r"""$), achieving a bypass rate $\ge """ + f"{min(bypass_02, bypass_05, bypass_10, bypass_20):.2f}\\%" + r"""$ without establishing a valid monotonic ROC curve.}} \\
\end{tabular}%
}
\end{table}"""

    md = "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] + ["| " + " | ".join(r) + " |" for r in rows])
    return tex, md, headers, rows


# ==============================================================================
# TABLE 12: Empirical Validation of Architectural Hardening (SQA)
# ==============================================================================
def generate_table12(summary_dir: Path) -> Tuple[str, str, List[str], List[List[Any]]]:
    sqa_data = safe_load_json([
        summary_dir / "hardening_evaluation_results.json",
        summary_dir / "defense_esm_evaluation.json",
        OUTPUTS_DIR / "summary_json" / "hardening_evaluation_results.json",
    ])
    sqa = sqa_data.get("results", sqa_data.get("sqa_hardening", sqa_data))

    cl_dyn = require_val(sqa, ["clean_baseline", "default_dynamic_only"], "sqa.clean.dyn")
    cl_sqa = require_val(sqa, ["clean_baseline", "hardened_static_augmented"], "sqa.clean.sqa")

    aq_dyn = require_val(sqa, ["aqsm_attack_eps16", "default_dynamic_only"], "sqa.aqsm.dyn")
    aq_sqa = require_val(sqa, ["aqsm_attack_eps16", "hardened_static_augmented"], "sqa.aqsm.sqa")

    pgd_dyn = require_val(sqa, ["taskpgd_attack_eps16", "default_dynamic_only"], "sqa.pgd.dyn")
    pgd_sqa = require_val(sqa, ["taskpgd_attack_eps16", "hardened_static_augmented"], "sqa.pgd.sqa")

    clean_dyn_m50 = float(require_val(cl_dyn, ["mAP@50"], "cl_dyn.m50"))
    clean_dyn_m95 = float(require_val(cl_dyn, ["mAP@50:95"], "cl_dyn.m95"))
    clean_dyn_rec = float(require_val(cl_dyn, ["recall_pct"], "cl_dyn.rec"))

    clean_pen = float(require_val(cl_sqa, ["clean_penalty_pp"], "cl_sqa.penalty"))
    clean_sqa_m50 = float(require_val(cl_sqa, ["mAP@50"], "cl_sqa.m50"))
    clean_sqa_m95 = float(require_val(cl_sqa, ["mAP@50:95"], "cl_sqa.m95"))
    clean_sqa_rec = float(require_val(cl_sqa, ["recall_pct"], "cl_sqa.rec"))

    aq_base_m50 = float(require_val(aq_dyn, ["mAP@50"], "aq_dyn.m50"))
    aq_base_m95 = float(require_val(aq_dyn, ["mAP@50:95"], "aq_dyn.m95"))
    aq_base_rec = float(require_val(aq_dyn, ["recall_pct"], "aq_dyn.rec"))
    aq_base_ret = float(aq_dyn.get("retention_pct", round((aq_base_m50 / max(clean_dyn_m50, 1e-6)) * 100.0, 1)))
    aq_m50 = float(require_val(aq_sqa, ["mAP@50"], "aq_sqa.m50"))
    aq_m95 = float(require_val(aq_sqa, ["mAP@50:95"], "aq_sqa.m95"))
    aq_rec = float(require_val(aq_sqa, ["recall_pct"], "aq_sqa.rec"))
    aq_recovery = aq_m50 - aq_base_m50

    lat_blk = sqa["edge_latency_jetson_orin_nano_ms"]
    lat_dyn_ms = float(lat_blk["default_dynamic_only"])
    lat_sqa_ms = float(lat_blk["hardened_static_augmented"])
    lat_ovh_pct = float(lat_blk["overhead_pct"])

    n_dyn_t12 = int(sqa_data.get("num_dynamic_queries", 300))
    n_stat_t12 = int(sqa_data.get("num_static_anchors", 30))
    n_tot_t12 = n_dyn_t12 + n_stat_t12
    model_t12 = str(sqa_data.get("model", "RT-DETR")).split("/")[-1]
    model_t12_tex = model_t12.replace("_", r"\_")

    has_pgd_sqa = isinstance(pgd_dyn, dict) and ("mAP@50" in pgd_dyn) and isinstance(pgd_sqa, dict) and ("mAP@50" in pgd_sqa)
    if has_pgd_sqa:
        pgd_base_m50 = float(pgd_dyn["mAP@50"])
        pgd_base_m95 = float(pgd_dyn.get("mAP@50:95", 0.0))
        pgd_base_rec = float(pgd_dyn.get("recall_pct", 0.0))
        pgd_base_ret = round((pgd_base_m50 / max(clean_dyn_m50, 1e-6)) * 100.0, 1)
        pgd_m50 = float(pgd_sqa["mAP@50"])
        pgd_m95 = float(pgd_sqa.get("mAP@50:95", 0.0))
        pgd_rec = float(pgd_sqa.get("recall_pct", 0.0))
        pgd_recovery = pgd_m50 - pgd_base_m50
        pgd_dyn_md = [f"Task PGD: Standard {model_t12} ({n_dyn_t12} dynamic)", f"{pgd_base_m50:.2f}% ({pgd_base_ret:.1f}% Ret.)", f"{pgd_base_m95:.2f}%", f"{pgd_base_rec:.2f}%", f"{lat_dyn_ms:.2f} ms"]
        pgd_sqa_md = [f"Task PGD: Hardened SQA Prototype ({n_dyn_t12} dyn. + {n_stat_t12} static)", f"{pgd_m50:.2f}% ({pgd_recovery:+.2f} pp)", f"{pgd_m95:.2f}%", f"{pgd_rec:.2f}%", f"{lat_sqa_ms:.2f} ms ({lat_ovh_pct:+.2f}%)"]
        pgd_dyn_tex = f"\\quad Standard {model_t12_tex} (${n_dyn_t12}$ dynamic) & {pgd_base_m50:.2f}\\% ({pgd_base_ret:.1f}\\% Ret.) & {pgd_base_m95:.2f}\\% & {pgd_base_rec:.2f}\\% & {lat_dyn_ms:.2f} ms \\\\"
        pgd_sqa_tex = f"\\quad Hardened SQA Prototype (${n_dyn_t12}$ dyn. $+ {n_stat_t12}$ static) & {pgd_m50:.2f}\\% ({pgd_recovery:+.2f} pp) & {pgd_m95:.2f}\\% & {pgd_rec:.2f}\\% & {lat_sqa_ms:.2f} ms ({lat_ovh_pct:+.2f}\\%) \\\\"
    else:
        pgd_dyn_md = [f"Task PGD: Standard {model_t12} ({n_dyn_t12} dynamic)", "---", "---", "---", f"{lat_dyn_ms:.2f} ms (Not evaluated)"]
        pgd_sqa_md = [f"Task PGD: Hardened SQA Prototype ({n_dyn_t12} dyn. + {n_stat_t12} static)", "---", "---", "---", f"{lat_sqa_ms:.2f} ms (Not evaluated)"]
        pgd_dyn_tex = f"\\quad Standard {model_t12_tex} (${n_dyn_t12}$ dynamic) & --- & --- & --- & {lat_dyn_ms:.2f} ms (Not evaluated) \\\\"
        pgd_sqa_tex = f"\\quad Hardened SQA Prototype (${n_dyn_t12}$ dyn. $+ {n_stat_t12}$ static) & --- & --- & --- & {lat_sqa_ms:.2f} ms (Not evaluated) \\\\"

    # Trích xuất động các nhóm đối chứng và tấn công thích ứng SQA (không fallback hardcode)
    ctrl_d = sqa.get("controls", {}) or sqa_data.get("controls", {}) or sqa_data.get("strategy_ablation", {})
    dyn330 = ctrl_d.get("dynamic_top330") or ctrl_d["dynamic_330_topk"]
    dyn330_m50 = float(require_any_key(dyn330, ["mAP@50", "map50"], "dyn330.m50"))
    dyn330_m95 = float(require_any_key(dyn330, ["mAP@50:95", "map50_95"], "dyn330.m95"))
    dyn330_rec = float(require_val(dyn330, ["recall_pct"], "dyn330.rec"))
    dyn330_recov = float(dyn330.get("recovery_pp", dyn330.get("recovery_vs_300_pp", dyn330_m50 - aq_base_m50)))

    rand30 = ctrl_d.get("random30_static") or ctrl_d.get("random_30_static") or ctrl_d["random"]
    rand30_m50 = float(require_any_key(rand30, ["mAP@50", "map50", "aqsm_recovered_map50"], "rand30.m50"))
    rand30_m95 = float(require_any_key(rand30, ["mAP@50:95", "map50_95", "aqsm_recovered_map50_95"], "rand30.m95"))
    rand30_rec = float(require_val(rand30, ["recall_pct"], "rand30.rec"))
    rand30_recov = float(rand30.get("recovery_pp", rand30.get("recovery_vs_300_pp", rand30_m50 - aq_base_m50)))

    adapt = ctrl_d.get("adaptive_evasion_sqa") or sqa_data.get("adaptive_threat_boundary") or sqa_data["adaptive_attack_carlini_tramer"]
    adapt_m50 = float(require_any_key(adapt, ["mAP@50", "adaptive_map50", "adaptive_post_sqa_map50"], "adapt.m50"))
    adapt_m95 = float(require_any_key(adapt, ["mAP@50:95", "adaptive_map50_95", "adaptive_post_sqa_map50_95"], "adapt.m95"))
    adapt_rec = float(require_any_key(adapt, ["recall_pct", "adaptive_recall_pct"], "adapt.rec"))
    adapt_bypass = float(require_any_key(adapt, ["bypass_pct", "evasion_success_pct"], "adapt.bypass"))
    adapt_delta = adapt_m50 - aq_m50

    headers = ["Model Architecture / Evaluated Condition", "COCO mAP@50", "COCO mAP@50:95", "Detection Recall", "Measured Latency / Note"]
    rows = [
        [f"Clean: Standard {model_t12} ({n_dyn_t12} dynamic)", f"{clean_dyn_m50:.2f}%", f"{clean_dyn_m95:.2f}%", f"{clean_dyn_rec:.2f}%", f"{lat_dyn_ms:.2f} ms (Ref.)"],
        [f"Clean: Hardened SQA Prototype ({n_dyn_t12} dyn. + {n_stat_t12} static)", f"{clean_sqa_m50:.2f}% ({clean_pen:+.2f} pp)", f"{clean_sqa_m95:.2f}%", f"{clean_sqa_rec:.2f}%", f"{lat_sqa_ms:.2f} ms ({lat_ovh_pct:+.2f}%)"],
        [f"AQSM: Standard {model_t12} ({n_dyn_t12} dynamic)", f"{aq_base_m50:.2f}% ({aq_base_ret:.1f}% Ret.)", f"{aq_base_m95:.2f}%", f"{aq_base_rec:.2f}%", f"{lat_dyn_ms:.2f} ms"],
        [f"AQSM Control 1: Dynamic Top-{n_tot_t12} (+{n_stat_t12} dynamic queries)", f"{dyn330_m50:.2f}% ({dyn330_recov:+.2f} pp)", f"{dyn330_m95:.2f}%", f"{dyn330_rec:.2f}%", "Eviction unmitigated"],
        [f"AQSM Control 2: Random-{n_stat_t12} Static Queries", f"{rand30_m50:.2f}% ({rand30_recov:+.2f} pp)", f"{rand30_m95:.2f}%", f"{rand30_rec:.2f}%", "Partial spatial retention"],
        [f"AQSM Proposed: Uniform Grid-{n_stat_t12} SQA Prototype", f"{aq_m50:.2f}% ({aq_recovery:+.2f} pp)", f"{aq_m95:.2f}%", f"{aq_rec:.2f}%", f"{lat_sqa_ms:.2f} ms (Canvas dispersion)"],
        ["AQSM Adaptive Evasion: White-box Joint Hungarian on SQA", f"{adapt_m50:.2f}% ({adapt_delta:+.2f} pp)", f"{adapt_m95:.2f}%", f"{adapt_rec:.2f}%", f"Evasion bypass {adapt_bypass:.1f}%"],
        pgd_dyn_md,
        pgd_sqa_md,
    ]

    eff_res_t12 = int(get_img_size())
    tex = r"""\begin{table}[!htbp]
\centering
\caption{Empirical evaluation of Architectural Hardening via Static Query Augmentation ($N_{\mathrm{static}} = """ + f"{n_stat_t12}" + r"""$), controls, and adaptive evasion boundary on COCO val2017 ($""" + f"{eff_res_t12} \\times {eff_res_t12}" + r"""$, model """ + model_t12.replace("_", r"\_") + r"""). Demonstrates partial recovery under stationary AQSM ($""" + f"{aq_recovery:+.2f}" + r"""$ pp, $""" + f"{aq_m50:.2f}" + r"""\%$) via canvas spatial dispersion, while exposing vulnerability under white-box adaptive evasion ($""" + f"{adapt_m50:.2f}" + r"""\%$).}
\label{tab:tab12_hardening_validation}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccc}
\toprule
\textbf{Model Architecture / Evaluated Condition} & \textbf{COCO mAP@50} & \textbf{COCO mAP@50:95} & \textbf{Detection Recall} & \textbf{Measured Latency / Note} \\
\midrule
\textbf{Clean Baseline Inference:} & & & & \\
\quad Standard """ + model_t12.replace("_", r"\_") + f" (${n_dyn_t12}$ dynamic)" + r""" & """ + f"{clean_dyn_m50:.2f}\\%" + r""" & """ + f"{clean_dyn_m95:.2f}\\%" + r""" & """ + f"{clean_dyn_rec:.2f}\\%" + r""" & """ + f"{lat_dyn_ms:.2f} ms (Ref.)" + r""" \\
\quad Hardened SQA Prototype (""" + f"${n_dyn_t12}$ dyn. $+ {n_stat_t12}$ static" + r""") & """ + f"{clean_sqa_m50:.2f}\\% ({clean_pen:+.2f} pp)" + r""" & """ + f"{clean_sqa_m95:.2f}\\%" + r""" & """ + f"{clean_sqa_rec:.2f}\\%" + r""" & """ + f"{lat_sqa_ms:.2f} ms ({lat_ovh_pct:+.2f}\\%)" + r""" \\
\midrule
\textbf{Stationary AQSM Attack ($\epsilon = 16/255$):} & & & & \\
\quad Standard """ + model_t12.replace("_", r"\_") + f" (${n_dyn_t12}$ dynamic)" + r""" & """ + f"{aq_base_m50:.2f}\\% ({aq_base_ret:.1f}\\% Ret.)" + r""" & """ + f"{aq_base_m95:.2f}\\%" + r""" & """ + f"{aq_base_rec:.2f}\\%" + r""" & """ + f"{lat_dyn_ms:.2f} ms" + r""" \\
\quad Control 1: Dynamic Top-""" + f"{n_tot_t12} ($+{n_stat_t12}$ dynamic ranking)" + r""" & """ + f"{dyn330_m50:.2f}\\% ({dyn330_recov:+.2f} pp)" + r""" & """ + f"{dyn330_m95:.2f}\\%" + r""" & """ + f"{dyn330_rec:.2f}\\%" + r""" & Eviction unmitigated \\
\quad Control 2: Random-""" + f"{n_stat_t12}" + r""" Static Anchors & """ + f"{rand30_m50:.2f}\\% ({rand30_recov:+.2f} pp)" + r""" & """ + f"{rand30_m95:.2f}\\%" + r""" & """ + f"{rand30_rec:.2f}\\%" + r""" & Partial spatial retention \\
\quad \textbf{Proposed SQA: Uniform Grid-""" + f"{n_stat_t12}" + r""" Prototype} & \textbf{""" + f"{aq_m50:.2f}\\% ({aq_recovery:+.2f} pp)" + r"""} & \textbf{""" + f"{aq_m95:.2f}\\%" + r"""} & \textbf{""" + f"{aq_rec:.2f}\\%" + r"""} & \textbf{""" + f"{lat_sqa_ms:.2f} ms (Dispersion)" + r"""} \\
\quad \textbf{Adaptive Evasion on SQA (White-Box Joint)} & \textbf{""" + f"{adapt_m50:.2f}\\% ({adapt_delta:+.2f} pp)" + r"""} & \textbf{""" + f"{adapt_m95:.2f}\\%" + r"""} & \textbf{""" + f"{adapt_rec:.2f}\\%" + r"""} & \textbf{Bypass: """ + f"{adapt_bypass:.1f}\\%" + r"""} \\
\midrule
\textbf{Standard Task PGD Baseline ($\epsilon = 16/255$):} & & & & \\
""" + pgd_dyn_tex + "\n" + pgd_sqa_tex + r"""
\bottomrule
\end{tabular}%
}
\end{table}"""

    md = "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] + ["| " + " | ".join(r) + " |" for r in rows])
    return tex, md, headers, rows


def generate_table13(summary_dir: Path) -> Tuple[str, str, List[str], List[List[str]]]:
    """
    Sinh Bảng 13: Multi-Seed Optimization Stability and Cross-Variant Architectural Generalization.
    Đọc hoàn toàn động từ multimodel_and_seed_stability_summary.json (không fix cứng fallback).
    """
    ms_data = safe_load_json([
        summary_dir / "multimodel_and_seed_stability_summary.json",
        OUTPUTS_DIR / "summary_json" / "multimodel_and_seed_stability_summary.json",
    ], default_dict={})
    records = ms_data.get("records", {})
    if not records:
        raise KeyError("Không có bản ghi nào trong multimodel_and_seed_stability_summary.json")

    headers = [
        "Model Architecture",
        "Backbone / Params",
        "Evaluation Seed / Protocol",
        "Clean mAP@50",
        "AQSM mAP@50 [Rel. Drop]",
        "Delta_Hungarian Set Drift",
        "Anchor FG Hit",
    ]
    rows: List[List[str]] = []
    tex_rows: List[str] = []
    seed_counts: List[int] = []

    # Thu thập danh sách kiến trúc động (ưu tiên bản multiseed nếu cùng một kiến trúc có cả canonical)
    arch_prefixes: List[str] = []
    for k in records.keys():
        if k.endswith("_multiseed"):
            pfx = k[:-len("_multiseed")]
        elif k.endswith("_canonical"):
            pfx = k[:-len("_canonical")]
        else:
            pfx = k
        if pfx not in arch_prefixes:
            arch_prefixes.append(pfx)

    for pfx in arch_prefixes:
        ms_rec = records.get(f"{pfx}_multiseed")
        can_rec = records.get(f"{pfx}_canonical")
        if isinstance(ms_rec, dict) and "aqsm_map50_mean" in ms_rec:
            n_s = len(ms_rec.get("seeds", []))
            if n_s > 0:
                seed_counts.append(n_s)
            proto_str = rf"{n_s} seeds ($\mu \pm \sigma$)" if n_s > 0 else str(ms_rec.get("protocol", "Multi-Seed"))
            arch_disp = f"{ms_rec.get('model', pfx.upper())} (Multi-Seed)"
            bb_str = str(ms_rec.get("backbone", pfx))
            clean_v = float(ms_rec["clean_map50"])
            aqsm_m = float(ms_rec["aqsm_map50_mean"])
            aqsm_s = float(ms_rec["aqsm_map50_std"])
            rel_v = float(ms_rec["relative_drop_pct"])
            drift_m = float(ms_rec["hungarian_set_drift_mean"])
            drift_s = float(ms_rec["hungarian_set_drift_std"])
            hit_m = float(ms_rec["anchor_fg_hit_pct_mean"])
            hit_s = float(ms_rec["anchor_fg_hit_pct_std"])
            rows.append([
                arch_disp,
                bb_str,
                proto_str,
                f"{clean_v:.2f}%",
                f"{aqsm_m:.2f} \u00b1 {aqsm_s:.2f}% [{rel_v:+.2f}%]",
                f"{drift_m:.4f} \u00b1 {drift_s:.4f}",
                f"{hit_m:.2f} \u00b1 {hit_s:.2f}%",
            ])
            arch_disp_tex = arch_disp.replace("_", r"\_")
            tex_rows.append(
                f"{arch_disp_tex} & {bb_str} & {proto_str} & {clean_v:.2f}\\% & "
                f"{aqsm_m:.2f} $\\pm$ {aqsm_s:.2f}\\% [$-${abs(rel_v):.2f}\\%] & "
                f"{drift_m:.4f} $\\pm$ {drift_s:.4f} & {hit_m:.2f} $\\pm$ {hit_s:.2f}\\% \\\\"
            )
        elif isinstance(can_rec, dict) and "aqsm_map50" in can_rec:
            arch_disp = str(can_rec.get("model", pfx.upper()))
            bb_str = str(can_rec.get("backbone", pfx))
            proto_str = str(can_rec.get("protocol", "Canonical"))
            clean_v = float(can_rec["clean_map50"])
            aqsm_v = float(can_rec["aqsm_map50"])
            rel_v = float(can_rec["relative_drop_pct"])
            drift_v = float(can_rec["hungarian_set_drift"])
            hit_v = float(can_rec["anchor_fg_hit_pct"])
            rows.append([
                arch_disp,
                bb_str,
                proto_str,
                f"{clean_v:.2f}%",
                f"{aqsm_v:.2f}% [{rel_v:+.2f}%]",
                f"{drift_v:.4f}",
                f"{hit_v:.2f}%",
            ])
            arch_disp_tex = arch_disp.replace("_", r"\_")
            tex_rows.append(
                f"{arch_disp_tex} & {bb_str} & {proto_str} & {clean_v:.2f}\\% & "
                f"{aqsm_v:.2f}\\% [$-${abs(rel_v):.2f}\\%] & {drift_v:.4f} & {hit_v:.2f}\\% \\\\"
            )

    if not rows:
        raise KeyError("Không tìm thấy bản ghi kiến trúc hợp lệ nào trong multimodel_and_seed_stability_summary.json")

    caption_k = f"$K={seed_counts[0]}$ seeds" if seed_counts else "multi-seed"
    eff_res_t13 = int(get_img_size())
    threat_note = str(ms_data.get("threat_budget", r"$\epsilon = 16/255, T = 15$"))
    tex = r"""\begin{table}[!htbp]
\centering
\caption{Multi-seed optimization stability (""" + caption_k + rf""") and cross-variant generalization on COCO val2017 (${eff_res_t13} \times {eff_res_t13}$, """ + threat_note + r""").}
\label{tab:tab13_multiseed_generalization}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccc}
\toprule
\textbf{Model Architecture} & \textbf{Backbone / Params} & \textbf{Evaluation Seed / Protocol} & \textbf{Clean mAP@50} & \textbf{AQSM mAP@50 [Rel. Drop]} & \textbf{$\Delta_{\mathrm{Hungarian}}$ Set Drift} & \textbf{Anchor FG Hit} \\
\midrule
""" + "\n".join(tex_rows) + r"""
\bottomrule
\end{tabular}%
}
\end{table}"""

    md = "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] + ["| " + " | ".join(r) + " |" for r in rows])
    return tex, md, headers, rows


# ==============================================================================
# MAIN ORCHESTRATOR
# ==============================================================================
def main():
    cur_outputs_dir = get_outputs_dir()
    parser = argparse.ArgumentParser(description="Tự động kết xuất toàn bộ 13 Bảng biểu thực nghiệm chuẩn Elsevier EAAI")
    parser.add_argument("--output_dir", type=str, default=str(cur_outputs_dir), help="Thư mục chứa summary_json")
    parser.add_argument("--tables_dir", type=str, default=None, help="Thư mục xuất bảng .tex, .md, .csv")
    parser.add_argument("--verify_tex", action="store_true", default=False, help="Kiểm tra đối chiếu với main.tex")
    args, _ = parser.parse_known_args()

    if args.output_dir:
        update_experiment_paths(output_dir=args.output_dir)

    out_root = Path(args.output_dir).resolve()
    tables_out_default = out_root / "tables"

    # Dò tìm thư mục summary_dir linh hoạt (ưu tiên --output_dir tường minh, tránh lẫn với thư mục global)
    explicit_candidates: List[Path] = [
        out_root / "summary_json",
        out_root,
        out_root.parent / "summary_json",
    ]
    global_candidates: List[Path] = [
        get_summary_json_dir(),
        get_outputs_dir() / "summary_json",
        get_outputs_dir(),
        WORKSPACE_DIR / "paper_1_aqsm_mechanism" / "outputs" / "summary_json",
        Path(__file__).resolve().parent.parent / "outputs" / "summary_json",
    ]

    summary_dir = None
    for cand in explicit_candidates:
        if cand.exists() and list(cand.glob("*.json")):
            summary_dir = cand
            break
    if summary_dir is None:
        for cand in explicit_candidates + global_candidates:
            if cand.exists() and (cand / "defense_esm_evaluation.json").exists():
                summary_dir = cand
                break
    if summary_dir is None:
        for cand in explicit_candidates + global_candidates:
            if cand.exists():
                summary_dir = cand
                break
    if summary_dir is None:
        summary_dir = out_root / "summary_json"

    tables_dir = Path(args.tables_dir) if args.tables_dir else tables_out_default
    tables_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = tables_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(" [AUTOMATED TABLE GENERATOR] TỰ ĐỘNG XUẤT 13 BẢNG BIỂU CHUẨN ELSEVIER EAAI")
    print(f" Nguồn dữ liệu: {summary_dir}")
    print(f" Thư mục đích  : {tables_dir}")
    print("=" * 80)

    generators = [
        ("table1_ablation", generate_table1),
        ("table2_main_benchmark", generate_table2),
        ("table3_causal_surgery", generate_table3),
        ("table4_anova_decomp", generate_table4),
        ("table5_memory_scales", generate_table5),
        ("table6_query_dynamics", generate_table6),
        ("table7_attention_telemetry", generate_table7),
        ("table8_controlled_misrouting", generate_table8),
        ("table9_matched_budget", generate_table9),
        ("table10_latency_sponge", generate_table10),
        ("table11_defense_results", generate_table11),
        ("table12_hardening_validation", generate_table12),
        ("table13_multiseed_generalization", generate_table13),
    ]

    all_tex = []
    all_md = []
    statuses: Dict[str, str] = {}

    for name, gen_fn in generators:
        try:
            tex, md, headers, rows = gen_fn(summary_dir)

            # 1. Lưu file .tex riêng
            tex_path = tables_dir / f"{name}.tex"
            save_text(tex_path, tex)

            # 2. Lưu file .csv
            csv_path = csv_dir / f"{name}.csv"
            save_csv_rows(csv_path, headers, rows)

            all_tex.append(f"% --- {name.upper()} ---\n{tex}\n")
            all_md.append(f"## {name.upper()}\n\n{md}\n")
            print(f" [OK] Đã sinh {name} -> {tex_path.name} & {csv_path.name}")
            statuses[name] = "PASS"
        except Exception as e:
            print(f" [LỖI] Thất bại khi sinh {name}: {e}")
            statuses[name] = f"FAIL ({e})"

    n_pass = sum(1 for s in statuses.values() if s == "PASS")
    n_total = len(statuses)

    # 3. Lưu file tổng hợp nếu có ít nhất 1 bảng thành công
    if n_pass > 0:
        combined_tex_path = tables_dir / "all_tables.tex"
        save_text(combined_tex_path, "\n\n".join(all_tex))

        combined_md_path = tables_dir / "all_tables.md"
        c_aq_hdr = safe_load_json([summary_dir / "causal_swap_results_aqsm.json"], default_dict={})
        n_hdr = c_aq_hdr.get("n_evaluated_images")
        hdr_tag = f" (N={int(n_hdr):,})" if n_hdr else ""
        save_text(combined_md_path, f"# TOÀN BỘ 13 BẢNG BIỂU THỰC NGHIỆM PAPER 1{hdr_tag}\n\n" + "\n\n".join(all_md))

        print("\n" + "=" * 80)
        print(f" [HOÀN TẤT] {n_pass}/{n_total} bảng biểu đã được sinh ra thành công:")
        print(f"   LaTeX snippets : {combined_tex_path}")
        print(f"   Markdown review: {combined_md_path}")
        print(f"   CSV tables     : {csv_dir}")
        print("=" * 80)

    if n_pass == 0:
        raise RuntimeError(f"Toàn bộ 13 bảng biểu đều thất bại trong {summary_dir}!")

    if args.verify_tex:
        tex_file = Path(args.output_dir).parent / "main.tex"
        if not tex_file.exists():
            tex_file = PAPER_DIR / "main.tex"
        if tex_file.exists():
            print(f"\n>>> [KIỂM TRA ĐỐI CHIẾU] Đang kiểm tra tính nhất quán với {tex_file.name}...")
            with open(tex_file, "r", encoding="utf-8") as f:
                tex_content = f.read()

            missing_keys = []
            test_patterns = []

            def _try_json(fname: str) -> Dict[str, Any]:
                fp = summary_dir / fname
                return load_json(fp) if fp.exists() else {}

            c_aq = _try_json("causal_swap_results_aqsm.json")
            mb_j = _try_json("matched_budget_sweep.json")
            mem_j = _try_json("encoder_memory_analysis.json")
            dyn_j = _try_json("query_selection_dynamics.json")
            tel_j = _try_json("deformable_attention_telemetry.json")
            mis_j = _try_json("controlled_misrouting_results.json")
            esm_j = _try_json("defense_esm_evaluation.json")
            sqa_j = _try_json("hardening_evaluation_results.json")
            ms_j = _try_json("multimodel_and_seed_stability_summary.json")

            if mb_j.get("clean_baseline"):
                test_patterns.append(("Table 1 Clean mAP", f"{mb_j['clean_baseline']['mAP@50']:.2f}%"))
            if mb_j.get("sweep_results", {}).get("aqsm"):
                test_patterns.extend([
                    ("Table 1 AQSM mAP", f"{mb_j['sweep_results']['aqsm']['eps_16']['mAP@50']:.2f}%"),
                    ("Table 9 AQSM eps4", f"{mb_j['sweep_results']['aqsm']['eps_4']['mAP@50']:.2f}%"),
                ])
            if c_aq.get("point_maps"):
                pm = c_aq["point_maps"]
                test_patterns.extend([
                    ("Table 3 Arm A", f"{pm['Arm_A']['mAP@50']:.2f}%"),
                    ("Table 3 Arm B", f"{pm['Arm_B']['mAP@50']:.2f}%"),
                    ("Table 3 Arm C_full", f"{pm['Arm_C_full']['mAP@50']:.2f}%"),
                    ("Table 3 Arm D_full", f"{pm['Arm_D_full']['mAP@50']:.2f}%"),
                    ("Table 3 Arm D_index", f"{pm['Arm_D_index']['mAP@50']:.2f}%"),
                ])
            if c_aq.get("anova_repeated_measures_full"):
                av = c_aq["anova_repeated_measures_full"]
                test_patterns.extend([
                    ("Table 4 ME_M", f"{av['marginal_effects']['ME_M']:.2f}"),
                    ("Table 4 ME_Q", f"{av['marginal_effects']['ME_Q']:.2f}"),
                    ("Table 4 SS_M", f"{av['anova_table']['Memory_M']['SS']:.1f}"),
                    ("Table 4 SS_Q", f"{av['anova_table']['Query_Q']['SS']:.1f}"),
                ])
            if mem_j.get("results", {}).get("aqsm"):
                test_patterns.append(("Table 5 Overall Cosine", f"{mem_j['results']['aqsm']['cosine_similarity']['Overall']['mean']:.4f}"))
            if dyn_j.get("results", {}).get("aqsm"):
                test_patterns.extend([
                    ("Table 6 Overlap", f"{dyn_j['results']['aqsm']['top300_overlap_pct']['mean']:.2f}%"),
                    ("Table 6 Evicted Top15", f"{dyn_j['results']['aqsm']['evicted_top15_fg_pct']['mean']:.2f}%"),
                ])
            if tel_j.get("aqsm"):
                test_patterns.append(("Table 7 FG Hit", f"{tel_j['aqsm']['mean_fg_hit_rate'] * 100.0:.2f}%"))
            if mis_j.get("table8_summary"):
                t8s = mis_j["table8_summary"]
                test_patterns.extend([
                    ("Table 8 Random Reassignment", f"{t8s['Random_Reassignment']['mAP@50_mean']:.2f}"),
                    ("Table 8 Random Shift", f"{t8s['Matched_L2_Random_Shift']['mAP@50_mean']:.2f}"),
                    ("Table 8 AQSM Misrouting", f"{t8s['AQSM_Spatial_Misrouting']['mAP@50']:.2f}%"),
                ])
            if esm_j.get("attacks", {}).get("aqsm"):
                test_patterns.extend([
                    ("Table 11 AQSM TPR", f"{esm_j['attacks']['aqsm']['tpr_detection_rate_pct']:.2f}%"),
                    ("Table 11 AUROC", f"{esm_j['attacks']['aqsm']['auroc']:.4f}"),
                ])
            if sqa_j.get("results", {}).get("aqsm_attack_eps16"):
                sqa_res = sqa_j["results"]
                if "default_dynamic_only" in sqa_res.get("edge_latency_jetson_orin_nano_ms", {}):
                    test_patterns.append(("Table 10 Jetson Orin", f"{sqa_res['edge_latency_jetson_orin_nano_ms']['default_dynamic_only']:.2f}"))
                test_patterns.append(("Table 12 SQA Recovery", f"{sqa_res['aqsm_attack_eps16']['hardened_static_augmented']['mAP@50']:.2f}%"))

            ms_recs = ms_j.get("records", {})
            if "rtdetr_r18vd_multiseed" in ms_recs:
                r18_r = ms_recs["rtdetr_r18vd_multiseed"]
                test_patterns.append(("Table 13 R18 Multi-Seed", f"{r18_r['aqsm_map50_mean']:.2f} $\\pm$ {r18_r['aqsm_map50_std']:.2f}%"))
            r50_r = ms_recs.get("rtdetr_r50vd_multiseed") or ms_recs.get("rtdetr_r50vd_canonical")
            if isinstance(r50_r, dict):
                test_patterns.append(("Table 13 R50 Clean", f"{r50_r['clean_map50']:.2f}%"))
                if "aqsm_map50_mean" in r50_r:
                    test_patterns.append(("Table 13 R50 AQSM", f"{r50_r['aqsm_map50_mean']:.2f} $\\pm$ {r50_r['aqsm_map50_std']:.2f}%"))
                elif "aqsm_map50" in r50_r:
                    test_patterns.append(("Table 13 R50 AQSM", f"{r50_r['aqsm_map50']:.2f}%"))
            for desc, pat in test_patterns:
                pat_escaped = pat.replace("%", r"\%")
                if pat not in tex_content and pat_escaped not in tex_content:
                    missing_keys.append(f"Thiếu {desc} ({pat})")

            if missing_keys:
                print(f" [CẢNH BÁO] Có {len(missing_keys)} số liệu chưa tìm thấy trong {tex_file.name}:")
                for k in missing_keys:
                    print(f"   - {k}")
            else:
                print(" [THÀNH CÔNG] Tất cả số liệu kiểm tra đều khớp 100% trong main.tex!")


if __name__ == "__main__":
    main()
