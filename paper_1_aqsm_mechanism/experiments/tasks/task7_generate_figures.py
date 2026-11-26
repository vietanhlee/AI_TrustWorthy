# -*- coding: utf-8 -*-
"""
Module: paper_1_aqsm_mechanism.experiments.tasks.task7_generate_figures
Mục đích: Đọc số liệu thực nghiệm THẬT 100% từ các file JSON trong summary_json/
được sinh ra từ quá trình chạy trên Kaggle (N = 1,000 COCO val2017), tự động vẽ
toàn bộ các biểu đồ vector PDF/PNG xuất bản chuẩn Elsevier EAAI:
  - Figure 3: Proposal Score Spectrum Across Top-300 Bottleneck & Logit Distribution
  - Figure 4 / 5: Repeated-Measures 2x2 Causal Tensor-Swap on COCO val2017 (N = 1,000) & ANOVA Effects
  - Figure 5 / 6: Multi-Scale Cosine Degradation (S3/S4/S5) & Proposal Overlap/Coverage Dynamics
  - Figure 6 / 7: Deformable Attention Telemetry (Normalized Entropy vs Spatial FG Hit Rate)
  - Figure 7 / 8: Matched-Budget Robustness Comparison across Perturbation Budgets
  - Figure 8 / 9: Empirical Defense Evaluation (ESM Detection Rate & Adaptive Evasion)

Đặc tính chuẩn công nghiệp (Production Standards):
  - Kích hoạt chế độ headless backend (matplotlib.use('Agg')) hoàn toàn tương thích Linux/Kaggle/Docker.
  - Tự động dò tìm đường dẫn artifact JSON thông minh (Smart Artifact Resolver) qua nhiều thư mục ứng viên.
  - Xử lý ngoại lệ độc lập từng biểu đồ (Graceful Fault Isolation), không làm đứt gãy luồng thực thi chung.
  - Tuyệt đối không hardcode, đọc động 100% từ các artifact JSON thực nghiệm.
"""

import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# BẮT BUỘC: Thiết lập matplotlib backend 'Agg' TRƯỚC KHI import pyplot
# để hoạt động ổn định trên môi trường máy chủ Linux headless (Kaggle / Colab / HPC) không có X-Server.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import pchip_interpolate

WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from paper_1_aqsm_mechanism.experiments.config import (
    FIG_DIR,
    OUTPUTS_DIR,
    PAPER_DIR,
    get_fig_dir,
    get_summary_json_dir,
    update_experiment_paths,
)
from paper_1_aqsm_mechanism.utils.cli import create_base_parser
from paper_1_aqsm_mechanism.utils.io_helpers import load_json

# Thiết lập font và style xuất bản chuẩn Elsevier EAAI với DejaVu Serif dự phòng toàn diện
plt.rcParams.update({
    "font.family": ["DejaVu Serif", "serif", "sans-serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def _to_pct(val: float) -> float:
    """Đảm bảo giá trị hiển thị ở thang phần trăm 0 - 100."""
    return val * 100.0 if 0.0 <= val <= 1.0 else val


def _require_stat(d: Any, label: str = "") -> float:
    """Trích xuất giá trị thống kê bắt buộc từ dict {'mean': ...} hoặc số (không dùng fallback tĩnh)."""
    if isinstance(d, dict):
        if "mean" in d and d["mean"] is not None:
            return float(d["mean"])
        raise KeyError(f"[STRICT PROVENANCE] Thiếu khóa 'mean' cho '{label}': {list(d.keys())}")
    if isinstance(d, (int, float)):
        return float(d)
    raise KeyError(f"[STRICT PROVENANCE] Giá trị thống kê không hợp lệ cho '{label}': {d!r}")


def _get_stat(d: Any, default: float = 0.0) -> float:
    """Trích xuất giá trị thống kê an toàn từ dict {'mean': ...} hoặc số nguyên / số thực."""
    if isinstance(d, dict):
        return float(d.get("mean", default))
    elif isinstance(d, (int, float)):
        return float(d)
    return default


def find_artifact_file(candidate_dirs: List[Path], filenames: List[str]) -> Optional[Path]:
    """
    Dò tìm file artifact đầu tiên tồn tại trong danh sách các thư mục và tên file tương đương.
    Hỗ trợ xử lý triệt để sự khác biệt đường dẫn giữa Kaggle, Colab và Local.
    """
    checked_paths = []
    for d in candidate_dirs:
        if not d.exists():
            continue
        for fn in filenames:
            cand = d / fn
            checked_paths.append(cand)
            if cand.exists() and cand.is_file():
                return cand
    return None


def plot_score_spectrum_figure(json_path: Path, output_pdf: Path, output_png: Path) -> None:
    """
    Vẽ Figure 3: Phổ điểm đề xuất Encoder và phân phối logit.
    Đọc trực tiếp 100% từ encoder_query_dynamics.json hoặc query_selection_dynamics.json.
    Sử dụng nội suy pchip đơn điệu khớp chính xác các mốc thực nghiệm s1, s50, s300.
    """
    data = load_json(json_path)

    results = data.get("results", {})
    aqsm_data = results.get("aqsm", {}) or data.get("aqsm", {})
    spec = aqsm_data.get("score_spectrum", {}) or data.get("score_spectrum", {})

    if not spec:
        raise KeyError(f"Không tìm thấy trường 'score_spectrum' trong {json_path}.")

    clean_s1 = _require_stat(spec["clean_s1"], "spec.clean_s1")
    clean_s50 = _require_stat(spec["clean_s50"], "spec.clean_s50")
    clean_s300 = _require_stat(spec["clean_s300"], "spec.clean_s300")

    adv_s1 = _require_stat(spec["adv_s1"], "spec.adv_s1")
    adv_s50 = _require_stat(spec["adv_s50"], "spec.adv_s50")
    adv_s300 = _require_stat(spec["adv_s300"], "spec.adv_s300")

    ranks = np.arange(1, 301)
    rank_anchors = [1, 50, 300]
    decay_clean = pchip_interpolate(rank_anchors, [clean_s1, clean_s50, clean_s300], ranks)
    decay_adv = pchip_interpolate(rank_anchors, [adv_s1, adv_s50, adv_s300], ranks)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.8))

    # Subplot 1: Toàn bộ Top-300
    ax1.plot(ranks, decay_clean, label="Clean Baseline", color="#2ca02c", linewidth=2.0)
    ax1.plot(ranks, decay_adv, label="AQSM (Ours)", color="#d62728", linewidth=2.0, linestyle="--")

    ax1.set_xlabel("Query Candidate Rank (Sorted by Confidence)")
    ax1.set_ylabel("Sigmoid Proposal Score $s^{(i)}$")
    ax1.set_title("(a) Proposal Score Spectrum Across Top-300", fontweight="bold", fontsize=10)
    ax1.set_xlim(1, 300)
    ax1.set_ylim(0.0, 1.0)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", frameon=True)

    # Subplot 2: Ranks 1-50 Zoom
    ax2.plot(ranks[:50], decay_clean[:50], label="Clean Baseline", color="#2ca02c", linewidth=2.0)
    ax2.plot(ranks[:50], decay_adv[:50], label="AQSM (Ours)", color="#d62728", linewidth=2.0, linestyle="--")
    ax2.set_xlabel("Top Candidate Rank (1 - 50)")
    ax2.set_ylabel("Sigmoid Score $s^{(i)}$")
    ax2.set_title("(b) Foreground Proposal Zoom (Ranks #1 to #50)", fontweight="bold", fontsize=10)
    ax2.set_xlim(1, 50)
    ax2.set_ylim(0.0, 1.0)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", frameon=True)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_pdf)
    plt.savefig(output_png)
    plt.close()
    print(f"   [OK] Đã xuất bản Figure Score Spectrum: {output_pdf.name}")


def plot_causal_swap_figure(json_path: Path, output_pdf: Path, output_png: Path) -> None:
    """
    Vẽ Figure 4 / 5: Biểu đồ hoán đổi nhân quả 2x2 Factorial Causal Swap ở Native 640x640 (N = 1,000).
    Bao gồm 2 subplots:
      (a) Global mAP@50 and mAP@50:95 across canonical intervention arms.
      (b) Repeated-measures ANOVA effect sizes (partial eta-squared eta_p^2).
    Đọc trực tiếp 100% từ causal_swap_results_aqsm.json.
    """
    data = load_json(json_path)

    pm = data["point_maps"]
    arms_order = ["Arm_A", "Arm_D_index", "Arm_D_full", "Arm_C_full", "Arm_B"]

    map50_vals = [float(pm[a]["mAP@50"]) for a in arms_order]
    map95_vals = [float(pm[a]["mAP@50:95"]) for a in arms_order]

    labels = [
        "Arm A\n(Clean Baseline)",
        "Arm D (Index)\n($M_{clean} + Q_{idx}^{adv}$)",
        "Arm D (Full)\n($M_{clean} + Q_{full}^{adv}$)",
        "Arm C (Full)\n($M_{adv} + Q_{full}^{clean}$)",
        "Arm B\n(Full AQSM)",
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2), gridspec_kw={"width_ratios": [1.4, 0.8]})

    # Subplot (a): Canonical Intervention Arms
    x = np.arange(len(labels))
    width = 0.35

    rects1 = ax1.bar(x - width / 2, map50_vals, width, label="mAP@50 (%)", color="#1f77b4", edgecolor="black", alpha=0.85)
    rects2 = ax1.bar(x + width / 2, map95_vals, width, label="mAP@50:95 (%)", color="#aec7e8", edgecolor="black", alpha=0.85)

    ax1.set_ylabel("Detection Accuracy (%)", fontweight="bold")
    ax1.set_title("(a) Canonical Intervention Arms (N = 1,000, 640x640)", fontweight="bold", fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8.5)
    ax1.legend(loc="upper right", frameon=True)
    ax1.set_ylim(0.0, 75.0)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    for rect in rects2:
        h = rect.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    # Subplot (b): ANOVA Factorial Effect Attribution
    anova = data.get("anova_repeated_measures_full") or data["anova_repeated_measures_index"]
    eta_sq = anova["attribution_shares"]["partial_eta_squared"]

    factors = ["Memory ($M$)", "Query ($Q$)", "Interaction\n($M \\times Q$)"]
    eta_vals = [
        float(eta_sq["Memory_M"]),
        float(eta_sq["Query_Q"]),
        float(eta_sq["Interaction_MxQ"]),
    ]

    colors = ["#d62728", "#ff7f0e", "#2ca02c"]
    bars_b = ax2.bar(factors, eta_vals, width=0.5, color=colors, edgecolor="black", alpha=0.85)
    ax2.set_ylabel("Partial Eta-Squared ($\\eta_p^2$)", fontweight="bold")
    ax2.set_title("(b) Effect Size (ANOVA Decomposition)", fontweight="bold", fontsize=10)
    ax2.set_ylim(0.0, 1.0)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars_b:
        h = bar.get_height()
        ax2.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_pdf)
    plt.savefig(output_png)
    plt.close()
    print(f"   [OK] Đã xuất bản Figure Causal Swap: {output_pdf.name}")


def plot_multiscale_and_overlap_figure(
    memory_json_path: Path,
    query_json_path: Path,
    output_pdf: Path,
    output_png: Path,
) -> None:
    """
    Vẽ Figure 5 / 6:
      (a) Multi-Scale Cosine Similarity qua S3 (80x80), S4 (40x40), S5 (20x20)
      (b) Query selection overlap, foreground eviction, Center-in-GT, và IoU>=0.30 coverage.
    """
    mem_data = load_json(memory_json_path)
    query_data = load_json(query_json_path)

    mem_res = mem_data.get("results", {}) or mem_data
    q_res = query_data.get("results", {}) or query_data

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.8))

    # Subplot (a): Multi-Scale Cosine
    scales = ["$S_3$ (80x80)", "$S_4$ (40x40)", "$S_5$ (20x20)"]
    x = np.arange(len(scales))
    width = 0.20

    def _req_cos_scale(atk_name: str, scale_name: str) -> float:
        return _require_stat(mem_res[atk_name]["cosine_similarity"][scale_name], f"{atk_name}.cosine.{scale_name}")

    aqsm_cos = [_req_cos_scale("aqsm", "S3"), _req_cos_scale("aqsm", "S4"), _req_cos_scale("aqsm", "S5")]
    vanish_cos = [_req_cos_scale("vanishing", "S3"), _req_cos_scale("vanishing", "S4"), _req_cos_scale("vanishing", "S5")]
    taskpgd_cos = [_req_cos_scale("taskpgd", "S3"), _req_cos_scale("taskpgd", "S4"), _req_cos_scale("taskpgd", "S5")]
    has_gaussian = "gaussian" in mem_res
    if has_gaussian:
        noise_cos = [_req_cos_scale("gaussian", "S3"), _req_cos_scale("gaussian", "S4"), _req_cos_scale("gaussian", "S5")]

    if has_gaussian:
        ax1.bar(x - 1.5 * width, noise_cos, width, label="Gaussian Noise", color="#7f7f7f", edgecolor="black", alpha=0.85)
    ax1.bar(x - 0.5 * width, vanish_cos, width, label="PGD Vanishing", color="#1f77b4", edgecolor="black", alpha=0.85)
    ax1.bar(x + 0.5 * width, taskpgd_cos, width, label="Task PGD", color="#ff7f0e", edgecolor="black", alpha=0.85)
    ax1.bar(x + 1.5 * width, aqsm_cos, width, label="AQSM (Ours)", color="#d62728", edgecolor="black", alpha=0.85)

    ax1.set_ylabel("Cosine Similarity vs. Clean", fontweight="bold")
    ax1.set_title("(a) Multi-Scale Feature Degradation ($S_3, S_4, S_5$)", fontweight="bold", fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(scales, fontsize=9)
    ax1.set_ylim(0.0, 1.1)
    ax1.legend(loc="lower right", frameon=True, fontsize=8)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # Subplot (b): Query Selection Dynamics & Proposal Coverage
    aqsm_q = q_res["aqsm"]
    cov = aqsm_q["object_coverage"]

    metrics = ["Top-300\nOverlap", "Top-15 FG\nEvicted", "Center-in-GT\nCoverage", "GT IoU $\\geq$ 0.30\nCoverage"]

    clean_vals = [
        100.0,
        0.0,
        _require_stat(cov["center_in_gt_clean_pct"], "cov.center_in_gt_clean_pct"),
        _require_stat(cov["iou30_clean_pct"], "cov.iou30_clean_pct"),
    ]

    aqsm_vals = [
        _require_stat(aqsm_q["top300_overlap_pct"], "aqsm.top300_overlap_pct"),
        _require_stat(aqsm_q["evicted_top15_fg_pct"], "aqsm.evicted_top15_fg_pct"),
        _require_stat(cov["center_in_gt_adv_pct"], "cov.center_in_gt_adv_pct"),
        _require_stat(cov["iou30_adv_pct"], "cov.iou30_adv_pct"),
    ]

    x2 = np.arange(len(metrics))
    width2 = 0.35

    bar_c = ax2.bar(x2 - width2 / 2, clean_vals, width2, label="Clean Baseline", color="#2ca02c", edgecolor="black", alpha=0.85)
    bar_a = ax2.bar(x2 + width2 / 2, aqsm_vals, width2, label="AQSM (Ours)", color="#d62728", edgecolor="black", alpha=0.85)

    ax2.set_ylabel("Percentage (%)", fontweight="bold")
    ax2.set_title("(b) Query Overlap & Proposal Coverage Dynamics", fontweight="bold", fontsize=10)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(metrics, fontsize=8.5)
    ax2.set_ylim(0.0, 115.0)
    ax2.legend(loc="upper right", frameon=True, fontsize=8.5)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bar_c:
        h = bar.get_height()
        ax2.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    for bar in bar_a:
        h = bar.get_height()
        ax2.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_pdf)
    plt.savefig(output_png)
    plt.close()
    print(f"   [OK] Đã xuất bản Figure Multi-Scale & Overlap: {output_pdf.name}")


def plot_attention_telemetry_figure(json_path: Path, output_pdf: Path, output_png: Path) -> None:
    """
    Vẽ Figure 6 / 7: So sánh Deformable Attention Entropy vs Spatial FG Hit Rate.
    Đọc trực tiếp từ deformable_attention_telemetry.json hoặc mechanism_telemetry_results.json.
    """
    data = load_json(json_path)

    conditions = ["clean", "vanishing", "taskpgd", "aqsm"]
    cond_labels = ["Clean Baseline", "Query Vanishing", "Task PGD", "AQSM (Ours)"]

    entropy_means = []
    entropy_stds = []
    hit_means = []
    hit_stds = []

    for c in conditions:
        cond_dict = data[c]
        ent = float(cond_dict["mean_entropy_l1"])
        ent_std = float(cond_dict["std_entropy_l1"])
        hit = float(_to_pct(cond_dict["mean_fg_hit_rate"]))
        hit_std = float(_to_pct(cond_dict["std_fg_hit_rate"]))

        entropy_means.append(ent)
        entropy_stds.append(ent_std)
        hit_means.append(hit)
        hit_stds.append(hit_std)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 3.8))

    x = np.arange(len(conditions))
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]

    # Subplot (a): Attention Entropy (Normalized)
    bars1 = ax1.bar(x, entropy_means, yerr=entropy_stds, capsize=4, width=0.5,
                    color=colors, edgecolor="black", alpha=0.85)
    ax1.set_ylabel("Shannon Entropy $H(A)$ (Layer 1)", fontweight="bold")
    ax1.set_title("(a) Cross-Attention Weight Entropy", fontweight="bold", fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(cond_labels, fontsize=8.5)
    ax1.set_ylim(0.65, 0.95)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars1:
        h = bar.get_height()
        ax1.annotate(f"{h:.4f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 5), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    # Subplot (b): Spatial FG Hit Rate
    bars2 = ax2.bar(x, hit_means, yerr=hit_stds, capsize=4, width=0.5,
                    color=colors, edgecolor="black", alpha=0.85)
    ax2.set_ylabel("Spatial Foreground Hit Rate (%)", fontweight="bold")
    ax2.set_title("(b) Sampling Location Spatial Alignment", fontweight="bold", fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(cond_labels, fontsize=8.5)
    ax2.set_ylim(0.0, 100.0)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars2:
        h = bar.get_height()
        ax2.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 5), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_pdf)
    plt.savefig(output_png)
    plt.close()
    print(f"   [OK] Đã xuất bản Figure Attention Telemetry: {output_pdf.name}")


def plot_matched_budget_figure(json_path: Path, output_pdf: Path, output_png: Path) -> None:
    """
    Vẽ Figure 7 / 8: Đường cong suy giảm mAP@50 và mAP@50:95 theo ngân sách nhiễu L_inf.
    Đọc trực tiếp 100% từ matched_budget_sweep.json hoặc matched_budget_results.json.
    """
    data = load_json(json_path)

    sweep = data["sweep_results"]
    clean_base = data["clean_baseline"]
    clean_map50 = float(clean_base["mAP@50"])
    clean_map95 = float(clean_base["mAP@50:95"])

    aqsm_sweep = sweep["aqsm"]
    vanish_sweep = sweep["vanishing"]
    taskpgd_sweep = sweep["taskpgd"]

    available_keys = sorted(
        [k for k in aqsm_sweep.keys() if k.startswith("eps_")],
        key=lambda x: int(x.split("_")[1]) if x.split("_")[1].isdigit() else 0
    )
    if not available_keys:
        raise KeyError(f"Thiếu khóa eps_* trong {json_path}")

    eps_numeric = [0] + [int(k.split("_")[1]) for k in available_keys]
    x_tick_labels = ["Clean (0)"] + [f"{int(k.split('_')[1])}/255" for k in available_keys]

    aqsm_pts = [clean_map50]
    vanish_pts = [clean_map50]
    taskpgd_pts = [clean_map50]

    aqsm_map95 = [clean_map95]
    vanish_map95 = [clean_map95]
    taskpgd_map95 = [clean_map95]

    for k in available_keys:
        aqsm_pts.append(float(aqsm_sweep[k]["mAP@50"]))
        vanish_pts.append(float(vanish_sweep[k]["mAP@50"]))
        taskpgd_pts.append(float(taskpgd_sweep[k]["mAP@50"]))

        aqsm_map95.append(float(aqsm_sweep[k]["mAP@50:95"]))
        vanish_map95.append(float(vanish_sweep[k]["mAP@50:95"]))
        taskpgd_map95.append(float(taskpgd_sweep[k]["mAP@50:95"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 3.8))

    # Subplot (a): mAP@50 Breakdown
    ax1.plot(eps_numeric, aqsm_pts, "o-", color="#d62728", linewidth=2.2, label="AQSM (Ours)")
    ax1.plot(eps_numeric, vanish_pts, "s--", color="#1f77b4", linewidth=1.8, label="Query Vanishing")
    ax1.plot(eps_numeric, taskpgd_pts, "^-.", color="#ff7f0e", linewidth=1.8, label="Task PGD")

    ax1.set_xlabel("Perturbation Budget $L_\\infty$ Epsilon", fontweight="bold")
    ax1.set_ylabel("COCO mAP@50 (%)", fontweight="bold")
    ax1.set_title("(a) Perturbation Budget Sweep: mAP@50", fontweight="bold", fontsize=10)
    ax1.set_xticks(eps_numeric)
    ax1.set_xticklabels(x_tick_labels)
    ax1.set_ylim(0.0, 75.0)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", frameon=True)

    # Subplot (b): mAP@50:95 Breakdown
    ax2.plot(eps_numeric, aqsm_map95, "o-", color="#d62728", linewidth=2.2, label="AQSM (Ours)")
    ax2.plot(eps_numeric, vanish_map95, "s--", color="#1f77b4", linewidth=1.8, label="Query Vanishing")
    ax2.plot(eps_numeric, taskpgd_map95, "^-.", color="#ff7f0e", linewidth=1.8, label="Task PGD")

    ax2.set_xlabel("Perturbation Budget $L_\\infty$ Epsilon", fontweight="bold")
    ax2.set_ylabel("COCO mAP@50:95 (%)", fontweight="bold")
    ax2.set_title("(b) Perturbation Budget Sweep: mAP@50:95", fontweight="bold", fontsize=10)
    ax2.set_xticks(eps_numeric)
    ax2.set_xticklabels(x_tick_labels)
    ax2.set_ylim(0.0, 55.0)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", frameon=True)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_pdf)
    plt.savefig(output_png)
    plt.close()
    print(f"   [OK] Đã xuất bản Figure Matched Budget Sweep: {output_pdf.name}")


def plot_defense_evaluation_figure(
    json_path: Path,
    output_pdf: Path,
    output_png: Path,
    hardening_json_path: Optional[Path] = None,
) -> None:
    """
    Vẽ Figure 8 / 9: Hiệu năng phòng thủ ESM & Đề xuất Gia cố Kiến trúc SQA (3 subplots).
      (a) ESM Detection Performance (Table 11)
      (b) White-Box Adaptive Robustness / Evasion (Carlini-Tramèr)
      (c) SQA Architectural Hardening (Table 12) - Canvas Spatial Dispersion
    """
    data = load_json(json_path)

    attacks = ["aqsm", "vanishing", "taskpgd"]
    labels = ["AQSM (Ours)", "Query Vanishing", "Task PGD"]

    atk_dict = data["attacks"]
    clean_fpr = float(data["empirical_clean_fpr_pct"])
    tpr_vals = [float(atk_dict[a]["tpr_detection_rate_pct"]) for a in attacks]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14.2, 3.9))

    # =========================================================================
    # Subplot (a): Bar chart TPR at empirical clean FPR
    # =========================================================================
    x = np.arange(len(labels))
    bars = ax1.bar(x, tpr_vals, width=0.45, color=["#d62728", "#1f77b4", "#ff7f0e"], edgecolor="black", alpha=0.85)
    ax1.set_ylabel(f"Detection TPR (%) [Clean FPR = {clean_fpr:.2f}%]", fontweight="bold")
    ax1.set_title("(a) ESM Detection Performance (Table 11)", fontweight="bold", fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8.5)
    ax1.set_ylim(0, 105)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # =========================================================================
    # Subplot (b): Line chart Adaptive Evasion
    # =========================================================================
    adapt = data.get("adaptive_evasion", {})
    if adapt:
        lambdas = []
        evade_tpr = []
        evasion_success = []
        for l_key, l_data in sorted(adapt.items()):
            if isinstance(l_data, dict) and any(k in l_data for k in ["detected_tpr_pct", "evasion_success_pct", "conditioned_evasion_success_pct", "raw_monitor_bypass_pct"]):
                try:
                    l_val = float(l_data.get("evasion_weight_lambda", l_key.replace("lambda_", "")))
                except ValueError:
                    l_val = len(lambdas) * 0.5
                lambdas.append(l_val)
                tpr_val = float(l_data["detected_tpr_pct"])
                evade_tpr.append(tpr_val)
                ev_succ = l_data.get(
                    "evasion_success_pct",
                    l_data.get(
                        "conditioned_evasion_success_pct",
                        l_data.get("raw_monitor_bypass_pct", 100.0 - tpr_val),
                    ),
                )
                evasion_success.append(float(ev_succ if ev_succ is not None else (100.0 - tpr_val)))

        if lambdas:
            ax2.plot(lambdas, evade_tpr, "o-", color="#d62728", linewidth=2.0, label="Detection TPR (%)")
            ax2.plot(lambdas, evasion_success, "s--", color="#2ca02c", linewidth=2.0, label="Evasion Success (%)")
            ax2.set_xlabel("Adaptive Evasion Weight $\\lambda$", fontweight="bold")
            ax2.set_ylabel("Rate (%)", fontweight="bold")
            ax2.set_title("(b) White-Box Adaptive Robustness", fontweight="bold", fontsize=10)
            ax2.set_ylim(-5, 110)
            ax2.grid(True, linestyle="--", alpha=0.5)
            ax2.legend(loc="center right", frameon=True, fontsize=8.5)
        else:
            ax2.axis("off")
    else:
        ax2.axis("off")

    # =========================================================================
    # Subplot (c): SQA Architectural Hardening (Table 12)
    # =========================================================================
    sqa_conditions = [
        "Standard\n(300 Dyn.)",
        "Control 1\n(Dyn. 330)",
        "Control 2\n(Random 30)",
        "Proposed SQA\n(Grid-30)",
        "Adaptive\n(on SQA)"
    ]
    if not hardening_json_path or not Path(hardening_json_path).exists():
        raise FileNotFoundError(
            f"Thiếu tệp hardening JSON bắt buộc ({hardening_json_path}) để vẽ Subplot (c) SQA — không dùng số liệu hardcode!"
        )
    h_data = load_json(hardening_json_path)
    if "results" in h_data:
        res_d = h_data["results"]
        aqsm_res = res_d["aqsm_attack_eps16"]
        ctrl_d = res_d["controls"]
        sqa_map50 = [
            float(aqsm_res["default_dynamic_only"]["mAP@50"]),
            float(ctrl_d["dynamic_top330"]["mAP@50"]),
            float(ctrl_d["random30_static"]["mAP@50"]),
            float(aqsm_res["hardened_static_augmented"]["mAP@50"]),
            float(ctrl_d["adaptive_evasion_sqa"]["mAP@50"]),
        ]
    elif "controls" in h_data and "baselines" in h_data:
        c2 = h_data["controls"]
        sqa_map50 = [
            float(h_data["baselines"]["aqsm_eps16"]["dynamic_300_map50"]),
            float(c2["dynamic_330_topk"]["map50"]),
            float(c2["random_30_static"]["map50"]),
            float(c2["uniform_grid_30_sqa"]["map50"]),
            float(h_data["adaptive_threat_boundary"]["adaptive_map50"]),
        ]
    else:
        raise KeyError(f"Cấu trúc JSON hardening không hợp lệ tại: {hardening_json_path}")

    x_sqa = np.arange(len(sqa_conditions))
    sqa_colors = ["#d62728", "#64748b", "#f59e0b", "#10b981", "#8b5cf6"]
    bars_sqa = ax3.bar(x_sqa, sqa_map50, width=0.45, color=sqa_colors, edgecolor="black", alpha=0.88)

    ax3.set_ylabel("COCO mAP@50 (%) under AQSM", fontweight="bold")
    ax3.set_title("(c) SQA Architectural Hardening (Table 12)", fontweight="bold", fontsize=10)
    ax3.set_xticks(x_sqa)
    ax3.set_xticklabels(sqa_conditions, fontsize=8.0)
    ax3.set_ylim(0, 24)
    ax3.grid(axis="y", linestyle="--", alpha=0.5)

    # Reference baseline line for unhardened model
    ax3.axhline(y=sqa_map50[0], color="#d62728", linestyle=":", alpha=0.7, linewidth=1.2)

    for bar in bars_sqa:
        h = bar.get_height()
        ax3.annotate(f"{h:.2f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.0, fontweight="bold")

    # Recovery highlight badge on Proposed SQA bar (dynamically calculated)
    recovery_delta = sqa_map50[3] - sqa_map50[0]
    badge_label = f"+{recovery_delta:.2f} pp\nRecovery" if recovery_delta >= 0 else f"{recovery_delta:.2f} pp"
    sqa_bar = bars_sqa[3]
    ax3.annotate(
        badge_label,
        xy=(sqa_bar.get_x() + sqa_bar.get_width() / 2, sqa_map50[3]),
        xytext=(0, 15), textcoords="offset points",
        ha="center", va="bottom", fontsize=8.0, fontweight="bold", color="#065f46",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#d1fae5", edgecolor="#10b981", alpha=0.95)
    )

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_pdf, dpi=300)
    plt.savefig(output_png, dpi=300)
    plt.close()
    print(f"   [OK] Đã xuất bản Figure Defense Evaluation (3-panel ESM + SQA): {output_pdf.name}")


def generate_all_figures_and_tables(
    summary_dir: Optional[Path] = None,
    fig_dir: Optional[Path] = None,
    raw_csv_dir: Optional[Path] = None,
) -> Dict[str, str]:
    """
    Sinh toàn bộ biểu đồ vector chuẩn xuất bản Elsevier EAAI từ summary JSON.
    Tự động dò tìm qua danh sách thư mục ứng viên để tương thích hoàn toàn với Kaggle / Colab / Local.
    """
    candidate_dirs: List[Path] = []
    if summary_dir is not None:
        p_in = Path(summary_dir).resolve()
        candidate_dirs.extend([
            p_in,
            p_in / "summary_json",
            p_in.parent / "summary_json",
            p_in.parent,
        ])

    candidate_dirs.extend([
        get_summary_json_dir(),
        OUTPUTS_DIR / "summary_json",
        OUTPUTS_DIR,
        PAPER_DIR / "outputs" / "summary_json",
        WORKSPACE_DIR / "paper_1_aqsm_mechanism" / "outputs" / "summary_json",
        Path(__file__).resolve().parent.parent.parent / "outputs" / "summary_json",
    ])

    # Loại bỏ các đường dẫn trùng lặp
    seen = set()
    valid_candidate_dirs = []
    for cd in candidate_dirs:
        cd_str = str(cd.resolve())
        if cd_str not in seen:
            seen.add(cd_str)
            valid_candidate_dirs.append(cd)

    f_dir = Path(fig_dir) if fig_dir is not None else get_fig_dir()
    f_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(" [Automated Figure Engine] Tự động xuất biểu đồ Vector PDF & PNG chuẩn Elsevier")
    print(f" Thư mục xuất ảnh: {f_dir.resolve()}")
    print(" Danh sách thư mục tìm kiếm dữ liệu:")
    for d in valid_candidate_dirs:
        exists_mark = "V" if d.exists() else "X"
        print(f"   [{exists_mark}] {d}")
    print("=" * 80)

    figure_statuses: Dict[str, str] = {}

    # 1. Figure 3 / 4: Proposal Score Spectrum
    spectrum_json = find_artifact_file(
        valid_candidate_dirs,
        ["encoder_query_dynamics.json", "query_selection_dynamics.json"],
    )
    if spectrum_json:
        try:
            plot_score_spectrum_figure(
                spectrum_json,
                f_dir / "fig3_score_spectrum_n300.pdf",
                f_dir / "fig3_score_spectrum_n300.png",
            )
            plot_score_spectrum_figure(
                spectrum_json,
                f_dir / "fig3_proposal_score_spectrum.pdf",
                f_dir / "fig3_proposal_score_spectrum.png",
            )
            figure_statuses["Figure 3 (Score Spectrum)"] = "PASS"
        except Exception as e:
            print(f"   [CẢNH BÁO] Không thể sinh Figure 3: {e}")
            figure_statuses["Figure 3 (Score Spectrum)"] = f"FAIL ({e})"
    else:
        print("   [BỎ QUA] Thiếu file encoder_query_dynamics.json / query_selection_dynamics.json cho Figure 3.")
        figure_statuses["Figure 3 (Score Spectrum)"] = "SKIPPED (Missing JSON)"

    # 2. Figure 4 / 5: Causal Tensor-Swap
    causal_json = find_artifact_file(
        valid_candidate_dirs,
        ["causal_swap_results_aqsm.json", "causal_swap_results_all.json"],
    )
    if causal_json:
        try:
            plot_causal_swap_figure(
                causal_json,
                f_dir / "fig4_causal_swap_n300.pdf",
                f_dir / "fig4_causal_swap_n300.png",
            )
            figure_statuses["Figure 4 (Causal Swap)"] = "PASS"
        except Exception as e:
            print(f"   [CẢNH BÁO] Không thể sinh Figure 4: {e}")
            figure_statuses["Figure 4 (Causal Swap)"] = f"FAIL ({e})"
    else:
        print("   [BỎ QUA] Thiếu file causal_swap_results_aqsm.json cho Figure 4.")
        figure_statuses["Figure 4 (Causal Swap)"] = "SKIPPED (Missing JSON)"

    # 3. Figure 5 / 6: Multi-Scale Cosine & Proposal Dynamics
    mem_json = find_artifact_file(valid_candidate_dirs, ["encoder_memory_analysis.json"])
    query_json = find_artifact_file(
        valid_candidate_dirs,
        ["encoder_query_dynamics.json", "query_selection_dynamics.json"],
    )
    if mem_json and query_json:
        try:
            plot_multiscale_and_overlap_figure(
                mem_json,
                query_json,
                f_dir / "fig5_multiscale_cosine_and_overlap.pdf",
                f_dir / "fig5_multiscale_cosine_and_overlap.png",
            )
            figure_statuses["Figure 5 (Multi-Scale & Overlap)"] = "PASS"
        except Exception as e:
            print(f"   [CẢNH BÁO] Không thể sinh Figure 5: {e}")
            figure_statuses["Figure 5 (Multi-Scale & Overlap)"] = f"FAIL ({e})"
    else:
        print("   [BỎ QUA] Thiếu encoder_memory_analysis.json hoặc encoder_query_dynamics.json cho Figure 5.")
        figure_statuses["Figure 5 (Multi-Scale & Overlap)"] = "SKIPPED (Missing JSON)"

    # 4. Figure 6 / 7: Attention Telemetry
    telemetry_json = find_artifact_file(
        valid_candidate_dirs,
        ["deformable_attention_telemetry.json", "mechanism_telemetry_results.json"],
    )
    if telemetry_json:
        try:
            plot_attention_telemetry_figure(
                telemetry_json,
                f_dir / "fig6_attention_scattering_comparison.pdf",
                f_dir / "fig6_attention_scattering_comparison.png",
            )
            figure_statuses["Figure 6 (Attention Telemetry)"] = "PASS"
        except Exception as e:
            print(f"   [CẢNH BÁO] Không thể sinh Figure 6: {e}")
            figure_statuses["Figure 6 (Attention Telemetry)"] = f"FAIL ({e})"
    else:
        print("   [BỎ QUA] Thiếu file deformable_attention_telemetry.json cho Figure 6.")
        figure_statuses["Figure 6 (Attention Telemetry)"] = "SKIPPED (Missing JSON)"

    # 5. Figure 7 / 8: Matched-Budget Sweep
    sweep_json = find_artifact_file(
        valid_candidate_dirs,
        ["matched_budget_sweep.json", "matched_budget_results.json"],
    )
    if sweep_json:
        try:
            plot_matched_budget_figure(
                sweep_json,
                f_dir / "fig7_matched_budget_sweep.pdf",
                f_dir / "fig7_matched_budget_sweep.png",
            )
            figure_statuses["Figure 7 (Matched Budget Sweep)"] = "PASS"
        except Exception as e:
            print(f"   [CẢNH BÁO] Không thể sinh Figure 7: {e}")
            figure_statuses["Figure 7 (Matched Budget Sweep)"] = f"FAIL ({e})"
    else:
        print("   [BỎ QUA] Thiếu file matched_budget_sweep.json cho Figure 7.")
        figure_statuses["Figure 7 (Matched Budget Sweep)"] = "SKIPPED (Missing JSON)"

    # 6. Figure 8 / 9: Defense Evaluation (ESM + SQA 3-panel)
    defense_json = find_artifact_file(
        valid_candidate_dirs,
        ["defense_esm_evaluation.json", "defense_evaluation_summary.json"],
    )
    hardening_json = find_artifact_file(
        valid_candidate_dirs,
        ["hardening_evaluation_results.json", "sqa_controls_and_adaptive_summary.json"],
    )
    if defense_json:
        try:
            plot_defense_evaluation_figure(
                defense_json,
                f_dir / "fig8_defense_evaluation.pdf",
                f_dir / "fig8_defense_evaluation.png",
                hardening_json_path=hardening_json,
            )
            # Đồng bộ hình ảnh sang assets/images của web showcase
            for web_img_dir in [
                WORKSPACE_DIR / "assets" / "images",
                WORKSPACE_DIR / "docs" / "assets" / "images",
            ]:
                if web_img_dir.exists():
                    shutil.copy2(f_dir / "fig8_defense_evaluation.png", web_img_dir / "fig8_defense_evaluation.png")
            figure_statuses["Figure 8 (Defense Evaluation)"] = "PASS"
        except Exception as e:
            print(f"   [CẢNH BÁO] Không thể sinh Figure 8: {e}")
            figure_statuses["Figure 8 (Defense Evaluation)"] = f"FAIL ({e})"
    else:
        print("   [BỎ QUA] Thiếu file defense_esm_evaluation.json cho Figure 8.")
        figure_statuses["Figure 8 (Defense Evaluation)"] = "SKIPPED (Missing JSON)"

    # Tổng kết kết quả xuất bản
    n_pass = sum(1 for s in figure_statuses.values() if s == "PASS")
    n_total = len(figure_statuses)

    print("\n" + "=" * 80)
    print(f" [TỔNG KẾT XUẤT BẢN HÌNH VẼ TASK 7] {n_pass}/{n_total} biểu đồ xuất bản thành công:")
    for fig_name, status in figure_statuses.items():
        icon = "[PASS]" if status == "PASS" else "[CẢNH BÁO]"
        print(f"   {icon} {fig_name:35s}: {status}")
    print("=" * 80)

    if n_pass == 0:
        raise RuntimeError(
            "Task 7 không thể sinh bất kỳ biểu đồ nào do thiếu toàn bộ các artifact JSON! "
            f"Đã tìm kiếm trong: {[str(d) for d in valid_candidate_dirs if d.exists()]}"
        )

    return figure_statuses


def main():
    parser = create_base_parser("Generate Publication Figures (PDF/PNG)", task_name="Task 7")
    parser.add_argument("--fig_dir", type=str, default=str(FIG_DIR), help="Thư mục xuất file ảnh PDF/PNG")
    args = parser.parse_args()

    update_experiment_paths(
        output_dir=args.output_dir,
        fig_dir=args.fig_dir,
    )

    print("\n=======================================================")
    print("--> [Task 7] Đang sinh toàn bộ biểu đồ vector PDF/PNG xuất bản...")
    print(f" Thư mục dữ liệu: {args.output_dir}")
    print(f" Thư mục xuất ảnh: {args.fig_dir}")
    print("=======================================================")

    s_dir = Path(args.output_dir)
    generate_all_figures_and_tables(summary_dir=s_dir, fig_dir=Path(args.fig_dir))

    print("\n[HOÀN THÀNH TOÀN BỘ] Task 7: Generate Publication Figures thành công!")


if __name__ == "__main__":
    main()
