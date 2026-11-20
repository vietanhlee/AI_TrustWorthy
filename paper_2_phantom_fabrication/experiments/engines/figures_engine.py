# -*- coding: utf-8 -*-
"""
Module: paper_2_phantom_fabrication.experiments.engines.figures_engine
Mục đích: Động cơ tự động xuất bản 4 đồ thị vector chuẩn Elsevier EAAI từ file telemetry JSON:
  - Figure 1: NMS-Free Paradox (False Positives vs mAP50 trên RT-DETR và YOLOv8).
  - Figure 2: Displacement Trade-off (EQF vs DQF qua các ngân sách epsilon).
  - Figure 3: Downstream Cyber-Physical Impact (AEB Phantom Braking & ByteTrack Fragmentation).
  - Figure 4: SQD Filter Defense Comparison (So sánh hiệu quả lọc vật thể ma trước 4 baselines).
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import matplotlib.pyplot as plt
import numpy as np

# Thiết lập style xuất bản khoa học
plt.rcParams.update({
    "font.family": "serif",
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


def generate_all_paper2_figures(
    telemetry_path: Union[str, Path],
    fig_dir: Union[str, Path],
) -> None:
    """
    Đọc dữ liệu từ file telemetry JSON và xuất toàn bộ 4 biểu đồ vector PDF & PNG.
    """
    t_path = Path(telemetry_path)
    f_dir = Path(fig_dir)
    f_dir.mkdir(parents=True, exist_ok=True)

    if not t_path.exists():
        print(f"[CẢNH BÁO] Không tìm thấy file telemetry tại: {t_path}. Không thể vẽ hình.")
        return

    with open(t_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("\n" + "=" * 80)
    print(" [Automated Figure Engine - Paper 2] Xuất 4 biểu đồ vector chuẩn Elsevier EAAI")
    print(f" Dữ liệu nguồn: {t_path}")
    print(f" Thư mục xuất:  {f_dir}")
    print("=" * 80)

    # 1. Figure 1: NMS-Free Paradox
    t1_data = data.get("table1_benchmarks", data.get("stage1_benchmark", data))
    if t1_data:
        try:
            rtdetr = t1_data.get("rtdetr_aggregated", {})
            yolo = t1_data.get("yolov8_aggregated", {})
            conditions = ["Clean", "Untargeted", "EQF", "DQF"]
            r_keys = ["Clean", "Untargeted_PGD", "EQF", "DQF"]

            def get_stat(d, key, metric):
                sub = d.get(key, d.get(key.lower(), d.get(key.capitalize(), {})))
                if not sub and key in ("EQF", "DQF"):
                    sub = d.get("Targeted_Fabrication", {})
                return sub.get(metric, sub.get(metric.replace("_mean", ""), 0))

            r_fps = [get_stat(rtdetr, k, "fp_mean") for k in r_keys]
            y_fps = [get_stat(yolo, k, "fp_mean") for k in r_keys]
            r_maps = [get_stat(rtdetr, k, "mAP50_mean") for k in r_keys]
            y_maps = [get_stat(yolo, k, "mAP50_mean") for k in r_keys]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.8))
            x = np.arange(len(conditions))
            w = 0.35

            ax1.bar(x - w / 2, r_fps, w, label="RT-DETR (NMS-free)", color="#d62728", edgecolor="black", alpha=0.85)
            ax1.bar(x + w / 2, y_fps, w, label="YOLOv8n (NMS)", color="#1f77b4", edgecolor="black", alpha=0.85)
            ax1.set_ylabel("False Positive Count (FP)")
            ax1.set_title("(a) False Positive Inflation", fontweight="bold")
            ax1.set_xticks(x)
            ax1.set_xticklabels(conditions)
            ax1.legend(frameon=True)
            ax1.grid(axis="y", linestyle="--", alpha=0.5)

            ax2.bar(x - w / 2, r_maps, w, label="RT-DETR", color="#d62728", edgecolor="black", alpha=0.85)
            ax2.bar(x + w / 2, y_maps, w, label="YOLOv8n", color="#1f77b4", edgecolor="black", alpha=0.85)
            ax2.set_ylabel("COCO mAP@50 (%)")
            ax2.set_title("(b) Detection mAP Degradation", fontweight="bold")
            ax2.set_xticks(x)
            ax2.set_xticklabels(conditions)
            ax2.legend(frameon=True)
            ax2.grid(axis="y", linestyle="--", alpha=0.5)

            plt.tight_layout()
            plt.savefig(f_dir / "fig1_nms_free_paradox.pdf")
            plt.savefig(f_dir / "fig1_nms_free_paradox.png")
            plt.close()
            print(">>> Đã sinh Figure 1: fig1_nms_free_paradox.pdf / .png")
        except Exception as e:
            print(f"[Cảnh báo] Lỗi khi vẽ Figure 1: {e}")

    # 2. Figure 2: Displacement Trade-off
    t2_data = data.get("table2_budget_and_iterations", data.get("sensitivity_sweeps", data))
    if t2_data:
        try:
            bsweep = t2_data.get("budget_sweep", {})
            eps_labels = [4, 8, 12, 16]
            eqf_fp = [bsweep.get(f"eps_{e}", {}).get("EQF", {}).get("fp_count", 0) for e in eps_labels]
            dqf_fp = [bsweep.get(f"eps_{e}", {}).get("DQF", {}).get("fp_count", 0) for e in eps_labels]

            fig, ax = plt.subplots(figsize=(6.8, 3.8))
            ax.plot(eps_labels, eqf_fp, "o--", color="#1f77b4", linewidth=2.0, label="Existing Query Fabrication (EQF)")
            ax.plot(eps_labels, dqf_fp, "s-", color="#d62728", linewidth=2.2, label="Dense Query Fabrication (DQF - Ours)")
            ax.set_xlabel(r"Perturbation Budget $L_\infty$ Epsilon (/255)", fontweight="bold")
            ax.set_ylabel("Phantom False Positive Count", fontweight="bold")
            ax.set_title("Perturbation Budget Sensitivity: EQF vs DQF", fontweight="bold")
            ax.set_xticks(eps_labels)
            ax.set_xticklabels([f"{e}/255" for e in eps_labels])
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(frameon=True)

            plt.tight_layout()
            plt.savefig(f_dir / "fig2_displacement_tradeoff.pdf")
            plt.savefig(f_dir / "fig2_displacement_tradeoff.png")
            plt.close()
            print(">>> Đã sinh Figure 2: fig2_displacement_tradeoff.pdf / .png")
        except Exception as e:
            print(f"[Cảnh báo] Lỗi khi vẽ Figure 2: {e}")

    # 3. Figure 3: Downstream Cyber-Physical Impact
    downstream = data.get("downstream_and_figures", data.get("table3_downstream", data))
    if downstream:
        try:
            aeb = downstream.get("aeb_simulation", {})
            bytetrack = downstream.get("bytetrack_simulation", {})

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.6))
            # (a) AEB False Braking Risk at 28.0m (TTC <= 1.4s)
            clean_aeb = float(aeb.get("clean_risk_28m", 0.0))
            adv_aeb = float(aeb.get("adv_risk_28m", aeb.get("dqf_risk_28m", 0.0)))
            aeb_rates = [clean_aeb, adv_aeb]
            ax1.bar(["Clean Baseline", "Under DQF"], aeb_rates, width=0.45, color=["#2ca02c", "#d62728"], edgecolor="black", alpha=0.85)
            ax1.set_ylabel("Critical Braking Risk at 28m (%)")
            ax1.set_title("(a) Autonomous Braking (AEB) Hazard", fontweight="bold")
            ax1.set_ylim(0, 105)
            ax1.grid(axis="y", linestyle="--", alpha=0.5)

            # (b) ByteTrack Hungarian Association Latency (ms)
            clean_lat = float(bytetrack.get("clean_latency_ms", 0.0))
            adv_lat = float(bytetrack.get("adv_latency_ms", 0.0))
            lat_vals = [clean_lat, adv_lat]
            ax2.bar(["Clean Baseline", "Under DQF"], lat_vals, width=0.45, color=["#2ca02c", "#d62728"], edgecolor="black", alpha=0.85)
            ax2.set_ylabel("Hungarian Matching Latency (ms)")
            ax2.set_title("(b) ByteTrack Association Complexity Proxy", fontweight="bold")
            ax2.grid(axis="y", linestyle="--", alpha=0.5)

            plt.tight_layout()
            plt.savefig(f_dir / "fig3_downstream_cyberphysical_impact.pdf")
            plt.savefig(f_dir / "fig3_downstream_cyberphysical_impact.png")
            plt.close()
            print(">>> Đã sinh Figure 3: fig3_downstream_cyberphysical_impact.pdf / .png")
        except Exception as e:
            print(f"[Cảnh báo] Lỗi khi vẽ Figure 3: {e}")

    # 4. Figure 4: SQD Filter vs Baseline Defenses (Table 4)
    raw_t4 = data.get("table4_defenses", None)
    t4_data = None
    if isinstance(raw_t4, dict) and "defenses_aggregated" in raw_t4:
        t4_data = raw_t4["defenses_aggregated"]
    elif raw_t4 is not None:
        t4_data = raw_t4
    elif "defense_evaluation" in data:
        t4_data = data["defense_evaluation"]
    if t4_data:
        try:
            if isinstance(t4_data, list):
                # Format từ Bảng 4 chuẩn Paper 2
                names = [d.get("name", "").split("(")[0].strip() for d in t4_data]
                rejection_rates = [d.get("dqf_rejection_mean", 0.0) for d in t4_data]
                latencies = [d.get("cpu_latency_mean", 0.1) for d in t4_data]
            elif isinstance(t4_data, dict):
                names = list(t4_data.keys())
                rejection_rates = [float(t4_data[k].get("dqf_rejection_mean", t4_data[k].get("fp_reduction_pct", 0.0))) for k in names]
                latencies = [float(t4_data[k].get("cpu_latency_mean", t4_data[k].get("latency_ms", 0.0))) for k in names]
            else:
                names, rejection_rates, latencies = [], [], []

            if names:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 4.0))

                colors = ["#d62728", "#7f7f7f", "#7f7f7f", "#1f77b4", "#2ca02c"]
                if len(colors) < len(names):
                    colors = ["#1f77b4"] * len(names)

                # Subplot 1: DQF Phantom Query Rejection Rate (%)
                bars1 = ax1.bar(names, rejection_rates, width=0.45, color=colors[:len(names)], edgecolor="black", alpha=0.85)
                ax1.set_ylabel("Phantom Query Rejection Rate (%)", fontweight="bold")
                ax1.set_title("(a) Adversarial Rejection Efficiency", fontweight="bold")
                ax1.set_ylim(0, 105)
                ax1.tick_params(axis="x", rotation=25)
                ax1.grid(axis="y", linestyle="--", alpha=0.5)
                for b in bars1:
                    h = b.get_height()
                    ax1.annotate(f"{h:.1f}%", xy=(b.get_x() + b.get_width() / 2, h),
                                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

                # Subplot 2: Processing Latency (ms)
                bars2 = ax2.bar(names, latencies, width=0.45, color=colors[:len(names)], edgecolor="black", alpha=0.85)
                ax2.set_ylabel("Processing Latency (ms)", fontweight="bold")
                ax2.set_title("(b) Edge Hardware Overhead", fontweight="bold")
                ax2.tick_params(axis="x", rotation=25)
                ax2.grid(axis="y", linestyle="--", alpha=0.5)
                for b in bars2:
                    h = b.get_height()
                    ax2.annotate(f"{h:.2f}", xy=(b.get_x() + b.get_width() / 2, h),
                                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

                plt.tight_layout()
                plt.savefig(f_dir / "fig4_sqd_filter_defense.pdf")
                plt.savefig(f_dir / "fig4_sqd_filter_defense.png")
                plt.close()
                print(">>> Đã sinh Figure 4: fig4_sqd_filter_defense.pdf / .png")
        except Exception as e:
            print(f"[Cảnh báo] Lỗi khi vẽ Figure 4: {e}")
