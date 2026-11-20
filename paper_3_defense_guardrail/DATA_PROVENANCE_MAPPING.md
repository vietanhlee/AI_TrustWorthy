# Bảng Ánh xạ Nguồn gốc Dữ liệu Thực nghiệm (Data Provenance & Audit Mapping)
## Paper 3: Split-Percentile Guardrails for Edge RT-DETR

Tài liệu này cung cấp **đường dẫn nguồn gốc dữ liệu khép kín (end-to-end data provenance)** cho từng Bảng biểu (Table), Hình vẽ (Figure) và các con số khẳng định trong bản thảo `main_defense.tex`. Mọi số liệu đều được tạo ra từ việc thực thi mã nguồn PyTorch thật trên tập dữ liệu chuẩn COCO val2017, lưu vết đầy đủ trong các file JSON tóm tắt và CSV kiểm toán từng mẫu ảnh (sample-level audit logs).

---

## 1. Bản đồ Nguồn gốc Bảng biểu (Tables Provenance Matrix)

| Bảng trong LaTeX | Tiêu đề Bảng / Nội dung Khoa học | Kịch bản Thực nghiệm (Experiment Script) | Tệp Metrics JSON Chính thức | Tệp CSV Dữ liệu Kiểm toán Từng Mẫu (Audit CSV) | Phương pháp Định lượng / Thống kê Suy luận |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Table 1**<br/>`tab:related_comparison` | So sánh phân loại lý thuyết với các giải pháp phòng thủ hiện hành | Tài liệu nghiên cứu tổng quan | N/A (Taxonomic Synthesis) | N/A | Khảo cứu tài liệu học thuật (Literature review) |
| **Table 2**<br/>`tab:fpr_stability` | Tỷ lệ Báo động Giả (FPR) dưới 4 điều kiện biến dạng tự nhiên (Gaussian, Blur, Patch sạch, Brightness) | `experiments/run_controls_and_fpr.py` | `results/step2_controls_fpr_metrics.json` | `results/step2_controls_and_fpr_samples.csv` ($N=600 \times 4$) | Tỷ lệ FPR thực nghiệm tại ngưỡng hiệu chuẩn danh định $\alpha=0.05$; khoảng tin cậy nhị thức Clopper-Pearson 95% |
| **Table 3**<br/>`tab:main_results` | Benchmark phát hiện toàn diện AUROC & TPR@5%FPR qua 10 cohorts tấn công | `experiments/run_calibration.py`<br/>`experiments/run_patch_attack_eval.py`<br/>`experiments/run_aqsm_and_esm.py`<br/>`experiments/run_query_attacks_eval.py`<br/>`experiments/run_adaptive_attacks_eval.py` | `results/step1_calibration_metrics.json`<br/>`results/step3_patch_attack_metrics.json`<br/>`results/step4_aqsm_esm_metrics.json`<br/>`results/step5_query_attacks_metrics.json`<br/>`results/step6_adaptive_attacks_metrics.json` | `results/step1_calibration_samples.csv`<br/>`results/step3_patch_attack_samples.csv`<br/>`results/step4_aqsm_and_esm_samples.csv`<br/>`results/step5_query_attacks_samples.csv`<br/>`results/step6_adaptive_attacks_samples.csv` | AUROC với Bootstrap 95% CI ($B=1000$); TPR tại ngưỡng cố định FPR 5%; Clopper-Pearson exact binomial CI |
| **Table 4**<br/>`tab:defensive_baselines` | Đối sánh hiệu quả và độ trễ với baseline không gian ảnh (TV Energy & JPEG Difference) | `experiments/run_defensive_baselines.py` | `results/step9_defensive_baselines_metrics.json` | `results/step9_defensive_baselines_samples.csv` | TPR bắt đòn AQSM vs FPR trên ảnh nhiễu cảm biến tự nhiên ($\sigma=16$); đo đạc thời gian xử lý từng khung hình |
| **Table 5**<br/>`tab:multitest_comparison` | So sánh các mô thức kiểm soát FWER đa kênh ($M=4$): Uncalibrated OR vs Bonferroni vs Calibrated Max-OR | `experiments/run_calibration.py`<br/>`experiments/run_query_attacks_eval.py` | `results/step1_calibration_metrics.json`<br/>`results/step5_query_attacks_metrics.json` | `results/step1_calibration_samples.csv` | Ngưỡng phân vị $\tau_{\mathrm{fused}} = 0.9842$; kiểm soát FWER trên tập test độc lập $N=600$ và năng lực giữ TPR |
| **Table 6**<br/>`tab:mcnemar_results` | Kiểm định thống kê ghép cặp McNemar ($2 \times 2$ contingency table) chứng minh ưu thế của ESM | `experiments/run_mcnemar_tests.py` | `results/step11_mcnemar_test_results.json` | Cặp dự đoán nhị phân từ Step 4 và Step 5 CSV ($N=300$) | Thống kê $\chi^2 = (|b - c| - 1)^2 / (b + c)$; kiểm định xác suất nhị thức chính xác hai phía (two-tailed exact binomial $p$-value) |
| **Table 7**<br/>`tab:runtime_overhead` | Đo đạc độ trễ phần cứng thực tế và phụ trội tài nguyên trên Edge CPU qua 50 lần lặp độc lập | `experiments/run_edge_latency.py` | `results/step8_edge_latency_metrics.json` | `results/step8_latency_runs_history.csv` | Trung bình $\pm$ độ lệch chuẩn (Mean $\pm$ Std), P95, P99, Tốc độ khung hình (FPS) trên CPU Intel Core i5-12600H (4 threads) |

---

## 2. Bản đồ Nguồn gốc Hình vẽ (Figures Provenance Matrix)

| Hình trong LaTeX | Tệp Hình Ảnh Vector (PDF) | Tệp Nguồn Sinh Đồ Thị | Dữ liệu Đầu vào Dùng để Vẽ Đồ Thị | Ý nghĩa Khoa học & Nội dung Minh họa |
| :--- | :--- | :--- | :--- | :--- |
| **Figure 1** | `fig/fig1_system_architecture.pdf` | `edge_sec_pro/visualization/generate_publication_figures.py` | Sơ đồ kiến trúc vector chuẩn | Kiến trúc tổng thể 3 tầng giám sát và cổng Calibrated Max-OR Gate trên RT-DETR |
| **Figure 6** | `fig/fig6_attack_taxonomy.pdf` | `edge_sec_pro/visualization/generate_publication_figures.py` | Sơ đồ phân loại vector chuẩn | Phân loại các bề mặt tấn công: Input, Encoder Selection, Deformable Decoder, Adaptive Evasion |
| **Figure 7** | `fig/fig7_calibration_mechanism.pdf` | `edge_sec_pro/visualization/generate_publication_figures.py` | `results/step1_calibration_samples.csv` | (a) Phân phối lệch phải của $r_{\mathrm{act}}$ trên 1200 ảnh sạch; (b) So sánh FPR giữa Boole bound, Naive OR, Bonferroni, và Max-OR |
| **Figure 2** | `fig/fig2_roc_curves.pdf` | `edge_sec_pro/visualization/generate_publication_figures.py` | `results/step4_aqsm_esm_metrics.json`<br/>`results/step5_query_attacks_metrics.json`<br/>`results/step3_patch_attack_metrics.json` | Tọa độ 100 điểm ROC thực tế cho AQSM, Query Vanishing, Query Fabrication, và Patch 48x48 |
| **Figure 3** | `fig/fig3_adaptive_tradeoff.pdf` | `edge_sec_pro/visualization/generate_publication_figures.py` | `results/step6_adaptive_attacks_metrics.json` | Đồ thị đánh đổi giữa tỷ lệ thành công của kẻ tấn công ASR và độ nhạy phát hiện TPR theo $\lambda$ |
| **Figure 4** | `fig/fig4_patch_size_sweep.pdf` | `edge_sec_pro/visualization/generate_publication_figures.py` | `results/step3_patch_attack_metrics.json` | Đồ thị độ nhạy phát hiện theo diện tích miếng dán ($16\times 16 \rightarrow 64\times 64$), vạch rõ điểm mù không gian |
| **Figure 5** | `fig/fig5_fpr_drift_stability.pdf` | `edge_sec_pro/visualization/generate_publication_figures.py` | `results/step2_controls_fpr_metrics.json` | Biểu đồ cột biểu diễn tính ổn định của FPR trên tập test sạch và 4 điều kiện biến dạng tự nhiên |
| **Figure 8** | `fig/fig8_cross_channel_correlation.pdf` | `edge_sec_pro/visualization/generate_publication_figures.py` | `results/step1_calibration_samples.csv` | Ma trận hệ số tương quan Spearman giữa $r_{\mathrm{act}}$, $\bar{p}_{\mathrm{top1}}$, $D_M^2$ và $S_{\mathrm{fused}}$ |
| **Figure 10** | `fig/fig10_latency_breakdown.pdf` | `edge_sec_pro/visualization/generate_publication_figures.py` | `results/step8_edge_latency_metrics.json` | Biểu đồ phân rã thời gian tính toán của Base Model vs Tensor Extraction vs Decision Scoring |
| **Figure 11** | `fig/fig11_qualitative_detections.pdf` | `edge_sec_pro/visualization/generate_publication_figures.py` | Bounding box coordinates & guardrail status | Trực quan hóa định tính hộp dự đoán và trạng thái cảnh báo trên ảnh COCO ID 159791 |

---

## 3. Phân chia Cohort Dữ liệu Chống Rò rỉ (Strict Disjoint Cohorts)

* **Tập Hiệu chuẩn ($\mathcal{D}_{\mathrm{cal}}$)**: 1200 ảnh COCO val2017 đầu tiên (indices $0 \rightarrow 1199$, cố định seed=42). Dùng độc quyền để tính ngưỡng phân vị và ma trận hiệp phương sai Mahalanobis.
* **Tập Kiểm thử Âm tính Sạch ($\mathcal{D}_{\mathrm{test}}$)**: 600 ảnh COCO val2017 tiếp theo (indices $1200 \rightarrow 1799$). Tuyệt đối không tham gia hiệu chuẩn, dùng độc quyền để đo lường FPR sạch khách quan ngoài mẫu.
* **Tập Đánh giá Đối kháng ($\mathcal{D}_{\mathrm{eval}}$)**: 300 ảnh COCO val2017 độc lập được trích xuất để sinh các đòn tấn công PGD, AQSM, Vanishing, DQF, Patch và Adaptive Evasion.

Mọi thực nghiệm đều được chạy tự động thông qua `run_batch_experiments.py` hoặc các file `experiments/run_*.py` với zero-mock guarantee.
