# RT-DETR Adversarial Security & Reliability-Aware Monitoring (Elsevier EAAI)

Dự án nghiên cứu: **An Empirical Analysis of Query Selection Vulnerability and Internal Monitoring in Real-Time DETR**  
Tạp chí mục tiêu: **Elsevier Engineering Applications of Artificial Intelligence (EAAI)**

---

## 1. Cấu Trúc Dự Án Đã Tái Cấu Trúc (Production Architecture)

```
edge_sec_pro/
│
├── __init__.py                                 # Package root initialization
├── README.md                                   # Hướng dẫn tái lập thực nghiệm & kiến trúc
│
├── core/                                       # Core modules (Detection, features, metrics)
│   ├── config.py                               # Cấu hình siêu tham số và đường dẫn
│   ├── dataset.py                              # Loader dữ liệu COCO val2017
│   ├── detector.py                             # Logic detector interface
│   ├── features.py                             # Trích xuất đặc trưng
│   ├── metrics.py                              # AP, mAP, FPR, TPR
│   └── profiler.py                             # Đo trễ và tài nguyên bộ nhớ
│
├── transformer_monitors/                       # Transformer monitors & Attack generators
│   ├── detr_wrapper.py                         # Wrapper cho RT-DETR (PekingU/rtdetr_r18vd)
│   ├── detr_attack_generator.py                # Attack generator (AQSM, Vanishing, DQF, Patch, Adaptive)
│   ├── detr_query_monitor.py                   # 1D monitors (ActiveQ, Top1Mean)
│   ├── rtdetr_advanced_features.py             # ESM, Mahalanobis layer features
│   ├── rtdetr_advanced_monitor.py              # Calibrated Max-OR Guardrail pipeline
│   └── query_attention_features.py             # Attention weights extractor
│
├── experiments/                                # BỘ THỰC NGHIỆM ĐƯỢC CHUẨN HÓA THEO 12 STEPS
│   ├── common_exp.py                           # Thư viện tiện ích dùng chung
│   ├── step1_extract_calibration.py            # Bước 1: Trích xuất phân phối hiệu chuẩn (N_cal = 1200)
│   ├── step2_evaluate_controls_and_fpr.py      # Bước 2: Đánh giá FPR trên clean test và 4 natural shifts
│   ├── step3_evaluate_patch_attack.py          # Bước 3: Đánh giá localized gradient patch (Tier 0 blindspot)
│   ├── step4_evaluate_aqsm_and_esm.py          # Bước 4: Đánh giá tấn công AQSM & ESM dropoff / margin (Tier 1)
│   ├── step5_evaluate_query_attacks.py         # Bước 5: Đánh giá Query Vanishing & DQF (Tier 2)
│   ├── step6_evaluate_adaptive_attacks.py      # Bước 6: Đánh giá White-box Adaptive Evasion (Tier 3)
│   ├── step7_causal_mechanism_anova.py         # Bước 7: Can thiệp nhân quả 4 Arms & Repeated Measures ANOVA thật
│   ├── step8_measure_edge_latency.py           # Bước 8: Đo đạc chi phí thời gian thực (Latency & Memory Overhead)
│   ├── step9_defensive_baselines.py            # Bước 9: Đánh giá đối sánh với TV Energy Detector & JPEG Differencing
│   ├── step10_epsilon_sensitivity_sweep.py     # Bước 10: Quét độ nhạy epsilon toàn diện (eps in [2, 4, 8, 16]/255)
│   ├── step11_statistical_hypothesis_tests.py  # Bước 11: Kiểm định giả thuyết ghép cặp McNemar & Multiple Testing
│   └── step12_aggregate_and_export_results.py  # Bước 12: Tổng hợp toàn bộ kết quả vào results/*.json và SSOT
│
├── pipelines/                                  # CÁC SCRIPT CHẠY HÀNG LOẠT (ORCHESTRATION PIPELINES)
│   ├── run_pipeline.py                         # Master python CLI runner
│   ├── run_master_pipeline.ps1                 # PowerShell master pipeline (Chạy tuần tự Step 1 -> 12 -> Render -> Compile)
│   └── run_parallel_experiments.ps1            # PowerShell parallel batch runner (chạy phân tán các worker)
│
├── visualization/                              # THƯ VIỆN & SCRIPT SINH HÌNH VẼ PUBLICATION
│   ├── generate_publication_figures.py         # Sinh trọn bộ 10 Figures Vector PDF + PNG 600 DPI
│   └── generate_qualitative_predictions.py     # Sinh hình định tính trực quan hóa phát hiện vật thể
│
├── tools/                                      # CÔNG CỤ XUẤT BẢN, KIỂM TRA & BIÊN DỊCH
│   ├── compile_paper.py                        # Biên dịch Paper EAAI (pdflatex + bibtex)
│   ├── compile_slide.py                        # Biên dịch Slide Beamer (xelatex)
│   ├── audit_paper_ssot.py                     # Kiểm tra tính nhất quán số liệu toàn bài (SSOT Auditor)
│   ├── verify_build_quality.py                 # Kiểm tra log biên dịch (0 error, 0 undefined, 0 overfull)
│   └── clean_math_environments.py              # Dọn dẹp lỗi formatting môi trường toán
│
├── paper/                                      # BÀI BÁO ELSEVIER EAAI (LATEX SOURCE & FIGURES)
│   ├── main.tex                                # File TeX chính Camera-Ready
│   ├── refs_EAAI.bib                           # CSDL trích dẫn BibTeX chuẩn
│   └── fig/                                    # 10 Figures Vector PDF & PNG
│
├── slides/                                     # SLIDE THUYẾT TRÌNH BEAMER
│   └── slide_rtdetr_security_eaai.tex          # File XeLaTeX Beamer Camera-Ready
│
├── results/                                    # CSDL KẾT QUẢ THỰC NGHIỆM SSOT (JSON & NPZ)
│   ├── rtdetr_advanced_defense_results.json
│   ├── rtdetr_factorial_anova_standardized.json
│   ├── rtdetr_factorial_causal_raw_per_image.npz
│   ├── rtdetr_defensive_baselines_results.json
│   └── rtdetr_mcnemar_test_results.json
│
└── scratch/                                    # Nơi lưu trữ archive lịch sử
    └── legacy_archive/                         # Lưu các script vá cũ để tra cứu lịch sử
```

---

## 2. Hướng Dẫn Sử Dụng Nhanh (Quick Start)

### 2.1. Chạy Từng Bước Thực Nghiệm (CLI)
```bash
# Chạy Bước 7: Can thiệp Nhân quả & Phân tích Repeated Measures ANOVA chuẩn mực
python -m edge_sec_pro.pipelines.run_pipeline --step 7

# Chạy Bước 9: Baseline Phòng thủ Đầu vào (TV Energy & JPEG Diff)
python -m edge_sec_pro.pipelines.run_pipeline --step 9

# Chạy Bước 11: Kiểm định McNemar Ghép cặp
python -m edge_sec_pro.pipelines.run_pipeline --step 11
```

### 2.2. Sinh 10 Figures Vector Chuẩn Xuất Bản
```bash
python -m edge_sec_pro.pipelines.run_pipeline --visualize
```

### 2.3. Biên Dịch Toàn Bộ Tài Liệu Học Thuật
```bash
python -m edge_sec_pro.pipelines.run_pipeline --compile
```

### 2.4. Kiểm Tra Chất Lượng Camera-Ready (QA Verification)
```bash
python -m edge_sec_pro.pipelines.run_pipeline --verify
```

### 2.5. Chạy Toàn Bộ Luồng Tự Động (Master Pipeline)
```powershell
pwsh run_master_pipeline.ps1
```
