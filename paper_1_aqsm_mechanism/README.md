# Mechanistic Breakdown of Dynamic Query Selection in Real-Time DEtection TRansformer

Mã nguồn thực nghiệm chuẩn công nghiệp (Production-Grade Experiment Suite) phục vụ bài báo khoa học:
> **"Mechanistic Breakdown of Dynamic Query Selection in Real-Time DEtection TRansformer"**
> *Submitted to Elsevier Expert Systems with Applications (EAAI).*

---

## 1. Cấu trúc thư mục (Project Architecture)

```
paper_1_aqsm_mechanism/
├── run_experiments.sh           # Entry point duy nhất cho Linux / WSL / Git Bash
├── run_experiments.ps1          # Entry point cho Windows PowerShell
├── run_all_experiments.py       # Master Experiment Orchestrator điều phối toàn diện
├── kaggle_run_experiments.py    # Runner tối ưu cho môi trường Kaggle GPU / Colab
├── main.tex                     # Bản thảo LaTeX hoàn chỉnh theo chuẩn Elsevier EAAI
├── main.pdf                     # Bản PDF xuất bản (26 trang A4)
├── refs_EAAI.bib                # Danh mục tài liệu tham khảo BibTeX chuẩn hóa
├── elsarticle.cls               # Document class chuẩn của nhà xuất bản Elsevier
│
├── core/                        # Các module thuật toán và cơ chế nền tảng
│   ├── dataset_loader.py        # Flexible Dataset Loader & Lazy Batch Loading (Zero-OOM)
│   ├── detr_wrapper.py          # RT-DETR Model Wrapper & Internal States Extractor
│   ├── attack_generator.py      # Diagnostic Attacks (AQSM, Vanishing, Task PGD, Adaptive)
│   ├── mechanism_analyzer.py    # Encoder Memory, Query Selection & Deformable Hooks
│   ├── defense_engine.py        # Encoder Selection Monitor (ESM) & Calibrated Guardrail
│   ├── adv_cache.py             # Quản lý đọc/nạp bộ nhớ đệm mẫu đối kháng .npz
│   ├── coco_gt.py               # Bộ đánh giá COCO mAP, ANOVA 2x2 & Bootstrap CI
│   └── metrics.py               # AUROC, TPR @ Fixed FPR, Clopper-Pearson CI
│
├── experiments/                 # Các kịch bản và engine thực nghiệm
│   ├── config.py                # Cấu hình tập trung, siêu tham số & Hardware Auto-Detection
│   ├── generate_adv_cache_640.py# Task 0: Sinh bộ nhớ đệm mẫu đối kháng Native 640x640
│   ├── causal_surgery_engine.py # Engine can thiệp nhân quả 2x2 Factorial Swap
│   ├── telemetry_analyzer.py    # Engine đo đạc Deformable Attention Telemetry
│   ├── matched_budget_sweep.py  # Engine quét ngân sách nhiễu Matched-Budget
│   ├── latency_profiler.py      # Engine đo đạc độ trễ phần cứng tự động theo thiết bị
│   ├── generate_figures_and_tables.py # Engine tự động kết xuất biểu đồ vector xuất bản
│   ├── audit_logger.py          # Audit stream logger chuẩn hóa JSONL
│   ├── attack_engine.py         # Diagnostic Threat Engine helper
│   └── tasks/                   # 8 Tasks thực nghiệm độc lập
│       ├── task1_causal_surgery.py          # Phẫu thuật nhân quả 6-Arm (Table 3, 4)
│       ├── task2_encoder_query_dynamics.py  # Động lực học Encoder & Query (Table 5, 6)
│       ├── task3_deformable_telemetry.py    # Attention Telemetry (Table 7)
│       ├── task4_controlled_misrouting.py   # Bác bỏ giả thuyết Tautology (Table 8)
│       ├── task5_matched_sweep.py           # Quét ngân sách đối kháng (Table 9)
│       ├── task6_latency_profiler.py        # Hardware Latency Telemetry (Table 10)
│       ├── task7_generate_figures.py        # Xuất bản Figures 3-8 (PDF & PNG)
│       └── task8_defense_evaluation.py      # Đánh giá ESM & Adaptive PGD (Table 11)
│
├── tools/                       # 9 công cụ chuẩn hóa phục vụ kiểm định và xuất bản
│   ├── audit_data_provenance.py                 # Kiểm toán tính toàn vẹn 100% dữ liệu N=1,000
│   ├── generate_publication_tables.py           # Tự động xuất 12 bảng biểu chuẩn Elsevier EAAI
│   ├── fill_experiment_results_to_paper.py      # Turnkey Tool tự động điền kết quả vào main.tex
│   ├── run_multimodel_benchmark.py              # Benchmark so sánh RT-DETR-R18 vs RT-DETR-R50
│   ├── evaluate_sqa_controls.py                 # Đánh giá controls và tấn công thích ứng SQA
│   ├── compute_set_distance_and_instance_coverage.py # Khoảng cách tập hợp Hungarian / nearest-anchor
│   ├── compute_nonparametric_statistics.py      # Thống kê phi tham số Wilcoxon & Paired Bootstrap CI
│   ├── generate_architecture_diagram.py         # Sinh sơ đồ kiến trúc vector
│   └── generate_publication_diagrams.py         # Sinh biểu đồ phân tích xuất bản
│
├── outputs/                     # Thư mục lưu trữ kết quả thực nghiệm tự động
│   ├── tables/                  # 12 bảng LaTeX (.tex), Markdown (.md) và bảng CSV
│   ├── summary_json/            # Dữ liệu JSON tổng hợp (Ánh xạ 1:1 với Tables trong bài báo)
│   ├── raw_csv/                 # Dữ liệu thô trên từng ảnh phục vụ kiểm toán khoa học
│   ├── audit_streams/           # Log kiểm toán chi tiết từng bước dạng JSONL
│   ├── logs/                    # Log thực thi của từng task
│   └── final_experiment_manifest.json # Nghiệm thu tổng thể chuỗi thực nghiệm
│
├── fig/                         # Hình ảnh biểu đồ vector PDF & PNG độ phân giải 300 DPI
├── cache/adv_cache_640/         # Bộ nhớ đệm mẫu đối kháng .npz nén theo model tag
└── legacy_archive/              # Thư mục lưu trữ các script lịch sử và patch tạm thời
```

---

## 2. Hướng dẫn chạy thực nghiệm & Đồng bộ bài báo

### Cách 1: Chạy toàn bộ chuỗi thực nghiệm trên GPU (Kaggle / Colab / Server)
```bash
# Chạy với RT-DETR-R18 (chuẩn N = 1,000 ảnh, 0 GB đĩa với VRAM streaming)
python kaggle_run_experiments.py \
  --model r18 \
  --max_samples 1000 \
  --batch_size 16 \
  --skip_cache_gen

# Chạy với RT-DETR-R50 (6 decoder layers)
python kaggle_run_experiments.py \
  --model r50 \
  --max_samples 1000 \
  --batch_size 8 \
  --skip_cache_gen
```

### Cách 2: Benchmark so sánh trực tiếp đa mô hình (R18 vs R50) với 1 lệnh
```bash
python tools/run_multimodel_benchmark.py \
  --models r18,r50 \
  --max_samples 1000 \
  --batch_size 16 \
  --device cuda
```

### Cách 3: Tự động điền 100% số liệu vào bài báo (Turnkey Tool)
Sau khi có kết quả tải về, chạy lệnh sau để tự động điền toàn bộ 12 bảng biểu vào `main.tex`:
```bash
python tools/fill_experiment_results_to_paper.py --results_dir /path/to/results/summary_json
```

---

## 3. Bản đồ ánh xạ kết quả thực nghiệm vào bài báo

| Bảng / Hình trong bài báo | File dữ liệu tổng hợp (JSON) | File số liệu thô (CSV) | Nội dung khoa học |
|---|---|---|---|
| **Table 1** | `summary_json/tripartite_ablation_results.json` | `raw_csv/table1_ablation.csv` | Ablation 3 thành phần hàm mất mát AQSM ($\mathcal{L}_{\mathrm{rank}}, \mathcal{L}_{\mathrm{suppr}}, \mathcal{L}_{\mathrm{elev}}$) |
| **Table 2** | `summary_json/main_benchmark_results.json` | `raw_csv/table2_main_benchmark.csv` | Benchmark COCO val2017 qua các cấp độ đối kháng và kiểm chứng môi trường |
| **Table 3 & Table 4** | `summary_json/causal_swap_results_*.json` | `raw_csv/causal_anova_per_image_*.csv` | 2×2 Factorial Tensor-Swap, Repeated-Measures ANOVA $F$-test, partial $\eta_p^2$ và 95% Paired Bootstrap CI |
| **Table 5** | `summary_json/encoder_memory_analysis.json` | `raw_csv/encoder_query_dynamics_raw.csv` | Multi-Scale Cosine Similarity qua các tầng $S_3, S_4, S_5$ và phân tầng Foreground vs Background |
| **Table 6** | `summary_json/query_selection_dynamics.json` | `raw_csv/encoder_query_dynamics_raw.csv` | Overlap Top-300, Top-100, Top-50; Tỷ lệ trục xuất Evicted Top-15; Object Coverage IoU $\ge 0.50$ |
| **Table 7** | `summary_json/deformable_attention_telemetry.json` | `raw_csv/attention_telemetry_per_image.csv` | Attention Weights Entropy ($H_{\text{norm}}$), Spatial Hit Rate, Reference Coordinate Drift |
| **Table 8** | `summary_json/tautology_control_results.json` | `raw_csv/tautology_refutation_raw.csv` | Bác bỏ giả thuyết Tautology: Random Reassignment & Matched-$L_2$ Random Shift qua 5 seeds |
| **Table 9** | `summary_json/matched_budget_sweep.json` | `raw_csv/matched_budget_sweep_raw.csv` | Quét ngân sách nhiễu tương thích $\varepsilon \in \{4, 8, 16\}/255$ so sánh Clean vs AQSM vs Vanishing vs Task PGD |
| **Table 10** | `summary_json/latency_profile.json` | `raw_csv/latency_profiler_raw.csv` | Phân rã độ trễ phần cứng từng stage (Backbone $\to$ Encoder $\to$ Query Selection $\to$ Decoder) & Timing Invariance |
| **Table 11** | `summary_json/defense_esm_evaluation.json` | — | Hiệu năng phát hiện của Encoder Selection Monitor (ESM) (TPR @ 5% FPR) và ranh giới né tránh thích ứng Adaptive PGD |
| **Table 12** | `summary_json/hardening_evaluation_results.json` | — | Đánh giá gia cố kiến trúc thông qua Tiêm Anchor Tĩnh (Static Query Augmentation - SQA) |
| **Figures 3 - 8** | `fig/*.pdf` và `fig/*.png` | — | Biểu đồ vector xuất bản chuẩn Elsevier EAAI |

---

## 4. Các điểm lưu ý về tính liêm chính khoa học
1. **Tuyệt đối không hardcode số liệu**: Mọi con số đều được tính toán trực tiếp từ trọng số mô hình gốc (`PekingU/rtdetr_r18vd` và `PekingU/rtdetr_r50vd`).
2. **Quy trình kiểm toán độc lập**: Hệ thống tích hợp sẵn script `audit_data_provenance.py` để kiểm tra chéo tự động giữa JSON artifacts và nội dung `main.tex`, đảm bảo độ tin cậy và khả năng tái lập 100%.
