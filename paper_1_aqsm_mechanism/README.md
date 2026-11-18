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
├── outputs/                     # Thư mục lưu trữ kết quả thực nghiệm tự động
│   ├── summary_json/            # Dữ liệu JSON tổng hợp (Ánh xạ 1:1 với Tables trong bài báo)
│   ├── raw_csv/                 # Dữ liệu thô trên từng ảnh phục vụ kiểm toán khoa học
│   ├── audit_streams/           # Log kiểm toán chi tiết từng bước dạng JSONL
│   ├── logs/                    # Log thực thi của từng task
│   └── final_experiment_manifest.json # Nghiệm thu tổng thể chuỗi thực nghiệm
│
├── fig/                         # Hình ảnh biểu đồ vector PDF & PNG độ phân giải 300 DPI
├── cache/adv_cache_640/         # Bộ nhớ đệm mẫu đối kháng .npz nén
└── legacy_archive/              # Thư mục lưu trữ các script lịch sử không dùng trong luồng mới
```

---

## 2. Hướng dẫn chạy thực nghiệm

### Cách 1: Trên Linux / WSL / Git Bash (Khuyến nghị)
```bash
# Chạy toàn bộ chuỗi thực nghiệm trên tập ảnh COCO val2017
bash run_experiments.sh \
  --folder_image /path/to/coco/val2017 \
  --annotations /path/to/instances_val2017.json \
  --device auto
```

### Cách 2: Trên Windows PowerShell
```powershell
.\run_experiments.ps1 `
  -FolderImage "G:\Edge_Sec\edge_sec_pro\data\coco5k\val2017" `
  -Annotations "G:\Edge_Sec\edge_sec_pro\data\coco5k\annotations\instances_val2017.json" `
  -Device "auto"
```

### Cách 3: Trên Kaggle GPU hoặc Google Colab (Tăng tốc ~15x)
```bash
python kaggle_run_experiments.py \
  --folder_image "/kaggle/input/coco-2017-dataset/coco2017/val2017" \
  --annotations "/kaggle/input/coco-2017-dataset/coco2017/annotations/instances_val2017.json" \
  --batch_size 16
```
*Script sẽ tự động nén toàn bộ thư mục `outputs` và `fig` thành file `paper1_experiment_results.zip` để tải về.*

---

## 3. Bản đồ ánh xạ kết quả thực nghiệm vào bài báo

| Bảng / Hình trong bài báo | File dữ liệu tổng hợp (JSON) | File số liệu thô (CSV) | Nội dung khoa học |
|---|---|---|---|
| **Table 3 & Table 4** | `summary_json/causal_swap_results_*.json` | `raw_csv/causal_anova_per_image_*.csv` | 2×2 Factorial Tensor-Swap, Repeated-Measures ANOVA $F$-test, partial $\eta_p^2$ và 95% Paired Bootstrap CI |
| **Table 5** | `summary_json/encoder_memory_analysis.json` | `raw_csv/encoder_query_dynamics_raw.csv` | Multi-Scale Cosine Similarity qua các tầng $S_3, S_4, S_5$ và phân tầng Foreground vs Background |
| **Table 6** | `summary_json/query_selection_dynamics.json` | `raw_csv/encoder_query_dynamics_raw.csv` | Overlap Top-300, Top-100, Top-50; Tỷ lệ trục xuất Evicted Top-15; Object Coverage IoU $\ge 0.50$ |
| **Table 7** | `summary_json/deformable_attention_telemetry.json` | `raw_csv/attention_telemetry_per_image.csv` | Attention Weights Entropy ($H_{\text{norm}}$), Spatial Hit Rate, Reference Coordinate Drift |
| **Table 8** | `summary_json/tautology_control_results.json` | `raw_csv/tautology_refutation_raw.csv` | Bác bỏ giả thuyết Tautology: Random Reassignment & Matched-$L_2$ Random Shift qua 5 seeds |
| **Table 9** | `summary_json/matched_budget_sweep.json` | `raw_csv/matched_budget_sweep_raw.csv` | Quét ngân sách nhiễu tương thích $\varepsilon \in \{4, 8, 16\}/255$ so sánh Clean vs AQSM vs Vanishing vs Task PGD |
| **Table 10** | `summary_json/latency_profile.json` | `raw_csv/latency_profiler_raw.csv` | Phân rã độ trễ phần cứng từng stage (Backbone $\to$ Encoder $\to$ Query Selection $\to$ Decoder) & Timing Invariance |
| **Table 11** | `summary_json/defense_esm_evaluation.json` | — | Hiệu năng phát hiện của Encoder Selection Monitor (ESM) (TPR @ 5% FPR) và ranh giới né tránh thích ứng Adaptive PGD |
| **Figures 3 - 8** | `fig/*.pdf` và `fig/*.png` | — | Biểu đồ vector xuất bản chuẩn Elsevier EAAI |

---

## 4. Các điểm lưu ý về tính liêm chính khoa học
1. **Tuyệt đối không hardcode số liệu**: Mọi con số đều được tính toán trực tiếp từ trọng số mô hình `PekingU/rtdetr_r18vd` gốc.
2. **Không dùng script tự động điền vào LaTeX**: Sau khi thực nghiệm hoàn tất, người nghiên cứu trực tiếp đọc dữ liệu từ JSON/CSV và cập nhật thủ công vào `main.tex` để bảo đảm tính chính xác và mạch lạc của văn phong học thuật.
