# Edge AI Security: Phantom Query Fabrication Attacks & Lightweight Guardrails on RT-DETR

Hệ thống nghiên cứu toàn diện về an ninh kiến trúc mô hình **Real-Time Detection Transformer (RT-DETR)** trên thiết bị tính toán biên (Edge Computing).

Repository được cấu trúc thành **2 bài báo khoa học độc lập**, mỗi bài sở hữu thư mục làm việc, mã nguồn thực nghiệm và bản thảo LaTeX riêng biệt, cùng thư mục phát hành tập trung `SUBMISSION_PDFS/`:

---

## 📚 Danh Sách 2 Bài Báo Độc Lập

### 1. Bài 1: Tấn Công Tạo Bão Query Ma Trên Mô Hình Không Dùng NMS (Phantom Queries & Fabrication)
* **Thư mục làm việc**: [`paper_2_phantom_fabrication/`](paper_2_phantom_fabrication/)
* **Tiêu đề**: *Phantom Queries: Exploiting NMS-Free Set Prediction for Adversarial Object Fabrication in Real-Time Transformers*
* **Định dạng & PDF Đầu Ra**:
  * Bản Elsevier Preprint (12 trang): [`SUBMISSION_PDFS/3_Paper_Phantom_Queries_Fabrication_Elsevier.pdf`](SUBMISSION_PDFS/3_Paper_Phantom_Queries_Fabrication_Elsevier.pdf)
* **Trọng tâm khoa học**:
  * **Nghịch lý An ninh NMS-Free (NMS-Free Security Paradox)**: Việc loại bỏ NMS giúp RT-DETR triệt tiêu các cuộc tấn công trễ tính toán (NMS sponge latency attacks), nhưng đồng thời tước bỏ lớp lọc không gian đầu ra, khiến $100\%$ các query ma lọt thẳng vào downstream controllers mà không bị bất kỳ rào cản nào ngăn chặn (ngược lại, YOLOv8 + NMS lọc được $78.4\%$ box ma nhưng bị vọt trễ $+242\%$).
  * **2 Véc-tơ Tấn công Chuyên sâu**:
    * **EQF (Encoder Query Fabrication)**: Bơm gradient vào phân loại candidate của Hybrid Encoder để đẩy các anchor nền vượt ngưỡng $s_{(300)}$ của top-300 selection ($\mathrm{ASR}_{\mathrm{fab}} = 100.0\%$, trung bình $14.85$ phantom boxes).
    * **DQF (Decoder Query Fabrication)**: Tối ưu hóa trực tiếp output logits của Deformable Decoder ($\mathrm{ASR}_{\mathrm{fab}} = 99.7\%$, trung bình $16.12$ phantom boxes).
  * **Phát hiện Cơ chế Tác Dụng Phụ (Foreground Proposal Displacement)**: Vì bottleneck cố định ở $K=300$, việc EQF nhồi nhét anchor nền đã đẩy văng (evict) **$64.2\%$** các anchor vật thể thật, giải thích vì sao EQF làm sụt giảm mAP@50 tới $-92.0\%$.
  * **Hiểm họa Vật lý Không gian Mạng (Downstream Cyber-Physical Impact)**: Làm tăng độ trễ liên kết theo dõi (ByteTrack Hungarian matching) gấp **$7.5\times$** và nâng xác suất phanh khẩn cấp ma (**Phantom Braking Risk**) lên trên **$90\%$** trong hành lang di chuyển của xe tự hành.
  * **Giải pháp phòng thủ SQD-Filter (Spatial Query Deduplication & Dispersion Filter)**: Bộ lọc hậu xử lý siêu nhẹ ($0.08$~ms trên GPU, $1.15$~ms trên CPU) loại bỏ $>91.5\%$ phantom boxes với tỷ lệ giữ lại vật thể thật đạt $>98.4\%$.

---

### 2. Bài 2: Hệ Thống Phòng Thủ Đa Tầng Bằng Guardrail Phân Vị (Split-Percentile Defense)
* **Thư mục làm việc**: [`paper_3_defense_guardrail/`](paper_3_defense_guardrail/)
* **Tiêu đề**: *Split-Percentile Multi-Scale Latency Guardrails for Real-Time Edge Vision Transformers*
* **Định dạng & PDF Đầu Ra**:
  * Bản Elsevier Preprint (22 trang): [`SUBMISSION_PDFS/4_Paper_Defense_Split_Percentile_Guardrails.pdf`](SUBMISSION_PDFS/4_Paper_Defense_Split_Percentile_Guardrails.pdf)
* **Trọng tâm khoa học**:
  * Kiến trúc phòng thủ chủ động kiểm soát phân vị đa thang đo (Split-Percentile Guardrail) giám sát năng lượng đặc trưng và độ tin cậy query của RT-DETR theo thời gian thực.
  * Phục hồi độ chính xác nhận dạng và phát hiện tấn công đối kháng với độ trễ cực thấp phù hợp triển khai trên vi điều khiển và edge accelerators.

---

## 📂 Cấu Trúc Workspace

```
g:\Edge_Sec\
├── SUBMISSION_PDFS/                          # Thư mục lưu trữ bản PDF chính thức
│   ├── 3_Paper_Phantom_Queries_Fabrication_Elsevier.pdf
│   └── 4_Paper_Defense_Split_Percentile_Guardrails.pdf
│
├── paper_2_phantom_fabrication/             # BÀI BÁO 1: Phantom Queries & NMS-Free Vulnerability
│   ├── main_phantom.tex                     # Toàn văn bản thảo LaTeX (13 trang)
│   ├── elsarticle.cls
│   ├── refs_EAAI.bib
│   ├── experiments/                         # Mã nguồn thực nghiệm độc lập
│   │   ├── phantom_attack_generator.py      # Module tấn công EQF & DQF
│   │   ├── sqd_filter.py                    # Bộ lọc phòng thủ SQD-Filter
│   │   ├── run_phantom_experiments.py       # Sinh 4 figures vector PDF chuẩn IEEE
│   │   └── run_extended_benchmarks.py       # Ablation sweep, multi-class, size breakdown
│   └── fig/                                 # 4 vector figures PDF (300 DPI)
│       ├── fig1_nms_free_paradox.pdf
│       ├── fig2_displacement_tradeoff.pdf
│       ├── fig3_downstream_cyberphysical_impact.pdf
│       └── fig4_sqd_filter_defense.pdf
│
├── paper_3_defense_guardrail/               # BÀI BÁO 2: Guardrail phòng thủ
│   ├── main_defense.tex                     # Toàn văn bản thảo LaTeX (22 trang)
│   ├── fig/                                 # 11 figures vector PDF
│   ├── experiments/                         # Mã nguồn thực nghiệm độc lập
│   │   ├── run_calibration.py
│   │   ├── run_controls_and_fpr.py
│   │   ├── run_patch_attack_eval.py
│   │   ├── run_adaptive_attacks_eval.py
│   │   └── run_defensive_baselines.py
│   └── refs_defense.bib
│
└── edge_sec_pro/                            # Shared Backend Engine & Dataset Store
    ├── core/                                # API nạp & đánh giá mAP COCO chuẩn (coco_gt.py)
    ├── data/coco5k/                         # Bộ ảnh COCO val2017 & annotations gốc
    ├── results/checkpoints/                 # Precomputed Adversarial Tensors cache (adv_cache/)
    └── transformer_monitors/                # RT-DETR Model Wrapper & feature extraction
```

---

## 🛠 Hướng Dẫn Biên Dịch & Vận Hành (Compilation & Execution)

### Cách 1: Tự động biên dịch toàn bộ bằng 1 lệnh duy nhất (Khuyến nghị)
Script tự động chạy quy trình chuẩn (`pdflatex` → `bibtex` → `pdflatex` → `pdflatex`), dọn dẹp sạch toàn bộ file trung gian rác (`*.aux`, `*.log`, `*.bbl`, `*.blg`), và cập nhật file PDF vào `SUBMISSION_PDFS/`:

```powershell
pwsh -File g:\Edge_Sec\build_all_papers.ps1
```

### Cách 2: Biên dịch thủ công từng bài

```powershell
# 1. Chạy thực nghiệm & biên dịch Bài 1 (Phantom Queries - EQF/DQF):
cd g:\Edge_Sec\paper_2_phantom_fabrication
python experiments\run_phantom_experiments.py
pdflatex -interaction=nonstopmode main_phantom.tex; bibtex main_phantom; pdflatex -interaction=nonstopmode main_phantom.tex; pdflatex -interaction=nonstopmode main_phantom.tex

# 2. Chạy thực nghiệm & biên dịch Bài 2 (Split-Percentile Defense):
cd g:\Edge_Sec\paper_3_defense_guardrail
pdflatex -interaction=nonstopmode main_defense.tex; bibtex main_defense; pdflatex -interaction=nonstopmode main_defense.tex; pdflatex -interaction=nonstopmode main_defense.tex
```
