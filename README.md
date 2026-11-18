# Edge AI Security: Mechanism Analysis, Vulnerabilities & Lightweight Guardrails on RT-DETR

Hệ thống nghiên cứu toàn diện về an ninh kiến trúc mô hình **Real-Time Detection Transformer (RT-DETR)** trên thiết bị tính toán biên (Edge Computing).

Repository được cấu trúc thành **3 bài báo khoa học độc lập**, mỗi bài sở hữu thư mục làm việc, mã nguồn thực nghiệm và bản thảo LaTeX riêng biệt, cùng thư mục phát hành tập trung `SUBMISSION_PDFS/`:

---

## 📚 Danh Sách 3 Bài Báo Độc Lập

### 1. Bài 1: Phẫu Thuật Nhân Quả & Cơ Chế Sụp Đổ Của Dynamic Query Selection
* **Thư mục làm việc**: [`paper_1_aqsm_mechanism/`](paper_1_aqsm_mechanism/)
* **Tiêu đề**: *Mechanistic Breakdown of Dynamic Query Selection in Real-Time Detection Transformers*
* **Tạp chí mục tiêu**: *Engineering Applications of Artificial Intelligence (Elsevier EAAI)*
* **Định dạng & PDF Đầu Ra**:
  * Bản Elsevier Preprint (26 trang): [`SUBMISSION_PDFS/1_Paper_AQSM_Mechanism_Elsevier_EAAI.pdf`](SUBMISSION_PDFS/1_Paper_AQSM_Mechanism_Elsevier_EAAI.pdf)
* **Trọng tâm khoa học**:
  * **Causal Mechanism Surgery (2×2 Factorial Tensor-Swap)**: Phẫu thuật hoán đổi tensor tại ranh giới Encoder $\to$ Decoder (Memory $M$ vs. Query Prior $Q$), chứng minh $M$ là động lực chi phối làm sụt giảm mAP ($\eta_p^2 = 0.902$), trong khi $Q$ đóng vai trò khuếch đại không gian ($\eta_p^2 = 0.820$).
  * **Động lực học tuyển chọn Query (Top-K Perturbation Dynamics)**: AQSM phân hạng bù trừ làm xáo trộn nghiêm trọng thứ tự $K=300$ đề xuất (Top-15 eviction $>93\%$, tương quan Spearman $\rho \approx 0.35$).
  * **Deformable Attention Telemetry**: Entropy chú ý bất biến nhưng Spatial Hit Rate vùng vật thể giảm sâu từ $61.2\%$ xuống $35.8\%$, phân biệt rõ với Query Vanishing.
  * **Phòng thủ siêu nhẹ ESM (Encoder Selection Monitor)**: Giám sát phổ năng lượng Encoder đạt TPR $>84\%$ tại $5\%$ FPR với độ trễ $<0.15$ ms.
* **Thực thi**:
  ```bash
  # Chạy toàn bộ thực nghiệm (Task 0 -> 8):
  bash paper_1_aqsm_mechanism/run_experiments.sh --folder_image "/path/to/coco/val2017"
  ```

---

### 2. Bài 2: Tấn Công Tạo Bão Query Ma Trên Mô Hình Không Dùng NMS (Phantom Queries & Fabrication)
* **Thư mục làm việc**: [`paper_2_phantom_fabrication/`](paper_2_phantom_fabrication/)
* **Tiêu đề**: *Phantom Queries: Exploiting NMS-Free Set Prediction for Adversarial Object Fabrication in Real-Time Transformers*
* **Định dạng & PDF Đầu Ra**:
  * Bản Elsevier Preprint (13 trang): [`SUBMISSION_PDFS/3_Paper_Phantom_Queries_Fabrication_Elsevier.pdf`](SUBMISSION_PDFS/3_Paper_Phantom_Queries_Fabrication_Elsevier.pdf)
* **Trọng tâm khoa học**:
  * **Nghịch lý An ninh NMS-Free (NMS-Free Security Paradox)**: Việc loại bỏ NMS giúp RT-DETR triệt tiêu các cuộc tấn công trễ tính toán, nhưng đồng thời tước bỏ lớp lọc không gian đầu ra, khiến $100\%$ các query ma lọt thẳng vào downstream controllers (YOLOv8 + NMS lọc được $78.4\%$ box ma nhưng bị vọt trễ $+242\%$).
  * **2 Véc-tơ Tấn công Chuyên sâu**: EQF (Encoder Query Fabrication) và DQF (Decoder Query Fabrication) đạt $\mathrm{ASR}_{\mathrm{fab}} \ge 99.7\%$.
  * **Giải pháp phòng thủ SQD-Filter (Spatial Query Deduplication Filter)**: Siêu nhẹ ($0.08$ ms GPU, $1.15$ ms CPU), loại bỏ $>91.5\%$ phantom boxes.

---

### 3. Bài 3: Hệ Thống Phòng Thủ Đa Tầng Bằng Guardrail Phân Vị (Split-Percentile Defense)
* **Thư mục làm việc**: [`paper_3_defense_guardrail/`](paper_3_defense_guardrail/)
* **Tiêu đề**: *Split-Percentile Multi-Scale Latency Guardrails for Real-Time Edge Vision Transformers*
* **Định dạng & PDF Đầu Ra**:
  * Bản Elsevier Preprint (22 trang): [`SUBMISSION_PDFS/4_Paper_Defense_Split_Percentile_Guardrails.pdf`](SUBMISSION_PDFS/4_Paper_Defense_Split_Percentile_Guardrails.pdf)
* **Trọng tâm khoa học**:
  * Kiến trúc phòng thủ chủ động kiểm soát phân vị đa thang đo (Split-Percentile Guardrail) giám sát năng lượng đặc trưng và độ tin cậy query của RT-DETR theo thời gian thực.

---

## 📂 Cấu Trúc Workspace

```
g:\Edge_Sec\
├── SUBMISSION_PDFS/                          # Thư mục lưu trữ bản PDF xuất bản chính thức
│   ├── 1_Paper_AQSM_Mechanism_Elsevier_EAAI.pdf
│   ├── 3_Paper_Phantom_Queries_Fabrication_Elsevier.pdf
│   └── 4_Paper_Defense_Split_Percentile_Guardrails.pdf
│
├── paper_1_aqsm_mechanism/                  # BÀI BÁO 1: Mechanistic Breakdown (Elsevier EAAI)
│   ├── main.tex                             # Toàn văn bản thảo LaTeX (26 trang chuẩn Elsevier)
│   ├── elsarticle.cls
│   ├── refs_EAAI.bib
│   ├── run_all_experiments.py               # Master Orchestrator toàn bộ Task 0 -> 8
│   ├── run_experiments.sh                   # Script Bash 1-click
│   ├── run_experiments.ps1                  # Script PowerShell 1-click
│   ├── kaggle_run_experiments.py            # Runner tối ưu hóa GPU Kaggle / Colab
│   ├── core/                                # Module nền tảng (Wrapper, Cache, Loader, Analyzer, Metrics)
│   ├── experiments/                         # Cấu hình, Logger, Engines thực nghiệm và Tasks 1-8
│   ├── outputs/                             # Summary JSON, Raw CSV, Logs kiểm toán
│   └── fig/                                 # Vector figures PDF & PNG xuất bản (Figures 3-8)
│
├── paper_2_phantom_fabrication/             # BÀI BÁO 2: Phantom Queries & Object Fabrication
│   ├── main_phantom.tex
│   └── experiments/
│
├── paper_3_defense_guardrail/               # BÀI BÁO 3: Guardrail Phòng Thủ Đa Tầng
│   ├── main_defense.tex
│   └── experiments/
│
├── edge_sec_pro/                            # Shared Backend Engine & Dataset Store
│   ├── core/                                # API nạp & đánh giá mAP COCO chuẩn (coco_gt.py)
│   └── data/coco5k/                         # Bộ ảnh COCO val2017 & annotations gốc
│
├── tools_archive/                           # Kho lưu trữ các công cụ vá lỗi / kiểm toán văn bản cũ
└── build_all_papers.ps1                     # Script tự động biên dịch trọn bộ 3 bài báo ra PDF
```

---

## 🛠 Hướng Dẫn Vận Hành & Biên Dịch (Compilation & Execution)

### 1. Chạy thực nghiệm Bài 1 (AQSM Mechanism Breakdown)
```bash
# Trên Linux / Git Bash / WSL:
bash paper_1_aqsm_mechanism/run_experiments.sh --folder_image "/path/to/coco/val2017"

# Trên Windows PowerShell:
.\paper_1_aqsm_mechanism\run_experiments.ps1 -FolderImage "G:\Edge_Sec\edge_sec_pro\data\coco5k\val2017"
```

### 2. Biên dịch toàn bộ 3 bài báo ra PDF xuất bản
```powershell
pwsh -File g:\Edge_Sec\build_all_papers.ps1
```
