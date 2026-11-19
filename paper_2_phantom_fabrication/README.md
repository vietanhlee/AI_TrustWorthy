# Paper 2: Phantom Queries Fabrication in NMS-Free RT-DETR

> **Target Venue**: *Elsevier Engineering Applications of Artificial Intelligence (EAAI)*  
> **Status**: Camera-Ready Production Codebase & Automated Telemetry Pipeline  
> **Scientific Integrity**: 100% Authentic Empirical Computation (Strictly Zero Hardcoding / Synthetic Noise)

---

## 1. Tổng quan Kiến trúc Thư mục (Architecture Overview)

Cấu trúc thư mục của Paper 2 được chuẩn hóa theo chuẩn Production, phân tách rõ ràng giữa mã nguồn lõi (`core/`), bộ công cụ thực nghiệm (`experiments/`), kết quả đo đạc thực chứng (`results/`), đồ thị xuất bản (`fig/`) và kho lưu trữ mã nguồn lịch sử (`legacy_archive/`).

```text
paper_2_phantom_fabrication/
├── README.md                              # Tài liệu hướng dẫn toàn diện
├── main_phantom.tex                       # Bản thảo LaTeX nộp Elsevier EAAI
├── main_phantom.pdf                       # Bản thảo PDF đã biên dịch hoàn chỉnh
├── refs_EAAI.bib                          # Danh mục trích dẫn tài liệu tham khảo
├── elsarticle.cls                         # Định dạng chuẩn tạp chí Elsevier
├── run_all.sh                             # Script thực thi toàn diện cho Linux / Kaggle / Colab
│
├── core/                                  # Mô-đun nền tảng (Domain Logic & Config)
│   ├── __init__.py                        # Định nghĩa package exports
│   ├── adv_cache.py                       # Quản lý bộ nhớ đệm đối kháng chuẩn IEEE 754 float32 (Zero uint8 Loss)
│   ├── coco_gt.py                         # Ingestion COCO, Streaming DataLoader, Hungarian Matcher, VOC mAP
│   └── config.py                          # Cấu hình siêu tham số, thread CPU, cache dir và thiết bị
│
├── experiments/                           # Hệ thống thực nghiệm chính (Production Engine)
│   ├── __init__.py                        # Package init
│   ├── run_full_paper2_suite.py           # Master Suite điều phối 3 Stage thực nghiệm & Streaming Loader
│   ├── plot_paper_figures.py              # Sinh đồ thị vector (Figures 1-4) từ telemetry
│   ├── sync_telemetry_to_paper.py         # Đồng bộ dữ liệu JSON vào Tables 1-4 & build PDF
│   └── sqd_filter.py                      # Bộ lọc phòng thủ Spatial Query Dispersion (SQD)
│
├── cache/                                 # Thư mục lưu trữ bộ nhớ đệm mẫu đối kháng
│   └── adv_cache_640/                     # Cache delta đối kháng .npz chuẩn float32 nguyên bản (tự động bỏ qua git)
│
├── fig/                                   # Đồ thị vector và hình minh họa bài báo
│   ├── fig1_concept_paradox.png           # Sơ đồ nghịch lý NMS-Free
│   ├── fig1_nms_free_paradox.pdf          # Đồ thị phân phối Query Top-K & IoU clustering
│   ├── fig2_displacement_tradeoff.pdf     # Đánh đổi ngân sách nhiễu L_inf vs Tỉ lệ chế tạo
│   ├── fig3_downstream_cyberphysical_impact.pdf # Mô phỏng phanh khẩn cấp AEB (TTC/Stopping)
│   ├── fig4_sqd_filter_defense.pdf        # Hiệu quả bộ lọc SQD theo bán kính r_filter
│   └── fig_qualitative_detections.png     # Ảnh minh họa định tính phát hiện bóng ma
│
├── results/                               # Dữ liệu thực chứng khoa học (Empirical Artifacts)
│   ├── paper2_complete_telemetry.json     # Telemetry tổng hợp 100% số liệu cho Paper
│   ├── multiseed_640_audit.json           # Dữ liệu kiểm toán multi-seed benchmark
│   ├── parallel_640_ablations_audit.json  # Dữ liệu kiểm toán ablations chuyên sâu
│   └── multiseed_checkpoints/             # Checkpoints từng seed (seed_42, seed_123, seed_456)
│
└── legacy_archive/                        # Kho lưu trữ các phiên bản thử nghiệm prototype trước đây
    ├── scripts/                           # Các script đơn lẻ đã được hợp nhất vào Master Suite
    ├── old_results/                       # Các kết quả kiểm thử cũ
    ├── old_logs/                          # Nhật ký chạy đa luồng trước đây
    └── adv_cache/                         # Cache nhiễu đối kháng prototype cũ
```

---

## 2. Chu trình Thực nghiệm 3 Giai đoạn (Three-Stage Empirical Pipeline)

Toàn bộ hệ thống thực nghiệm được tự động hóa thông qua `run_full_paper2_suite.py`:

```
               [ Input COCO Images (--folder_image) ]
                               │
       ┌───────────────────────┴───────────────────────┐
       ▼                                               ▼
[ Stage 1: Multi-Seed Benchmark ]          [ Stage 2: In-Depth Ablations ]
 - Clean Baseline (AP50, Latency)           - Object Size Breakdown (S / M / L)
 - AQSM Attack (ASR, mAP Drop)              - Cross-Class Transferability (5 classes)
 - PGD / Patch Baselines                    - ByteTrack MOT Multi-Object Tracking
 - 5 Defenses (Median, Gauss, NMS, SQD)     - AEB 3D Pinhole Distance & TTC Simulation
 - Multi-GPU DataParallel Acceleration      - SQD Radius Hyperparameter Grid Sweep
       │                                               │
       └───────────────────────┬───────────────────────┘
                               │
                               ▼
                [ Stage 3: Vector Graphics Gen ]
                 - Reads compiled metrics
                 - Emits publication PDFs (Figs 1-4)
                               │
                               ▼
             [ results/paper2_complete_telemetry.json ]
```

---

## 3. Hướng dẫn Thực thi (Execution Guide)

### Cách 1: Chạy toàn bộ bằng Bash Script (Khuyến nghị trên Kaggle / Linux Server)
Script `run_all.sh` tự động nhận diện phần cứng, cấp phát tiến trình và hỗ trợ tham số linh hoạt:

```bash
# 1. Cấp quyền thực thi
chmod +x run_all.sh

# 2. Chạy toàn bộ ảnh COCO (mặc định N=0: chạy 100% full dataset)
./run_all.sh \
  --folder_image "/kaggle/input/datasets/huecute/coco5k/coco5k/val2017" \
  --annotations "/kaggle/input/datasets/huecute/coco5k/coco5k/annotations/instances_val2017.json"

# Tùy chọn giới hạn số lượng ảnh nếu muốn kiểm thử nhanh (quick dry-run):
./run_all.sh \
  --folder_image "/kaggle/input/datasets/huecute/coco5k/coco5k/val2017" \
  --num_benchmark 100 \
  --num_ablation 300
```

### Cách 2: Chạy trực tiếp qua Python CLI

```bash
python experiments/run_full_paper2_suite.py \
  --folder_image /kaggle/input/datasets/huecute/coco5k/coco5k/val2017 \
  --annotations /kaggle/input/datasets/huecute/coco5k/coco5k/annotations/instances_val2017.json \
  --num_images_benchmark 0 \
  --num_images_ablation 0 \
  --seeds 42,123,456 \
  --device auto \
  --cache_dir cache/adv_cache_640 \
  --output_json results/paper2_complete_telemetry.json
```

**Các tham số chính:**
- `--folder_image`: Đường dẫn thư mục chứa ảnh JPEG đánh giá (bắt buộc).
- `--annotations`: Đường dẫn file `instances_val2017.json` (tự động dò tìm nếu để trống).
- `--num_images_benchmark`: Số lượng ảnh đánh giá trên mỗi seed ở Stage 1 (mặc định: `0` = chạy toàn bộ ảnh).
- `--num_images_ablation`: Số lượng ảnh đánh giá ở Stage 2 (mặc định: `0` = chạy toàn bộ ảnh).
- `--seeds`: Danh sách random seeds độc lập (mặc định: `42,123,456`).
- `--device`: Thiết bị tính toán (`auto`, `cuda`, `cpu`).
- `--cache_dir`: Thư mục lưu và tái sử dụng bộ nhớ đệm đối kháng `.npz` chuẩn float32 (mặc định: `cache/adv_cache_640`).
- `--skip_cache_gen`: Chỉ nạp cache có sẵn, không sinh cache mới.
- `--force_recompute`: Ép buộc tính toán lại và ghi đè cache.

---

## 4. Kiến trúc Streaming & Bộ nhớ đệm Đối kháng (Adversarial Cache Architecture)

Theo chuẩn mực kỹ thuật đã thiết lập tại Paper 1:
1. **Lưu trữ chuẩn IEEE 754 float32 nguyên bản (Zero uint8 Loss)**:
   - Perturbation delta $\delta = x_{adv} - x_{clean}$ được lưu giữ chính xác tuyệt đối ở định dạng `float32`.
   - Tuyệt đối không ép kiểu hay làm tròn sang `uint8`, tránh làm sai lệch gradient và tính chất toán học của các cuộc tấn công đối kháng.
2. **Loại bỏ hoàn toàn nguy cơ tràn RAM (Out-Of-Memory SIGKILL 57)**:
   - Dùng `CocoCohortDataset` nạp ảnh lười theo dòng (Streaming DataLoader) kết hợp `np.memmap` trên đĩa.
   - Mức tiêu thụ RAM thực tế luôn được giữ ổn định dưới 1.5 GB ngay cả khi xử lý toàn bộ 5,000 ảnh COCO.
3. **Tái sử dụng tức thì (Instant Cache Hit)**:
   - Khi chạy đa hạt giống (Multi-Seed) trên toàn bộ tập dữ liệu, seed đầu tiên sẽ sinh và ghi cache vào đĩa.
   - Các seed tiếp theo và các bài kiểm tra chuyên sâu tự động phát hiện `[Cache Hit]` và nạp trực tiếp qua `mmap_mode="r"`, giảm thời gian thực thi từ hàng giờ xuống vài chục giây.

---

## 4. Cơ chế Đo lường Độ trễ Chuyên sâu theo Thiết bị (Device-Aware Latency)

Để đảm bảo tính chuẩn xác và loại trừ hoàn toàn overhead đồng bộ giữa CPU và GPU:
- **Trên NVIDIA CUDA GPU**:
  - Thực hiện 5 lần warm-up forward pass để ổn định xung nhịp GPU.
  - Sử dụng cặp sự kiện `torch.cuda.Event(enable_timing=True)` đo thời gian thực thi thuần túy của CUDA kernels.
  - Tự động kích hoạt `torch.nn.DataParallel` khi phát hiện `device_count > 1` để tối ưu hóa băng thông.
- **Trên CPU Biên (Edge CPU)**:
  - Sử dụng đồng hồ độ phân giải cao `time.perf_counter()` đo lường forward passes.
  - Trích xuất thông tin phần cứng CPU/VRAM/GPU Model vào trường `metadata.hardware` của JSON.

---

## 5. Quy trình Đồng bộ Tự động vào Manuscript & Biên dịch PDF

Sau khi chạy xong và có file `results/paper2_complete_telemetry.json`, bạn chỉ cần chạy:

```bash
python experiments/sync_telemetry_to_paper.py --telemetry results/paper2_complete_telemetry.json
```

**Tự động cập nhật:**
1. **Table 1** (Multi-seed Empirical Benchmark): Điền Clean AP50, AQSM AP50, ASR, Target IoU, FPS, VRAM (Mean ± Std trên 3 seeds).
2. **Table 2** (Defensive Guardrails Evaluation): Điền Clean mAP, Under Attack mAP, Attack ASR, Latency Overhead cho RT-DETR gốc, Median Blur, Gaussian Blur, Post-hoc NMS và SQD Filter.
3. **Table 3** (Multi-dimensional Ablation Studies): Cập nhật độ nhạy theo kích thước (Small / Medium / Large), khả năng chuyển giao đa lớp (Car, Person, Bicycle, Bus, Truck), và so sánh phòng thủ.
4. **Table 4** (Downstream Cyber-Physical Impact): Cập nhật khoảng cách ước lượng AEB (m), độ suy giảm TTC (s), tỉ lệ phanh ma (%) và số lần chuyển đổi định danh ByteTrack (ID Switches).
5. **Biên dịch LaTeX**: Tự động gọi chuỗi `pdflatex` + `bibtex` + `pdflatex` (2 lần) và xuất bản tệp nộp bài sạch tại `SUBMISSION_PDFS/3_Paper_Phantom_Queries_Fabrication_Elsevier.pdf`.
