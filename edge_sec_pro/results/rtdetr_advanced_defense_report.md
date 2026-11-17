# Báo Cáo Thực Nghiệm Khoa Học Chuẩn Mực: RT-DETR Streamlined Security Evaluation (320x320)

- **Mô hình**: `PekingU/rtdetr_r18vd` (Pretrained COCO, 80 Classes, 300 Queries, 20.17M tham số)
- **Độ phân giải**: **$320 \times 320$** (Chuẩn thị giác biên Edge AI)
- **Hạ tầng**: Edge-class CPU execution profile (x86-64 embedded workstation / Intel Core i5-12600H)
- **Tập dữ liệu**: COCO val2017 ($N_{cal} = 1200$, $N_{test} = 600$, Zero Data Leakage)

## 1. Bảng Kiểm Định Tỷ Lệ Báo Động Oan (FPR) Dưới Biến Dạng Tự Nhiên & Clean Test

| Kịch bản Kiểm Định | N | ActiveQ Split | Top1Mean Split | ESM Dropoff | LayerMaha | **Calibrated Max-OR** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Clean Test Set (Ảnh sạch độc lập)** | 600 | 5.0% | 6.7% | 5.0% | 6.5% | **5.2%** |
| **Control: Gaussian Noise (sigma=16/255)** | 300 | 5.0% | 5.7% | 6.3% | 8.0% | **6.0%** |
| **Control: Random Patch (48x48 Random Pos)** | 300 | 4.7% | 4.7% | 4.7% | 5.3% | **4.3%** |
| **Clean Shift: Gaussian Blur (5x5, sigma=1.5)** | 300 | 4.3% | 1.7% | 6.3% | 5.0% | **5.7%** |
| **Clean Shift: Brightness (-30%)** | 300 | 4.7% | 5.7% | 6.0% | 8.0% | **6.0%** |


## 2. Bảng Đối Đầu Toàn Diện Tại Ngưỡng Khớp FPR (Matched 5% Nominal FPR)

| Kịch bản Thử nghiệm | N | ASR [95% CI] | ActiveQ Split | Top1Mean Split | ESM Dropoff | LayerMaha | **Calibrated Max-OR** | Oracle Best Single |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Control: Gaussian Noise (sigma=16/255)** | 300 | - | 0.477 (5.0%) | 0.490 (5.7%) | 0.511 (6.3%) | 0.498 (8.0%) | **0.515** (**6.0%**) | - |
| **Control: Random Patch (48x48 Random Pos)** | 300 | - | 0.502 (4.7%) | 0.483 (4.7%) | 0.503 (4.7%) | 0.480 (5.3%) | **0.494** (**4.3%**) | - |
| **Clean Shift: Gaussian Blur (5x5, sigma=1.5)** | 300 | - | 0.466 (4.3%) | 0.444 (1.7%) | 0.511 (6.3%) | 0.502 (5.0%) | **0.484** (**5.7%**) | - |
| **Clean Shift: Brightness (-30%)** | 300 | - | 0.488 (4.7%) | 0.495 (5.7%) | 0.514 (6.0%) | 0.494 (8.0%) | **0.505** (**6.0%**) | - |
| **Attack: Gradient Patch (48x48 Random Pos)** | 300 | 95.0% [91.9%, 97.2%] | 0.414 (2.7%) | 0.419 (4.7%) | 0.458 (3.0%) | 0.434 (3.7%) | **0.425** (**2.3%**) | Top1Mean_Split (4.7%) |
| **Attack: Query Selection Manipulation (AQSM)** | 300 | 73.2% [67.7%, 78.2%] | 0.711 (25.0%) | 0.813 (50.0%) | 0.889 (56.0%) | 0.534 (9.7%) | **0.833** (**47.7%**) | ESM_Margin (57.7%) |
| **Attack: Encoder Query Fabrication (EQF, 15-step PGD)** | 300 | 100.0% [98.8%, 100.0%] | 0.929 (56.7%) | 0.897 (49.3%) | 0.826 (14.7%) | 0.929 (66.3%) | **0.922** (**52.7%**) | ESM_Margin (86.7%) |
| **Attack: PGD Query Vanishing (15 steps)** | 300 | 46.7% [40.9%, 52.5%] | 0.956 (82.3%) | 0.692 (23.3%) | 0.622 (16.7%) | 0.720 (13.3%) | **0.918** (**72.7%**) | ActiveQ_Split (82.3%) |
| **Attack: PGD Query Fabrication (DQF, 15 steps)** | 300 | 100.0% [98.8%, 100.0%] | 0.839 (31.7%) | 0.765 (27.3%) | 0.708 (12.0%) | 0.846 (30.0%) | **0.826** (**21.3%**) | ActiveQ_Split (31.7%) |
| **Attack: Adaptive Evasion PGD (Hinge, lambda=0.5)** | 300 | 99.3% [97.6%, 99.9%] | 0.769 (2.0%) | 0.687 (5.0%) | 0.657 (5.7%) | 0.804 (27.3%) | **0.744** (**9.0%**) | Layer_Maha (27.3%) |
| **Attack: Joint Adaptive Evasion (1D + Maha Awareness)** | 300 | 98.0% [95.7%, 99.3%] | 0.501 (0.3%) | 0.430 (2.3%) | 0.530 (7.0%) | 0.838 (38.0%) | **0.746** (**13.7%**) | Layer_Maha (38.0%) |
| **Attack: Adaptive ESM Evasion (Dropoff-Aware, lambda=2.0)** | 300 | 71.5% [65.9%, 76.6%] | 0.577 (5.0%) | 0.691 (17.7%) | 0.769 (16.0%) | 0.468 (5.0%) | **0.683** (**8.3%**) | ESM_Margin (25.3%) |
| **Attack: Adaptive Fused Ensemble Evasion (All-Tiers)** | 300 | 98.0% [95.7%, 99.3%] | 0.505 (2.0%) | 0.452 (2.7%) | 0.526 (4.7%) | 0.816 (30.7%) | **0.725** (**12.7%**) | Layer_Maha (30.7%) |


## 3. Thực Nghiệm Cô Lập Cơ Chế Nhân Quả AQSM (Causal Mechanism Isolation)

- **Tỷ lệ Proposal sạch bị văng khỏi Top-300 (GT Eviction Rate)**: **56.4%** (169.3 / 300 queries bị thay thế bởi nền ảo giác)
- **Top-1 Proposal Score Drop**: Sạch 0.772 -> AQSM 0.684 (Proposal đứng đầu vẫn giữ điểm cao nhưng GT bị đẩy khỏi top-300)

| Nhánh Thử Nghiệm (Arm) | Cấu Hình Feature & Query Index | mAP@50 | mAP@50:95 | Ý Nghĩa Cơ Chế |
|---|---|:---:|:---:|---|
| **Arm A (Clean Baseline)** | $F_{\mathrm{clean}} + \mathcal{I}_{\mathrm{clean}}$ | 100.0% | 100.0% | Chuẩn đối chứng mAP sạch |
| **Arm B (AQSM Attack)** | $F_{\mathrm{adv}} + \mathcal{I}_{\mathrm{adv}}$ | 10.9% | 5.8% | Sụp đổ mAP (-89.1%) |
| **Arm C (Oracle Selection)** | $F_{\mathrm{adv}} + \mathcal{I}_{\mathrm{clean}}$ | **16.5%** | **9.2%** | **Hồi phục +51.4% (Chứng minh Information Starvation)** |
| **Arm D (Reverse Swap)** | $F_{\mathrm{clean}} + \mathcal{I}_{\mathrm{adv}}$ | 96.1% | 94.2% | Khả năng tự bảo vệ khi feature sạch |

### Phân Tích Đóng Góp Thành Phần Hàm Mất Mát (Loss Ablation):

- **Thuần Ranking Margin (L_rank + L_bg)**: mAP@50 = **21.6%** (Một mình ranking margin làm sập 78.4% mAP)
- **Thuần Suppression (L_supp)**: mAP@50 = **13.0%**
- **AQSM Phối Hợp (Combined)**: mAP@50 = **10.9%**


## 4. Quét Độ Nhạy Ngân Sách Nhiễu (Epsilon Sensitivity Sweep: 2, 4, 8, 16 / 255)

| Ngân Sách Nhiễu (ε) | AQSM mAP@50 | ESM Dropoff | Vanishing mAP@50 | ActiveQ Split | Nhận Xét |
|:---:|:---:|:---:|:---:|:---:|---|
| **2/255** | 80.0% | 39.05 | 0.0% | 0.00 | AQSM sụt giảm nhanh hơn Vanishing |
| **4/255** | 55.8% | 27.92 | 0.0% | 0.00 | AQSM sụt giảm nhanh hơn Vanishing |
| **8/255** | 32.3% | 22.43 | 0.0% | 0.00 | AQSM sụt giảm nhanh hơn Vanishing |
| **16/255** | 13.9% | 20.13 | 0.0% | 0.00 | AQSM sụt giảm nhanh hơn Vanishing |


## 5. Kiểm Định Thống Kê Ghép Cặp McNemar (Paired Hypothesis Testing, N=300)

| Cặp So Sánh Giả Thuyết | Chi-square (χ²) | Odds Ratio | p-value | Kết Luận Thống Kê |
|---|:---:|:---:|:---:|---|
| **AQSM: ESM Dropoff (56.0%) vs Top1Mean (50.0%)** | 7.605 | 2.80 | 0.0051 | **Có ý nghĩa thống kê (p < 0.01)** |
| **AQSM: ESM Dropoff (56.0%) vs ActiveQ Split (25.0%)** | 74.903 | 10.30 | 1.3214e-20 | **Có ý nghĩa thống kê (p < 0.01)** |
| **AQSM: ESM Margin (57.7%) vs Top1Mean (50.0%)** | 16.690 | 8.67 | 1.5236e-05 | **Có ý nghĩa thống kê (p < 0.01)** |
| **Query Vanishing: ActiveQ Split (82.3%) vs Top1Mean (23.3%)** | 175.006 | inf | 1.0440e-53 | **Có ý nghĩa thống kê (p < 0.01)** |
| **Query Vanishing: ActiveQ Split (82.3%) vs ESM Dropoff (16.7%)** | 193.045 | 198.00 | 4.9784e-58 | **Có ý nghĩa thống kê (p < 0.01)** |
