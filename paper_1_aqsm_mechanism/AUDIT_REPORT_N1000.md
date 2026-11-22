# BÁO CÁO KIỂM TOÁN VÀ ĐỒNG BỘ SỐ LIỆU TOÀN DIỆN $N=1000$ (AUDIT & SCIENTIFIC FEEDBACK REPORT)
**Dự án:** Paper 1 - Adversarial Query Selection Manipulation (AQSM) in Real-Time Detection Transformers  
**Tập tin kiểm toán:** `G:\Edge_Sec\paper_1_aqsm_mechanism\main.tex`  
**Nguồn dữ liệu chân lý (Ground Truth Telemetry):** `G:\Edge_Sec\paper_1_aqsm_mechanism\outputs\summary_json\` ($N=1000$ images, $N_{\mathrm{valid}} = 992$, $S = 8,400$ candidate anchors tại độ phân giải Native $640 \times 640$)  
**Kiểm toán viên:** Agent 1 (Auditor & Scientific Feedback Specialist) & Agent 2 (Writer & Synchronizer)  
**Ngày lập:** 21/11/2026 (Đồng bộ hoàn thiện: 22/11/2026)  

---

## I. TỔNG QUAN KẾT QUẢ KIỂM TOÁN (EXECUTIVE SUMMARY)

Sau khi đối chiếu toàn diện file mã nguồn `main.tex` với bộ dữ liệu thực nghiệm mới $N=1,000$ ảnh từ Kaggle cluster (kết quả gốc `results (4)`), kiểm toán viên xác nhận:
1. **Nâng cấp quy mô mẫu chuẩn mực:** Bộ số liệu mới nâng quy mô từ $N=400$ ($N_{\mathrm{valid}} = 397$) lên $N=1,000$ ($N_{\mathrm{valid}} = 992$ ảnh có đối tượng ground-truth, $df = [1, 991]$). Toàn bộ các vị trí xuất hiện `$N = 400$` hoặc `$N=400$` trong văn bản đã được cập nhật đồng bộ sang `$N = 1,000$`.
2. **Liêm chính học thuật và triệt tiêu Overclaiming:**
   - **Về Cơ chế Phân rã Causal Surgery (Table 3 & 4):**  
     Bản mới $N=1,000$ xác nhận: Tác động biên có dấu (signed fraction of total drop) là **Memory $M$: $61.43\%$** ($-44.97$ pp) và **Query $Q$: $38.57\%$** ($-28.24$ pp).  
     Về tỷ trọng tổng bình phương phân tích phương sai (ANOVA Condition SS Share): **Memory $M$: $67.97\%$** ($SS = 2,006,395.4, \eta_p^2 = 0.806$), **Query $Q$: $26.81\%$** ($SS = 791,287.8, \eta_p^2 = 0.735$), và **Tương tác $M \times Q$: $5.23\%$** ($SS = 154,315.4, \eta_p^2 = 0.291$) với $F_{M} = 4112.11, F_{Q} = 2753.39, F_{M \times Q} = 406.30$, tất cả $p < 10^{-15}$.  
     Kiểm định phi tham số Wilcoxon Signed-Rank xác nhận tính vững chắc tuyệt đối:
     - Arm $\text{D}_{\text{full}}$ vs Arm B: phục hồi $+22.62$ pp ($W = 236.5, p = 1.19 \times 10^{-130}$, Cohen's $d = 1.07$, 95% BCa CI $[+21.25, +25.00]$ pp).
     - Arm $\text{D}_{\text{index}}$ vs Arm A: sụt giảm $-32.46$ pp ($W = 4669.5, p = 3.61 \times 10^{-122}$, Cohen's $d = -1.09$, 95% BCa CI $[-34.53, -29.94]$ pp).
   - **Về Deformable Attention Entropy & Foreground Hits (Table 7):**  
     Entropy Layer 1 trên $N=1000$ dịch chuyển từ $0.8263$ (Clean) sang $0.8146$ (AQSM), trung bình 3 decoder layers là $0.8009 \to 0.7901$ ($d = -0.21$, Wilcoxon $W = 186,431.0, p = 3.38 \times 10^{-11}$). Tỷ lệ lấy mẫu trúng foreground (FG hit rate) sụp đổ từ $63.51\%$ xuống $38.77\%$ ($d = -1.03$, $t = 32.62, p = 1.77 \times 10^{-159}$). Dropoff Ratio về đúng $6.90 \pm 3.72$, Score Margin về đúng $0.6580 \pm 0.0907$, Reference Drift $0.4924 \pm 0.0605$.  
     *Nguyên tắc hành văn:* Khẳng định Deformable Attention không phải là nguyên nhân khởi phát mà là nạn nhân thụ động của việc tráo đổi tọa độ anchor từ tầng query selection (Spatial Target Misrouting).
   - **Về Hardware Latency Profiling (Table 10):**  
     Tesla T4 profiling đạt $26.75\text{ FPS}$, tổng độ trễ $37.39 \pm 0.89\text{ ms}$ (Backbone: $19.34\text{ ms}$ [$51.73\%$], Hybrid Encoder: $10.10\text{ ms}$ [$27.01\%$], Query Selection: $1.64\text{ ms}$ [$4.38\%$], Deformable Decoder: $6.04\text{ ms}$ [$16.14\%$]). Độ trễ duy trì tính bất biến giữa đầu vào sạch và đầu vào đối kháng.

---

## II. ĐỐI CHIẾU SỐ LIỆU CHI TIẾT 12 BẢNG BIỂU ($N=1000$ CHÂN LÝ RESULTS 4)

### 1. Clean Baseline & Tấn công Chính (Abstract, Sec 1, Table 1 & 2)
- **Clean mAP@50:** **$65.50\%$** (mAP@50:95: **$48.89\%$**)
- **AQSM Arm B mAP@50:** **$2.50\%$** (giảm tương đối **$-96.19\%$**), mAP@50:95: **$1.37\%$**
- **Query Vanishing mAP@50:** **$1.16\%$** (giảm tương đối **$-98.22\%$**), mAP@50:95: **$0.68\%$**
- **Standard Task PGD mAP@50:** **$0.67\%$** (giảm tương đối **$-98.98\%$**), mAP@50:95: **$0.41\%$**
- **AQSM Eviction Rate:** **$87.79 \pm 8.86\%$** (top-300 overall replacement), Top-15 FG Eviction: **$98.50\%$**
- **TaskPGD Eviction Rate:** **$73.62 \pm 10.58\%$**
- **Vanishing Eviction Rate:** **$69.61 \pm 12.65\%$**

### 2. Table 3: Causal Surgery 6-Arm Matrix
| Cấu hình | AQSM ($N=1000$) | TaskPGD ($N=1000$) | Vanishing ($N=1000$) |
|---|---|---|---|
| **Arm A (Clean M + Clean Q)** | $65.50 / 48.89$ | $65.50 / 48.89$ | $65.50 / 48.89$ |
| **Arm B (Adv M + Adv Q)** | $2.50 / 1.37$ | $0.67 / 0.41$ | $1.16 / 0.68$ |
| **Arm C_full (Adv M + Clean Q)** | $6.25 / 3.41$ | $1.40 / 0.85$ | $3.79 / 2.27$ |
| **Arm C_index (Adv M + Clean Idx)** | $3.07 / 1.69$ | $0.87 / 0.52$ | $1.49 / 0.84$ |
| **Arm D_full (Clean M + Adv Q)** | $25.11 / 17.13$ | $39.50 / 28.97$ | $38.71 / 28.60$ |
| **Arm D_index (Clean M + Adv Idx)** | $33.04 / 24.17$ | $49.89 / 38.02$ | $51.43 / 38.81$ |

### 3. Table 4: Repeated-Measures ANOVA Decomposition ($N_{\mathrm{valid}} = 992$, $df = [1, 991]$)
- **Tác động biên (Marginal Effects):**
  - Memory $M$: $\text{ME}_M = -44.97$ pp [$-46.34, -43.54$], $\eta_p^2 = 0.806$ ($F = 4112.1, p < 10^{-15}$)
  - Query $Q$: $\text{ME}_Q = -28.24$ pp [$-29.31, -27.23$], $\eta_p^2 = 0.735$ ($F = 2753.4, p < 10^{-15}$)
  - Tương tác $M \times Q$: $+12.47$ pp [$+11.32, +13.68$], $\eta_p^2 = 0.291$ ($F = 406.3, p < 10^{-15}$)
- **Tổng bình phương ANOVA (Sum of Squares):**
  - Memory $M$: $2,006,395.4$ ($67.97\%$ treatment factor SS)
  - Query $Q$: $791,287.8$ ($26.81\%$ treatment factor SS)
  - Tương tác $M \times Q$: $154,315.4$ ($5.23\%$ treatment factor SS)

### 4. Non-Parametric Contrasts & BCa Bootstrap Differences (95% CI)
- **Arm D_full - Arm B:** **$+22.62\text{ pp}$** [$+21.25, +25.00$] ($W = 236.5, p = 1.19 \times 10^{-130}$, $d = 1.07$)
- **Arm D_index - Arm A:** **$-32.46\text{ pp}$** [$-34.53, -29.94$] ($W = 4669.5, p = 3.61 \times 10^{-122}$, $d = -1.09$)
- **Arm C_full - Arm A:** **$-59.25\text{ pp}$** [$-61.60, -57.32$] ($W = 822.0, p = 1.07 \times 10^{-154}$, $d = -2.02$)
- **Arm B - Arm A:** **$-63.01\text{ pp}$** [$-65.69, -61.26$] ($W = 0.0, p = 6.50 \times 10^{-163}$, $d = -2.95$)

### 5. Table 5: Multi-Scale Encoder Memory Deformation
- **Cosine Similarity (AQSM):**
  - Level S3 ($80 \times 80$): **$0.5917$** ($0.592$)
  - Level S4 ($40 \times 40$): **$0.5074$** ($0.507$)
  - Level S5 ($20 \times 20$): **$0.5846$** ($0.585$)
  - Foreground Tokens: **$0.5964$**, Background Tokens: **$0.5800$**
  - Overall Cosine: **$0.5753$**
- **Relative L2 Perturbation (AQSM):**
  - S3: **$0.9407$**, S4: **$1.0261$**, S5: **$0.9315$**, Foreground: **$0.8727$**, Background: **$0.9782$**, Overall: **$0.9565$**
  - FG/BG Ratio: **$0.8921$**, Cosine Gap ($S_5 - S_4$): **$+0.0772$**

### 6. Table 6: Query Selection Perturbation Dynamics
- **Top-300 Overlap:** AQSM **$12.21\%$**, Vanishing **$30.39\%$**, Task PGD **$26.38\%$**
- **Evicted Top-15 FG:** AQSM **$98.50\%$**, Vanishing **$51.06\%$**, Task PGD **$56.01\%$**
- **Tương quan thứ hạng:** AQSM Spearman $\rho = 0.165$, Kendall $\tau = 0.115$
- **Độ bao phủ mục tiêu:** Center-in-GT Clean **$98.98\%$** $\to$ Adv **$74.70\%$** (nominal) / **$73.92\%$** (mean)

### 7. Table 7: Deformable Attention Telemetry
- **Attention Entropy:**
  - Clean: L1 = **$0.8263$**, L2 = **$0.7993$**, L3 = **$0.7771$** (mean: **$0.8009 \pm 0.0453$**)
  - AQSM: L1 = **$0.8146$**, L2 = **$0.7888$**, L3 = **$0.7669$** (mean: **$0.7901 \pm 0.0260$**, $d = -0.21$)
  - Vanishing: L1 = **$0.8495$** (mean: **$0.8495 \pm 0.0283$**)
  - TaskPGD: L1 = **$0.8366$** (mean: **$0.8366 \pm 0.0293$**)
- **Spatial Sampling Hit Rate (FG):** Clean **$63.51\%$** $\to$ AQSM **$38.77\%$** ($d = -1.03$, $p = 1.77 \times 10^{-159}$)
- **Dropoff Ratio:** AQSM **$6.90 \pm 3.72$**, Clean **$37.51 \pm 25.80$**
- **Score Margin:** AQSM **$0.6580 \pm 0.0907$**, Clean **$0.7984 \pm 0.1281$**
- **Reference Drift:** AQSM **$0.4924 \pm 0.0605$**

### 8. Table 8: Spatial Specificity Verification & Tautology Refutation
- **Clean Baseline:** $\text{mAP}_{50} = \mathbf{65.50\%}$, Recall = $\mathbf{78.61\%}$, FG Hit = $\mathbf{68.93\%}$
- **Random Reassignment (5 seeds):** $\text{mAP}_{50} = \mathbf{60.87 \pm 0.33\%}$ ($92.9\%$ retain), Recall = $\mathbf{74.67\%}$, FG Hit = $\mathbf{68.93\%}$
- **Matched-L2 Random Shift:** $\text{mAP}_{50} = \mathbf{21.55 \pm 0.42\%}$ ($32.9\%$ retain), Recall = $\mathbf{40.87\%}$, FG Hit = $\mathbf{32.90\%}$
- **AQSM Spatial Misrouting:** $\text{mAP}_{50} = \mathbf{23.90\%}$ ($36.5\%$ retain), Recall = $\mathbf{30.48\%}$, FG Hit = $\mathbf{39.86\%}$

### 9. Table 9: Matched-Budget Perturbation Sweep
- **$\epsilon = 4/255$:** Task PGD: $3.01\%$ ($-95.41\%$), AQSM: $7.34\%$ ($-88.80\%$), Vanishing: $3.18\%$ ($-95.14\%$)
- **$\epsilon = 8/255$:** Task PGD: $0.81\%$ ($-98.76\%$), AQSM: $3.89\%$ ($-94.07\%$), Vanishing: $2.28\%$ ($-96.53\%$)
- **$\epsilon = 16/255$:** Task PGD: $0.67\%$ ($-98.98\%$), AQSM: $2.50\%$ ($-96.19\%$), Vanishing: $1.16\%$ ($-98.22\%$)

### 10. Table 10: Hardware Execution Latency Breakdown
- **Thông lượng Tesla T4:** $\mathbf{26.75\text{ FPS}}$, Tổng độ trễ $\mathbf{37.39 \pm 0.89\text{ ms}}$
  - Backbone: $\mathbf{19.34 \pm 0.45\text{ ms}}$ ($51.73\%$)
  - Hybrid Encoder: $\mathbf{10.10 \pm 0.34\text{ ms}}$ ($27.01\%$)
  - Query Selection: $\mathbf{1.64 \pm 0.07\text{ ms}}$ ($4.38\%$)
  - Deformable Decoder: $\mathbf{6.04 \pm 0.49\text{ ms}}$ ($16.14\%$)
- **Jetson Orin Nano:** Clean $14.20\text{ ms}$, Adversarial $14.22\text{ ms}$ (Immune, NMS sponge immunity)
- **Raspberry Pi 5:** Clean $418.20\text{ ms}$, Adversarial $418.50\text{ ms}$ (Immune)

### 11. Table 11: ESM Defense Evaluation & Adaptive Evasion ($N_{\mathrm{test}} = 700$)
- **Clean FPR thực nghiệm:** $\mathbf{6.57\%}$ (ngưỡng hiệu chuẩn $\tau_{\mathrm{fused}} = 0.97$, $N_{\mathrm{cal}} = 300$)
- **AQSM TPR:** $\mathbf{64.57\%}$, AUROC = $\mathbf{0.9161}$
- **Vanishing TPR:** $\mathbf{8.86\%}$, AUROC = $\mathbf{0.5700}$
- **TaskPGD TPR:** $\mathbf{7.14\%}$, AUROC = $\mathbf{0.4840}$
- **Adaptive Evasion ($N_{\mathrm{test}} = 700$):**
  - $\lambda = 0.2$: TPR $1.57\%$ (né tránh $98.43\%$)
  - $\lambda = 0.5$: TPR $\mathbf{1.71\%}$ (né tránh $\mathbf{98.29\%}$), post-attack mAP@50: $4.12\%$
  - $\lambda = 1.0$: TPR $1.00\%$ (né tránh $99.00\%$)
  - $\lambda = 2.0$: TPR $1.14\%$ (né tránh $98.86\%$)

### 12. Table 12: Architectural Hardening via Static Query Augmentation (SQA)
- **Clean Baseline:** Dynamic-only: $65.50\%$ $\to$ Hardened SQA: $64.83\%$ (Clean penalty chỉ $\mathbf{-0.67\text{ pp}}$)
- **AQSM Attack ($\epsilon=16$):** Dynamic-only: $2.50\%$ $\to$ Hardened SQA: $\mathbf{18.42\%}$ (Phục hồi $\mathbf{+15.92\text{ pp}}$)
- **TaskPGD Attack ($\epsilon=16$):** Dynamic-only: $0.67\%$ $\to$ Hardened SQA: $\mathbf{3.12\%}$ (Phục hồi $+2.45\text{ pp}$)
- **Độ trễ Edge Jetson Orin Nano:** $14.20\text{ ms} \to \mathbf{14.58\text{ ms}}$ (chỉ thêm $0.38\text{ ms}$, $+2.68\%$)

---

## III. CHỈ THỊ VIẾT LẠI DÀNH CHO AGENT 2 (WRITER DIRECTIVES) - HOÀN THÀNH 100%

1. **Tuyệt đối nhất quán:** Mọi giá trị số trong `main.tex` đã khớp chính xác đến chữ số thập phân thứ hai với telemetry gốc từ `results (4)`.
2. **Cập nhật mẫu số liệu:** Toàn bộ cụm `$N=400$`, `$N = 400$` đã được thay thế thành `$N=1,000$`, `$N = 1,000$`.
3. **Cập nhật bậc tự do ANOVA:** `$df=[1, 991]$` trên $N_{\mathrm{valid}} = 992$ ảnh có đối tượng ground-truth.
4. **Không overclaim:** Tuân thủ chuẩn mực học thuật trong việc diễn giải phân rã phương sai và tương tác encoder memory - query selection.
