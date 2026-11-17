# Báo Cáo Hiệu Năng Phát Hiện Vật Thể (COCO mAP) - Streamlined Benchmark

Mô hình: `PekingU/rtdetr_r18vd` (320x320) | Dataset: COCO val2017

| Kịch bản Đánh giá | mAP@50 (%) | mAP@50:95 (%) | Mức Suy giảm mAP@50 (Drop %) | Tác động Thực tiễn (Real-world Impact) | Với Cơ chế Phòng thủ (Guardrail) |
|:---|:---:|:---:|:---:|:---|:---:|
| **Clean Baseline** | **51.2%** | **35.4%** | 0.0% (Chuẩn) | Nhận diện vật thể chính xác, an toàn | **Bảo toàn 100% mAP (51.2%)** |
| **Control: Gaussian Noise (sigma=16)** | **37.5%** | **22.3%** | -26.7% | Nhiễu cảm biến camera tự nhiên | **Cho qua bình thường (FPR = 7.0%)** |
| **Control: Random Patch (48x48)** | **51.1%** | **35.2%** | -0.3% | Vết bẩn / bùn đất che góc ống kính | **Cho qua bình thường (FPR = 7.0%)** |
| **Clean Shift: Gaussian Blur (5x5)** | **36.2%** | **22.3%** | -29.2% | Mờ nét do rung lắc hoặc sương mù | **Cho qua bình thường (FPR = 6.5%)** |
| **Clean Shift: Brightness (-30%)** | **50.5%** | **34.5%** | -1.5% | Ánh sáng yếu / trời tối chập choạng | **Cho qua bình thường (FPR = 5.0%)** |
| **Attack: Gradient Patch (48x48)** | **49.7%** | **34.1%** | -3.0% | Bẫy Attention cục bộ, phá vỡ định vị | **1D bị mù (5%), Mahalanobis bắt 18%** |
| **Attack: Query Selection Manipulation (AQSM)** | **8.3%** | **2.5%** | -83.7% | Bỏ đói thông tin Decoder (Information Starvation) | **Bắt 98.0% TPR (Kích hoạt Safe Fallback)** |
| **Attack: PGD Query Vanishing (20 steps)** | **6.6%** | **2.9%** | -87.1% | Mù hoàn toàn vật thể thật (AEB hỏng, đâm va) | **Bắt 95.0% TPR (Kích hoạt Safe Fallback)** |
| **Attack: PGD Query Fabrication (20 steps)** | **8.1%** | **2.8%** | -84.1% | Tràn ngập hộp ma do thiếu NMS (Phanh gấp) | **Bắt 31.0% TPR (Khắc phục việc thiếu NMS)** |
| **Attack: Adaptive Evasion PGD (lambda=0.5)** | **8.8%** | **3.1%** | -82.8% | Né 1D monitor, hủy diệt 94% task mAP | **1D sụp đổ (1%), Mahalanobis giữ 41% TPR** |
