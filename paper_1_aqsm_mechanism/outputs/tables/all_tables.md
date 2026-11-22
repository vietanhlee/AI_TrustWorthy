# TOÀN BỘ 12 BẢNG BIỂU THỰC NGHIỆM PAPER 1 (N=1,000)

## TABLE1_ABLATION

| AQSM Formulation / Loss Variant | mAP@50 [Rel. Drop] | Top-300 Overlap | Evicted Top-15 FG | Dominant Failure Mode |
| --- | --- | --- | --- | --- |
| Clean Baseline (No Attack) | 65.50% [---] | 100.0% | 0.0% | Nominal inference |
| Margin-Ranking Loss only (L_rank) | 24.15% [-63.1%] | 38.5% | 62.4% | Partial proposal reordering |
| Foreground Suppression Loss only (L_suppr) | 32.80% [-49.9%] | 52.1% | 44.2% | Foreground confidence suppression |
| Distal Background Elevation only (L_elev) | 19.45% [-70.3%] | 31.8% | 74.8% | Background cutoff elevation |
| L_suppr + L_rank | 14.20% [-78.3%] | 22.4% | 86.5% | Compound foreground suppression |
| L_rank + L_elev | 6.85% [-89.5%] | 15.6% | 94.2% | Dual-boundary selection compression |
| Full Tripartite AQSM | 2.50% [-96.2%] | 12.2% [10.0, 14.5] | 98.5% [97.1, 99.5] | Compound Eviction & Misrouting |


## TABLE2_MAIN_BENCHMARK

| Threat Tier & Condition | mAP@50 [Rel. Drop] | mAP@50:95 [Rel.] | Active Boxes | ASR_obj [CI 95%] | ASR_silence [CI 95%] | ASR_starve [CI 95%] |
| --- | --- | --- | --- | --- | --- | --- |
| Nominal Clean Baseline | 65.50% [Ref.] | 48.89% [Ref.] | 4.52 ± 3.10 | 0.0% [Reference] | 0.0% [Reference] | 0.0% [Reference] |
| Gaussian Noise (sigma = 16/255) | 58.12% [-11.3%] | 40.15% [-17.9%] | 4.25 ± 2.65 | 11.2% [8.5, 14.2] | 0.8% [0.2, 2.1] | 6.4% [4.1, 9.5] |
| Uniform Noise (± 16/255) | 60.85% [-7.1%] | 42.40% [-13.3%] | 4.38 ± 2.80 | 7.8% [5.4, 10.6] | 0.4% [0.1, 1.5] | 4.2% [2.4, 6.8] |
| Tier 1: AQSM (eps = 16/255) | 2.50% [-96.2%] | 1.37% [-97.2%] | 7.15 ± 4.30 | 98.5% [97.1, 99.5] | 12.8% [9.5, 16.8] | 87.8% [84.2, 91.0] |
| Tier 2: Query Vanishing (eps = 16/255) | 1.16% [-98.2%] | 0.68% [-98.6%] | 0.18 ± 0.52 | 97.8% [95.5, 99.1] | 52.4% [46.8, 58.1] | 69.6% [65.2, 74.0] |
| Baseline: Standard Task PGD (eps = 16/255) | 0.67% [-99.0%] | 0.41% [-99.2%] | 29.40 ± 8.70 | 96.2% [93.1, 98.4] | 0.0% [0.0, 1.0] | 73.6% [68.9, 78.3] |


## TABLE3_CAUSAL_SURGERY

| Intervention Arm / Factor | AQSM mAP@50 | PGD Vanishing mAP@50 | Standard Task PGD mAP@50 |
| --- | --- | --- | --- |
| Arm A: Clean Baseline | 65.50% (100.0%) | 65.50% (100.0%) | 65.50% (100.0%) |
| Arm B: Full Attack | 2.50% (3.8%) [-63.01 pp] | 1.16% (1.8%) [-64.34 pp] | 0.67% (1.0%) [-64.83 pp] |
| Arm C_full: Adv Mem + Clean Prior | 6.25% (9.5%) [-59.25 pp] | 3.79% (5.8%) [-61.71 pp] | 1.40% (2.1%) [-64.10 pp] |
| Arm C_index: Adv Mem + Clean Idx | 3.07% (4.7%) [-62.43 pp] | 1.49% (2.3%) [-64.01 pp] | 0.87% (1.3%) [-64.63 pp] |
| Arm D_full: Clean Mem + Adv Prior | 25.11% (38.3%) [-40.39 pp] | 38.71% (59.1%) [-26.80 pp] | 39.50% (60.3%) [-26.00 pp] |
| Arm D_index: Clean Mem + Adv Idx | 33.04% (50.4%) [-32.46 pp] | 51.43% (78.5%) [-14.07 pp] | 49.89% (76.2%) [-15.61 pp] |
| ME_M (Memory Main Effect) | -44.97 pp (eta_p^2 = 0.806) | -54.11 pp (eta_p^2 = 0.835) | -63.42 pp (eta_p^2 = 0.868) |
| ME_Q (Query Main Effect) | -28.24 pp (eta_p^2 = 0.735) | -18.87 pp (eta_p^2 = 0.601) | -14.39 pp (eta_p^2 = 0.531) |
| Interaction M x Q | +12.47 pp | +1.64 pp | +6.98 pp |


## TABLE4_ANOVA_DECOMP

| Attack Formulation | Factor | Sum of Squares | df | Mean Square | F-Statistic | Partial eta_p^2 |
| --- | --- | --- | --- | --- | --- | --- |
| AQSM (Ours) | Memory (M) | 2006395.4 | 1 | 2006395.4 | 4112.1 | 0.806 (p < 0.001) |
| AQSM (Ours) | Query Prior (Q) | 791287.8 | 1 | 791287.8 | 2753.4 | 0.735 (p < 0.001) |
| AQSM (Ours) | Interaction (M x Q) | 154315.4 | 1 | 154315.4 | 406.3 | 0.291 (p < 0.001) |
| PGD Vanishing | Memory (M) | 2904618.6 | 1 | 2904618.6 | 5006.4 | 0.835 (p < 0.001) |
| PGD Vanishing | Query Prior (Q) | 353224.0 | 1 | 353224.0 | 1491.7 | 0.601 (p < 0.001) |
| PGD Vanishing | Interaction (M x Q) | 2665.8 | 1 | 2665.8 | 9.4 | 0.009 (p = 0.0022) |
| Standard Task PGD | Memory (M) | 3989350.8 | 1 | 3989350.8 | 6509.4 | 0.868 (p < 0.001) |
| Standard Task PGD | Query Prior (Q) | 205538.2 | 1 | 205538.2 | 1121.9 | 0.531 (p < 0.001) |
| Standard Task PGD | Interaction (M x Q) | 48337.5 | 1 | 48337.5 | 217.6 | 0.180 (p < 0.001) |


## TABLE5_MEMORY_SCALES

| Condition | Overall Cosine | Per-Scale Cosine (S3/S4/S5) | FG Rel. L2 | BG Rel. L2 | FG/BG Ratio | Cosine Gap (S5 - S4) |
| --- | --- | --- | --- | --- | --- | --- |
| Gaussian Noise (sigma = 16/255) | 0.9240 | 0.915 / 0.942 / 0.962 | 0.3210 | 0.3840 | 0.8359 | +0.020 |
| PGD Vanishing (T = 15) | 0.6874 | 0.702 / 0.639 / 0.649 | 0.7746 | 0.7997 | 0.9686 | +0.010 |
| AQSM (Ours, T = 15) | 0.5753 | 0.592 / 0.507 / 0.585 | 0.8727 | 0.9782 | 0.8921 | +0.077 |
| Standard Task PGD (T = 15) | 0.6626 | 0.681 / 0.608 / 0.592 | 0.8141 | 0.8287 | 0.9824 | -0.016 |


## TABLE6_QUERY_DYNAMICS

| Condition | Top-300 Overlap | Spearman rho | Kendall tau | Evicted Top-15 FG | Center-in-GT | GT IoU >= 0.30 Cov. |
| --- | --- | --- | --- | --- | --- | --- |
| Clean Baseline | 100.0% [Ref.] | 1.000 | 1.000 | 0.0% | 98.98% | 98.46% |
| Gaussian Noise (sigma = 16/255) | 82.40% | 0.892 | 0.735 | 11.20% | 94.50% | 92.10% |
| PGD Vanishing (T = 15) | 30.39% | 0.424 | 0.295 | 51.06% | 89.83% | 78.89% |
| AQSM (Ours, T = 15) | 12.21% | 0.165 | 0.115 | 98.50% | 73.92% | 53.11% |
| Standard Task PGD (T = 15) | 26.38% | 0.387 | 0.268 | 56.01% | 89.43% | 78.04% |


## TABLE7_ATTENTION_TELEMETRY

| Condition | Norm. Attention Entropy | Spatial Sampling Hit Rate (FG) | Reference Drift (Delta p) | Dropoff Ratio | Score Margin |
| --- | --- | --- | --- | --- | --- |
| Clean Baseline | 0.8009 ± 0.0453 | 63.51% [61.8, 65.2] | 0.0000 | 37.51 ± 25.80 | 0.7984 ± 0.1281 |
| Gaussian Noise (sigma = 16/255) | 0.8005 ± 0.0430 | 58.40% [55.8, 61.0] | 0.0150 | 32.10 ± 21.40 | 0.7650 ± 0.1250 |
| PGD Vanishing (T = 15) | 0.8495 ± 0.0283 | 48.82% [45.8, 51.8] | 0.4302 ± 0.0505 | 18.71 ± 13.01 | 0.3351 ± 0.1691 |
| AQSM (Ours, T = 15) | 0.7901 ± 0.0260 | 38.77% [36.7, 40.8] | 0.4924 ± 0.0605 | 6.90 ± 3.72 | 0.6580 ± 0.0907 |
| Standard Task PGD (T = 15) | 0.8366 ± 0.0293 | 52.07% [49.2, 54.9] | 0.4285 ± 0.0524 | 42.42 ± 29.40 | 0.7372 ± 0.1582 |


## TABLE8_CONTROLLED_MISROUTING

| Condition / Strategy | Seed | mAP@50 (Rel.) | mAP@50:95 | Recall | Displ. (L2) | Center Disp. (L2) | Size Disp. (L2) | Ref FG Hit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Clean Reference Baseline | --- | 65.50% (100.0%) | 48.89% | 78.61% | 0.0000 | 0.0000 | 0.0000 | 68.93% |
| Random Reassignment (Mean ± Std) | 5 seeds | 60.87 ± 0.33% (92.9%) | 45.51 ± 0.19% | 74.67 ± 0.35% | 0.5508 | 0.3812 | 0.3122 | 68.93% |
| Random Shift (Matched L2, Mean ± Std) | 5 seeds | 21.55 ± 0.42% (32.9%) | 14.30 ± 0.23% | 40.87 ± 0.52% | 0.5176 ± 0.0010 | 0.3690 | 0.3080 | 32.90% |
| AQSM Spatial Misrouting | --- | 23.90% (36.5%) | 16.06% | 30.48% | 0.6468 | 0.4924 | 0.3068 | 39.86% |


## TABLE9_MATCHED_BUDGET

| Perturb. Budget (eps) | Metric / Telemetry Indicator | Standard Task PGD | AQSM (Encoder Ranking) | PGD Vanishing |
| --- | --- | --- | --- | --- |
| 4/255 (0.0157) | COCO mAP@50 (mAP@50:95) | 3.01% (1.97%) | 7.34% (4.08%) | 3.18% (2.00%) |
|  | Relative Drop / Retention | -95.41% (4.6% Ret.) | -88.80% (11.2% Ret.) | -95.14% (4.9% Ret.) |
| 8/255 (0.0314) | COCO mAP@50 (mAP@50:95) | 0.81% (0.49%) | 3.89% (1.88%) | 2.28% (1.35%) |
|  | Relative Drop / Retention | -98.76% (1.2% Ret.) | -94.07% (5.9% Ret.) | -96.53% (3.5% Ret.) |
| 16/255 (0.0627) | COCO mAP@50 (mAP@50:95) | 0.67% (0.41%) | 2.50% (1.37%) | 1.16% (0.68%) |
|  | Relative Drop / Retention | -98.98% (1.0% Ret.) | -96.19% (3.8% Ret.) | -98.22% (1.8% Ret.) |


## TABLE10_LATENCY_SPONGE

| Model Architecture | Operational Condition | Jetson Orin Nano (ms / FPS) | Raspberry Pi 5 (ms / FPS) | Active Detections (tau >= 0.35) | Sponge Attack Vulnerability |
| --- | --- | --- | --- | --- | --- |
| RT-DETR-R18VD | Clean Baseline | 14.20 ± 0.35 (70.4) | 418.20 ± 12.40 (2.39) | 4.52 ± 3.10 | Immune (NMS-free set prediction) |
| RT-DETR-R18VD | Query Vanishing (eps = 16) | 14.21 ± 0.36 (70.4) | 418.80 ± 11.80 (2.39) | 0.18 ± 0.52 | Immune (Delta t = 0.01 ms) |
| RT-DETR-R18VD | AQSM Attack (eps = 16) | 14.22 ± 0.38 (70.3) | 418.50 ± 12.10 (2.39) | 7.15 ± 4.30 | Immune (Delta t = 0.02 ms) |
| RT-DETR-R18VD | Standard Task PGD (eps = 16) | 14.25 ± 0.40 (70.2) | 419.50 ± 12.80 (2.38) | 29.40 ± 8.70 | Immune (Delta t = 0.05 ms) |
| YOLOv8-S (Baseline) | Clean Baseline | 11.50 ± 0.28 (87.0) | 245.30 ± 8.50 (4.08) | 4.65 ± 2.90 | Vulnerable (NMS quadratic clustering) |
| YOLOv8-S (Baseline) | Dense Sponge Attack (eps = 16) | 39.40 ± 2.45 (25.4) | 815.40 ± 34.20 (1.23) | 286.40 ± 36.10 | Vulnerable (+232% to +242% surge) |


## TABLE11_DEFENSE_RESULTS

| Evaluated Condition | Detection TPR (%) | Empirical FPR (%) | AUROC | Mean Anomaly Score | Post-Attack mAP@50 (%) |
| --- | --- | --- | --- | --- | --- |
| Clean Test Baseline | --- | 6.57% [Union Bound 10.0%] | --- | 0.4950 | 65.50% |
| Stationary AQSM (eps = 16/255) | 64.57% | 6.57% | 0.9161 | 0.9562 | 2.50% |
| Stationary Standard Task PGD (eps = 16/255) | 7.14% | 6.57% | 0.4840 | 0.6594 | 0.67% |
| Stationary Query Vanishing (eps = 16/255) | 8.86% | 6.57% | 0.5700 | 0.7321 | 1.16% |
| Adaptive Evasion (lambda = 0.2) | 1.57% | --- | < 0.50 | Evasion: 98.43% | 3.65% |
| Adaptive Evasion (lambda = 0.5) | 1.71% | --- | < 0.50 | Evasion: 98.29% | 4.12% |
| Adaptive Evasion (lambda = 1.0) | 1.00% | --- | < 0.50 | Evasion: 99.00% | 5.20% |
| Adaptive Evasion (lambda = 2.0) | 1.14% | --- | < 0.50 | Evasion: 98.86% | 6.85% |


## TABLE12_HARDENING_VALIDATION

| Model Architecture / Evaluated Condition | COCO mAP@50 | COCO mAP@50:95 | Detection Recall | Jetson Orin Latency |
| --- | --- | --- | --- | --- |
| Clean Baseline: Standard RT-DETR-R18VD (300 dynamic) | 65.50% | 48.89% | 78.61% | 14.20 ms |
| Clean Baseline: Hardened SQA Prototype (300 dyn. + 30 static) | 64.83% (-0.67 pp) | 48.22% | 78.85% | 14.58 ms (+2.68%) |
| Stationary AQSM: Standard RT-DETR-R18VD (300 dynamic) | 2.50% (3.8% Ret.) | 1.37% | 30.48% | 14.20 ms |
| Stationary AQSM: Hardened SQA Prototype (300 dyn. + 30 static) | 18.42% (+15.92 pp) | 11.25% | 46.80% | 14.58 ms |
| Standard Task PGD: Standard RT-DETR-R18VD (300 dynamic) | 0.67% (1.0% Ret.) | 0.41% | 14.20% | 14.20 ms |
| Standard Task PGD: Hardened SQA Prototype (300 dyn. + 30 static) | 3.12% (+2.45 pp) | 1.84% | 19.50% | 14.58 ms (+2.68%) |
