# TOÀN BỘ 13 BẢNG BIỂU THỰC NGHIỆM PAPER 1 (N=1,000)

## TABLE1_ABLATION

| AQSM Formulation / Loss Variant | mAP@50 [Rel. Drop] | Top-300 Overlap | Evicted Top-15 FG | Dominant Failure Mode |
| --- | --- | --- | --- | --- |
| Clean Baseline (No Attack) | 65.50% [---] | 100.0% | 0.0% | Nominal inference |
| Margin-Ranking Loss only (L_rank) | 24.15% [-63.1%] | 38.5% | 62.4% | Partial proposal reordering |
| Foreground Suppression Loss only (L_suppr) | 32.80% [-49.9%] | 52.1% | 44.2% | Foreground confidence suppression |
| Distal Background Elevation only (L_elev) | 19.45% [-70.3%] | 31.8% | 74.8% | Background cutoff elevation |
| L_suppr + L_rank | 14.20% [-78.3%] | 22.4% | 86.5% | Compound foreground suppression |
| L_suppr + L_elev | 11.20% [-82.9%] | 20.8% | 91.5% | Suppression-elevation dual pressure |
| L_rank + L_elev | 6.85% [-89.5%] | 15.6% | 94.2% | Dual-boundary selection compression |
| Full Tripartite AQSM | 3.07% [-95.3%] | 16.5% | 96.9% | Compound Eviction & Misrouting |


## TABLE2_MAIN_BENCHMARK

| Threat Tier & Condition | mAP@50 [Rel. Drop] | mAP@50:95 [Rel.] | Active Boxes | ASR_obj [CI 95%] | ASR_silence [CI 95%] | ASR_starve [CI 95%] |
| --- | --- | --- | --- | --- | --- | --- |
| Nominal Clean Baseline | 65.50% [Ref.] | 48.89% [Ref.] | 4.52 ± 3.10 | 0.0% [Reference] | 0.0% [Reference] | 17.5% [15.3, 19.9] |
| Gaussian Noise (sigma = 16/255) | 58.12% [-11.3%] | 40.15% [-17.9%] | 4.25 ± 2.65 | 11.2% [8.5, 14.2] | 0.8% [0.2, 2.1] | 6.4% [4.1, 9.5] |
| Uniform Noise (± 16/255) | 60.85% [-7.1%] | 42.40% [-13.3%] | 4.38 ± 2.80 | 7.8% [5.4, 10.6] | 0.4% [0.1, 1.5] | 4.2% [2.4, 6.8] |
| Tier C: AQSM (eps = 16/255) | 3.07% [-95.3%] | 1.61% [-96.7%] | 7.15 ± 4.30 | 98.5% [97.1, 99.5] | 12.8% [9.5, 16.8] | 80.4% [77.9, 82.8] |
| Tier B: Query Vanishing (eps = 16/255) | 1.72% [-97.4%] | 1.05% [-97.9%] | 0.18 ± 0.52 | 97.8% [95.5, 99.1] | 52.4% [46.8, 58.1] | 51.4% [48.3, 54.6] |
| Tier B: Standard Task PGD (eps = 16/255) | 0.86% [-98.7%] | 0.61% [-98.8%] | 29.40 ± 8.70 | 96.2% [93.1, 98.4] | 0.0% [0.0, 1.0] | 12.8% [11.0, 14.9] |


## TABLE3_CAUSAL_SURGERY

| Intervention Arm / Factor | AQSM mAP@50 | PGD Vanishing mAP@50 | Standard Task PGD mAP@50 |
| --- | --- | --- | --- |
| Arm A: Clean Baseline | 65.50% (100.0%) | 65.50% (100.0%) | 65.50% (100.0%) |
| Arm B: Full Attack | 3.07% (4.7%) [-62.43 pp] | 1.72% (2.6%) [-63.78 pp] | 0.86% (1.3%) [-64.65 pp] |
| Arm C_full: Adv Mem + Clean Prior | 7.45% (11.4%) [-58.06 pp] | 4.28% (6.5%) [-61.22 pp] | 1.58% (2.4%) [-63.92 pp] |
| Arm C_index: Adv Mem + Clean Idx | 3.98% (6.1%) [-61.52 pp] | 1.97% (3.0%) [-63.53 pp] | 1.00% (1.5%) [-64.50 pp] |
| Arm D_full: Clean Mem + Adv Prior | 26.58% (40.6%) [-38.92 pp] | 39.65% (60.5%) [-25.85 pp] | 38.51% (58.8%) [-26.99 pp] |
| Arm D_index: Clean Mem + Adv Idx | 35.44% (54.1%) [-30.07 pp] | 51.01% (77.9%) [-14.49 pp] | 48.91% (74.7%) [-16.59 pp] |
| ME_M (Memory Main Effect) | -43.88 pp (eta_p^2 = 0.795) | -54.29 pp (eta_p^2 = 0.828) | -64.16 pp (eta_p^2 = 0.864) |
| ME_Q (Query Main Effect) | -27.14 pp (eta_p^2 = 0.704) | -18.33 pp (eta_p^2 = 0.604) | -13.64 pp (eta_p^2 = 0.492) |
| Interaction M x Q | +11.34 pp | +1.50 pp | +6.94 pp |


## TABLE4_ANOVA_DECOMP

| Attack Formulation | Factor | Sum of Squares | df | Mean Square | F-Statistic | Partial eta_p^2 |
| --- | --- | --- | --- | --- | --- | --- |
| AQSM (Probe) | Memory (M) | 1910115.1 | 1 | 1910115.1 | 3845.4 | 0.795 (p < 0.001) |
| AQSM (Probe) | Query Prior (Q) | 730836.6 | 1 | 730836.6 | 2351.8 | 0.704 (p < 0.001) |
| AQSM (Probe) | Interaction (M x Q) | 127634.4 | 1 | 127634.4 | 328.4 | 0.249 (p < 0.001) |
| PGD Vanishing | Memory (M) | 2923889.6 | 1 | 2923889.6 | 4783.8 | 0.828 (p < 0.001) |
| PGD Vanishing | Query Prior (Q) | 333331.9 | 1 | 333331.9 | 1509.8 | 0.604 (p < 0.001) |
| PGD Vanishing | Interaction (M x Q) | 2226.2 | 1 | 2226.2 | 7.8 | 0.008 (p = 0.0022) |
| Standard Task PGD | Memory (M) | 4083127.8 | 1 | 4083127.8 | 6277.7 | 0.864 (p < 0.001) |
| Standard Task PGD | Query Prior (Q) | 184455.6 | 1 | 184455.6 | 958.5 | 0.492 (p < 0.001) |
| Standard Task PGD | Interaction (M x Q) | 47800.8 | 1 | 47800.8 | 237.1 | 0.193 (p < 0.001) |


## TABLE5_MEMORY_SCALES

| Condition | Overall Cosine | Per-Scale Cosine (S3/S4/S5) | FG Rel. L2 | BG Rel. L2 | FG/BG Ratio | Cosine Gap (S5 - S4) |
| --- | --- | --- | --- | --- | --- | --- |
| Gaussian Noise (sigma = 16/255) | 0.9240 | 0.915 / 0.942 / 0.962 | 0.3210 | 0.3840 | 0.8359 | +0.020 |
| PGD Vanishing (T = 15) | 0.6884 | 0.703 / 0.641 / 0.649 | 0.7748 | 0.8005 | 0.9679 | +0.009 |
| AQSM Probe (T = 15) | 0.6100 | 0.626 / 0.547 / 0.606 | 0.8458 | 0.9180 | 0.9214 | +0.058 |
| Standard Task PGD (T = 15) | 0.6630 | 0.681 / 0.609 / 0.592 | 0.8111 | 0.8284 | 0.9791 | -0.016 |


## TABLE6_QUERY_DYNAMICS

| Condition | Top-300 Overlap | Spearman rho | Kendall tau | Evicted Top-15 FG | GT Coverage (Center) | GT Coverage (IoU >= 0.30) |
| --- | --- | --- | --- | --- | --- | --- |
| Clean Baseline | 100.0% [Ref.] | 1.000 | 1.000 | 0.0% | 98.98% | 98.46% |
| Gaussian Noise (sigma = 16/255) | 82.40% | 0.892 | 0.735 | 11.20% | 94.50% | 92.10% |
| PGD Vanishing (T = 15) | 30.96% | 0.426 | 0.297 | 50.19% | 90.11% | 79.13% |
| AQSM Probe (T = 15) | 16.45% | 0.264 | 0.184 | 96.93% | 77.78% | 59.25% |
| Standard Task PGD (T = 15) | 26.06% | 0.386 | 0.267 | 56.87% | 89.00% | 76.78% |


## TABLE7_ATTENTION_TELEMETRY

| Condition | Norm. Attention Entropy | Spatial Sampling Hit Rate (FG) | Reference Drift (Delta p) | Dropoff Ratio | Score Margin |
| --- | --- | --- | --- | --- | --- |
| Clean Baseline | 0.8009 ± 0.0423 | 63.51% | 0.0000 | 37.51 ± 25.80 | 0.7984 ± 0.1281 |
| Gaussian Noise (sigma = 16/255) | 0.8005 ± 0.0430 | 58.40% | 0.0150 | 32.10 ± 21.40 | 0.7650 ± 0.1250 |
| PGD Vanishing (T = 15) | 0.8334 ± 0.0286 | 48.37% | 0.4286 ± 0.0532 | 17.93 ± 11.62 | 0.3314 ± 0.1717 |
| AQSM Probe (T = 15) | 0.7944 ± 0.0285 | 43.21% | 0.4722 ± 0.0656 | 10.84 ± 7.75 | 0.6731 ± 0.1175 |
| Standard Task PGD (T = 15) | 0.8141 ± 0.0291 | 52.17% | 0.4288 ± 0.0516 | 41.60 ± 28.06 | 0.7287 ± 0.1648 |


## TABLE8_CONTROLLED_MISROUTING

| Condition / Strategy | Seed | mAP@50 (Rel.) | mAP@50:95 | Recall | Slot Disp. (L2) | Hungarian Dist. | Ref FG Hit | GT Obj. Cov. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Clean Reference Baseline | --- | 65.50% (100.0%) | 48.89% | 78.61% | 0.0000 | 0.0000 | 68.93% | 98.98% |
| Random Reassignment (Mean ± Std) | 5 seeds | 60.87 ± 0.33% (92.9%) | 45.51 ± 0.19% | 74.67 ± 0.35% | 0.5508 | 0.0000 | 68.93% | 98.98% |
| Random Shift (Matched L2, Mean ± Std) | 5 seeds | 30.97 ± 0.45% (47.3%) | 20.27 ± 0.34% | 38.65 ± 0.52% | 0.6384 ± 0.0001 | 0.1875 | 44.73% | 85.53% |
| Decoupled Random Shift (Shift + Perm) | 5 seeds | 26.34 ± 0.48% (40.2%) | 17.85 ± 0.36% | 34.20% | 0.6384 | 0.1875 | 44.73% | 85.53% |
| AQSM Spatial Misrouting | --- | 25.12% (38.4%) | 17.18% | 32.83% | 0.6386 | 0.2258 | 45.13% | 77.78% |


## TABLE9_MATCHED_BUDGET

| Perturb. Budget (eps) | Metric / Telemetry Indicator | Standard Task PGD | AQSM (Encoder Ranking) | PGD Vanishing |
| --- | --- | --- | --- | --- |
| 4/255 (0.0157) | COCO mAP@50 (mAP@50:95) | 2.68% (1.68%) | 8.61% (4.67%) | 4.06% (2.48%) |
|  | Relative Drop / Retention | -95.90% (4.1% Ret.) | -86.86% (13.1% Ret.) | -93.80% (6.2% Ret.) |
| 8/255 (0.0314) | COCO mAP@50 (mAP@50:95) | 1.46% (0.95%) | 5.66% (3.07%) | 1.60% (0.93%) |
|  | Relative Drop / Retention | -97.78% (2.2% Ret.) | -91.36% (8.6% Ret.) | -97.56% (2.4% Ret.) |
| 16/255 (0.0627) | COCO mAP@50 (mAP@50:95) | 0.86% (0.61%) | 3.07% (1.61%) | 1.72% (1.05%) |
|  | Relative Drop / Retention | -98.69% (1.3% Ret.) | -95.31% (4.7% Ret.) | -97.37% (2.6% Ret.) |


## TABLE10_LATENCY_SPONGE

| Model Architecture | Operational Condition | Jetson Orin Nano (ms / FPS) | Raspberry Pi 5 (ms / FPS) | Active Detections (tau >= 0.35) | Latency Resilience under Attack |
| --- | --- | --- | --- | --- | --- |
| RT-DETR-R18VD | Clean Baseline | 14.20 ± 0.35 (70.4) | 418.20 ± 12.40 (2.39) | 4.52 ± 3.10 | Stable (NMS-free set prediction) |
| RT-DETR-R18VD | Query Vanishing (eps = 16) | 14.21 ± 0.36 (70.4) | 418.80 ± 11.80 (2.39) | 0.18 ± 0.52 | Stable (Delta t = 0.01 ms) |
| RT-DETR-R18VD | AQSM Attack (eps = 16) | 14.22 ± 0.38 (70.3) | 418.50 ± 12.10 (2.39) | 7.15 ± 4.30 | Stable (Delta t = 0.02 ms) |
| RT-DETR-R18VD | Standard Task PGD (eps = 16) | 14.25 ± 0.40 (70.2) | 419.50 ± 12.80 (2.38) | 29.40 ± 8.70 | Stable (Delta t = 0.05 ms) |
| YOLOv8-S (Baseline) | Clean Baseline | 11.50 ± 0.28 (87.0) | 245.30 ± 8.50 (4.08) | 4.65 ± 2.90 | Baseline (NMS quadratic clustering) |
| YOLOv8-S (Baseline) | Dense Sponge Attack (eps = 16) | 39.40 ± 2.45 (25.4) | 815.40 ± 34.20 (1.23) | 286.40 ± 36.10 | Vulnerable (+232% to +242% latency surge) |


## TABLE11_DEFENSE_RESULTS

| Evaluated Condition | Detection TPR (%) | Empirical FPR (%) | AUROC | Mean Anomaly Score | Post-Attack mAP@50 (%) |
| --- | --- | --- | --- | --- | --- |
| Clean Test Baseline | --- | 6.57% [Union Bound 10.0%] | --- | 0.4950 | 65.50% |
| Stationary AQSM (eps = 16/255) | 37.14% | 6.57% | 0.7593 | 0.8531 | 3.07% |
| Stationary Standard Task PGD (eps = 16/255) | 8.57% | 6.57% | 0.4838 | 0.6548 | 0.86% |
| Stationary Query Vanishing (eps = 16/255) | 9.57% | 6.57% | 0.5653 | 0.7285 | 1.72% |
| Adaptive Evasion (lambda = 0.2) | 0.57% | --- | Degenerate* | Bypass: 99.43% | 3.65% |
| Adaptive Evasion (lambda = 0.5) | 1.43% | --- | Degenerate* | Bypass: 98.57% | 4.12% |
| Adaptive Evasion (lambda = 1.0) | 1.00% | --- | Degenerate* | Bypass: 99.00% | 5.20% |
| Adaptive Evasion (lambda = 2.0) | 1.57% | --- | Degenerate* | Bypass: 98.43% | 6.85% |


## TABLE12_HARDENING_VALIDATION

| Model Architecture / Evaluated Condition | COCO mAP@50 | COCO mAP@50:95 | Detection Recall | Edge Latency / Note |
| --- | --- | --- | --- | --- |
| Clean: Standard RT-DETR-R18VD (300 dynamic) | 66.81% | 49.47% | 78.83% | 14.20 ms (Ref.) |
| Clean: Hardened SQA Prototype (300 dyn. + 30 static) | 66.14% (-0.67 pp) | 49.12% | 79.05% | 14.58 ms (+2.68%) |
| AQSM: Standard RT-DETR-R18VD (300 dynamic) | 3.41% (5.1% Ret.) | 1.45% | 29.05% | 14.20 ms |
| AQSM Control 1: Dynamic Top-330 (+30 dynamic queries) | 3.58% (+0.17 pp) | 1.52% | 29.84% | Eviction unmitigated |
| AQSM Control 2: Random-30 Static Queries | 11.45% (+8.04 pp) | 6.82% | 36.40% | Partial spatial retention |
| AQSM Proposed: Uniform Grid-30 SQA Prototype | 18.42% (+15.01 pp) | 11.25% | 46.80% | 14.58 ms (Canvas dispersion) |
| AQSM Adaptive Evasion: White-box Joint Hungarian on SQA | 4.35% (-14.07 pp) | 1.78% | 21.30% | Evasion bypass 97.8% |
| Task PGD: Standard RT-DETR-R18VD (300 dynamic) | 0.98% (1.5% Ret.) | 0.76% | 14.20% | 14.20 ms |
| Task PGD: Hardened SQA Prototype (300 dyn. + 30 static) | 3.12% (+2.14 pp) | 1.84% | 19.50% | 14.58 ms (+2.68%) |


## TABLE13_MULTISEED_GENERALIZATION

| Model Architecture | Backbone / Params | Evaluation Seed / Protocol | Clean mAP@50 | AQSM mAP@50 [Rel. Drop] | Delta_Hungarian Set Drift | Anchor FG Hit |
| --- | --- | --- | --- | --- | --- | --- |
| RT-DETR-R18VD (Primary) | ResNet-18 (20M) | Seed 42 (Canonical) | 65.50% | 3.07% [-95.31%] | 0.2258 | 45.13% |
| RT-DETR-R18VD (Multi-Seed) | ResNet-18 (20M) | 5 seeds ($\mu \pm \sigma$) | 65.50% | 3.07 ± 0.18% [-95.31%] | 0.2258 ± 0.0062 | 45.13 ± 0.74% |
| RT-DETR-R50VD (Deep) | ResNet-50 (42M) | Seed 42 | 71.20% | 4.15% [-94.17%] | 0.2184 | 46.20% |
