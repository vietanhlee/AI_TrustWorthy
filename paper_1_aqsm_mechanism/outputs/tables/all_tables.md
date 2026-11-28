# TOÀN BỘ 13 BẢNG BIỂU THỰC NGHIỆM PAPER 1 (N=300)

## TABLE1_ABLATION

| AQSM Formulation / Loss Variant | mAP@50 [Rel. Drop] | Top-300 Overlap | Evicted Top-15 FG | Dominant Failure Mode |
| --- | --- | --- | --- | --- |
| Clean Baseline (No Attack) | 66.93% [---] | 100.0% | 0.0% | Nominal inference |
| Margin-Ranking Loss only (L_rank) | 7.03% [-89.5%] | 19.0% | 96.7% | Margin-ranking inversion only |
| Foreground Suppression Loss only (L_suppr) | 3.71% [-94.5%] | 17.5% | 97.9% | Foreground confidence suppression only |
| Distal Background Elevation only (L_elev) | 7.51% [-88.8%] | 18.0% | 59.4% | Distal background elevation only |
| L_suppr + L_rank | 5.54% [-91.7%] | 18.7% | 96.4% | Suppression + margin ranking |
| L_suppr + L_elev | 2.10% [-96.9%] | 12.9% | 98.2% | Suppression + distal elevation |
| L_rank + L_elev | 4.96% [-92.6%] | 17.2% | 95.8% | Dual-boundary ranking + elevation |
| Full Tripartite AQSM | 4.05% [-94.0%] | 16.1% | 97.2% | Compound Eviction & Misrouting |


## TABLE2_MAIN_BENCHMARK

| Threat Tier & Condition | mAP@50 [Rel. Drop] | mAP@50:95 [Rel.] | Active Boxes | ASR_obj [CI 95%] | ASR_silence [CI 95%] | ASR_starve [CI 95%] |
| --- | --- | --- | --- | --- | --- | --- |
| Nominal Clean Baseline | 66.93% [Ref.] | 50.24% [Ref.] | 8.54 ± 9.06 | 0.0% [Reference] | 0.0% [Reference] | --- |
| Tier C: AQSM (eps = 16/255) | 4.05% [-94.0%] | 2.10% [-95.8%] | 17.68 ± 11.14 | --- | --- | --- |
| Tier B: Query Vanishing (eps = 16/255) | 1.32% [-98.0%] | 0.65% [-98.7%] | 0.74 ± 1.05 | --- | --- | --- |
| Tier B: Standard Task PGD (eps = 16/255) | 0.35% [-99.5%] | 0.05% [-99.9%] | 46.43 ± 19.91 | --- | --- | --- |


## TABLE3_CAUSAL_SURGERY

| Intervention Arm / Factor | AQSM mAP@50 | PGD Vanishing mAP@50 | Standard Task PGD mAP@50 |
| --- | --- | --- | --- |
| Arm A: Clean Baseline | 66.93% (100.0%) | 66.93% (100.0%) | 66.93% (100.0%) |
| Arm B: Full Attack | 4.05% (6.0%) [-62.88 pp] | 1.32% (2.0%) [-65.61 pp] | 0.35% (0.5%) [-66.58 pp] |
| Arm C_full: Adv Mem + Clean Prior | 9.31% (13.9%) [-57.62 pp] | 5.23% (7.8%) [-61.70 pp] | 2.16% (3.2%) [-64.77 pp] |
| Arm C_index: Adv Mem + Clean Idx | 4.86% (7.3%) [-62.07 pp] | 1.95% (2.9%) [-64.98 pp] | 1.08% (1.6%) [-65.84 pp] |
| Arm D_full: Clean Mem + Adv Prior | 30.37% (45.4%) [-36.55 pp] | 42.92% (64.1%) [-24.00 pp] | 26.78% (40.0%) [-40.14 pp] |
| Arm D_index: Clean Mem + Adv Idx | 39.11% (58.4%) [-27.82 pp] | 52.94% (79.1%) [-13.99 pp] | 42.52% (63.5%) [-24.41 pp] |
| ME_M (Memory Main Effect) | -42.95 pp (eta_p^2 = 0.784) | -54.60 pp (eta_p^2 = 0.830) | -60.72 pp (eta_p^2 = 0.877) |
| ME_Q (Query Main Effect) | -27.94 pp (eta_p^2 = 0.709) | -19.19 pp (eta_p^2 = 0.622) | -19.90 pp (eta_p^2 = 0.692) |
| Interaction M x Q | +10.06 pp | +2.82 pp | +13.61 pp |


## TABLE4_ANOVA_DECOMP

| Attack Formulation | Factor | Sum of Squares | df | Mean Square | F-Statistic | Partial eta_p^2 |
| --- | --- | --- | --- | --- | --- | --- |
| AQSM (Probe) | Memory (M) | 551657.9 | 1 | 551657.9 | 1084.1 | 0.784 (N/A) |
| AQSM (Probe) | Query Prior (Q) | 233340.7 | 1 | 233340.7 | 725.2 | 0.709 (N/A) |
| AQSM (Probe) | Interaction (M x Q) | 30251.9 | 1 | 30251.9 | 75.9 | 0.203 (N/A) |
| PGD Vanishing | Memory (M) | 891463.7 | 1 | 891463.7 | 1452.8 | 0.830 (N/A) |
| PGD Vanishing | Query Prior (Q) | 110114.1 | 1 | 110114.1 | 490.1 | 0.622 (N/A) |
| PGD Vanishing | Interaction (M x Q) | 2373.3 | 1 | 2373.3 | 7.6 | 0.025 (N/A) |
| Standard Task PGD | Memory (M) | 1102402.9 | 1 | 1102402.9 | 2131.2 | 0.877 (N/A) |
| Standard Task PGD | Query Prior (Q) | 118415.8 | 1 | 118415.8 | 668.7 | 0.692 (N/A) |
| Standard Task PGD | Interaction (M x Q) | 55357.7 | 1 | 55357.7 | 272.6 | 0.478 (N/A) |


## TABLE5_MEMORY_SCALES

| Condition | Overall Cosine | Per-Scale Cosine (S3/S4/S5) | FG Rel. L2 | BG Rel. L2 | FG/BG Ratio | Cosine Gap (S5 - S4) |
| --- | --- | --- | --- | --- | --- | --- |
| PGD Vanishing (T = 15) | 0.6888 | 0.704 / 0.639 / 0.646 | 0.7732 | 0.7955 | 0.9720 | +0.008 |
| AQSM Probe (T = 15) | 0.6140 | 0.630 / 0.551 / 0.612 | 0.8358 | 0.9084 | 0.9201 | +0.061 |
| Standard Task PGD (T = 15) | 0.6031 | 0.628 / 0.512 / 0.562 | 0.8996 | 0.9739 | 0.9237 | +0.049 |


## TABLE6_QUERY_DYNAMICS

| Condition | Top-300 Overlap | Spearman rho | Kendall tau | Evicted Top-15 FG | GT Coverage (Center) | GT Coverage (IoU >= 0.30) |
| --- | --- | --- | --- | --- | --- | --- |
| Clean Baseline | 100.0% [Ref.] | 1.000 | 1.000 | 0.0% | 99.05% | 98.64% |
| PGD Vanishing (T = 15) | 31.11% | 0.426 | 0.297 | 50.53% | 89.61% | 77.26% |
| AQSM Probe (T = 15) | 16.11% | 0.266 | 0.185 | 97.22% | 78.25% | 60.29% |
| Standard Task PGD (T = 15) | 23.62% | 0.373 | 0.259 | 60.58% | 87.14% | 61.82% |


## TABLE7_ATTENTION_TELEMETRY

| Condition | Norm. Attention Entropy | Spatial Sampling Hit Rate (FG) | Reference Drift (Delta p) | Dropoff Ratio | Score Margin |
| --- | --- | --- | --- | --- | --- |
| Clean Baseline | 0.8032 ± 0.0461 | 63.53% | 0.0000 | 34.73 ± 22.80 | 0.7923 ± 0.1337 |
| PGD Vanishing (T = 15) | 0.8340 ± 0.0284 | 51.43% | 0.4264 ± 0.0525 | 17.67 ± 12.35 | 0.3254 ± 0.1687 |
| AQSM Probe (T = 15) | 0.7945 ± 0.0316 | 45.44% | 0.4672 ± 0.0689 | 11.21 ± 7.98 | 0.6749 ± 0.1224 |
| Standard Task PGD (T = 15) | 0.7853 ± 0.0351 | 47.42% | 0.4534 ± 0.0537 | 14.43 ± 8.41 | 0.8526 ± 0.0381 |


## TABLE8_CONTROLLED_MISROUTING

| Condition / Strategy | Seed | mAP@50 (Rel.) | mAP@50:95 | Recall | Slot Disp. (L2) | Hungarian Dist. | Ref FG Hit | GT Obj. Cov. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Clean Reference Baseline | --- | 66.93% (100.0%) | 50.24% | 79.52% | 0.0000 | 0.0000 | 68.95% | 99.31% |
| Random Reassignment (Mean ± Std) | 5 seeds | 63.20 ± 0.45% (94.4%) | 47.81 ± 0.39% | 75.70 ± 0.26% | 0.5529 | 0.0000 | 68.95% | 99.31% |
| Random Shift (Matched L2, Mean ± Std) | 5 seeds | 33.92 ± 1.38% (50.7%) | 22.72 ± 0.83% | 38.33 ± 0.63% | 0.6360 ± 0.0000 | 0.1948 | 46.86% | 85.62% |
| Decoupled Random Shift (Shift + Perm) | 5 seeds | 33.48 ± 0.76% (50.0%) | 22.74 ± 0.79% | 37.98% | 0.6620 | 0.1948 | 46.86% | 85.62% |
| AQSM Spatial Misrouting | --- | 30.24% (45.2%) | 21.43% | 34.84% | 0.6360 | 0.2411 | 47.43% | 77.21% |


## TABLE9_MATCHED_BUDGET

| Perturb. Budget (eps) | Metric / Telemetry Indicator | Standard Task PGD | AQSM (Encoder Ranking) | PGD Vanishing |
| --- | --- | --- | --- | --- |
| 4/255 (0.0157) | COCO mAP@50 (mAP@50:95) | 2.71% (1.12%) | 10.80% (6.22%) | 6.46% (3.98%) |
|  | Relative Drop / Retention | -95.94% (4.1% Ret.) | -83.86% (16.1% Ret.) | -90.35% (9.7% Ret.) |
| 8/255 (0.0314) | COCO mAP@50 (mAP@50:95) | 1.18% (0.38%) | 8.93% (4.91%) | 2.37% (1.38%) |
|  | Relative Drop / Retention | -98.24% (1.8% Ret.) | -86.66% (13.3% Ret.) | -96.45% (3.5% Ret.) |
| 16/255 (0.0627) | COCO mAP@50 (mAP@50:95) | 0.35% (0.05%) | 4.05% (2.10%) | 1.32% (0.65%) |
|  | Relative Drop / Retention | -99.48% (0.5% Ret.) | -93.96% (6.0% Ret.) | -98.03% (2.0% Ret.) |


## TABLE10_LATENCY_SPONGE

| Stage / Operational Condition | Mean ± Std (ms) | P95 (ms) | Throughput (FPS) | Active Boxes (tau >= 0.35) | Share / Latency Delta |
| --- | --- | --- | --- | --- | --- |
| Stage 1: Backbone | 18.943 ± 0.425 | 19.715 | --- | --- | 51.2% of E2E |
| Stage 2: Hybrid Encoder | 9.979 ± 0.349 | 10.528 | --- | --- | 27.0% of E2E |
| Stage 3: Query Selection | 1.611 ± 0.071 | 1.684 | --- | --- | 4.4% of E2E |
| Stage 4: Deformable Decoder | 6.163 ± 0.595 | 7.030 | --- | --- | 16.7% of E2E |
| End-to-End Clean Stage Sum | 36.987 ± 0.992 | 38.427 | 27.04 | --- | 100.0% Reference |
| Condition: Clean Baseline | 36.598 ± 0.952 | 37.996 | 27.32 | 7.95 | Baseline |
| Condition: AQSM Attack (eps=16/255) | 36.465 ± 0.689 | 37.573 | 27.42 | 19.90 | Delta t = -0.133 ms (-0.36%) |
| Condition: Standard Task PGD (eps=16/255) | 36.469 ± 0.762 | 37.573 | 27.42 | 44.80 | Delta t = -0.129 ms (-0.35%) |
| Condition: Proposal-Flooding Sponge | 36.366 ± 0.957 | 37.485 | 27.50 | 154.20 | Delta t = -0.232 ms (-0.63%) |


## TABLE11_DEFENSE_RESULTS

| Evaluated Condition | Detection TPR (%) | Empirical FPR (%) | AUROC | Mean Anomaly Score | Post-Attack mAP@50 (%) |
| --- | --- | --- | --- | --- | --- |
| Clean Test Baseline | --- | 10.00% [Union Bound 22.4%] | --- | 0.6813 | 66.93% |
| Stationary AQSM (eps = 16/255) | 46.67% | 10.00% | 0.7743 | 0.8818 | 4.05% |
| Stationary Standard Task PGD (eps = 16/255) | 12.86% | 10.00% | 0.6505 | 0.8021 | 0.35% |
| Stationary Query Vanishing (eps = 16/255) | 15.71% | 10.00% | 0.6098 | 0.7722 | 1.32% |
| Adaptive Evasion (lambda = 0.2) | 4.76% | --- | Degenerate* | Bypass: 95.24% | 9.77% |
| Adaptive Evasion (lambda = 0.5) | 2.38% | --- | Degenerate* | Bypass: 97.62% | 10.97% |
| Adaptive Evasion (lambda = 1.0) | 2.86% | --- | Degenerate* | Bypass: 97.14% | 12.45% |
| Adaptive Evasion (lambda = 2.0) | 2.38% | --- | Degenerate* | Bypass: 97.62% | 12.08% |


## TABLE12_HARDENING_VALIDATION

| Model Architecture / Evaluated Condition | COCO mAP@50 | COCO mAP@50:95 | Detection Recall | Measured Latency / Note |
| --- | --- | --- | --- | --- |
| Clean: Standard rtdetr_r18vd (300 dynamic) | 66.93% | 50.24% | 86.48% | 34.23 ms (Ref.) |
| Clean: Hardened SQA Prototype (300 dyn. + 30 static) | 66.83% (-0.10 pp) | 50.27% | 86.53% | 34.61 ms (+1.08%) |
| AQSM: Standard rtdetr_r18vd (300 dynamic) | 4.08% (6.1% Ret.) | 2.15% | 18.09% | 34.23 ms |
| AQSM Control 1: Dynamic Top-330 (+30 dynamic queries) | 4.05% (-0.03 pp) | 2.12% | 18.14% | Eviction unmitigated |
| AQSM Control 2: Random-30 Static Queries | 4.01% (-0.07 pp) | 2.18% | 18.18% | Partial spatial retention |
| AQSM Proposed: Uniform Grid-30 SQA Prototype | 4.01% (-0.07 pp) | 2.16% | 18.14% | 34.61 ms (Canvas dispersion) |
| AQSM Adaptive Evasion: White-box Joint Hungarian on SQA | 1.79% (-2.22 pp) | 0.77% | 11.39% | Evasion bypass 97.3% |
| Task PGD: Standard rtdetr_r18vd (300 dynamic) | 0.35% (0.5% Ret.) | 0.09% | 5.36% | 34.23 ms |
| Task PGD: Hardened SQA Prototype (300 dyn. + 30 static) | 0.36% (+0.01 pp) | 0.09% | 5.36% | 34.61 ms (+1.08%) |


## TABLE13_MULTISEED_GENERALIZATION

| Model Architecture | Backbone / Params | Evaluation Seed / Protocol | Clean mAP@50 | AQSM mAP@50 [Rel. Drop] | Delta_Hungarian Set Drift | Anchor FG Hit |
| --- | --- | --- | --- | --- | --- | --- |
| RT-DETR-R18VD (Multi-Seed) | ResNet-18 (3 dec.) | 5 seeds ($\mu \pm \sigma$) | 66.93% | 4.00 ± 0.74% [-94.02%] | 0.2345 ± 0.0037 | 47.12 ± 0.48% |
