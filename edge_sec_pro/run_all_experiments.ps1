# PowerShell Master Runner: Chạy tuần tự toàn bộ các bước thực nghiệm và tự động tổng hợp kết quả
$ErrorActionPreference = "Stop"

$PYTHON_EXE = "G:\Edge_Sec\.venv\Scripts\python.exe"
if (-not (Test-Path $PYTHON_EXE)) {
    $PYTHON_EXE = "python"
}

Write-Host "=================================================================================" -ForegroundColor Cyan
Write-Host "   RT-DETR STREAMLINED SECURITY EVALUATION: MASTER AUTOMATED RUNNER (FULL)       " -ForegroundColor Cyan
Write-Host "=================================================================================" -ForegroundColor Cyan

$steps = @(
    @{ Name = "Bước 1: Trích xuất Calibration & Huấn luyện Detectors"; Script = "edge_sec_pro/experiments/step1_extract_calibration.py" },
    @{ Name = "Bước 2: Đánh giá FPR Stability trên Controls & Biến dạng"; Script = "edge_sec_pro/experiments/step2_evaluate_controls_and_fpr.py" },
    @{ Name = "Bước 3: Đánh giá Adversarial Patch 48x48 & Patch Sweep"; Script = "edge_sec_pro/experiments/step3_evaluate_patch_attack.py" },
    @{ Name = "Bước 4: Đánh giá Tấn công AQSM & Tuyến phòng thủ ESM"; Script = "edge_sec_pro/experiments/step4_evaluate_aqsm_and_esm.py" },
    @{ Name = "Bước 5: Đánh giá Tấn công Query (EQF, Vanishing, DQF)"; Script = "edge_sec_pro/experiments/step5_evaluate_query_attacks.py" },
    @{ Name = "Bước 6: Đánh giá Adaptive Evasion (1D vs Joint Latent Evasion)"; Script = "edge_sec_pro/experiments/step6_evaluate_adaptive_attacks.py" },
    @{ Name = "Bước 7: Thực nghiệm Can thiệp Nhân quả (Causal Query Swap)"; Script = "edge_sec_pro/experiments/step7_causal_query_swap.py" },
    @{ Name = "Bước 8: Đo lường Thời gian Thực thi (Edge Latency Profile)"; Script = "edge_sec_pro/experiments/step8_measure_edge_latency.py" },
    @{ Name = "Bước 9: Đánh giá Định lượng 2 Loại mAP COCO (Dual mAP Benchmark)"; Script = "edge_sec_pro/run_map_evaluation.py" },
    @{ Name = "Bước 10: Cô lập Cơ chế Nhân quả AQSM (Oracle Swap, GT Eviction, Loss Ablation)"; Script = "edge_sec_pro/experiments/exp_causal_mechanism_isolation.py" },
    @{ Name = "Bước 11: Quét Ngân sách Nhiễu (Epsilon Sensitivity Sweep)"; Script = "edge_sec_pro/experiments/exp_epsilon_sweep.py" },
    @{ Name = "Bước 12: Kiểm định Thống kê Ghép cặp McNemar (Paired Hypothesis Testing)"; Script = "edge_sec_pro/experiments/compute_mcnemar_tests.py" },
    @{ Name = "Bước 13: Tổng hợp Dữ liệu Master SSOT (JSON & NPZ)"; Script = "edge_sec_pro/experiments/aggregate_all_results.py" },
    @{ Name = "Bước 14: Sinh Toàn bộ 12 Publication Figures Vector"; Script = "edge_sec_pro/generate_publication_figures.py" },
    @{ Name = "Bước 15: Biên dịch Bài báo Elsevier EAAI Camera-Ready"; Script = "edge_sec_pro/paper/compile_paper.py" },
    @{ Name = "Bước 16: Biên dịch Slide Báo cáo Beamer XeLaTeX Camera-Ready"; Script = "edge_sec_pro/slides/compile_slide.py" }
)

$t_total_start = Get-Date

foreach ($s in $steps) {
    Write-Host "`n>>> BẮT ĐẦU: $($s.Name)" -ForegroundColor Yellow
    $t_step_start = Get-Date
    $scriptPath = $s.Script
    if (-not (Test-Path $scriptPath)) {
        if ($scriptPath.StartsWith("edge_sec_pro/") -and (Test-Path $scriptPath.Substring(13))) {
            $scriptPath = $scriptPath.Substring(13)
        } elseif (Test-Path "$PSScriptRoot/$($s.Script)") {
            $scriptPath = "$PSScriptRoot/$($s.Script)"
        } elseif (Test-Path "$PSScriptRoot/../$($s.Script)") {
            $scriptPath = "$PSScriptRoot/../$($s.Script)"
        }
    }
    & $PYTHON_EXE $scriptPath
    if ($LASTEXITCODE -ne 0) {
        Write-Host "LỖI TẠI: $($s.Name) (Exit Code: $LASTEXITCODE)" -ForegroundColor Red
        exit $LASTEXITCODE
    }
    $t_step_end = Get-Date
    $dur = ($t_step_end - $t_step_start).TotalSeconds
    Write-Host ">>> HOÀN THÀNH: $($s.Name) trong [${dur}s]" -ForegroundColor Green
}

$t_total_end = Get-Date
$total_dur = ($t_total_end - $t_total_start).TotalSeconds
Write-Host "`n=================================================================================" -ForegroundColor Cyan
Write-Host "   TOÀN BỘ 16 BƯỚC THỰC NGHIỆM ĐÃ HOÀN TẤT TRONG [${total_dur}s]!                " -ForegroundColor Cyan
Write-Host "   File kết quả: edge_sec_pro/results/rtdetr_advanced_defense_results.json                   " -ForegroundColor Cyan
Write-Host "   Báo cáo Markdown: edge_sec_pro/results/rtdetr_advanced_defense_report.md                  " -ForegroundColor Cyan
Write-Host "=================================================================================" -ForegroundColor Cyan
