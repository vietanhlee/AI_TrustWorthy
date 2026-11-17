# PowerShell Parallel Runner: Chay thuc nghiem RT-DETR theo tung me (batch) song song 4 processes
$ErrorActionPreference = "Stop"

$PYTHON_EXE = "G:\Edge_Sec\.venv\Scripts\python.exe"
if (-not (Test-Path $PYTHON_EXE)) {
    $PYTHON_EXE = "python"
}

$BASE_DIR = "G:\Edge_Sec\edge_sec_pro"
$EXP_DIR = "$BASE_DIR\experiments"
$CKPT_DIR = "$BASE_DIR\results\checkpoints"
$BACKUP_DIR = "$BASE_DIR\results\checkpoints_prev"
$LOG_DIR = "$BASE_DIR\results\logs_run"

if (Test-Path $CKPT_DIR) {
    if (-not (Test-Path $BACKUP_DIR)) {
        New-Item -ItemType Directory -Path $BACKUP_DIR -Force | Out-Null
    }
    Copy-Item -Path "$CKPT_DIR\*" -Destination $BACKUP_DIR -Recurse -Force
    Remove-Item -Path "$CKPT_DIR\*" -Recurse -Force
} else {
    New-Item -ItemType Directory -Path $CKPT_DIR -Force | Out-Null
}

if (-not (Test-Path $LOG_DIR)) {
    New-Item -ItemType Directory -Path $LOG_DIR -Force | Out-Null
}

Write-Host "=================================================================================" -ForegroundColor Cyan
Write-Host "   RT-DETR STREAMLINED SECURITY EVALUATION: PARALLEL RUNNER (4 WORKERS)         " -ForegroundColor Cyan
Write-Host "   Hardware: 12-Core / 16-Logical Processors Intel CPU                          " -ForegroundColor Cyan
Write-Host "=================================================================================" -ForegroundColor Cyan

function Run-SingleScript {
    param (
        [string]$Name,
        [string]$ScriptPath
    )
    Write-Host "`n>>> [RUNNING SINGLE] $Name" -ForegroundColor Yellow
    $t_start = Get-Date
    & $PYTHON_EXE $ScriptPath
    if ($LASTEXITCODE -ne 0) {
        Write-Host ">>> [ERROR] $Name failed with exit code $LASTEXITCODE" -ForegroundColor Red
        exit $LASTEXITCODE
    }
    $dur = ((Get-Date) - $t_start).TotalSeconds
    Write-Host ">>> [SUCCESS] $Name completed in [${dur}s]" -ForegroundColor Green
}

function Run-ParallelBatch {
    param (
        [string]$BatchName,
        [array]$Tasks
    )
    Write-Host "`n=================================================================================" -ForegroundColor Magenta
    Write-Host ">>> BAT DAU ME CHAY SONG SONG: $BatchName (So tien trinh: $($Tasks.Count))" -ForegroundColor Magenta
    Write-Host "=================================================================================" -ForegroundColor Magenta
    $t_batch_start = Get-Date

    $processes = @()
    foreach ($task in $Tasks) {
        $taskName = $task.Name
        $script = $task.Script
        $outFile = "$LOG_DIR\$taskName.out.log"
        $errFile = "$LOG_DIR\$taskName.err.log"
        if (Test-Path $outFile) { Remove-Item $outFile -Force }
        if (Test-Path $errFile) { Remove-Item $errFile -Force }

        Write-Host "  -> Khoi chay tien trinh: $taskName ($script)" -ForegroundColor Cyan
        $proc = Start-Process -FilePath $PYTHON_EXE -ArgumentList $script -NoNewWindow -PassThru -RedirectStandardOutput $outFile -RedirectStandardError $errFile
        $processes += @{ Name = $taskName; Process = $proc; OutFile = $outFile; ErrFile = $errFile }
    }

    Write-Host "`n  Dang cho cac tien trinh trong me '$BatchName' hoan thanh..." -ForegroundColor Gray
    $failed = $false
    foreach ($p in $processes) {
        $p.Process.WaitForExit()
        $exitCode = $p.Process.ExitCode
        if ($exitCode -ne 0) {
            Write-Host "  [FAIL] $($p.Name) ket thuc voi ma loi: $exitCode" -ForegroundColor Red
            if (Test-Path $p.ErrFile) {
                Get-Content $p.ErrFile -Tail 20 | Write-Host -ForegroundColor DarkRed
            }
            $failed = $true
        } else {
            Write-Host "  [DONE] $($p.Name) da hoan tat thanh cong!" -ForegroundColor Green
        }
    }

    if ($failed) {
        Write-Host "`n>>> [ERROR] Me chay song song '$BatchName' co loi phat sinh!" -ForegroundColor Red
        exit 1
    }

    $batch_dur = ((Get-Date) - $t_batch_start).TotalSeconds
    Write-Host ">>> HOAN THANH ME: $BatchName trong [${batch_dur}s]" -ForegroundColor Green
}

$t_all_start = Get-Date

# Giai doan 1: Step 1 Calibration (chay don le vi cac step sau phu thuoc vao calibration_and_detectors.pkl)
Run-SingleScript -Name "Step 1: Calibration & Detectors Extraction" -ScriptPath "$EXP_DIR\step1_extract_calibration.py"

# Giai doan 2: Me 1 - 4 tien trinh song song
$batch1 = @(
    @{ Name = "step2_controls_fpr"; Script = "$EXP_DIR\step2_evaluate_controls_and_fpr.py" },
    @{ Name = "step3_patch_attack"; Script = "$EXP_DIR\step3_evaluate_patch_attack.py" },
    @{ Name = "step4_aqsm_esm"; Script = "$EXP_DIR\step4_evaluate_aqsm_and_esm.py" },
    @{ Name = "step5_query_attacks"; Script = "$EXP_DIR\step5_evaluate_query_attacks.py" }
)
Run-ParallelBatch -BatchName "Me 1 (Controls, Patch 48x48, AQSM, Query Attacks)" -Tasks $batch1

# Giai doan 3: Me 2 - 3 tien trinh song song
$batch2 = @(
    @{ Name = "step6_adaptive_attacks"; Script = "$EXP_DIR\step6_evaluate_adaptive_attacks.py" },
    @{ Name = "step7_causal_query_swap"; Script = "$EXP_DIR\step7_causal_query_swap.py" },
    @{ Name = "step8_edge_latency"; Script = "$EXP_DIR\step8_measure_edge_latency.py" }
)
Run-ParallelBatch -BatchName "Me 2 (Adaptive Evasion, Causal Swap, Edge Latency)" -Tasks $batch2

# Giai doan 4: Thuc nghiem Phan bien & Danh gia mAP COCO
Run-SingleScript -Name "Danh gia Dinh luong 2 Loai mAP COCO" -ScriptPath "$BASE_DIR\run_map_evaluation.py"
Run-SingleScript -Name "Co lap Co che Nhan qua AQSM (Oracle Swap, GT Eviction, Loss Ablation)" -ScriptPath "$EXP_DIR\exp_causal_mechanism_isolation.py"
Run-SingleScript -Name "Quet Ngan sach Nhieu Epsilon (2, 4, 8, 16 / 255)" -ScriptPath "$EXP_DIR\exp_epsilon_sweep.py"
Run-SingleScript -Name "Kiem dinh Thong ke Ghep cap McNemar" -ScriptPath "$EXP_DIR\compute_mcnemar_tests.py"

# Giai doan 5: Hop nhat ket qua tong the (Master SSOT)
Run-SingleScript -Name "Hop nhat Ket qua Toan dien (JSON & NPZ)" -ScriptPath "$EXP_DIR\aggregate_all_results.py"

# Giai doan 6: Sinh 12 Publication Figures & Bien dich Camera-Ready
Run-SingleScript -Name "Sinh Toan bo 12 Publication Figures Vector" -ScriptPath "$BASE_DIR\generate_publication_figures.py"
Run-SingleScript -Name "Bien dich Bai bao Elsevier EAAI Camera-Ready" -ScriptPath "$BASE_DIR\paper\compile_paper.py"
Run-SingleScript -Name "Bien dich Slide Bao cao Beamer Camera-Ready" -ScriptPath "$BASE_DIR\slides\compile_slide.py"

$t_all_dur = ((Get-Date) - $t_all_start).TotalSeconds
Write-Host "`n=================================================================================" -ForegroundColor Green
Write-Host "   TOAN BO 16 BUOC THUC NGHIEM VA BIEN DICH DA HOAN TAT TRONG [${t_all_dur}s]!   " -ForegroundColor Green
Write-Host "   File ket qua: edge_sec_pro/results/rtdetr_advanced_defense_results.json       " -ForegroundColor Green
Write-Host "   Bao cao Markdown: edge_sec_pro/results/rtdetr_advanced_defense_report.md      " -ForegroundColor Green
Write-Host "   Paper PDF: edge_sec_pro/paper/main.pdf                                        " -ForegroundColor Green
Write-Host "   Slide PDF: edge_sec_pro/slides/slide_rtdetr_security_eaai.pdf                 " -ForegroundColor Green
Write-Host "=================================================================================" -ForegroundColor Green
