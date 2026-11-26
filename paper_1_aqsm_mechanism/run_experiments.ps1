<#
.SYNOPSIS
    Script PowerShell điều phối chạy toàn bộ thực nghiệm Paper 1 (AQSM Mechanism Breakdown) trên Windows.
.DESCRIPTION
    Hỗ trợ chạy linh hoạt trên toàn bộ 5,000 ảnh COCO val2017 hoặc tập ảnh bất kỳ.
.EXAMPLE
    .\run_experiments.ps1 -FolderImage "G:\Edge_Sec\edge_sec_pro\data\coco5k\val2017"
#>

[CmdletBinding()]
param (
    [Parameter(Mandatory = $false)]
    [string]$FolderImage = "",

    [Parameter(Mandatory = $false)]
    [string]$Annotations = "",

    [Parameter(Mandatory = $false)]
    [string]$OutputDir = "",

    [Parameter(Mandatory = $false)]
    [string]$CacheDir = "",

    [Parameter(Mandatory = $false)]
    [string]$FigDir = "",

    [Parameter(Mandatory = $false)]
    [string]$Device = "auto",

    [Parameter(Mandatory = $false)]
    [int]$BatchSize = 0,

    [Parameter(Mandatory = $false)]
    [int]$Threads = 8,

    [Parameter(Mandatory = $false)]
    [int]$MaxSamples = 1000,

    [Parameter(Mandatory = $false)]
    [string]$Model = "r18",

    [Parameter(Mandatory = $false)]
    [string]$Tasks = "0,1,2,3,4,5,6,8,10,7,9",

    [Parameter(Mandatory = $false)]
    [switch]$SkipCacheGen,

    [Parameter(Mandatory = $false)]
    [switch]$ForceRecompute,

    [Parameter(Mandatory = $false)]
    [switch]$IncludeSweep,

    [Parameter(Mandatory = $false)]
    [switch]$RunSqaAblation,

    [Parameter(Mandatory = $false)]
    [switch]$RunMultiseed,

    [Parameter(Mandatory = $false)]
    [string]$Seeds = "",

    [Parameter(Mandatory = $false)]
    [int]$PgdSteps = 15
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host " 🚀 KHỞI ĐỘNG HỆ THỐNG THỰC NGHIỆM PAPER 1: DYNAMIC QUERY SELECTION (RT-DETR)" -ForegroundColor Cyan
Write-Host " Thư mục làm việc: $ScriptDir" -ForegroundColor Cyan
Write-Host "==============================================================================" -ForegroundColor Cyan

# 1. Tìm Python interpreter
$PythonBin = ""
$VenvPython = Join-Path (Split-Path $ScriptDir -Parent) ".venv\Scripts\python.exe"
if (Test-Path $VenvPython) {
    $PythonBin = $VenvPython
} elseif (Get-Command "python" -ErrorAction SilentlyContinue) {
    $PythonBin = "python"
} else {
    Write-Error "Không tìm thấy trình thông dịch Python hợp lệ!"
    exit 1
}

Write-Host " Đang sử dụng Python: $PythonBin" -ForegroundColor Green
& $PythonBin -c "import torch; print(f' PyTorch: {torch.__version__} | CUDA Available: {torch.cuda.is_available()} | GPU Count: {torch.cuda.device_count()}')"

# 2. Xây dựng danh sách tham số
$PyArgs = @((Join-Path $ScriptDir "run_all_experiments.py"))

if ($FolderImage -ne "") { $PyArgs += @("--folder_image", $FolderImage) }
if ($Annotations -ne "") { $PyArgs += @("--annotations", $Annotations) }
if ($OutputDir -ne "") { $PyArgs += @("--output_dir", $OutputDir) }
if ($CacheDir -ne "") { $PyArgs += @("--cache_dir", $CacheDir) }
if ($FigDir -ne "") { $PyArgs += @("--fig_dir", $FigDir) }
if ($Device -ne "") { $PyArgs += @("--device", $Device) }
if ($BatchSize -gt 0) { $PyArgs += @("--batch_size", $BatchSize) }
if ($Threads -gt 0) { $PyArgs += @("--threads", $Threads) }
if ($MaxSamples -ge 0) { $PyArgs += @("--max_samples", $MaxSamples) }
if ($Model -ne "") { $PyArgs += @("--model", $Model) }
if ($Tasks -ne "") { $PyArgs += @("--tasks", $Tasks) }
if ($SkipCacheGen) { $PyArgs += @("--skip_cache_gen") }
if ($ForceRecompute) { $PyArgs += @("--force_recompute") }
if ($IncludeSweep) { $PyArgs += @("--include_sweep") }
if ($RunSqaAblation) { $PyArgs += @("--run_sqa_ablation") }
if ($RunMultiseed) { $PyArgs += @("--run_multiseed") }
if ($Seeds -ne "") { $PyArgs += @("--seeds", $Seeds) }
if ($PgdSteps -gt 0) { $PyArgs += @("--pgd_steps", $PgdSteps) }

Write-Host ""
Write-Host " Đang thực thi Master Orchestrator (run_all_experiments.py)..." -ForegroundColor Yellow
& $PythonBin $PyArgs

$ExitCode = $LASTEXITCODE
if ($ExitCode -eq 0) {
    Write-Host ""
    Write-Host "==============================================================================" -ForegroundColor Green
    Write-Host " ✅ TOÀN BỘ CHUỖI THỰC NGHIỆM ĐÃ HOÀN TẤT THÀNH CÔNG RỰC RỠ!" -ForegroundColor Green
    Write-Host " Dữ liệu số liệu Table và Figures đã sẵn sàng trong thư mục outputs/ và fig/." -ForegroundColor Green
    Write-Host "==============================================================================" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "==============================================================================" -ForegroundColor Red
    Write-Host " ❌ THỰC NGHIỆM BỊ GIÁN ĐOẠN HOẶC GẶP LỖI (Exit code: $ExitCode)" -ForegroundColor Red
    Write-Host " Kiểm tra chi tiết tại thư mục outputs/logs/." -ForegroundColor Red
    Write-Host "==============================================================================" -ForegroundColor Red
}

exit $ExitCode
