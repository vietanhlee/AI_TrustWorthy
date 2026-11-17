# ==============================================================================
# Master Parallel Benchmark Runner Script (PowerShell)
# Bài báo 1: AQSM Mechanism & Query Selection Vulnerability (Elsevier EAAI)
# Khả năng: Chạy song song tối đa 4 task đồng thời, kiểm soát luồng và xuất log kiểm toán.
# ==============================================================================
param (
    [int]$MaxWorkers = 4,
    [int]$CohortSize = 300,
    [int]$ThreadsPerTask = 2,
    [string]$Tasks = "1,2,3,4,5,6,7",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$scriptDir = $PSScriptRoot
$rootDir = (Resolve-Path (Join-Path $scriptDir "..\..")).Path

# Xác định Python executable từ virtualenv .venv
$pythonExe = Join-Path $rootDir ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    $pythonExe = "python"
}

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host " HỆ THỐNG ĐIỀU PHỐI THỰC NGHIỆM SONG SONG (MAX $MaxWorkers WORKERS)" -ForegroundColor Yellow
Write-Host " Tác phẩm: Towards Reliable Real-Time DETR (Elsevier EAAI)" -ForegroundColor White
Write-Host " Thư mục gốc: $rootDir" -ForegroundColor Gray
Write-Host " Python runtime: $pythonExe" -ForegroundColor Gray
Write-Host " Cấu hình: MaxWorkers=$MaxWorkers | CohortSize=$CohortSize | Tasks=$Tasks" -ForegroundColor Gray
Write-Host "=================================================================" -ForegroundColor Cyan

$masterScript = Join-Path $scriptDir "run_master_benchmark_suite.py"

$cmdArgs = @(
    $masterScript,
    "--max_workers", $MaxWorkers.ToString(),
    "--cohort_size", $CohortSize.ToString(),
    "--threads", $ThreadsPerTask.ToString(),
    "--tasks", $Tasks
)

if ($DryRun) {
    $cmdArgs += "--dry_run"
}

$pinfo = New-Object System.Diagnostics.ProcessStartInfo
$pinfo.FileName = $pythonExe
$pinfo.Arguments = ($cmdArgs -join " ")
$pinfo.WorkingDirectory = $rootDir
$pinfo.UseShellExecute = $false
$pinfo.RedirectStandardOutput = $false
$pinfo.RedirectStandardError = $false

$process = [System.Diagnostics.Process]::Start($pinfo)
$process.WaitForExit()

if ($process.ExitCode -eq 0) {
    Write-Host "`n>>> [THÀNH CÔNG] Toàn bộ thực nghiệm song song đã hoàn tất chuẩn chỉ!" -ForegroundColor Green
} else {
    Write-Host "`n>>> [CẢNH BÁO] Quá trình chạy thực nghiệm kết thúc với mã lỗi $($process.ExitCode)." -ForegroundColor Red
    exit $process.ExitCode
}
