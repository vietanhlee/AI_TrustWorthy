[CmdletBinding()]
param (
    [string]$FolderImage = "",
    [string]$Annotations = "",
    [string]$Tasks = "0,1,2,3,4,5,6",
    [string]$Stage = "all",
    [int]$NumImages = 0,
    [string]$Seeds = "42,123,456",
    [string]$Device = "auto",
    [int]$BatchSize = 0,
    [string]$OutputDir = "outputs",
    [string]$CacheDir = "cache/adv_cache_640",
    [string]$FigDir = "fig",
    [switch]$SkipCacheGen,
    [switch]$ForceRecompute,
    [switch]$SyncOnly,
    [switch]$SkipPaperCompile
)

$ErrorActionPreference = "Stop"

if ($SyncOnly -or ($Stage -eq "figures") -or ($Stage -eq "sync")) {
    $Tasks = "5,6"
} elseif ($Stage -eq "benchmark") {
    $Tasks = "0,1,4,5,6"
} elseif ($Stage -eq "ablations") {
    $Tasks = "2,3,5,6"
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$PythonCmd = "python"
if (Test-Path "..\.venv\Scripts\python.exe") {
    $PythonCmd = (Resolve-Path "..\.venv\Scripts\python.exe").Path
} elseif (Test-Path ".venv\Scripts\python.exe") {
    $PythonCmd = (Resolve-Path ".venv\Scripts\python.exe").Path
}

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " KHỞI ĐỘNG HỆ THỐNG THỰC NGHIỆM PAPER 2 (WINDOWS POWERSHELL)" -ForegroundColor Cyan
Write-Host "----------------------------------------------------------------------"
if ($FolderImage) {
    Write-Host " Thư mục ảnh COCO : $FolderImage"
} else {
    Write-Host " Chế độ thực thi  : Hậu xử lý / Đồng bộ (Sync Only)"
}
Write-Host " Python Engine    : $PythonCmd"
Write-Host " Danh sách Tasks  : $Tasks"
Write-Host " Thiết bị         : $Device"
Write-Host " Thư mục Output   : $OutputDir"
Write-Host "======================================================================"

$CmdArgs = @(
    "run_all_experiments.py",
    "--tasks", $Tasks,
    "--num_images", $NumImages,
    "--seeds", $Seeds,
    "--device", $Device,
    "--batch_size", $BatchSize,
    "--cache_dir", $CacheDir,
    "--output_dir", $OutputDir,
    "--fig_dir", $FigDir
)

if ($FolderImage) {
    $CmdArgs += @("--folder_image", $FolderImage)
}
if ($Annotations) {
    $CmdArgs += @("--annotations", $Annotations)
}
if ($SkipCacheGen) {
    $CmdArgs += @("--skip_cache_gen")
}
if ($ForceRecompute) {
    $CmdArgs += @("--force_recompute")
}
if ($SyncOnly) {
    $CmdArgs += @("--sync_only")
}
if ($SkipPaperCompile) {
    $CmdArgs += @("--skip_paper_compile")
}

& $PythonCmd $CmdArgs
