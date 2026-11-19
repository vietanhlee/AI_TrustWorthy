<#
.SYNOPSIS
    Master Execution Script for Paper 2 on Windows PowerShell.
    Chạy chuỗi thực nghiệm Paper 2 (Task 0 -> Task 5) với kiến trúc Zero-OOM.
#>

[CmdletBinding()]
param (
    [Parameter(Mandatory=$true)]
    [string]$FolderImage,

    [string]$Annotations = "",
    [string]$Tasks = "0,1,2,3,4,5",
    [string]$Stage = "all",
    [int]$NumImages = 0,
    [string]$Seeds = "42,123,456",
    [string]$Device = "auto",
    [int]$BatchSize = 0,
    [string]$OutputDir = "outputs",
    [string]$CacheDir = "cache/adv_cache_640",
    [string]$FigDir = "fig",
    [switch]$SkipCacheGen,
    [switch]$ForceRecompute
)

$ErrorActionPreference = "Stop"

if ($Stage -eq "benchmark") {
    $Tasks = "0,1,4"
} elseif ($Stage -eq "ablations") {
    $Tasks = "2,3"
} elseif ($Stage -eq "figures") {
    $Tasks = "5"
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
Write-Host " Thư mục ảnh COCO : $FolderImage"
Write-Host " Python Engine    : $PythonCmd"
Write-Host " Danh sách Tasks  : $Tasks"
Write-Host " Thiết bị         : $Device"
Write-Host " Thư mục Output   : $OutputDir"
Write-Host "======================================================================"

$CmdArgs = @(
    "run_all_experiments.py",
    "--folder_image", $FolderImage,
    "--tasks", $Tasks,
    "--num_images", $NumImages,
    "--seeds", $Seeds,
    "--device", $Device,
    "--batch_size", $BatchSize,
    "--cache_dir", $CacheDir,
    "--output_dir", $OutputDir,
    "--fig_dir", $FigDir
)

if ($Annotations) {
    $CmdArgs += @("--annotations", $Annotations)
}
if ($SkipCacheGen) {
    $CmdArgs += @("--skip_cache_gen")
}
if ($ForceRecompute) {
    $CmdArgs += @("--force_recompute")
}

& $PythonCmd $CmdArgs
