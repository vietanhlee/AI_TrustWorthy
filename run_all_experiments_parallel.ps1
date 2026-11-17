# ==============================================================================
# Master Shortcut Runner: Chạy toàn bộ thực nghiệm song song (Tối đa 4 task đồng thời)
# ==============================================================================
param (
    [int]$MaxWorkers = 4,
    [int]$CohortSize = 300,
    [string]$Tasks = "1,2,3,4,5,6,7",
    [switch]$DryRun
)

$targetScript = Join-Path $PSScriptRoot "paper_1_aqsm_mechanism\experiments\run_parallel_benchmarks.ps1"

$params = @{
    MaxWorkers = $MaxWorkers
    CohortSize = $CohortSize
    Tasks = $Tasks
}

if ($DryRun) {
    $params["DryRun"] = $true
}

& $targetScript @params
