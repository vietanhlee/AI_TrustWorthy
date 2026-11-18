# ==============================================================================
# Master Shortcut Runner: Chạy toàn bộ thực nghiệm Paper 1 từ thư mục gốc
# ==============================================================================
param (
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ArgsRemaining
)

$targetScript = Join-Path $PSScriptRoot "paper_1_aqsm_mechanism\run_experiments.ps1"
& $targetScript @ArgsRemaining
