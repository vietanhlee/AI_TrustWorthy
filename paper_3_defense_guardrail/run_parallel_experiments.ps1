# ==============================================================================
# PowerShell Parallel Runner: Chay Toan Bo Thuc Nghiem Paper 3 (Max 4 Workers)
# Tich hop Live Heartbeat Ticker va Hien thi Tien do theo Thoi gian Thuc
# ==============================================================================
param (
    [int]$MaxWorkers = 4,
    [switch]$SkipCalibration = $false
)

$ErrorActionPreference = "Stop"
$ScriptDir = $PSScriptRoot
$RootDir = Split-Path -Parent $ScriptDir
$ExpDir = Join-Path $ScriptDir "experiments"
$LogsDir = Join-Path $ScriptDir "logs"
$ResultsDir = Join-Path $ScriptDir "results"

if (-not (Test-Path $LogsDir)) { New-Item -ItemType Directory -Path $LogsDir -Force | Out-Null }
if (-not (Test-Path $ResultsDir)) { New-Item -ItemType Directory -Path $ResultsDir -Force | Out-Null }

# Tim Python Executable
$PythonExe = Join-Path $RootDir ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}

Write-Host "=================================================================================" -ForegroundColor Cyan
Write-Host "   PAPER 3 PARALLEL EXPERIMENT RUNNER (MAX $MaxWorkers CONCURRENT WORKERS)       " -ForegroundColor Cyan
Write-Host "   Python: $PythonExe                                                            " -ForegroundColor Cyan
Write-Host "   Logs  : $LogsDir                                                              " -ForegroundColor Cyan
Write-Host "=================================================================================" -ForegroundColor Cyan

$TotalStartTime = Get-Date

# Function chay don le dong bo (Synchronous) voi Live Output
function Run-StepSynchronous {
    param (
        [string]$TaskName,
        [string]$ScriptPath
    )
    Write-Host "`n>>> [DONG BO] DANG KHOI CHAY LIVE TQDM: $TaskName..." -ForegroundColor Yellow
    $t_start = Get-Date
    & $PythonExe $ScriptPath
    if ($LASTEXITCODE -ne 0) {
        Write-Host ">>> [LOI] $TaskName that bai voi ma loi $LASTEXITCODE!" -ForegroundColor Red
        exit $LASTEXITCODE
    }
    $dur = [math]::Round(((Get-Date) - $t_start).TotalSeconds, 1)
    Write-Host ">>> [HOAN TAT] $TaskName thanh cong trong [${dur}s]!" -ForegroundColor Green
}

# ------------------------------------------------------------------------------
# GIAI DOAN 1: BƯỚC 1 CALIBRATION
# ------------------------------------------------------------------------------
$CkptFile = Join-Path $RootDir "edge_sec_pro\results\checkpoints\calibration_and_detectors.pkl"
if (-not $SkipCalibration -or -not (Test-Path $CkptFile)) {
    Run-StepSynchronous -TaskName "Step 1: Calibration & Detectors Fitting" -ScriptPath (Join-Path $ExpDir "run_calibration.py")
} else {
    Write-Host "`n>>> [GIAI DOAN 1] Bo qua Buoc 1 vi checkpoint da ton tai san: $CkptFile" -ForegroundColor DarkCyan
}

# ------------------------------------------------------------------------------
# GIAI DOAN 2: CHAY SONG SONG MAX 4 WORKERS QUA HANG DOI (PROCESS QUEUE)
# ------------------------------------------------------------------------------
$TasksQueue = [System.Collections.Generic.Queue[hashtable]]::new()

$TasksQueue.Enqueue(@{ Name = "step2_controls_fpr"; Script = (Join-Path $ExpDir "run_controls_and_fpr.py") })
$TasksQueue.Enqueue(@{ Name = "step3_patch_attack"; Script = (Join-Path $ExpDir "run_patch_attack_eval.py") })
$TasksQueue.Enqueue(@{ Name = "step4_aqsm_esm"; Script = (Join-Path $ExpDir "run_aqsm_and_esm.py") })
$TasksQueue.Enqueue(@{ Name = "step5_query_attacks"; Script = (Join-Path $ExpDir "run_query_attacks_eval.py") })
$TasksQueue.Enqueue(@{ Name = "step6_adaptive_attacks"; Script = (Join-Path $ExpDir "run_adaptive_attacks_eval.py") })
$TasksQueue.Enqueue(@{ Name = "step9_defensive_baselines"; Script = (Join-Path $ExpDir "run_defensive_baselines.py") })
$TasksQueue.Enqueue(@{ Name = "step8_edge_latency"; Script = (Join-Path $ExpDir "run_edge_latency.py") })
$TasksQueue.Enqueue(@{ Name = "step10_epsilon_sweep"; Script = (Join-Path $ExpDir "run_epsilon_sweep.py") })
$TasksQueue.Enqueue(@{ Name = "step11_mcnemar_tests"; Script = (Join-Path $ExpDir "run_mcnemar_tests.py") })

$TotalParallelTasks = $TasksQueue.Count
Write-Host "`n=================================================================================" -ForegroundColor Magenta
Write-Host ">>> GIAI DOAN 2: BAT DAU DIEU PHOI $TotalParallelTasks THUC NGHIEM SONG SONG (MAX $MaxWorkers DONG THOI)..." -ForegroundColor Magenta
Write-Host "=================================================================================" -ForegroundColor Magenta

$ActiveJobs = [System.Collections.Generic.List[hashtable]]::new()
$CompletedJobs = [System.Collections.Generic.List[hashtable]]::new()
$LastHeartbeat = Get-Date

while ($TasksQueue.Count -gt 0 -or $ActiveJobs.Count -gt 0) {
    # 1. Day them job vao active slot neu con cho trong
    while ($ActiveJobs.Count -lt $MaxWorkers -and $TasksQueue.Count -gt 0) {
        $task = $TasksQueue.Dequeue()
        $tName = $task.Name
        $tScript = $task.Script
        $timestamp = (Get-Date).ToString("yyyyMMdd_HHmmss")
        $outFile = Join-Path $LogsDir "${tName}_pwsh_${timestamp}.out.log"
        $errFile = Join-Path $LogsDir "${tName}_pwsh_${timestamp}.err.log"

        Write-Host "  -> [DISPATCH] Khoi chay worker: $tName" -ForegroundColor Cyan
        $proc = Start-Process -FilePath $PythonExe -ArgumentList $tScript -NoNewWindow -PassThru -RedirectStandardOutput $outFile -RedirectStandardError $errFile
        $startTime = Get-Date

        $ActiveJobs.Add(@{
            Name = $tName;
            Process = $proc;
            OutFile = $outFile;
            ErrFile = $errFile;
            StartTime = $startTime
        })
    }

    Start-Sleep -Milliseconds 800

    # 2. Kiem tra cac tien trinh da hoan tat
    for ($i = $ActiveJobs.Count - 1; $i -ge 0; $i--) {
        $job = $ActiveJobs[$i]
        if ($job.Process.HasExited) {
            $exitCode = $job.Process.ExitCode
            $dur = [math]::Round(((Get-Date) - $job.StartTime).TotalSeconds, 1)

            if ($exitCode -eq 0) {
                Write-Host "  [+] [SUCCESS] ($($CompletedJobs.Count + 1)/$TotalParallelTasks) $($job.Name) hoan tat thanh cong trong [${dur}s]!" -ForegroundColor Green
            } else {
                Write-Host "  [!] [FAILED]  ($($CompletedJobs.Count + 1)/$TotalParallelTasks) $($job.Name) that bai voi ma loi $exitCode sau [${dur}s]!" -ForegroundColor Red
                if (Test-Path $job.ErrFile) {
                    Get-Content $job.ErrFile -Tail 10 | Write-Host -ForegroundColor DarkRed
                }
            }

            $job["ExitCode"] = $exitCode
            $job["Duration"] = $dur
            $CompletedJobs.Add($job)
            $ActiveJobs.RemoveAt($i)
        }
    }

    # 3. Heartbeat Monitor: In ra trang thai tien do dinh ky moi 4 giay
    if ($ActiveJobs.Count -gt 0 -and ((Get-Date) - $LastHeartbeat).TotalSeconds -ge 4.0) {
        $LastHeartbeat = Get-Date
        $statusList = @()
        foreach ($j in $ActiveJobs) {
            $curSec = [math]::Round(((Get-Date) - $j.StartTime).TotalSeconds, 0)
            $lastLog = "Dang chay..."
            if (Test-Path $j.OutFile) {
                $lines = Get-Content $j.OutFile -Tail 3 -ErrorAction SilentlyContinue
                if ($lines -and $lines.Count -gt 0) {
                    $raw = ($lines | Select-Object -Last 1).Trim()
                    if ($raw.Length -gt 50) { $raw = $raw.Substring(0, 47) + "..." }
                    $lastLog = $raw
                }
            }
            $statusList += "[$($j.Name) (${curSec}s): $lastLog]"
        }
        Write-Host ("    >>> [TIEN DO: $($CompletedJobs.Count)/$TotalParallelTasks Hoan tat] " + ($statusList -join " | ")) -ForegroundColor DarkYellow
    }
}

# ------------------------------------------------------------------------------
# GIAI DOAN 3: TONG KET
# ------------------------------------------------------------------------------
$TotalDur = [math]::Round(((Get-Date) - $TotalStartTime).TotalSeconds, 1)

Write-Host "`n=================================================================================" -ForegroundColor Cyan
Write-Host "                TONG KET TIEN DO THUC NGHIEM PAPER 3 PARALLEL                     " -ForegroundColor Cyan
Write-Host "=================================================================================" -ForegroundColor Cyan
foreach ($job in $CompletedJobs) {
    $st = if ($job.ExitCode -eq 0) { "THANH CONG" } else { "THAT BAI" }
    $color = if ($job.ExitCode -eq 0) { "Green" } else { "Red" }
    Write-Host ("{0,-28} | {1,-12} | {2,8}s | Log: {3}" -f $job.Name, $st, $job.Duration, (Split-Path $job.OutFile -Leaf)) -ForegroundColor $color
}
Write-Host "---------------------------------------------------------------------------------" -ForegroundColor Cyan
Write-Host "Tong thoi gian hoan tat: ${TotalDur}s (~$([math]::Round($TotalDur/60, 2)) phut)." -ForegroundColor Cyan
Write-Host "=================================================================================" -ForegroundColor Cyan
