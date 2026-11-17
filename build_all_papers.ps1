# ==============================================================================
# Master Build Script: Compile All 3 Papers & Clean Build Artifacts
# Paper 1: AQSM Mechanism Breakdown (Elsevier EAAI)
# Paper 2: Phantom Queries Fabrication (Elsevier)
# Paper 3: Split-Percentile Defense Guardrails
# ==============================================================================
$ErrorActionPreference = "Stop"
$rootDir = $PSScriptRoot
$submissionDir = Join-Path $rootDir "SUBMISSION_PDFS"

if (-not (Test-Path $submissionDir)) {
    New-Item -ItemType Directory -Path $submissionDir -Force | Out-Null
}

function Clean-LatexArtifacts($dir) {
    Get-ChildItem -Path $dir -Include *.aux, *.log, *.bbl, *.blg, *.out, *.toc, *.fls, *.fdb_latexmk, *.synctex.gz -Recurse -File -ErrorAction SilentlyContinue | Remove-Item -Force
}

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host " [1/3] Compiling Paper 1 (AQSM Mechanism - Elsevier EAAI)..." -ForegroundColor Yellow
Write-Host "=================================================================" -ForegroundColor Cyan
$p1Dir = Join-Path $rootDir "paper_1_aqsm_mechanism"
Push-Location $p1Dir
try {
    pdflatex -interaction=nonstopmode main.tex | Out-Null
    bibtex main | Out-Null
    pdflatex -interaction=nonstopmode main.tex | Out-Null
    pdflatex -interaction=nonstopmode main.tex | Out-Null
    Copy-Item "main.pdf" (Join-Path $submissionDir "1_Paper_AQSM_Mechanism_Elsevier_EAAI.pdf") -Force
    Clean-LatexArtifacts $p1Dir
    Write-Host ">>> Done: Paper 1 (AQSM Mechanism - Elsevier EAAI)" -ForegroundColor Green
} finally {
    Pop-Location
}

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host " [2/3] Compiling Paper 2 (Phantom Queries - Elsevier)..." -ForegroundColor Yellow
Write-Host "=================================================================" -ForegroundColor Cyan
$p2Dir = Join-Path $rootDir "paper_2_phantom_fabrication"
Push-Location $p2Dir
try {
    pdflatex -interaction=nonstopmode main_phantom.tex | Out-Null
    bibtex main_phantom | Out-Null
    pdflatex -interaction=nonstopmode main_phantom.tex | Out-Null
    pdflatex -interaction=nonstopmode main_phantom.tex | Out-Null
    Copy-Item "main_phantom.pdf" (Join-Path $submissionDir "3_Paper_Phantom_Queries_Fabrication_Elsevier.pdf") -Force
    Clean-LatexArtifacts $p2Dir
    Write-Host ">>> Done: Paper 2 (Phantom Queries)" -ForegroundColor Green
} finally {
    Pop-Location
}

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host " [3/3] Compiling Paper 3 (Split-Percentile Guardrail)..." -ForegroundColor Yellow
Write-Host "=================================================================" -ForegroundColor Cyan
$p3Dir = Join-Path $rootDir "paper_3_defense_guardrail"
Push-Location $p3Dir
try {
    pdflatex -interaction=nonstopmode main_defense.tex | Out-Null
    bibtex main_defense | Out-Null
    pdflatex -interaction=nonstopmode main_defense.tex | Out-Null
    pdflatex -interaction=nonstopmode main_defense.tex | Out-Null
    Copy-Item "main_defense.pdf" (Join-Path $submissionDir "4_Paper_Defense_Split_Percentile_Guardrails.pdf") -Force
    Clean-LatexArtifacts $p3Dir
    Write-Host ">>> Done: Paper 3 (Defense Guardrail)" -ForegroundColor Green
} finally {
    Pop-Location
}

Write-Host "`n=================================================================" -ForegroundColor Cyan
Write-Host "   ALL 3 PAPERS COMPILED AND READY IN SUBMISSION_PDFS!" -ForegroundColor Green
Write-Host "=================================================================" -ForegroundColor Cyan
