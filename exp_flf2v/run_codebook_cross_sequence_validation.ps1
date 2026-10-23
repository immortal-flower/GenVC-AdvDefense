param(
    [string]$Sequence = 'Beauty',
    [string]$Data = 'D:\yzb and lmk\dataset-720P\UVG\Beauty_720p.yuv',
    [string]$CleanRun = 'crossseq_clean',
    [int[]]$Counts = @(16),
    [int[]]$AttackSeeds = @(42,123,2026),
    [string]$Plan = 'exp_flf2v/validation_cells_jockey.csv',
    [string]$CondaExe = 'D:\anaconda\Scripts\conda.exe',
    [string]$ResultsRoot = 'exp_flf2v/results_720p',
    [switch]$SaveVideo,
    [switch]$PlanOnly
)

# Transfer the Jockey hotspot/control coordinates to another UVG sequence.
# The clean GOP and codebook are reused, while Wan, conditions, prompt and
# LPIPS are each loaded or encoded only once for the whole validation batch.
Set-Location (Split-Path -Parent $PSScriptRoot)

$cleanDir = Join-Path $ResultsRoot "$CleanRun/$Sequence/gop0"
$cleanCodebook = Join-Path $cleanDir 'codebook.tdcm'
$cleanMetrics = Join-Path $cleanDir 'metrics.json'

foreach($required in @($Plan, $Data)){
    if(-not (Test-Path -LiteralPath $required)){
        throw "Required input not found: $required"
    }
}
foreach($baseline in @($cleanCodebook, $cleanMetrics)){
    if(-not (Test-Path -LiteralPath $baseline)){
        throw ("Cross-sequence clean baseline not found: $baseline`n" +
               "Create it first with run_codebook_cross_sequence.ps1.")
    }
}

Write-Host "Cross-sequence targeted validation: $Sequence"
Write-Host "Data: $Data"
Write-Host "Clean codebook: $cleanCodebook"
Write-Host "Clean metrics: $cleanMetrics"
Write-Host "Plan: $Plan"

$batchArgs = @{
    Counts = $Counts
    AttackSeeds = $AttackSeeds
    Sequence = $Sequence
    CondaExe = $CondaExe
    Data = $Data
    ResultsRoot = $ResultsRoot
    CleanCodebook = $cleanCodebook
    CleanMetrics = $cleanMetrics
    Plan = $Plan
}
if($SaveVideo){ $batchArgs.SaveVideo = $true }
if($PlanOnly){ $batchArgs.PlanOnly = $true }

& "$PSScriptRoot\run_codebook_validation_batch.ps1" @batchArgs
