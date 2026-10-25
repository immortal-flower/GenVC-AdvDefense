param(
    [ValidateSet(1,2)]
    [int]$SourceGop = 1,
    [string]$Sequence = 'Jockey',
    [string]$Data = 'D:\yzb and lmk\dataset-720P\UVG\Jockey_720p.yuv',
    [string]$CleanRoot = 'exp_flf2v/results_720p/Jockey',
    [int[]]$Counts = @(16),
    [int[]]$AttackSeeds = @(42,123,2026),
    [string]$Plan = 'exp_flf2v/validation_cells_jockey.csv',
    [string]$CondaExe = 'D:\anaconda\Scripts\conda.exe',
    [string]$ResultsRoot = 'exp_flf2v/results_720p',
    [switch]$SaveVideo,
    [switch]$PlanOnly
)

# Validate GOP-0 hotspot coordinates on later source GOPs while keeping their
# source frames, boundary conditions, clean codebooks and output paths separate.
Set-Location (Split-Path -Parent $PSScriptRoot)

$cleanCodebook = Join-Path $CleanRoot "gop$SourceGop/codebook.tdcm"
$cleanMetrics = Join-Path $CleanRoot "gop$SourceGop/metrics.json"
foreach($required in @($Plan, $Data, $cleanCodebook, $cleanMetrics)){
    if(-not (Test-Path -LiteralPath $required)){
        throw "Required cross-GOP input not found: $required"
    }
}

Write-Host "Persistent targeted validation: $Sequence GOP $SourceGop"
Write-Host "Clean codebook: $cleanCodebook"
Write-Host "Clean metrics: $cleanMetrics"

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
    SourceGop = $SourceGop
}
if($SaveVideo){ $batchArgs.SaveVideo = $true }
if($PlanOnly){ $batchArgs.PlanOnly = $true }

& "$PSScriptRoot\run_codebook_validation_batch.ps1" @batchArgs
