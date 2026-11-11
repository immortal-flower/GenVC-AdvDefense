param(
    [string]$Python = 'C:\Users\Admin\.conda\envs\GVCC-5090\python.exe',
    [string]$Data = 'D:\yzb and lmk\dataset-720P\UVG\Jockey_720p.yuv',
    [string]$WanCkpt = '.\exp_flf2v\Wan2.1-FLF2V-14B-720P',
    [string]$OutputRoot = '.\exp_flf2v\results_720p',
    [int]$ProxySteps = 12,
    [int[]]$AttackSeeds = @(42, 123, 2026),
    [switch]$PlanOnly
)

$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
if ($AttackSeeds.Count -lt 1 -or ($AttackSeeds | Select-Object -Unique).Count -ne $AttackSeeds.Count) {
    throw 'Supply at least one unique attack seed.'
}
$noiseTag = '0p3062'
$runTags = foreach ($attackSeed in $AttackSeeds) {
    "input_codebook_proxy_step1pos3_eps2_seed42_aseed${attackSeed}_p${ProxySteps}_n${noiseTag}_gnative"
}
Write-Host "Conditional native-proxy follow-up: $($AttackSeeds.Count) attack seeds, $ProxySteps optimization steps."
Write-Host 'The next seed runs only if every earlier seed beats matched random by >=0.3 dB PSNR and >=0.005 LPIPS.'
Write-Host 'Each pair uses fresh GVCC re-encoding; prior runs with metrics.json are resumed.'
if ($PlanOnly) {
    $runTags | ForEach-Object { Write-Host "[PLAN] $_" }
    return
}

$outcomes = @()
foreach ($i in 0..($AttackSeeds.Count - 1)) {
    $attackSeed = $AttackSeeds[$i]
    $tag = $runTags[$i]
    Write-Host "[STAGE $($i+1)/$($AttackSeeds.Count)] attack seed $attackSeed"
    & '.\exp_flf2v\run_codebook_input_pilot.ps1' `
        -Python $Python -Data $Data -WanCkpt $WanCkpt -OutputRoot $OutputRoot `
        -Seed 42 -AttackSeed $attackSeed -ProxySteps $ProxySteps -ProxyGeometry native
    if (-not $?) { throw "Pilot failed: $tag" }
    $analysis = Join-Path $OutputRoot "${tag}_analysis.json"
    if (-not (Test-Path -LiteralPath $analysis -PathType Leaf)) {
        throw "Analysis is missing: $analysis"
    }
    $report = Get-Content -LiteralPath $analysis -Raw | ConvertFrom-Json
    $paired = $report.paired_comparison
    $outcomes += [pscustomobject]@{
        AttackSeed = $attackSeed
        CandidateExtraPSNRDrop_dB = [double]$paired.candidate_extra_PSNR_drop_dB
        CandidateExtraLPIPS = [double]$paired.candidate_extra_LPIPS
        CandidateExtraPSNRDropVsSubmitted_dB = [double]$paired.candidate_extra_PSNR_drop_vs_submitted_dB
        InputMSEMatched = [bool]$paired.input_MSE_matched_within_1pct
        WithinBound = [bool]$paired.both_within_2_255_bound
        ScreeningResult = [string]$paired.screening_result
        Analysis = $analysis
    }
    $outcomes | Format-Table -AutoSize
    $outcomes | Export-Csv (Join-Path $OutputRoot "codebook_input_followup_p${ProxySteps}_summary.csv") `
        -NoTypeInformation -Encoding UTF8
    if ($paired.screening_result -ne 'promising_single_GOP') {
        Write-Host "[STOP] $tag did not pass the predeclared screening gate. Remaining seeds were not run."
        break
    }
}
