param(
    [string]$Python = 'C:\Users\Admin\.conda\envs\GVCC-5090\python.exe',
    [string]$Data = 'D:\yzb and lmk\dataset-720P\UVG\Jockey_720p.yuv',
    [string]$WanCkpt = '.\exp_flf2v\Wan2.1-FLF2V-14B-720P',
    [string]$OldAttack = '.\exp_flf2v\attack_assets\jockey_vae_temporal_pgd_eps2.npz',
    [string]$OutputRoot = '.\exp_flf2v\results_720p',
    [int]$Seed = 42,
    [Nullable[int]]$InitSeed = $null,
    [switch]$PairedOnly,
    [int]$Epsilon = 2,
    [int]$OptimizerSteps = 20,
    [switch]$PlanOnly
)

# One unattended entry point. Each finished stage is resumable; failed stages
# stop the pipeline instead of silently comparing incomplete experiments.
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:CUDA_VISIBLE_DEVICES = '0,1'
foreach ($path in @($Python, $Data, $WanCkpt)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Missing prerequisite: $path" }
}
if ($Epsilon -lt 1 -or $Epsilon -gt 8 -or $OptimizerSteps -lt 1) {
    throw 'Expected pixel epsilon in [1,8] and positive optimizer steps.'
}

$vae = Join-Path $WanCkpt 'Wan2.1_VAE.pth'
if (-not (Test-Path -LiteralPath $vae)) { throw "Missing VAE: $vae" }
$cleanTag = "input_pilot_v2_eps${Epsilon}_seed${Seed}_steps${OptimizerSteps}"
if ($null -eq $InitSeed) { $InitSeed = $Seed }
$tag = $cleanTag
if ($PairedOnly -or $InitSeed -ne $Seed) { $tag += "_init${InitSeed}" }
$suffix = if ($tag -ne $cleanTag) { "_init${InitSeed}" } else { '' }
$asset = ".\exp_flf2v\attack_assets\input_v2_balanced_eps${Epsilon}_seed${Seed}_steps${OptimizerSteps}${suffix}.npz"
$cleanRun = "${tag}_clean"
$meanAsset = $asset.Replace('_balanced_', '_mean_')
$randomAsset = $asset.Replace('.npz', '_initial.npz')
$runs = @(
    [pscustomobject]@{ Name="${tag}_clean"; Attack='none'; File=$null },
    [pscustomobject]@{ Name="${tag}_random"; Attack='vae-pgd'; File=$randomAsset },
    [pscustomobject]@{ Name="${tag}_mean_vae"; Attack='vae-pgd'; File=$meanAsset }
)
if ($Epsilon -eq 2 -and (Test-Path -LiteralPath $OldAttack)) {
    $runs += [pscustomobject]@{ Name="${tag}_old_vae"; Attack='vae-pgd'; File=$OldAttack }
} else {
    Write-Warning "Skipping old VAE comparison: requires an existing epsilon=2 asset ($OldAttack)"
}
$runs += [pscustomobject]@{ Name="${tag}_balanced_vae"; Attack='vae-pgd'; File=$asset }
if ($PairedOnly) {
    $cleanRun = "${cleanTag}_clean"
    $runs = @($runs | Where-Object { $_.Name -in @("${tag}_mean_vae", "${tag}_balanced_vae") })
    $baseline = Join-Path $OutputRoot "$cleanRun\Jockey\gop0"
    foreach ($file in @('metrics.json','codebook.tdcm','input_attack_trace.npz')) {
        if (-not (Test-Path -LiteralPath (Join-Path $baseline $file) -PathType Leaf)) {
            throw "Missing reusable clean result: $baseline\$file"
        }
    }
    Write-Host "Reusing clean comparison only: $baseline (candidate encodings remain independent)"
}

Write-Host "Input attack pilot: $Data | codec seed=$Seed | perturbation initialization seed=$InitSeed | epsilon=$Epsilon/255"
Write-Host "All codec evaluations re-encode the perturbed input; no clean codebook is reused."
$runs | ForEach-Object { Write-Host "  $($_.Name): $($_.Attack)" }
if ($PlanOnly) { Write-Host '[PLAN ONLY] No model loaded.'; return }

foreach ($objective in @('balanced','mean')) {
    $candidate = if ($objective -eq 'balanced') { $asset } else { $meanAsset }
    $initialFile = $candidate.Replace('.npz', '_initial.npz')
    if ((Test-Path -LiteralPath $candidate) -and (Test-Path -LiteralPath $initialFile)) { continue }
    & $Python '.\exp_flf2v\generate_vae_fullcontext_pgd.py' `
        --data $Data --vae $vae --output $candidate --height 720 --width 1280 `
        --frames 33 --anchors 3 --steps $OptimizerSteps --epsilon $Epsilon `
        --alpha 0.25 --objective $objective --balance_weight 1.0 `
        --balance_temperature 0.01 --seed $InitSeed
    if ($LASTEXITCODE -ne 0) { throw "Perturbation generation failed: $objective" }
}

foreach ($run in $runs) {
    $metrics = Join-Path $OutputRoot "$($run.Name)\Jockey\gop0\metrics.json"
    if (Test-Path -LiteralPath $metrics) {
        $existing = Get-Content -LiteralPath $metrics -Raw | ConvertFrom-Json
        if ($null -ne $existing.reconstruction_vs_submitted_PSNR_dB) {
            Write-Host "[SKIP] $($run.Name) already has complete v2 metrics"
            continue
        }
    }
    $argv = @('.\exp_flf2v\run_flf2v_experiment.py',
        '--wan_ckpt', $WanCkpt, '--data_dir', $Data,
        '--output_dir', $OutputRoot, '--run_name', $run.Name,
        '--num_frames_per_gop', '33', '--num_gops', '1',
        '--height', '720', '--width', '1280', '--M', '64', '--K', '16384',
        '--steps', '20', '--ddim_tail', '3', '--g_scale', '3.0',
        '--ref_codec', 'compressai', '--ref_quality', '4',
        '--seed', "$Seed", '--sequences', 'Jockey',
        '--attack', $run.Attack, '--epsilon', "$Epsilon",
        '--attack_scope', 'all', '--defense', 'none', '--input_attack_diagnostics')
    if ($run.File) { $argv += @('--attack_file', $run.File) }
    Write-Host "[RUN] $($run.Name)"
    & $Python @argv
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $metrics)) {
        throw "Incomplete codec evaluation: $($run.Name)"
    }
}

$cleanPath = Join-Path $OutputRoot "$cleanRun\Jockey\gop0\metrics.json"
$clean = Get-Content -LiteralPath $cleanPath -Raw | ConvertFrom-Json
$summary = foreach ($run in $runs) {
    $path = Join-Path $OutputRoot "$($run.Name)\Jockey\gop0\metrics.json"
    $m = Get-Content -LiteralPath $path -Raw | ConvertFrom-Json
    [pscustomobject]@{
        Run = $run.Name
        Attack = $run.Attack
        PSNR_dB = $m.PSNR_dB
        PSNR_drop_dB = [math]::Round([double]$clean.PSNR_dB - [double]$m.PSNR_dB, 3)
        LPIPS = $m.LPIPS
        Delta_LPIPS = [math]::Round([double]$m.LPIPS - [double]$clean.LPIPS, 4)
        BPP = $m.BPP
        InputPSNR = $m.input_PSNR_dB
        InputLinfPixel = $m.input_Linf_pixel
        PSNR_vs_Submitted = $m.reconstruction_vs_submitted_PSNR_dB
        CodecMSEIncrease = [double]$m.reconstruction_vs_submitted_MSE - [double]$clean.reconstruction_vs_submitted_MSE
    }
}
$summaryPath = Join-Path $OutputRoot "${tag}_summary.csv"
$summary | Export-Csv -LiteralPath $summaryPath -NoTypeInformation -Encoding UTF8
$summary | Format-Table -AutoSize
Write-Host "Saved: $summaryPath"
& $Python '.\exp_flf2v\analyze_input_attack_pilot.py' `
    --results_root $OutputRoot --tag $tag --clean-run $cleanRun --sequence Jockey --gop 0
if ($LASTEXITCODE -ne 0) { throw 'Post-hoc codebook analysis failed.' }
Write-Host 'Note: this is a VAE-balanced input attack pilot, not yet a codec-gradient attack.'
