param(
    [string]$Python = 'C:\Users\Admin\.conda\envs\GVCC-5090\python.exe',
    [string]$Data = 'D:\yzb and lmk\dataset-720P\UVG\Jockey_720p.yuv',
    [string]$WanCkpt = '.\exp_flf2v\Wan2.1-FLF2V-14B-720P',
    [string]$OldAttack = '.\exp_flf2v\attack_assets\jockey_vae_temporal_pgd_eps2.npz',
    [string]$OutputRoot = '.\exp_flf2v\results_720p',
    [switch]$PlanOnly
)

$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:CUDA_VISIBLE_DEVICES = '0,1'
$tag = 'input_direction_oldvae_eps2_seed42'
$variants = @('full_positive', 'full_positive_random',
              'full_negative', 'full_negative_random',
              'half_positive', 'half_positive_random',
              'half_negative', 'half_negative_random')
$assetDir = Join-Path $OutputRoot "${tag}_assets"
$manifestPath = Join-Path $assetDir 'manifest.json'
$cleanPath = Join-Path $OutputRoot 'input_pilot_v2_eps2_seed42_steps20_clean\Jockey\gop0\metrics.json'
Write-Host "Calibration: Jockey GOP0, codec seed 42, eight full re-encodes."
Write-Host "Clean reference: $cleanPath"
foreach ($variant in $variants) { Write-Host "  ${tag}_${variant}" }
if ($PlanOnly) { Write-Host '[PLAN ONLY] No data or GPU used.'; return }

foreach ($path in @($Python, $Data, $WanCkpt, $OldAttack, $cleanPath)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf) -and
        -not (Test-Path -LiteralPath $path -PathType Container)) {
        throw "Missing prerequisite: $path"
    }
}
$dataItem = Get-Item -LiteralPath $Data
if ($dataItem.PSIsContainer -or $dataItem.Length -lt 33 * 1382400 -or
    $dataItem.Length % 1382400 -ne 0) {
    throw 'Expected 1280x720 planar YUV420 with at least 33 complete frames.'
}
if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
    & $Python '.\exp_flf2v\prepare_input_direction_calibration.py' `
        --data $Data --old_attack $OldAttack --output_dir $assetDir --seed 42
    if ($LASTEXITCODE -ne 0) { throw 'Direction preparation failed; no codec run started.' }
}
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$sourceHash = (Get-FileHash -LiteralPath $OldAttack -Algorithm SHA256).Hash
if ($manifest.source_sha256 -ne $sourceHash) {
    throw 'Old attack changed after assets were prepared; do not mix experiments.'
}
if (([IO.Path]::GetFullPath($manifest.data)) -ne ([IO.Path]::GetFullPath($Data))) {
    throw 'Prepared assets use a different input file.'
}
foreach ($variant in $variants) {
    $asset = Join-Path $assetDir "${variant}.npz"
    if (-not (Test-Path -LiteralPath $asset -PathType Leaf)) { throw "Missing asset: $asset" }
}
$clean = Get-Content -LiteralPath $cleanPath -Raw | ConvertFrom-Json
if ($null -eq $clean.reconstruction_vs_submitted_PSNR_dB) {
    throw 'Clean reference lacks input/submitted diagnostics.'
}

foreach ($variant in $variants) {
    $run = "${tag}_${variant}"
    $metrics = Join-Path $OutputRoot "$run\Jockey\gop0\metrics.json"
    if (Test-Path -LiteralPath $metrics -PathType Leaf) {
        $existing = Get-Content -LiteralPath $metrics -Raw | ConvertFrom-Json
        if ($null -ne $existing.reconstruction_vs_submitted_PSNR_dB) {
            Write-Host "[SKIP] $run"
            continue
        }
    }
    $asset = Join-Path $assetDir "${variant}.npz"
    Write-Host "[RUN] $run"
    & $Python '.\exp_flf2v\run_flf2v_experiment.py' `
        --wan_ckpt $WanCkpt --data_dir $Data --output_dir $OutputRoot `
        --run_name $run --num_frames_per_gop 33 --num_gops 1 `
        --height 720 --width 1280 --M 64 --K 16384 --steps 20 `
        --ddim_tail 3 --g_scale 3.0 --ref_codec compressai --ref_quality 4 `
        --seed 42 --sequences Jockey --attack vae-pgd --attack_file $asset `
        --epsilon 2 --attack_scope all --defense none --input_attack_diagnostics
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $metrics -PathType Leaf)) {
        throw "Incomplete codec run: $run. Re-run this same command to resume."
    }
}

$rows = foreach ($variant in $variants) {
    $metrics = Join-Path $OutputRoot "${tag}_${variant}\Jockey\gop0\metrics.json"
    $m = Get-Content -LiteralPath $metrics -Raw | ConvertFrom-Json
    [pscustomobject]@{
        Variant = $variant
        PSNR_dB = [double]$m.PSNR_dB
        PSNR_Drop_dB = [math]::Round([double]$clean.PSNR_dB - [double]$m.PSNR_dB, 3)
        LPIPS = [double]$m.LPIPS
        Delta_LPIPS = [math]::Round([double]$m.LPIPS - [double]$clean.LPIPS, 4)
        PSNR_vs_Submitted_dB = [double]$m.reconstruction_vs_submitted_PSNR_dB
        Input_MSE = [double]$m.input_MSE
        Input_Linf_Pixel = [double]$m.input_Linf_pixel
        BPP = [double]$m.BPP
    }
}
foreach ($variant in @('full_positive','full_negative','half_positive','half_negative')) {
    $candidate = $rows | Where-Object Variant -eq $variant
    $control = $rows | Where-Object Variant -eq "${variant}_random"
    if ([math]::Abs($candidate.Input_MSE - $control.Input_MSE) -gt 1e-8 -or
        [math]::Abs($candidate.Input_Linf_Pixel - $control.Input_Linf_Pixel) -gt 0.01) {
        throw "Input perturbation budget mismatch: $variant"
    }
}
$summaryPath = Join-Path $OutputRoot "${tag}_summary.csv"
$rows | Export-Csv -LiteralPath $summaryPath -NoTypeInformation -Encoding UTF8
$paired = foreach ($variant in @('full_positive','full_negative','half_positive','half_negative')) {
    $candidate = $rows | Where-Object Variant -eq $variant
    $control = $rows | Where-Object Variant -eq "${variant}_random"
    [pscustomobject]@{
        Direction = $variant
        Extra_PSNR_Drop_vs_Random_dB = [math]::Round($control.PSNR_dB - $candidate.PSNR_dB, 3)
        Extra_LPIPS_vs_Random = [math]::Round($candidate.LPIPS - $control.LPIPS, 4)
        Extra_Submitted_PSNR_Drop_vs_Random_dB = [math]::Round($control.PSNR_vs_Submitted_dB - $candidate.PSNR_vs_Submitted_dB, 3)
        Input_MSE_Difference = [math]::Round($candidate.Input_MSE - $control.Input_MSE, 10)
    }
}
$pairedPath = Join-Path $OutputRoot "${tag}_paired.csv"
$paired | Export-Csv -LiteralPath $pairedPath -NoTypeInformation -Encoding UTF8
$rows | Format-Table Variant,PSNR_dB,PSNR_Drop_dB,LPIPS,Delta_LPIPS,Input_MSE -AutoSize
$paired | Format-Table -AutoSize
Write-Host "Saved: $summaryPath"
Write-Host "Saved: $pairedPath"
