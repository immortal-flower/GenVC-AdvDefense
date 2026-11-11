param(
    [string]$Python = 'C:\Users\Admin\.conda\envs\GVCC-5090\python.exe',
    [string]$Data = 'D:\yzb and lmk\dataset-720P\UVG\Jockey_720p.yuv',
    [string]$WanCkpt = '.\exp_flf2v\Wan2.1-FLF2V-14B-720P',
    [string]$OutputRoot = '.\exp_flf2v\results_720p',
    [string]$CleanRun = 'input_pilot_v2_eps2_seed42_steps20_clean',
    [int]$Seed = 42,
    [int]$AttackSeed = 42,
    [int]$ProxySteps = 4,
    [double]$NoiseCoeff = 0.3062,
    [ValidateSet('native', 'lowres')]
    [string]$ProxyGeometry = 'native',
    [switch]$AnalyzeOnly,
    [switch]$PlanOnly
)

$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:CUDA_VISIBLE_DEVICES = '0,1'
$cleanDir = Join-Path $OutputRoot "$CleanRun\Jockey\gop0"
$vae = Join-Path $WanCkpt 'Wan2.1_VAE.pth'
$noiseTag = $NoiseCoeff.ToString('0.####', [Globalization.CultureInfo]::InvariantCulture).Replace('.', 'p')
$prefix = "input_codebook_proxy_step1pos3_eps2_seed${Seed}_aseed${AttackSeed}_p${ProxySteps}_n${noiseTag}_g${ProxyGeometry}"
$legacyPrefix = "input_codebook_proxy_step1pos3_eps2_seed${Seed}_p${ProxySteps}_n${noiseTag}"
if ($AnalyzeOnly -and $AttackSeed -eq 42 -and
    -not (Test-Path -LiteralPath (Join-Path $OutputRoot "${prefix}_candidate\Jockey\gop0\metrics.json") -PathType Leaf) -and
    (Test-Path -LiteralPath (Join-Path $OutputRoot "${legacyPrefix}_candidate\Jockey\gop0\metrics.json") -PathType Leaf)) {
    $prefix = $legacyPrefix
}
$asset = ".\exp_flf2v\attack_assets\${prefix}.npz"
$randomAsset = $asset.Replace('.npz', '_matched.npz')
$runs = @(
    [pscustomobject]@{ Label = 'random'; File = $randomAsset },
    [pscustomobject]@{ Label = 'candidate'; File = $asset }
)
$required = @($Python, $Data, $WanCkpt, $vae,
              (Join-Path $cleanDir 'metrics.json'),
              (Join-Path $cleanDir 'codebook.tdcm'),
              (Join-Path $cleanDir 'input_attack_trace.npz'))
foreach ($path in $required) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf) -and
        $path -ne $WanCkpt) { throw "Missing prerequisite: $path" }
}
if (-not (Test-Path -LiteralPath $WanCkpt -PathType Container)) {
    throw "Missing Wan checkpoint directory: $WanCkpt"
}
if ($ProxySteps -lt 1 -or $NoiseCoeff -le 0) { throw 'Invalid proxy steps or noise coefficient' }
if ($Seed -ne 42) { throw 'This pilot requires model/clean-stream seed 42; use -Seed 42' }
Write-Host 'Experimental objective: fixed-clean-trajectory, reduced-resolution codebook proxy.'
Write-Host 'Both candidate and matched random control undergo fresh normal GVCC encoding.'
if ($PlanOnly) {
    Write-Host "Clean: $cleanDir"
    Write-Host "Candidate asset: $asset"
    Write-Host "Run names: ${prefix}_random, ${prefix}_candidate"
    return
}

if (-not $AnalyzeOnly) {
& $Python '.\exp_flf2v\validate_codebook_proxy.py' `
    --bitstream (Join-Path $cleanDir 'codebook.tdcm') `
    --trace (Join-Path $cleanDir 'input_attack_trace.npz') `
    --step 1 --latent_position 3 `
    --output (Join-Path $OutputRoot "${prefix}_preflight.json")
if ($LASTEXITCODE -ne 0) { throw 'Clean trace/bitstream validation failed' }

if (-not ((Test-Path -LiteralPath $asset -PathType Leaf) -and
          (Test-Path -LiteralPath $randomAsset -PathType Leaf))) {
    & $Python '.\exp_flf2v\generate_codebook_input_pilot.py' `
        --data $Data --vae $vae `
        --bitstream (Join-Path $cleanDir 'codebook.tdcm') `
        --trace (Join-Path $cleanDir 'input_attack_trace.npz') `
        --output $asset --steps $ProxySteps `
        --noise_coeff $NoiseCoeff --seed $AttackSeed --proxy_geometry $ProxyGeometry
    if ($LASTEXITCODE -ne 0) { throw 'Candidate generation failed; no codec run started' }
}
if (-not ((Test-Path -LiteralPath $asset -PathType Leaf) -and
          (Test-Path -LiteralPath $randomAsset -PathType Leaf) -and
          (Test-Path -LiteralPath $asset.Replace('.npz', '.json') -PathType Leaf))) {
    throw 'Candidate or matched control asset is incomplete; no codec run started'
}

foreach ($run in $runs) {
    $name = "${prefix}_$($run.Label)"
    $metrics = Join-Path $OutputRoot "$name\Jockey\gop0\metrics.json"
    if (Test-Path -LiteralPath $metrics -PathType Leaf) {
        Write-Host "[SKIP] $name has metrics.json"
        continue
    }
    & $Python '.\exp_flf2v\run_flf2v_experiment.py' `
        --wan_ckpt $WanCkpt --data_dir $Data `
        --output_dir $OutputRoot --run_name $name `
        --num_frames_per_gop 33 --num_gops 1 `
        --height 720 --width 1280 --M 64 --K 16384 `
        --steps 20 --ddim_tail 3 --g_scale 3.0 `
        --ref_codec compressai --ref_quality 4 `
        --seed $Seed --sequences Jockey `
        --attack vae-pgd --attack_file $run.File `
        --input_attack_diagnostics `
        --epsilon 2 --attack_scope all --defense none
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $metrics -PathType Leaf)) {
        throw "Incomplete full re-encoding: $name"
    }
}
}

$clean = Get-Content -LiteralPath (Join-Path $cleanDir 'metrics.json') -Raw | ConvertFrom-Json
$rows = foreach ($run in $runs) {
    $name = "${prefix}_$($run.Label)"
    $path = Join-Path $OutputRoot "$name\Jockey\gop0\metrics.json"
    $m = Get-Content -LiteralPath $path -Raw | ConvertFrom-Json
    [pscustomobject]@{
        Run = $name
        PSNR = $m.PSNR_dB
        PSNRDrop = [math]::Round([double]$clean.PSNR_dB - [double]$m.PSNR_dB, 3)
        LPIPS = $m.LPIPS
        DeltaLPIPS = [math]::Round([double]$m.LPIPS - [double]$clean.LPIPS, 4)
        PSNRvsSubmitted = $m.reconstruction_vs_submitted_PSNR_dB
        InputMSE = $m.input_MSE
        InputLinfPixel = $m.input_Linf_pixel
        BPP = $m.BPP
    }
}
$rows | Format-Table -AutoSize
$rows | Export-Csv (Join-Path $OutputRoot "${prefix}_summary.csv") -NoTypeInformation -Encoding UTF8
& $Python '.\exp_flf2v\analyze_codebook_input_pilot.py' `
    --results_root $OutputRoot --clean_run $CleanRun --prefix $prefix `
    --output_prefix (Join-Path $OutputRoot "${prefix}_analysis")
if ($LASTEXITCODE -ne 0) { throw 'Post-hoc pilot analysis failed' }
