param(
    [string]$Python = 'C:\Users\Admin\.conda\envs\GVCC-5090\python.exe',
    [string]$Data = 'D:\yzb and lmk\dataset-720P\UVG\Jockey_720p.yuv',
    [string]$WanCkpt = '.\exp_flf2v\Wan2.1-FLF2V-14B-720P',
    [string]$OutputRoot = '.\exp_flf2v\results_720p',
    [string]$InitialAttack = '.\exp_flf2v\results_720p\input_direction_oldvae_eps2_seed42_assets\full_positive.npz',
    [int]$Iterations = 4,
    [int]$RolloutSteps = 2,
    [ValidateSet('early-latent', 'final-rgb')]
    [string]$Objective = 'early-latent',
    [ValidateRange(4,20)]
    [int]$BackwardSteps = 6,
    [ValidateRange(2,255)]
    [int]$Epsilon = 2,
    [double]$Alpha = 0.5,
    [int]$ProxyHeight = 128,
    [int]$ProxyWidth = 224,
    [switch]$GridRandom,
    [switch]$PairedOnly,
    [ValidatePattern('^[1-9][0-9]*GiB$')]
    [string]$Gpu0Memory = '20GiB',
    [ValidatePattern('^[1-9][0-9]*GiB$')]
    [string]$Gpu1Memory = '20GiB',
    [switch]$VaeCpuActivations,
    [switch]$StepwiseBackward,
    [switch]$GenerateOnly,
    [switch]$PlanOnly
)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:CUDA_VISIBLE_DEVICES = '0,1'
$alphaArg = $Alpha.ToString('R', [Globalization.CultureInfo]::InvariantCulture)
$alphaTag = $alphaArg.Replace('.', 'p')
$tag = if ($Objective -eq 'final-rgb') {
    "trajectory_rgb_jockey_gop0_eps${Epsilon}_seed42_b${BackwardSteps}_p${Iterations}_a${alphaTag}"
} elseif ($Epsilon -eq 2 -and $Alpha -eq 0.5) {
    "trajectory_v1_jockey_gop0_eps2_seed42_r${RolloutSteps}_p${Iterations}"
} else {
    "trajectory_v1_jockey_gop0_eps${Epsilon}_seed42_r${RolloutSteps}_p${Iterations}_a${alphaTag}"
}
$gridArgs = @()
if ($GridRandom) {
    $gridArgs = @('--grid_random', '--control_height', '128', '--control_width', '224')
}
if ($GridRandom -or $ProxyHeight -ne 128 -or $ProxyWidth -ne 224) {
    $tag += "_h${ProxyHeight}w${ProxyWidth}"
    if ($GridRandom) { $tag += '_grid128x224' }
}
$memoryArgs = @('--gpu0_memory', $Gpu0Memory, '--gpu1_memory', $Gpu1Memory)
if ($Gpu0Memory -ne '20GiB' -or $Gpu1Memory -ne '20GiB') {
    $tag += "_m${Gpu0Memory}_${Gpu1Memory}"
}
if ($VaeCpuActivations) {
    $tag += '_vaecpu'
    $memoryArgs += '--vae_cpu_activations'
}
if ($StepwiseBackward) {
    if ($Objective -ne 'final-rgb') { throw 'StepwiseBackward requires final-rgb' }
    $tag += '_stepvjp'
    $memoryArgs += '--stepwise_backward'
}
$assets = Join-Path $OutputRoot "${tag}_assets"
foreach ($p in @($Python, $Data, $InitialAttack, (Join-Path $WanCkpt 'Wan2.1_VAE.pth'))) {
    if (-not (Test-Path -LiteralPath $p -PathType Leaf)) { throw "Missing: $p" }
}
if ($Alpha -le 0 -or [double]::IsNaN($Alpha) -or [double]::IsInfinity($Alpha)) { throw 'Invalid Alpha' }
if ($ProxyHeight -lt 32 -or $ProxyWidth -lt 32 -or $ProxyHeight % 16 -ne 0 -or $ProxyWidth % 16 -ne 0) {
    throw 'Proxy height/width must be multiples of 16 and at least 32.'
}
if ($GenerateOnly) { Write-Host 'Plan: GPU candidate generation only; no native 720p runs.' }
else { Write-Host 'Plan: candidate generation, then fresh 720p encoding/decoding if the proxy improves.' }
Write-Host "Tag: $tag. Generation resumes from iteration checkpoints; completed codec cases are hash-verified."
if ($PlanOnly) { return }
& $Python -m unittest discover -s tests -p test_trajectory_input_objective.py -v
if ($LASTEXITCODE -ne 0) { throw 'Trajectory tests failed; GPU experiment not started' }
$reportPath = Join-Path $assets 'generation.json'
$startedUtc = [datetime]::UtcNow
& $Python '.\exp_flf2v\generate_trajectory_input_pilot.py' `
    --data $Data --wan_ckpt $WanCkpt --initial_attack_file $InitialAttack `
    --output_dir $assets --iterations $Iterations --rollout_steps $RolloutSteps `
    --objective $Objective --backward_steps $BackwardSteps --epsilon $Epsilon --alpha $alphaArg `
    --height $ProxyHeight --width $ProxyWidth @gridArgs @memoryArgs
if ($LASTEXITCODE -ne 0) {
    $failedReport = $null
    if ((Test-Path -LiteralPath $reportPath -PathType Leaf) -and
        ((Get-Item -LiteralPath $reportPath).LastWriteTimeUtc -ge $startedUtc)) {
        $failedReport = Get-Content -Raw -LiteralPath $reportPath | ConvertFrom-Json
    }
    if ($failedReport.status -eq 'failed' -and $failedReport.error -match '\bCUDA out of memory\b') {
        throw [System.OutOfMemoryException]::new("Candidate CUDA OOM: $($failedReport.error)")
    }
    throw 'Candidate generation failed. Resume with the same command after fixing the error.'
}
$report = Get-Content -Raw -LiteralPath $reportPath | ConvertFrom-Json
if ($report.status -ne 'completed') { throw 'Incomplete generation' }
if ($GenerateOnly) {
    Write-Host "Report: $(Join-Path $assets 'generation.json')"
    return
}
if (-not $report.proxy_improved) {
    Write-Warning 'Hard surrogate did not improve. Stop without spending GPU time on identical/unsuccessful candidate acceptance.'
    return
}
$variants = if ($PairedOnly) { @('candidate', 'random') } else { @('initial', 'candidate', 'random') }
$rows = foreach ($variant in $variants) {
    $asset = Join-Path $assets "$variant.npz"
    $run = "${tag}_${variant}"
    $metric = Join-Path $OutputRoot "$run\Jockey\gop0\metrics.json"
    $receipt = Join-Path $OutputRoot "${run}_accepted.json"
    $signature = "$($report.fingerprint)|$((Get-FileHash -LiteralPath $asset).Hash)|codec20-M64-K16384-q4-seed42-720p"
    $skip = $false
    if (Test-Path -LiteralPath $receipt) {
        $r = Get-Content -Raw -LiteralPath $receipt | ConvertFrom-Json
        if ($r.signature -ne $signature) { throw "Acceptance configuration changed: $run" }
        if (Test-Path -LiteralPath $metric) {
            $skip = ($r.metrics_sha256 -eq (Get-FileHash -LiteralPath $metric).Hash)
        }
    }
    if (-not $skip) {
        & $Python '.\exp_flf2v\run_flf2v_experiment.py' `
            --wan_ckpt $WanCkpt --data_dir $Data --output_dir $OutputRoot --run_name $run `
            --num_frames_per_gop 33 --num_gops 1 --height 720 --width 1280 `
            --M 64 --K 16384 --steps 20 --ddim_tail 3 --g_scale 3.0 `
            --ref_codec compressai --ref_quality 4 --seed 42 --sequences Jockey `
            --attack vae-pgd --attack_file $asset --epsilon $Epsilon --attack_scope all --defense none `
            --input_attack_diagnostics | Out-Host
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $metric)) { throw "Codec acceptance failed: $run" }
        @{ signature=$signature; metrics_sha256=(Get-FileHash -LiteralPath $metric).Hash } |
            ConvertTo-Json | Set-Content -LiteralPath $receipt -Encoding UTF8
    } else { Write-Host "[SKIP] Verified: $run" }
    $m = Get-Content -Raw -LiteralPath $metric | ConvertFrom-Json
    [pscustomobject]@{
        Variant=$variant; PSNR=[double]$m.PSNR_dB; LPIPS=[double]$m.LPIPS
        PSNRvsSubmitted=$m.reconstruction_vs_submitted_PSNR_dB
        InputMSE=[double]$m.input_MSE; InputLinfPixel=[double]$m.input_Linf_pixel; BPP=$m.BPP
    }
}
$candidate = $rows | Where-Object Variant -eq 'candidate'
$random = $rows | Where-Object Variant -eq 'random'
$initial = $rows | Where-Object Variant -eq 'initial'
$mseTol = if ($GridRandom) { $candidate.InputMSE * 0.01 } else { 1e-9 }
if ([math]::Abs($candidate.InputMSE - $random.InputMSE) -gt $mseTol) { throw 'Native input MSE matching failed' }
$rows | Format-Table -AutoSize
$rows | Export-Csv (Join-Path $OutputRoot "${tag}_summary.csv") -NoTypeInformation -Encoding UTF8
Write-Host "Candidate extra PSNR drop vs random: $($random.PSNR-$candidate.PSNR) dB"
Write-Host "Candidate LPIPS increase vs random: $($candidate.LPIPS-$random.LPIPS)"
if ($initial) {
    Write-Host "Candidate extra PSNR drop vs initial: $($initial.PSNR-$candidate.PSNR) dB (input MSE may differ)"
}
Write-Host 'One GOP is a pilot, not evidence of generalizable attack success.'
