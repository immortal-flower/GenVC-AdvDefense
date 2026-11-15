param(
    [string]$LogPath = '.\trajectory_rgb_full_chain_overnight.log',
    [switch]$IncludeFullBackward,
    [switch]$IncludeControls,
    [switch]$PlanOnly
)

$ErrorActionPreference = 'Stop'
$runner = Join-Path $PSScriptRoot 'run_trajectory_input_pilot.ps1'
$common = @{
    Objective = 'final-rgb'
    Epsilon = 4
    Alpha = 1
    Iterations = 1
    Gpu0Memory = '23GiB'
    Gpu1Memory = '15GiB'
    VaeCpuActivations = $true
    GridRandom = $true
    PairedOnly = $true
}
if ($PlanOnly) {
    Write-Host 'Jockey GOP0: 192x336 RGB candidate, matched grid random, then two full 720p codec runs.'
    Write-Host '6-step backward: one retry at 160x288 only if candidate generation reports CUDA OOM.'
    if ($IncludeFullBackward) {
        Write-Host '20-step backward follows at the same resolved size; no geometry fallback for this stage.'
    }
    if ($IncludeControls) {
        Write-Host 'Native controls: clean, plus one shared old-VAE initialization (verified reuse when available).'
    }
    return
}

function Show-StageResult([int]$Horizon, [string]$Geometry) {
    $tag = "trajectory_rgb_jockey_gop0_eps4_seed42_b${Horizon}_p1_a1_h" + $Geometry.Replace('x','w')
    $tag += '_grid128x224_m23GiB_15GiB_vaecpu'
    $root = '.\exp_flf2v\results_720p'
    $report = Join-Path $root "${tag}_assets\generation.json"
    $summary = Join-Path $root "${tag}_summary.csv"
    if (Test-Path -LiteralPath $summary -PathType Leaf) {
        Write-Host "[OVERNIGHT DONE] b$Horizon $Geometry completed two full 720p codec runs: $summary"
    } else {
        $gen = Get-Content -LiteralPath $report -Raw | ConvertFrom-Json
        if ($gen.status -ne 'completed' -or $gen.proxy_improved) {
            throw "Expected 720p summary missing: $summary"
        }
        Write-Warning "[OVERNIGHT NO PROXY GAIN] b$Horizon $Geometry produced no accepted update; no 720p run. $report"
    }
}

function Show-NativeControls {
    $root = '.\exp_flf2v\results_720p'
    $python = 'C:\Users\Admin\.conda\envs\GVCC-5090\python.exe'
    $data = 'D:\yzb and lmk\dataset-720P\UVG\Jockey_720p.yuv'
    $ckpt = '.\exp_flf2v\Wan2.1-FLF2V-14B-720P'
    $init = Join-Path $root 'input_direction_oldvae_eps2_seed42_assets\full_positive.npz'
    $dataHash = (Get-FileHash -LiteralPath $data).Hash
    $initHash = (Get-FileHash -LiteralPath $init).Hash
    $shared = "native-control-v1|$dataHash|$((Resolve-Path -LiteralPath $ckpt).Path)|720p33-S20-M64-K16384-q4-seed42"
    $env:CUDA_VISIBLE_DEVICES = '0,1'
    $runs = @(
        @{ Label = 'clean'; Name = 'trajectory_rgb_control_jockey_gop0_clean'; Extra = @('--attack', 'none') },
        @{ Label = 'old-vae-init'; Name = 'trajectory_rgb_control_jockey_gop0_oldinit';
           Extra = @('--attack', 'vae-pgd', '--attack_file', $init, '--epsilon', '4') }
    )
    $rows = foreach ($control in $runs) {
        $run = $control.Name
        $extra = $control.Extra
        $metric = Join-Path $root "$run\Jockey\gop0\metrics.json"
        $receipt = Join-Path $root "${run}_accepted.json"
        $key = if ($control.Label -eq 'clean') { "$shared|clean" } else { "$shared|old-init|$initHash" }
        $reuse = $false
        if ((Test-Path -LiteralPath $metric -PathType Leaf) -and (Test-Path -LiteralPath $receipt -PathType Leaf)) {
            $r = Get-Content -Raw -LiteralPath $receipt | ConvertFrom-Json
            $reuse = ($r.signature -eq $key -and $r.metrics_sha256 -eq (Get-FileHash -LiteralPath $metric).Hash)
        }
        if (-not $reuse -and $control.Label -eq 'old-vae-init') {
            $oldTag = 'trajectory_rgb_jockey_gop0_eps4_seed42_b6_p1_a1'
            $oldAssets = Join-Path $root "${oldTag}_assets"
            $oldReport = Join-Path $oldAssets 'generation.json'
            $oldMetric = Join-Path $root "${oldTag}_initial\Jockey\gop0\metrics.json"
            $oldReceipt = Join-Path $root "${oldTag}_initial_accepted.json"
            if ((Test-Path -LiteralPath $oldReport -PathType Leaf) -and
                (Test-Path -LiteralPath $oldMetric -PathType Leaf) -and
                (Test-Path -LiteralPath $oldReceipt -PathType Leaf)) {
                $g = Get-Content -Raw -LiteralPath $oldReport | ConvertFrom-Json
                $r = Get-Content -Raw -LiteralPath $oldReceipt | ConvertFrom-Json
                $initAssetHash = $g.asset_sha256.'initial.npz'
                $expected = "$($g.fingerprint)|$initAssetHash|codec20-M64-K16384-q4-seed42-720p"
                if ($g.status -eq 'completed' -and $g.configuration.data_sha256 -eq $dataHash -and
                    $g.configuration.init_sha256 -eq $initHash -and $g.configuration.wan_ckpt -eq $ckpt -and
                    $g.configuration.seed -eq 42 -and $g.configuration.epsilon -eq 4 -and $initAssetHash -and
                    $r.signature -eq $expected -and $r.metrics_sha256 -eq (Get-FileHash -LiteralPath $oldMetric).Hash) {
                    $metric = $oldMetric
                    $reuse = $true
                }
            }
        }
        if (-not $reuse) {
            Write-Host "[CONTROL START] $($control.Label)"
            & $python '.\exp_flf2v\run_flf2v_experiment.py' `
                --wan_ckpt $ckpt --data_dir $data --output_dir $root --run_name $run `
                --num_frames_per_gop 33 --num_gops 1 --height 720 --width 1280 `
                --M 64 --K 16384 --steps 20 --ddim_tail 3 --g_scale 3.0 `
                --ref_codec compressai --ref_quality 4 --seed 42 --sequences Jockey `
                --attack_scope all --defense none --input_attack_diagnostics @extra | Out-Host
            if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $metric -PathType Leaf)) {
                throw "Native control failed: $($control.Label)"
            }
            @{ signature=$key; metrics_sha256=(Get-FileHash -LiteralPath $metric).Hash } |
                ConvertTo-Json | Set-Content -LiteralPath $receipt -Encoding UTF8
        } else {
            Write-Host "[CONTROL REUSE] $($control.Label): $metric"
        }
        $m = Get-Content -Raw -LiteralPath $metric | ConvertFrom-Json
        [pscustomobject]@{
            Control = $control.Label; PSNR = $m.PSNR_dB; LPIPS = $m.LPIPS
            InputMSE = $m.input_MSE; BPP = $m.BPP; MetricsPath = $metric
        }
    }
    $rows | Format-Table Control, PSNR, LPIPS, InputMSE, BPP -AutoSize | Out-Host
    $out = Join-Path $root 'trajectory_rgb_overnight_native_controls.csv'
    $rows | Export-Csv -LiteralPath $out -NoTypeInformation -Encoding UTF8
    Write-Host "[OVERNIGHT CONTROLS DONE] $out"
}

Push-Location (Split-Path -Parent $PSScriptRoot)
$transcribing = $false
try {
    Start-Transcript -Path $LogPath -Append | Out-Host
    $transcribing = $true
    $h = 192
    $w = 336
    Write-Host '[OVERNIGHT START] b6 192x336'
    try {
        & $runner @common -BackwardSteps 6 -ProxyHeight $h -ProxyWidth $w
    } catch [System.OutOfMemoryException] {
        # Only switch geometry for the explicitly classified generation failure.
        if ($_.Exception.Message -notlike 'Candidate CUDA OOM:*') { throw }
        Write-Warning '192x336 candidate generation ran out of CUDA memory. Retrying once at 160x288.'
        $h = 160
        $w = 288
        & $runner @common -BackwardSteps 6 -ProxyHeight $h -ProxyWidth $w
    }
    $size = "${h}x${w}"
    Show-StageResult 6 $size

    if ($IncludeFullBackward) {
        Write-Host "[OVERNIGHT START] b20 $size"
        try {
            & $runner @common -BackwardSteps 20 -ProxyHeight $h -ProxyWidth $w
            Show-StageResult 20 $size
        } catch [System.OutOfMemoryException] {
            if ($_.Exception.Message -notlike 'Candidate CUDA OOM:*') { throw }
            Write-Warning "[OVERNIGHT FULL BACKWARD OOM] b20 $size; b6 results retained. $($_.Exception.Message)"
        }
    }
    if ($IncludeControls) { Show-NativeControls }
} finally {
    try {
        if ($transcribing) { Stop-Transcript | Out-Host }
    } finally {
        Pop-Location
    }
}
