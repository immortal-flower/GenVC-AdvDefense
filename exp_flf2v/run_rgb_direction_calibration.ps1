param(
    [string]$Python = 'C:\Users\Admin\.conda\envs\GVCC-5090\python.exe',
    [string]$OutputRoot = '.\exp_flf2v\results_720p',
    [string]$SourceTag = 'trajectory_rgb_jockey_gop0_eps4_seed42_b20_p1_a1_h192w336_grid128x224_m20GiB_18GiB_vaecpu_stepvjp',
    [switch]$PlanOnly
)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
$sourceAssets = Join-Path $OutputRoot "${SourceTag}_assets"
$sourceReport = Join-Path $sourceAssets 'generation.json'
$source = Get-Content -LiteralPath $sourceReport -Raw | ConvertFrom-Json
$cfg = $source.configuration
if ($source.status -ne 'completed' -or $cfg.iterations -ne 1 -or $cfg.objective -ne 'final-rgb') {
    throw 'Expected one completed final-rgb update as the source.'
}
if (-not $cfg.grid_random) { throw 'Expected the shared-control-grid experiment.' }
$tag = "${SourceTag}_dircheck"
$assets = Join-Path $OutputRoot "${tag}_assets"
if ($PlanOnly) {
    Write-Host 'Recover the saved first PGD sign direction; no new backward pass.'
    Write-Host 'Evaluate initial, +0.5, -0.5, +1 and -1 pixel steps. No proxy-based selection.'
    Write-Host 'Verify/reuse native initial and +1 results when available; normally 3 new 720p codec runs.'
    return
}
$env:CUDA_VISIBLE_DEVICES = '0,1'
Start-Transcript '.\trajectory_rgb_direction_calibration.log' -Append | Out-Host
try {
    & $Python -m unittest discover -s tests -p test_rgb_direction_audit.py -v
    if ($LASTEXITCODE -ne 0) { throw 'Direction recovery tests failed.' }
    $extra = @()
    if ($cfg.vae_cpu_activations) { $extra += '--vae_cpu_activations' }
    if ($cfg.stepwise_backward) { $extra += '--stepwise_backward' }
    & $Python '.\exp_flf2v\generate_trajectory_input_pilot.py' `
        --data $cfg.data --wan_ckpt $cfg.wan_ckpt `
        --initial_attack_file $cfg.initial_attack_file --output_dir $assets `
        --objective final-rgb --backward_steps $cfg.backward_steps --iterations 1 `
        --epsilon $cfg.epsilon --alpha $cfg.alpha --seed $cfg.seed --pool_size $cfg.pool_size `
        --height $cfg.height --width $cfg.width `
        --control_height $cfg.control_height --control_width $cfg.control_width --grid_random `
        --gpu0_memory $cfg.gpu0_memory --gpu1_memory $cfg.gpu1_memory `
        --direction_audit_from $sourceAssets @extra | Out-Host
    if ($LASTEXITCODE -ne 0) { throw 'Direction replay failed; no native codec runs started.' }
    $report = Get-Content -LiteralPath (Join-Path $assets 'generation.json') -Raw | ConvertFrom-Json
    if ($report.status -ne 'completed' -or -not $report.direction_reproduces_candidate) {
        throw 'Direction replay verification did not complete.'
    }

    function Find-VerifiedReuse([string]$OtherTag, [string]$Variant, [string]$AssetHash) {
        $gPath = Join-Path $OutputRoot "${OtherTag}_assets\generation.json"
        $mPath = Join-Path $OutputRoot "${OtherTag}_${Variant}\Jockey\gop0\metrics.json"
        $rPath = Join-Path $OutputRoot "${OtherTag}_${Variant}_accepted.json"
        if (-not ((Test-Path -LiteralPath $gPath) -and (Test-Path -LiteralPath $mPath) -and (Test-Path -LiteralPath $rPath))) { return $null }
        $g = Get-Content -LiteralPath $gPath -Raw | ConvertFrom-Json
        $r = Get-Content -LiteralPath $rPath -Raw | ConvertFrom-Json
        $sha = $g.asset_sha256."${Variant}.npz"
        if ($g.status -ne 'completed' -or $sha -ne $AssetHash -or
            $g.configuration.data_sha256 -ne $cfg.data_sha256 -or
            $g.configuration.wan_ckpt -ne $cfg.wan_ckpt -or
            $g.configuration.seed -ne 42 -or $g.configuration.epsilon -ne $cfg.epsilon) { return $null }
        $oldAsset = Join-Path $OutputRoot "${OtherTag}_assets\${Variant}.npz"
        if ((Get-FileHash -LiteralPath $oldAsset).Hash -ne $AssetHash) { return $null }
        $expected = "$($g.fingerprint)|$AssetHash|codec20-M64-K16384-q4-seed42-720p"
        if ($r.signature -ne $expected -or $r.metrics_sha256 -ne (Get-FileHash -LiteralPath $mPath).Hash) { return $null }
        return $mPath
    }

    if ($cfg.seed -ne 42) { throw 'Native receipt format currently requires model seed 42.' }
    $rows = foreach ($case in $report.rows) {
        $name = $case.variant
        $asset = Join-Path $assets "${name}.npz"
        $sha = (Get-FileHash -LiteralPath $asset).Hash
        if ($sha -ne $report.asset_sha256."${name}.npz") { throw "Asset hash mismatch: $name" }
        $run = "${tag}_${name}"
        $metric = Join-Path $OutputRoot "$run\Jockey\gop0\metrics.json"
        $receipt = Join-Path $OutputRoot "${run}_accepted.json"
        $signature = "$($report.fingerprint)|$sha|codec20-M64-K16384-q4-seed42-720p"
        $skip = $false
        if (Test-Path -LiteralPath $receipt) {
            $r = Get-Content -LiteralPath $receipt -Raw | ConvertFrom-Json
            if ($r.signature -ne $signature) { throw "Native configuration changed: $name" }
            if (Test-Path -LiteralPath $metric) {
                $skip = ($r.metrics_sha256 -eq (Get-FileHash -LiteralPath $metric).Hash)
            }
        }
        if (-not $skip) {
            $reuse = $null
            if ($name -eq 'positive_one' -and $report.source_accepted_step_pixel -eq 1) {
                $reuse = Find-VerifiedReuse $SourceTag 'candidate' $sha
            } elseif ($name -eq 'initial') {
                $reuse = Find-VerifiedReuse $SourceTag 'initial' $sha
                if (-not $reuse) {
                    $reuse = Find-VerifiedReuse 'trajectory_rgb_jockey_gop0_eps4_seed42_b6_p1_a1' 'initial' $sha
                }
            }
            if ($reuse) { $metric = $reuse; $skip = $true }
        }
        if ($skip) {
            Write-Host "[REUSE] $name : $metric"
        } else {
            Write-Host "[NATIVE] $name"
            & $Python '.\exp_flf2v\run_flf2v_experiment.py' `
                --wan_ckpt $cfg.wan_ckpt --data_dir $cfg.data --output_dir $OutputRoot --run_name $run `
                --num_frames_per_gop 33 --num_gops 1 --height 720 --width 1280 `
                --M 64 --K 16384 --steps 20 --ddim_tail 3 --g_scale 3.0 `
                --ref_codec compressai --ref_quality 4 --seed 42 --sequences Jockey `
                --attack vae-pgd --attack_file $asset --epsilon $cfg.epsilon `
                --attack_scope all --defense none --input_attack_diagnostics | Out-Host
            if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $metric)) { throw "Native run failed: $name" }
            @{ signature=$signature; metrics_sha256=(Get-FileHash -LiteralPath $metric).Hash } |
                ConvertTo-Json | Set-Content -LiteralPath $receipt -Encoding UTF8
        }
        $m = Get-Content -LiteralPath $metric -Raw | ConvertFrom-Json
        if ([math]::Abs([double]$m.input_MSE - [double]$case.input_MSE) -gt 1e-9 -or
            [double]$m.input_Linf_pixel -gt [double]$cfg.epsilon + 0.001) { throw "Native input verification failed: $name" }
        [pscustomobject]@{
            Variant=$name; StepPixel=$case.step_pixel; ProxyPSNR=[double]$case.proxy_rgb_psnr_db
            NativePSNR=[double]$m.PSNR_dB; LPIPS=[double]$m.LPIPS
            PSNRvsSubmitted=$m.reconstruction_vs_submitted_PSNR_dB
            InputMSE=[double]$m.input_MSE; InputLinfPixel=[double]$m.input_Linf_pixel
            BPP=$m.BPP; MetricsPath=$metric
        }
    }
    $initial = $rows | Where-Object Variant -eq 'initial'
    foreach ($row in $rows) {
        $row | Add-Member -NotePropertyName ProxyDrop -NotePropertyValue ($initial.ProxyPSNR - $row.ProxyPSNR)
        $row | Add-Member -NotePropertyName NativeDrop -NotePropertyValue ($initial.NativePSNR - $row.NativePSNR)
        $row | Add-Member -NotePropertyName DeltaLPIPS -NotePropertyValue ($row.LPIPS - $initial.LPIPS)
    }
    $out = Join-Path $OutputRoot "${tag}_summary.csv"
    $rows | Export-Csv -LiteralPath $out -NoTypeInformation -Encoding UTF8
    $rows | Format-Table Variant,StepPixel,ProxyDrop,NativeDrop,DeltaLPIPS,InputMSE -AutoSize | Out-Host
    Write-Host "[DONE] $out"
    Write-Host 'Positive drops mean worse reconstruction relative to the old-VAE initialization, NOT clean video.'
    Write-Host 'Signed pairs are not energy matched; inspect input MSE and quantization before interpreting direction.'
} finally {
    Stop-Transcript | Out-Host
}
