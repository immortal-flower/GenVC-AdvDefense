param([string]$Python='C:\Users\Admin\.conda\envs\GVCC-5090\python.exe')
$ErrorActionPreference='Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:CUDA_VISIBLE_DEVICES='0,1'
$root='.\exp_flf2v\results_720p'
$sourceAssets=Join-Path $root 'vae_source_rgb_eps4_r36_jockey_gop0_v1_assets'
$g=Get-Content -LiteralPath (Join-Path $sourceAssets 'generation.json') -Raw | ConvertFrom-Json
$entry=@($g.assets | Where-Object name -eq 'eps4_r44_candidate')
if($g.status -ne 'completed' -or $entry.Count -ne 1){throw 'Expected completed r44 candidate'}
$initial=Join-Path $sourceAssets $entry[0].file
if((Get-FileHash -LiteralPath $initial).Hash -ne $entry[0].sha256){throw 'r44 input hash mismatch'}
$tag='trajectory_rgb_jockey_gop0_eps4_seed42_b6_p1_a0p5_h720w1280_grid128x224_m20GiB_16GiB_vaecpu_stepvjp_r44_native_v1'
Start-Transcript '.\native720_rgb_queue.log' -Append
try {
    # Complete 720p forward, one update through final 6 steps, then native acceptance.
    & '.\exp_flf2v\run_trajectory_input_pilot.ps1' `
        -Python $Python -InitialAttack $initial `
        -Objective final-rgb -Epsilon 4 -Alpha 0.5 `
        -BackwardSteps 6 -Iterations 1 -ProxyHeight 720 -ProxyWidth 1280 `
        -Gpu0Memory 20GiB -Gpu1Memory 16GiB `
        -VaeCpuActivations -StepwiseBackward -GridRandom `
        -TagSuffix r44_native_v1
    # Even if the candidate fails the proxy gate, evaluate signed directions.
    & '.\exp_flf2v\run_rgb_direction_calibration.ps1' `
        -Python $Python -OutputRoot $root -SourceTag $tag
    if($LASTEXITCODE -ne 0){throw 'Direction calibration failed'}
    Write-Host 'Completed 720p suffix-gradient candidate, paired control and signed direction audit.'
} catch {
    Write-Warning 'Queue stopped. No lower-resolution fallback was run. Resume the same command after diagnosis.'
    throw
} finally {Stop-Transcript}
