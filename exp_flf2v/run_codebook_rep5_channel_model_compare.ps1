param(
    [string]$CondaExe = 'D:\anaconda\Scripts\conda.exe',
    [string]$Data = 'D:\yzb and lmk\dataset-720P\UVG\Jockey_720p.yuv',
    [string]$CleanCodebook = 'exp_flf2v/results_720p/codebook_sign_bitflip_rate0p02_step1/Jockey/gop0/codebook_clean.tdcm'
)
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:CUDA_VISIBLE_DEVICES = '0,1'

$common=@('run','--no-capture-output','-n','GVCC-5090','python',
    'exp_flf2v/run_flf2v_experiment.py',
    '--wan_ckpt','exp_flf2v/Wan2.1-FLF2V-14B-720P',
    '--data_dir',$Data,'--output_dir','exp_flf2v/results_720p',
    '--num_frames_per_gop','33','--num_gops','1','--height','720','--width','1280',
    '--M','64','--K','16384','--steps','20','--ddim_tail','3','--g_scale','3.0',
    '--ref_codec','compressai','--ref_quality','4','--seed','42',
    '--sequences','Jockey','--attack','none','--defense','none',
    '--reuse_codebook',$CleanCodebook,
    '--stream_attack','sign-bitflip','--stream_attack_count','128',
    '--stream_attack_target_steps','1','--stream_attack_target_frames','3','4',
    '--stream_attack_transport','protected-serialized',
    '--stream_defense','sign-repetition','--stream_defense_steps','1',
    '--stream_defense_frames','2','3','4','5',
    '--stream_defense_repetitions','5')

$randomRuns=@(
    @{Seed=42;   Name='robustness_rep5_serialized_random128_seed42'},
    @{Seed=123;  Name='robustness_rep5_serialized_random128_seed123'},
    @{Seed=2026; Name='robustness_rep5_serialized_random128_seed2026'}
)
foreach($run in $randomRuns){
    $metrics="exp_flf2v/results_720p/$($run.Name)/Jockey/gop0/metrics.json"
    if(Test-Path $metrics){Write-Host "[SKIP] $($run.Name)";continue}
    Write-Host ('='*78)
    Write-Host "[RUN] dispersed channel changes, seed $($run.Seed)"
    Write-Host ('='*78)
    & $CondaExe @common '--stream_attack_seed' ([string]$run.Seed) `
        '--stream_defense_attack_mode' 'random' '--run_name' $run.Name
    if($LASTEXITCODE -ne 0){throw "Channel-model comparison failed: $($run.Name)"}
}

$rows=@()
$adaptivePath='exp_flf2v/results_720p/robustness_rep5_serialized_budget128/Jockey/gop0/metrics.json'
if(Test-Path $adaptivePath){
    $m=Get-Content $adaptivePath -Raw | ConvertFrom-Json
    $rows+=[pscustomobject]@{
        Model='concentrated'; Seed=42
        PhysicalFlips=$m.stream_attack.channel_flipped_bits
        ResidualSigns=$m.stream_attack.changed_symbols
        CorrectedTouched=$m.stream_attack.corrected_logical_symbols
        PSNR=[double]$m.PSNR_dB; LPIPS=[double]$m.LPIPS; BPP=[double]$m.BPP
    }
}
foreach($run in $randomRuns){
    $path="exp_flf2v/results_720p/$($run.Name)/Jockey/gop0/metrics.json"
    if(-not (Test-Path $path)){continue}
    $m=Get-Content $path -Raw | ConvertFrom-Json
    $rows+=[pscustomobject]@{
        Model='dispersed'; Seed=$run.Seed
        PhysicalFlips=$m.stream_attack.channel_flipped_bits
        ResidualSigns=$m.stream_attack.changed_symbols
        CorrectedTouched=$m.stream_attack.corrected_logical_symbols
        PSNR=[double]$m.PSNR_dB; LPIPS=[double]$m.LPIPS; BPP=[double]$m.BPP
    }
}
Write-Host ('='*78)
Write-Host 'Five-copy channel-model comparison'
Write-Host ('='*78)
$rows | Format-Table -AutoSize

$dispersed=@($rows | Where-Object {$_.Model -eq 'dispersed'})
if($dispersed.Count -gt 0){
    $meanResidual=($dispersed | Measure-Object ResidualSigns -Average).Average
    $meanPSNR=($dispersed | Measure-Object PSNR -Average).Average
    $meanLPIPS=($dispersed | Measure-Object LPIPS -Average).Average
    Write-Host ("Dispersed mean: residual={0:N2}, PSNR={1:N2}, LPIPS={2:N4}" -f `
        $meanResidual,$meanPSNR,$meanLPIPS)
}
$summaryPath='exp_flf2v/results_720p/rep5_channel_model_summary.json'
$rows | ConvertTo-Json -Depth 4 | Set-Content $summaryPath -Encoding UTF8
Write-Host "Saved: $summaryPath"
