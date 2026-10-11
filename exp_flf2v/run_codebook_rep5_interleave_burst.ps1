param(
    [string]$CondaExe = 'D:\anaconda\Scripts\conda.exe',
    [string]$Data = 'D:\yzb and lmk\dataset-720P\UVG\Jockey_720p.yuv',
    [string]$CleanCodebook = 'exp_flf2v/results_720p/codebook_sign_bitflip_rate0p02_step1/Jockey/gop0/codebook_clean.tdcm'
)
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:CUDA_VISIBLE_DEVICES = '0,1'

# In the non-interleaved layout, redundancy bits [256,383] are concentrated
# on the first 32 logical signs of latent position 3.  The shared-seed layout
# sends the same physical interval to signs distributed across positions 2--5.
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
    '--stream_attack_seed','42','--stream_attack_transport','protected-serialized',
    '--stream_defense','sign-repetition','--stream_defense_steps','1',
    '--stream_defense_frames','2','3','4','5','--stream_defense_repetitions','5',
    '--stream_defense_attack_mode','contiguous-burst',
    '--stream_disturbance_burst_start','256')

$runs=@(
    @{Layout='sequential'; Seed=0; Name='robustness_rep5_burst128_sequential'},
    @{Layout='interleaved'; Seed=314159; Name='robustness_rep5_burst128_interleaved'}
)
foreach($run in $runs){
    $metrics="exp_flf2v/results_720p/$($run.Name)/Jockey/gop0/metrics.json"
    if(Test-Path $metrics){Write-Host "[SKIP] $($run.Name)";continue}
    Write-Host ('='*78)
    Write-Host "[RUN] $($run.Layout) redundancy layout, contiguous 128-bit channel burst"
    Write-Host ('='*78)
    & $CondaExe @common '--stream_defense_interleave_seed' ([string]$run.Seed) `
        '--run_name' $run.Name
    if($LASTEXITCODE -ne 0){throw "Interleave comparison failed: $($run.Name)"}
}

$rows=@()
foreach($run in $runs){
    $path="exp_flf2v/results_720p/$($run.Name)/Jockey/gop0/metrics.json"
    if(-not (Test-Path $path)){continue}
    $m=Get-Content $path -Raw | ConvertFrom-Json
    $a=$m.stream_attack
    $rows+=[pscustomobject]@{
        Layout=$run.Layout
        InterleaveSeed=$a.interleave_seed
        BurstBits=$a.channel_flipped_bits
        ResidualSigns=$a.changed_symbols
        CorrectedTouched=$a.corrected_logical_symbols
        PSNR=[double]$m.PSNR_dB
        LPIPS=[double]$m.LPIPS
        BPP=[double]$m.BPP
    }
}
Write-Host ('='*78)
Write-Host 'Shared-seed redundancy interleaving summary'
Write-Host ('='*78)
$rows | Format-Table -AutoSize
$summaryPath='exp_flf2v/results_720p/rep5_interleave_burst_summary.json'
$rows | ConvertTo-Json -Depth 4 | Set-Content $summaryPath -Encoding UTF8
Write-Host "Saved: $summaryPath"
