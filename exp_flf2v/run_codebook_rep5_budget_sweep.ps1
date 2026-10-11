param(
    [string]$CondaExe = 'D:\anaconda\Scripts\conda.exe',
    [string]$Data = 'D:\yzb and lmk\dataset-720P\UVG\Jockey_720p.yuv',
    [string]$CleanCodebook = 'exp_flf2v/results_720p/codebook_sign_bitflip_rate0p02_step1/Jockey/gop0/codebook_clean.tdcm'
)
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:CUDA_VISIBLE_DEVICES = '0,1'

# Fixed robustness-test setting: protect the first SDE step and latent
# positions 2--5 with five-copy majority coding.  Only the physical channel
# disturbance budget changes between runs.
$common=@('run','--no-capture-output','-n','GVCC-5090','python',
    'exp_flf2v/run_flf2v_experiment.py','--wan_ckpt','exp_flf2v/Wan2.1-FLF2V-14B-720P',
    '--data_dir',$Data,'--output_dir','exp_flf2v/results_720p','--num_frames_per_gop','33',
    '--num_gops','1','--height','720','--width','1280','--M','64','--K','16384',
    '--steps','20','--ddim_tail','3','--g_scale','3.0','--ref_codec','compressai',
    '--ref_quality','4','--seed','42','--sequences','Jockey','--attack','none',
    '--defense','none','--reuse_codebook',$CleanCodebook,
    '--stream_attack','sign-bitflip','--stream_attack_target_steps','1',
    '--stream_attack_target_frames','3','4','--stream_attack_seed','42',
    '--stream_defense','sign-repetition','--stream_defense_steps','1',
    '--stream_defense_frames','2','3','4','5',
    '--stream_defense_attack_mode','adaptive-pairs',
    '--stream_defense_repetitions','5')

$runs=@(
    @{Budget=64;  Name='robustness_rep5_budget64'},
    # Reuse the factor-sweep result already generated for budget 128.
    @{Budget=128; Name='defense_rep5_frames2_5_budget128'},
    @{Budget=192; Name='robustness_rep5_budget192'},
    @{Budget=256; Name='robustness_rep5_budget256'}
)
foreach($run in $runs){
    $metrics="exp_flf2v/results_720p/$($run.Name)/Jockey/gop0/metrics.json"
    if(Test-Path $metrics){Write-Host "[SKIP] $($run.Name)";continue}
    Write-Host ('='*78)
    Write-Host "[RUN] five-copy robustness test, physical disturbance budget $($run.Budget)"
    Write-Host ('='*78)
    & $CondaExe @common '--stream_attack_count' ([string]$run.Budget) '--run_name' $run.Name
    if($LASTEXITCODE -ne 0){throw "Five-copy budget sweep failed: $($run.Name)"}
}

$rows=@()
foreach($run in $runs){
    $path="exp_flf2v/results_720p/$($run.Name)/Jockey/gop0/metrics.json"
    if(-not (Test-Path $path)){continue}
    $m=Get-Content $path -Raw | ConvertFrom-Json
    $a=$m.stream_attack
    $rows+=[pscustomobject]@{
        PhysicalBudget=$run.Budget
        ResidualSigns=$a.changed_symbols
        CorrectedPartialSigns=$a.corrected_logical_symbols
        ChannelBER=[math]::Round([double]$a.payload_BER,7)
        OverheadBits=$a.protection_overhead_bits
        PSNR=[math]::Round([double]$m.PSNR_dB,2)
        LPIPS=[math]::Round([double]$m.LPIPS,4)
        BPP=[math]::Round([double]$m.BPP,6)
    }
}
Write-Host ('='*78)
Write-Host 'Five-copy robustness curve: fixed protected region'
Write-Host ('='*78)
$rows | Sort-Object PhysicalBudget | Format-Table -AutoSize

$summaryPath='exp_flf2v/results_720p/rep5_budget_sweep_summary.json'
$rows | Sort-Object PhysicalBudget | ConvertTo-Json -Depth 4 | Set-Content $summaryPath -Encoding UTF8
Write-Host "Saved: $summaryPath"
