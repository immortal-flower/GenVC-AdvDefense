param(
    [string]$CondaExe = 'D:\anaconda\Scripts\conda.exe',
    [string]$Data = 'D:\yzb and lmk\dataset-720P\UVG\Jockey_720p.yuv',
    [string]$CleanCodebook = 'exp_flf2v/results_720p/codebook_sign_bitflip_rate0p02_step1/Jockey/gop0/codebook_clean.tdcm'
)
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:CUDA_VISIBLE_DEVICES = '0,1'

$common=@('run','--no-capture-output','-n','GVCC-5090','python',
    'exp_flf2v/run_flf2v_experiment.py','--wan_ckpt','exp_flf2v/Wan2.1-FLF2V-14B-720P',
    '--data_dir',$Data,'--output_dir','exp_flf2v/results_720p','--num_frames_per_gop','33',
    '--num_gops','1','--height','720','--width','1280','--M','64','--K','16384',
    '--steps','20','--ddim_tail','3','--g_scale','3.0','--ref_codec','compressai',
    '--ref_quality','4','--seed','42','--sequences','Jockey','--attack','none',
    '--defense','none','--reuse_codebook',$CleanCodebook,
    '--stream_attack','sign-bitflip','--stream_attack_target_steps','1',
    '--stream_attack_target_frames','3','4','--stream_attack_seed','42',
    '--stream_defense','sign-repetition3','--stream_defense_steps','1',
    '--stream_defense_frames','2','3','4','5',
    '--stream_defense_attack_mode','adaptive-pairs')

$runs=@(
    @{Name='defense_rep3_frames2_5_adaptive128'; Count=128},
    @{Name='defense_rep3_frames2_5_adaptive256'; Count=256}
)
foreach($run in $runs){
    $metrics="exp_flf2v/results_720p/$($run.Name)/Jockey/gop0/metrics.json"
    if(Test-Path $metrics){Write-Host "[SKIP] $($run.Name)";continue}
    Write-Host ('='*78)
    Write-Host "[RUN] $($run.Name): $($run.Count) physical flips"
    Write-Host ('='*78)
    & $CondaExe @common '--stream_attack_count' ([string]$run.Count) '--run_name' $run.Name
    if($LASTEXITCODE -ne 0){throw "Targeted repetition defense failed: $($run.Name)"}
}

Write-Host ('='*78)
Write-Host 'Targeted unequal-protection summary'
Write-Host ('='*78)
$rows=@()
foreach($run in $runs){
    $path="exp_flf2v/results_720p/$($run.Name)/Jockey/gop0/metrics.json"
    if(-not (Test-Path $path)){continue}
    $m=Get-Content $path -Raw | ConvertFrom-Json
    $a=$m.stream_attack
    $rows+=[pscustomobject]@{
        Run=$run.Name
        PhysicalFlips=$a.channel_flipped_bits
        ResidualSigns=$a.changed_symbols
        CorrectedSigns=$a.corrected_logical_symbols
        OverheadBits=$a.protection_overhead_bits
        OverheadPct=[math]::Round(100.0*[double]$a.protection_overhead_bits/[double]$a.total_payload_bits,3)
        ProtectedBER=$a.payload_BER
        PSNR=[math]::Round([double]$m.PSNR_dB,2)
        LPIPS=[math]::Round([double]$m.LPIPS,4)
        BPP=[math]::Round([double]$m.BPP,6)
    }
}
$rows | Format-Table -AutoSize
