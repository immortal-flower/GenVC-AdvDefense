param(
    [string]$CondaExe = 'D:\anaconda\Scripts\conda.exe',
    [string]$Data = 'D:\yzb and lmk\dataset-720P\UVG\Jockey_720p.yuv',
    [string]$CleanCodebook = 'exp_flf2v/results_720p/codebook_sign_bitflip_rate0p02_step1/Jockey/gop0/codebook_clean.tdcm'
)
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:CUDA_VISIBLE_DEVICES = '0,1'
$runName='robustness_rep5_serialized_budget128'
$metrics="exp_flf2v/results_720p/$runName/Jockey/gop0/metrics.json"

if(-not (Test-Path $metrics)){
    $argsList=@('run','--no-capture-output','-n','GVCC-5090','python',
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
        '--stream_attack_seed','42','--stream_attack_transport','protected-serialized',
        '--stream_defense','sign-repetition','--stream_defense_steps','1',
        '--stream_defense_frames','2','3','4','5',
        '--stream_defense_attack_mode','adaptive-pairs',
        '--stream_defense_repetitions','5','--run_name',$runName)
    & $CondaExe @argsList
    if($LASTEXITCODE -ne 0){throw 'Serialized five-copy robustness validation failed.'}
} else {
    Write-Host "[SKIP] $runName"
}

$m=Get-Content $metrics -Raw | ConvertFrom-Json
$a=$m.stream_attack
$gopDir="exp_flf2v/results_720p/$runName/Jockey/gop0"
$logicalPath='exp_flf2v/results_720p/defense_rep5_frames2_5_budget128/Jockey/gop0/metrics.json'
$logical=$null
if(Test-Path $logicalPath){$logical=Get-Content $logicalPath -Raw | ConvertFrom-Json}
$row=[pscustomobject]@{
    Transport=$a.transport
    PhysicalFlips=$a.channel_flipped_bits
    ResidualSigns=$a.changed_symbols
    RedundancyBits=$a.protection_overhead_bits
    CleanTDCMBytes=(Get-Item "$gopDir/codebook_clean.tdcm").Length
    ProtectedFileBytes=(Get-Item "$gopDir/codebook_protected_disturbed.tdcmr").Length
    RecoveredTDCMBytes=(Get-Item "$gopDir/codebook.tdcm").Length
    PSNR=[math]::Round([double]$m.PSNR_dB,2)
    LPIPS=[math]::Round([double]$m.LPIPS,4)
    LogicalPSNR=if($logical){[math]::Round([double]$logical.PSNR_dB,2)}else{$null}
    LogicalLPIPS=if($logical){[math]::Round([double]$logical.LPIPS,4)}else{$null}
}
Write-Host ('='*78)
Write-Host 'Serialized majority-protection validation'
Write-Host ('='*78)
$row | Format-List
