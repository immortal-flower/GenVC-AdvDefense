param(
    [string]$CondaExe = 'D:\anaconda\Scripts\conda.exe',
    [string]$Data = 'D:\yzb and lmk\dataset-720P\UVG\Jockey_720p.yuv',
    [string]$CleanCodebook = 'exp_flf2v/results_720p/codebook_sign_bitflip_rate0p02_step1/Jockey/gop0/codebook_clean.tdcm'
)

# Decode the exact clean codebook used by the sensitivity sweeps and save the
# modern metrics.json, including all 33 per-frame PSNR values.  Encoding is
# skipped, so this costs one decode only.
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:CUDA_VISIBLE_DEVICES = '0,1'

if(-not (Test-Path -LiteralPath $CleanCodebook)){
    throw "Clean codebook not found: $CleanCodebook"
}

$metricsPath = 'exp_flf2v/results_720p/sensitivity_clean_redecode/Jockey/gop0/metrics.json'
if(Test-Path -LiteralPath $metricsPath){
    $metrics = Get-Content -LiteralPath $metricsPath -Raw | ConvertFrom-Json
    if($null -ne $metrics.per_frame_PSNR_dB){
        Write-Host '[SKIP] Clean per-frame reference already exists.'
        exit 0
    }
}

$experimentArgs = @(
    'run','--no-capture-output','-n','GVCC-5090','python',
    'exp_flf2v/run_flf2v_experiment.py',
    '--wan_ckpt','exp_flf2v/Wan2.1-FLF2V-14B-720P',
    '--data_dir',$Data,'--output_dir','exp_flf2v/results_720p',
    '--num_frames_per_gop','33','--num_gops','1',
    '--height','720','--width','1280','--M','64','--K','16384',
    '--steps','20','--ddim_tail','3','--g_scale','3.0',
    '--ref_codec','compressai','--ref_quality','4','--seed','42',
    '--sequences','Jockey','--attack','none','--defense','none',
    '--reuse_codebook',$CleanCodebook,
    '--run_name','sensitivity_clean_redecode'
)
& $CondaExe @experimentArgs
if($LASTEXITCODE -ne 0){throw 'Clean reused-codebook reference failed.'}
