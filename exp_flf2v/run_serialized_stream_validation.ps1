param(
    [string]$CondaExe = 'D:\anaconda\Scripts\conda.exe',
    [string]$Data = 'D:\yzb and lmk\dataset-720P\UVG\Jockey_720p.yuv',
    [string]$CleanCodebook = 'exp_flf2v/results_720p/codebook_sign_bitflip_rate0p02_step1/Jockey/gop0/codebook_clean.tdcm'
)

# Validate that attacks on actual serialized .tdcm bytes reproduce the earlier
# in-memory ablation.  Both runs reuse one clean codebook, so expensive encoding
# and Top-M search are skipped; only the corrupted trajectory is decoded.
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:CUDA_VISIBLE_DEVICES = '0,1'

if(-not (Test-Path -LiteralPath $CleanCodebook)){
    throw "Clean codebook not found: $CleanCodebook"
}

$experiments = @(
    [pscustomobject]@{
        Name = 'serialized_sign_rate0p02_step1'
        Attack = 'sign-bitflip'
    },
    [pscustomobject]@{
        Name = 'serialized_index_rate0p02_step1'
        Attack = 'index-bitflip'
    }
)

foreach($experiment in $experiments){
    Write-Host ('=' * 78)
    Write-Host "[RUN] $($experiment.Name) (reuse clean codebook; skip encoding)"
    Write-Host ('=' * 78)
    $experimentArgs = @(
        'run','--no-capture-output','-n','GVCC-5090','python',
        'exp_flf2v/run_flf2v_experiment.py',
        '--wan_ckpt','exp_flf2v/Wan2.1-FLF2V-14B-720P',
        '--data_dir',$Data,
        '--output_dir','exp_flf2v/results_720p',
        '--num_frames_per_gop','33','--num_gops','1',
        '--height','720','--width','1280','--M','64','--K','16384',
        '--steps','20','--ddim_tail','3','--g_scale','3.0',
        '--ref_codec','compressai','--ref_quality','4','--seed','42',
        '--sequences','Jockey','--attack','none','--defense','none',
        '--reuse_codebook',$CleanCodebook,
        '--stream_attack',$experiment.Attack,
        '--stream_attack_transport','serialized',
        '--stream_attack_rate','0.02',
        '--stream_attack_early_steps','1',
        '--stream_attack_seed','42',
        '--run_name',$experiment.Name
    )
    & $CondaExe @experimentArgs
    if($LASTEXITCODE -ne 0){
        throw "Serialized stream validation failed: $($experiment.Name)"
    }
}

Write-Host ('=' * 78)
Write-Host 'Serialized .tdcm validation summary'
Write-Host ('=' * 78)
$summary = foreach($experiment in $experiments){
    $metricsPath = Join-Path 'exp_flf2v/results_720p' "$($experiment.Name)/Jockey/gop0/metrics.json"
    $metrics = Get-Content -LiteralPath $metricsPath -Raw | ConvertFrom-Json
    [pscustomobject]@{
        Run = $experiment.Name
        Transport = $metrics.stream_attack.transport
        ChangedBits = $metrics.stream_attack.changed_bits
        PayloadBER = $metrics.stream_attack.payload_BER
        PSNR = $metrics.PSNR_dB
        LPIPS = $metrics.LPIPS
        EncodeSeconds = $metrics.encode_seconds
    }
}
$summary | Format-Table -AutoSize
