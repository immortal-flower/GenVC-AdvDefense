param(
    [string]$CondaExe = 'D:\anaconda\Scripts\conda.exe',
    [string]$Data = 'D:\yzb and lmk\dataset-720P\UVG\Jockey_720p.yuv',
    [string]$CleanRoot = 'exp_flf2v/results_720p/Jockey'
)

# Cross-GOP validation of the strongest structured threat found on GOP 0:
# invert all 64 sign bits at latent positions 3 and 4 in the first SDE step.
# GOP 0 reuses the completed result; GOP 1/2 reuse their own clean payloads.
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:CUDA_VISIBLE_DEVICES = '0,1'

$experiments = @(
    [pscustomobject]@{
        SourceGop=0
        Run='burst_step1_frames3_4_all'
        Clean=''
        Existing=$true
    },
    [pscustomobject]@{
        SourceGop=1
        Run='cross_gop1_step1_frames3_4_all'
        Clean=(Join-Path $CleanRoot 'gop1/codebook.tdcm')
        Existing=$false
    },
    [pscustomobject]@{
        SourceGop=2
        Run='cross_gop2_step1_frames3_4_all'
        Clean=(Join-Path $CleanRoot 'gop2/codebook.tdcm')
        Existing=$false
    }
)

foreach($experiment in $experiments){
    $metricsPath = "exp_flf2v/results_720p/$($experiment.Run)/Jockey/gop$($experiment.SourceGop)/metrics.json"
    if(Test-Path -LiteralPath $metricsPath){
        Write-Host "[SKIP] $($experiment.Run) already has metrics.json"
        continue
    }
    if($experiment.Existing){
        throw "Expected completed GOP 0 baseline is missing: $metricsPath"
    }
    if(-not (Test-Path -LiteralPath $experiment.Clean)){
        throw "Clean GOP $($experiment.SourceGop) codebook not found: $($experiment.Clean)"
    }
    Write-Host ('=' * 78)
    Write-Host "[RUN] Jockey GOP $($experiment.SourceGop): step 1, frames 3-4, 128 sign bits"
    Write-Host ('=' * 78)
    $experimentArgs = @(
        'run','--no-capture-output','-n','GVCC-5090','python',
        'exp_flf2v/run_flf2v_experiment.py',
        '--wan_ckpt','exp_flf2v/Wan2.1-FLF2V-14B-720P',
        '--data_dir',$Data,'--output_dir','exp_flf2v/results_720p',
        '--num_frames_per_gop','33','--num_gops','1',
        '--start_gop',[string]$experiment.SourceGop,
        '--height','720','--width','1280','--M','64','--K','16384',
        '--steps','20','--ddim_tail','3','--g_scale','3.0',
        '--ref_codec','compressai','--ref_quality','4','--seed','42',
        '--sequences','Jockey','--attack','none','--defense','none',
        '--reuse_codebook',$experiment.Clean,
        '--stream_attack','sign-bitflip',
        '--stream_attack_transport','serialized',
        '--stream_attack_count','128',
        '--stream_attack_target_steps','1',
        '--stream_attack_target_frames','3','4',
        '--stream_attack_seed','42','--run_name',$experiment.Run
    )
    & $CondaExe @experimentArgs
    if($LASTEXITCODE -ne 0){throw "Cross-GOP attack failed: $($experiment.Run)"}
}

Write-Host ('=' * 78)
Write-Host 'Cross-GOP structured sign-inversion summary'
Write-Host ('=' * 78)
$summary = foreach($experiment in $experiments){
    $metricsPath = "exp_flf2v/results_720p/$($experiment.Run)/Jockey/gop$($experiment.SourceGop)/metrics.json"
    $metrics = Get-Content -LiteralPath $metricsPath -Raw | ConvertFrom-Json
    $cleanMetricsPath = Join-Path $CleanRoot "gop$($experiment.SourceGop)/metrics.json"
    if(-not (Test-Path -LiteralPath $cleanMetricsPath)){
        throw "Clean metrics missing for GOP $($experiment.SourceGop): $cleanMetricsPath"
    }
    $cleanMetrics = Get-Content -LiteralPath $cleanMetricsPath -Raw | ConvertFrom-Json
    [pscustomobject]@{
        GOP = $experiment.SourceGop
        Bits = [int]$metrics.stream_attack.channel_flipped_bits
        PayloadBER = [double]$metrics.stream_attack.payload_BER
        CleanPSNR = [double]$cleanMetrics.PSNR_dB
        PSNR = [double]$metrics.PSNR_dB
        DeltaPSNR = [math]::Round([double]$metrics.PSNR_dB - [double]$cleanMetrics.PSNR_dB, 4)
        CleanLPIPS = [double]$cleanMetrics.LPIPS
        LPIPS = [double]$metrics.LPIPS
        DeltaLPIPS = [math]::Round([double]$metrics.LPIPS - [double]$cleanMetrics.LPIPS, 4)
        BPP = [double]$metrics.BPP
    }
}
$summary | Format-Table -AutoSize
$summary | ConvertTo-Json | Set-Content -Encoding UTF8 'exp_flf2v/results_720p/cross_gop_burst_summary.json'
