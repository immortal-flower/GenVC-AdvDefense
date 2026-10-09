param(
    [ValidateSet('step','dose','frame')]
    [string]$Mode = 'step',
    [string]$CondaExe = 'D:\anaconda\Scripts\conda.exe',
    [string]$Data = 'D:\yzb and lmk\dataset-720P\UVG\Jockey_720p.yuv',
    [string]$CleanCodebook = 'exp_flf2v/results_720p/codebook_sign_bitflip_rate0p02_step1/Jockey/gop0/codebook_clean.tdcm'
)

# Serialized-codebook sensitivity experiments.  Every run loads the same clean
# .tdcm and skips encoding.  Run one mode at a time to keep turnaround bounded:
#   step  : six representative SDE steps, 49 sign errors each
#   dose  : five error counts at SDE step 1
#   frame : nine latent-time positions at SDE step 1, 16 errors each
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:CUDA_VISIBLE_DEVICES = '0,1'

if(-not (Test-Path -LiteralPath $CleanCodebook)){
    throw "Clean codebook not found: $CleanCodebook"
}

$experiments = @()
if($Mode -eq 'step'){
    foreach($step in @(1,2,4,8,12,17)){
        $experiments += [pscustomobject]@{
            Name = "sensitivity_step$($step)_sign49"
            Step = $step; Frame = $null; Count = 49
        }
    }
} elseif($Mode -eq 'dose'){
    foreach($count in @(12,25,49,98,196)){
        $experiments += [pscustomobject]@{
            Name = "sensitivity_step1_sign$($count)"
            Step = 1; Frame = $null; Count = $count
        }
    }
} else {
    foreach($frame in 0..8){
        $experiments += [pscustomobject]@{
            Name = "sensitivity_step1_frame$($frame)_sign16"
            Step = 1; Frame = $frame; Count = 16
        }
    }
}

foreach($experiment in $experiments){
    Write-Host ('=' * 78)
    Write-Host "[RUN] $($experiment.Name)"
    Write-Host ('=' * 78)
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
        '--stream_attack','sign-bitflip',
        '--stream_attack_transport','serialized',
        '--stream_attack_count',[string]$experiment.Count,
        '--stream_attack_target_steps',[string]$experiment.Step,
        '--stream_attack_seed','42','--run_name',$experiment.Name
    )
    if($null -ne $experiment.Frame){
        $experimentArgs += @('--stream_attack_target_frames',[string]$experiment.Frame)
    }
    & $CondaExe @experimentArgs
    if($LASTEXITCODE -ne 0){throw "Sensitivity run failed: $($experiment.Name)"}
}

Write-Host ('=' * 78)
Write-Host "Codebook sensitivity summary: $Mode"
Write-Host ('=' * 78)
$summary = foreach($experiment in $experiments){
    $metricsPath = Join-Path 'exp_flf2v/results_720p' "$($experiment.Name)/Jockey/gop0/metrics.json"
    $metrics = Get-Content -LiteralPath $metricsPath -Raw | ConvertFrom-Json
    [pscustomobject]@{
        Run = $experiment.Name
        Step = $experiment.Step
        Frame = $experiment.Frame
        FlippedBits = $metrics.stream_attack.changed_bits
        PayloadBER = $metrics.stream_attack.payload_BER
        PSNR = $metrics.PSNR_dB
        LPIPS = $metrics.LPIPS
    }
}
$summary | Format-Table -AutoSize
$summary | ConvertTo-Json | Set-Content -Encoding UTF8 "exp_flf2v/results_720p/sensitivity_$($Mode)_summary.json"
