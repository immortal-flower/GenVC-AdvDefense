param(
    [ValidateSet('frames','steps')]
    [string]$Mode = 'frames',
    [string]$CondaExe = 'D:\anaconda\Scripts\conda.exe',
    [string]$Data = 'D:\yzb and lmk\dataset-720P\UVG\Jockey_720p.yuv',
    [string]$CleanCodebook = 'exp_flf2v/results_720p/codebook_sign_bitflip_rate0p02_step1/Jockey/gop0/codebook_clean.tdcm'
)

# Structured bitstream robustness test.  Each experiment flips every sign bit
# in the selected step/frame rectangle.  Because M=64, the exact count is
# 64 * (#steps) * (#latent frames); no random subset remains when count equals
# the eligible capacity.  This isolates burst location and extent.
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:CUDA_VISIBLE_DEVICES = '0,1'

if(-not (Test-Path -LiteralPath $CleanCodebook)){
    throw "Clean codebook not found: $CleanCodebook"
}

if($Mode -eq 'frames'){
    $experiments = @(
        # Reuse the completed 64-bit dose endpoint; flipping all 64 signs is
        # identical regardless of random or ranked selection.
        [pscustomobject]@{Name='impact_frame3_ranked64'; Steps=@(1); Frames=@(3)},
        [pscustomobject]@{Name='burst_step1_frames3_4_all'; Steps=@(1); Frames=@(3,4)},
        [pscustomobject]@{Name='burst_step1_frames2_4_all'; Steps=@(1); Frames=@(2,3,4)},
        [pscustomobject]@{Name='burst_step1_frames2_5_all'; Steps=@(1); Frames=@(2,3,4,5)},
        [pscustomobject]@{Name='burst_step1_frames1_5_all'; Steps=@(1); Frames=@(1,2,3,4,5)},
        [pscustomobject]@{Name='burst_step1_all_frames'; Steps=@(1); Frames=@(0,1,2,3,4,5,6,7,8)}
    )
} else {
    $experiments = @(
        [pscustomobject]@{Name='impact_frame3_ranked64'; Steps=@(1); Frames=@(3)},
        [pscustomobject]@{Name='burst_frame3_steps1_2_all'; Steps=@(1,2); Frames=@(3)},
        [pscustomobject]@{Name='burst_frame3_steps1_4_all'; Steps=@(1,2,3,4); Frames=@(3)},
        [pscustomobject]@{Name='burst_frame3_steps1_8_all'; Steps=@(1,2,3,4,5,6,7,8); Frames=@(3)}
    )
}

foreach($experiment in $experiments){
    $metricsPath = Join-Path 'exp_flf2v/results_720p' "$($experiment.Name)/Jockey/gop0/metrics.json"
    if(Test-Path -LiteralPath $metricsPath){
        Write-Host "[SKIP] $($experiment.Name) already has metrics.json"
        continue
    }
    $count = 64 * $experiment.Steps.Count * $experiment.Frames.Count
    Write-Host ('=' * 78)
    Write-Host "[RUN] $($experiment.Name), full sign inversion: $count bits"
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
        '--stream_attack_count',[string]$count,
        '--stream_attack_target_steps'
    )
    foreach($step in $experiment.Steps){$experimentArgs += [string]$step}
    $experimentArgs += '--stream_attack_target_frames'
    foreach($frame in $experiment.Frames){$experimentArgs += [string]$frame}
    $experimentArgs += @('--stream_attack_seed','42','--run_name',$experiment.Name)
    & $CondaExe @experimentArgs
    if($LASTEXITCODE -ne 0){throw "Burst attack failed: $($experiment.Name)"}
}

Write-Host ('=' * 78)
Write-Host "Structured full-sign inversion summary: $Mode"
Write-Host ('=' * 78)
$cleanPSNR = 33.47
$cleanLPIPS = 0.0757
$summary = foreach($experiment in $experiments){
    $metricsPath = Join-Path 'exp_flf2v/results_720p' "$($experiment.Name)/Jockey/gop0/metrics.json"
    $metrics = Get-Content -LiteralPath $metricsPath -Raw | ConvertFrom-Json
    $attack = $metrics.stream_attack
    [pscustomobject]@{
        Run = $experiment.Name
        Steps = ($experiment.Steps -join ',')
        Frames = ($experiment.Frames -join ',')
        Bits = [int]$attack.channel_flipped_bits
        PayloadBER = [double]$attack.payload_BER
        PSNR = [double]$metrics.PSNR_dB
        DeltaPSNR = [math]::Round([double]$metrics.PSNR_dB - $cleanPSNR, 4)
        LPIPS = [double]$metrics.LPIPS
        DeltaLPIPS = [math]::Round([double]$metrics.LPIPS - $cleanLPIPS, 4)
    }
}
$summary | Format-Table -AutoSize
$summary | ConvertTo-Json | Set-Content -Encoding UTF8 "exp_flf2v/results_720p/burst_$($Mode)_summary.json"
