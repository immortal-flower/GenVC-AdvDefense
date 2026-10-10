param(
    [ValidateSet('single','joint','dose')]
    [string]$Mode = 'single',
    [string]$CondaExe = 'D:\anaconda\Scripts\conda.exe',
    [string]$Data = 'D:\yzb and lmk\dataset-720P\UVG\Jockey_720p.yuv',
    [string]$CleanCodebook = 'exp_flf2v/results_720p/codebook_sign_bitflip_rate0p02_step1/Jockey/gop0/codebook_clean.tdcm'
)

# White-box bitstream robustness test.  ``single`` targets the consistently
# vulnerable latent position 4; ``joint`` lets the geometric objective allocate
# one shared budget between positions 3 and 4.  Random and impact-ranked flips
# always use the same physical bit counts.
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:CUDA_VISIBLE_DEVICES = '0,1'

if(-not (Test-Path -LiteralPath $CleanCodebook)){
    throw "Clean codebook not found: $CleanCodebook"
}

if($Mode -eq 'single'){
    $experiments = @(
        [pscustomobject]@{Name='impact_frame4_random4'; Attack='sign-bitflip'; Count=4; Frames=@(4)},
        [pscustomobject]@{Name='impact_frame4_ranked4'; Attack='sign-impact-bitflip'; Count=4; Frames=@(4)},
        [pscustomobject]@{Name='impact_frame4_random8'; Attack='sign-bitflip'; Count=8; Frames=@(4)},
        [pscustomobject]@{Name='impact_frame4_ranked8'; Attack='sign-impact-bitflip'; Count=8; Frames=@(4)},
        [pscustomobject]@{Name='sensitivity_step1_frame4_sign16'; Attack='sign-bitflip'; Count=16; Frames=@(4)},
        [pscustomobject]@{Name='impact_frame4_ranked16'; Attack='sign-impact-bitflip'; Count=16; Frames=@(4)}
    )
} elseif($Mode -eq 'joint') {
    $experiments = @(
        [pscustomobject]@{Name='impact_frames3_4_random8'; Attack='sign-bitflip'; Count=8; Frames=@(3,4)},
        [pscustomobject]@{Name='impact_frames3_4_ranked8'; Attack='sign-impact-bitflip'; Count=8; Frames=@(3,4)},
        [pscustomobject]@{Name='impact_frames3_4_random16'; Attack='sign-bitflip'; Count=16; Frames=@(3,4)},
        [pscustomobject]@{Name='impact_frames3_4_ranked16'; Attack='sign-impact-bitflip'; Count=16; Frames=@(3,4)}
    )
} else {
    # Frame 3 is the most vulnerable latent temporal position in the earlier
    # sweep.  This dose curve compares random and geometry-ranked corruption
    # under exactly the same physical sign-bit budget.  The wider range is
    # intended to expose a possible nonlinear failure threshold.
    $experiments = foreach($count in @(4,8,12,16,24,32,48,64)){
        [pscustomobject]@{
            Name="impact_frame3_random$count"
            Attack='sign-bitflip'
            Count=$count
            Frames=@(3)
        }
        [pscustomobject]@{
            Name="impact_frame3_ranked$count"
            Attack='sign-impact-bitflip'
            Count=$count
            Frames=@(3)
        }
    }
}

foreach($experiment in $experiments){
    $metricsPath = Join-Path 'exp_flf2v/results_720p' "$($experiment.Name)/Jockey/gop0/metrics.json"
    if(Test-Path -LiteralPath $metricsPath){
        Write-Host "[SKIP] $($experiment.Name) already has metrics.json"
        continue
    }
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
        '--stream_attack',$experiment.Attack,
        '--stream_attack_transport','serialized',
        '--stream_attack_count',[string]$experiment.Count,
        '--stream_attack_target_steps','1',
        '--stream_attack_target_frames'
    )
    foreach($frame in $experiment.Frames){$experimentArgs += [string]$frame}
    $experimentArgs += @('--stream_attack_seed','42','--run_name',$experiment.Name)
    & $CondaExe @experimentArgs
    if($LASTEXITCODE -ne 0){throw "Impact attack failed: $($experiment.Name)"}
}

Write-Host ('=' * 78)
Write-Host "Random versus impact-ranked sign flips: $Mode"
Write-Host ('=' * 78)
$cleanPSNR = 33.47
$cleanLPIPS = 0.0757
$summary = foreach($experiment in $experiments){
    $metricsPath = Join-Path 'exp_flf2v/results_720p' "$($experiment.Name)/Jockey/gop0/metrics.json"
    $metrics = Get-Content -LiteralPath $metricsPath -Raw | ConvertFrom-Json
    $selection = $metrics.stream_attack.selection_metadata
    [pscustomobject]@{
        Run = $experiment.Name
        Method = $experiment.Attack
        Frames = ($experiment.Frames -join ',')
        Bits = $experiment.Count
        NoiseMSE = if($null -ne $selection){$selection.final_noise_mse}else{$null}
        NoiseCosine = if($null -ne $selection){$selection.final_noise_cosine}else{$null}
        PSNR = $metrics.PSNR_dB
        DeltaPSNR = [math]::Round([double]$metrics.PSNR_dB - $cleanPSNR, 4)
        LPIPS = $metrics.LPIPS
        DeltaLPIPS = [math]::Round([double]$metrics.LPIPS - $cleanLPIPS, 4)
        Allocation = if($null -ne $selection){
            (($selection.allocation_by_frame.psobject.Properties | ForEach-Object {
                "$($_.Name):$($_.Value)"
            }) -join ',')
        }else{$null}
    }
}
$summary | Format-Table -AutoSize
$summary | ConvertTo-Json | Set-Content -Encoding UTF8 "exp_flf2v/results_720p/impact_$($Mode)_summary.json"
