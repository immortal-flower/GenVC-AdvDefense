param(
    [string]$CondaExe = 'D:\anaconda\Scripts\conda.exe',
    [string]$Data = 'D:\yzb and lmk\dataset-720P\UVG\Jockey_720p.yuv',
    [string]$CleanCodebook = 'exp_flf2v/results_720p/codebook_sign_bitflip_rate0p02_step1/Jockey/gop0/codebook_clean.tdcm'
)

# White-box bitstream attack on the consistently vulnerable location:
# SDE step 1, latent-time position 4.  Random and atom-impact-ranked flips use
# exactly the same physical bit budgets.  The existing random-16 result is
# reused automatically.
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:CUDA_VISIBLE_DEVICES = '0,1'

if(-not (Test-Path -LiteralPath $CleanCodebook)){
    throw "Clean codebook not found: $CleanCodebook"
}

$experiments = @(
    [pscustomobject]@{Name='impact_frame4_random4'; Attack='sign-bitflip'; Count=4},
    [pscustomobject]@{Name='impact_frame4_ranked4'; Attack='sign-impact-bitflip'; Count=4},
    [pscustomobject]@{Name='impact_frame4_random8'; Attack='sign-bitflip'; Count=8},
    [pscustomobject]@{Name='impact_frame4_ranked8'; Attack='sign-impact-bitflip'; Count=8},
    [pscustomobject]@{Name='sensitivity_step1_frame4_sign16'; Attack='sign-bitflip'; Count=16},
    [pscustomobject]@{Name='impact_frame4_ranked16'; Attack='sign-impact-bitflip'; Count=16}
)

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
        '--stream_attack_target_frames','4',
        '--stream_attack_seed','42','--run_name',$experiment.Name
    )
    & $CondaExe @experimentArgs
    if($LASTEXITCODE -ne 0){throw "Impact attack failed: $($experiment.Name)"}
}

Write-Host ('=' * 78)
Write-Host 'Random versus impact-ranked sign flips (step 1, latent frame 4)'
Write-Host ('=' * 78)
$summary = foreach($experiment in $experiments){
    $metricsPath = Join-Path 'exp_flf2v/results_720p' "$($experiment.Name)/Jockey/gop0/metrics.json"
    $metrics = Get-Content -LiteralPath $metricsPath -Raw | ConvertFrom-Json
    $selection = $metrics.stream_attack.selection_metadata
    [pscustomobject]@{
        Run = $experiment.Name
        Method = $experiment.Attack
        Bits = $experiment.Count
        NoiseMSE = if($null -ne $selection){$selection.final_noise_mse}else{$null}
        NoiseCosine = if($null -ne $selection){$selection.final_noise_cosine}else{$null}
        PSNR = $metrics.PSNR_dB
        LPIPS = $metrics.LPIPS
    }
}
$summary | Format-Table -AutoSize
$summary | ConvertTo-Json | Set-Content -Encoding UTF8 'exp_flf2v/results_720p/impact_frame4_summary.json'
