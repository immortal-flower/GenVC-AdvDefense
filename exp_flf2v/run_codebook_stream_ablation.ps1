param(
    [string]$CondaExe = 'D:\anaconda\Scripts\conda.exe',
    [string]$Data = 'D:\yzb and lmk\dataset-720P\UVG\Jockey_720p.yuv'
)
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:CUDA_VISIBLE_DEVICES = '0,1'

$experiments=@(
    [pscustomobject]@{
        Name='codebook_sign_bitflip_rate0p005_step1'
        Attack='sign-bitflip';Rate='0.005';Early='1'
    },
    [pscustomobject]@{
        Name='codebook_index_bitflip_rate0p02_step1'
        Attack='index-bitflip';Rate='0.02';Early='1'
    },
    [pscustomobject]@{
        Name='codebook_sign_bitflip_rate0p02_allsteps'
        Attack='sign-bitflip';Rate='0.02';Early='0'
    }
)

foreach($experiment in $experiments){
    Write-Host ('='*78)
    Write-Host "[RUN] $($experiment.Name)"
    Write-Host ('='*78)
    $experimentArgs=@('run','--no-capture-output','-n','GVCC-5090','python',
        'exp_flf2v/run_flf2v_experiment.py','--wan_ckpt','exp_flf2v/Wan2.1-FLF2V-14B-720P',
        '--data_dir',$Data,'--output_dir','exp_flf2v/results_720p','--num_frames_per_gop','33',
        '--num_gops','1','--height','720','--width','1280','--M','64','--K','16384',
        '--steps','20','--ddim_tail','3','--g_scale','3.0','--ref_codec','compressai',
        '--ref_quality','4','--seed','42','--sequences','Jockey','--attack','none',
        '--defense','none','--stream_attack',$experiment.Attack,
        '--stream_attack_rate',$experiment.Rate,
        '--stream_attack_early_steps',$experiment.Early,
        '--stream_attack_seed','42','--run_name',$experiment.Name)
    & $CondaExe @experimentArgs
    if($LASTEXITCODE -ne 0){throw "Ablation failed: $($experiment.Name)"}
}

Write-Host ('='*78)
Write-Host 'Codebook stream ablation summary'
Write-Host ('='*78)
$summary=@()
foreach($experiment in $experiments){
    $metricsPath=Join-Path 'exp_flf2v/results_720p' "$($experiment.Name)/Jockey/gop0/metrics.json"
    $metrics=Get-Content -LiteralPath $metricsPath -Raw | ConvertFrom-Json
    $summary += [pscustomobject]@{
        Run=$experiment.Name
        ChangedBits=$metrics.stream_attack.changed_bits
        PayloadBER=$metrics.stream_attack.payload_BER
        PSNR=$metrics.PSNR_dB
        LPIPS=$metrics.LPIPS
        BPP=$metrics.BPP
    }
}
$summary | Format-Table -AutoSize
