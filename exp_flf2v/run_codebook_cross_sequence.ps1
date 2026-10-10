param(
    [string[]]$Sequences = @('Beauty','YachtRide'),
    [string]$CondaExe = 'D:\anaconda\Scripts\conda.exe',
    [string]$DataDir = 'D:\yzb and lmk\dataset-720P\UVG',
    [string]$CleanRun = 'crossseq_clean',
    [string]$AttackRun = 'crossseq_burst_step1_frames3_4'
)

# Cross-content validation of the structured codebook weakness.  A clean GOP
# is encoded once per sequence when needed, then its exact payload is reused
# while all 128 sign bits in (SDE step 1) x (latent frames 3,4) are inverted.
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:CUDA_VISIBLE_DEVICES = '0,1'

$commonArgs = @(
    '--wan_ckpt','exp_flf2v/Wan2.1-FLF2V-14B-720P',
    '--data_dir',$DataDir,'--output_dir','exp_flf2v/results_720p',
    '--num_frames_per_gop','33','--num_gops','1','--start_gop','0',
    '--height','720','--width','1280','--M','64','--K','16384',
    '--steps','20','--ddim_tail','3','--g_scale','3.0',
    '--ref_codec','compressai','--ref_quality','4','--seed','42',
    '--attack','none','--defense','none'
)

foreach($sequence in $Sequences){
    $cleanDir = "exp_flf2v/results_720p/$CleanRun/$sequence/gop0"
    $cleanCodebook = Join-Path $cleanDir 'codebook.tdcm'
    $cleanMetrics = Join-Path $cleanDir 'metrics.json'
    if((Test-Path -LiteralPath $cleanCodebook) -and
       (Test-Path -LiteralPath $cleanMetrics)){
        Write-Host "[SKIP] $sequence clean GOP already exists"
    } else {
        Write-Host ('=' * 78)
        Write-Host "[ENCODE CLEAN] $sequence GOP 0"
        Write-Host ('=' * 78)
        $cleanArgs = @(
            'run','--no-capture-output','-n','GVCC-5090','python',
            'exp_flf2v/run_flf2v_experiment.py'
        ) + $commonArgs + @(
            '--sequences',$sequence,
            '--stream_attack','none',
            '--run_name',$CleanRun
        )
        & $CondaExe @cleanArgs
        if($LASTEXITCODE -ne 0){throw "Clean encoding failed: $sequence"}
        if(-not (Test-Path -LiteralPath $cleanCodebook)){
            throw "Clean codebook was not produced for $sequence; check the sequence name/data path"
        }
    }

    $attackMetrics = "exp_flf2v/results_720p/$AttackRun/$sequence/gop0/metrics.json"
    if(Test-Path -LiteralPath $attackMetrics){
        Write-Host "[SKIP] $sequence structured attack already exists"
        continue
    }
    Write-Host ('=' * 78)
    Write-Host "[ATTACK] $sequence GOP 0: step 1, latent frames 3-4, 128 sign bits"
    Write-Host ('=' * 78)
    $attackArgs = @(
        'run','--no-capture-output','-n','GVCC-5090','python',
        'exp_flf2v/run_flf2v_experiment.py'
    ) + $commonArgs + @(
        '--sequences',$sequence,
        '--reuse_codebook',$cleanCodebook,
        '--stream_attack','sign-bitflip',
        '--stream_attack_transport','serialized',
        '--stream_attack_count','128',
        '--stream_attack_target_steps','1',
        '--stream_attack_target_frames','3','4',
        '--stream_attack_seed','42',
        '--run_name',$AttackRun
    )
    & $CondaExe @attackArgs
    if($LASTEXITCODE -ne 0){throw "Cross-sequence attack failed: $sequence"}
}

Write-Host ('=' * 78)
Write-Host 'Cross-sequence structured sign-inversion summary'
Write-Host ('=' * 78)
$summary = foreach($sequence in $Sequences){
    $cleanPath = "exp_flf2v/results_720p/$CleanRun/$sequence/gop0/metrics.json"
    $attackPath = "exp_flf2v/results_720p/$AttackRun/$sequence/gop0/metrics.json"
    if(-not (Test-Path -LiteralPath $cleanPath)){
        throw "Missing clean metrics: $cleanPath"
    }
    if(-not (Test-Path -LiteralPath $attackPath)){
        throw "Missing attack metrics: $attackPath"
    }
    $clean = Get-Content -LiteralPath $cleanPath -Raw | ConvertFrom-Json
    $attack = Get-Content -LiteralPath $attackPath -Raw | ConvertFrom-Json
    [pscustomobject]@{
        Sequence = $sequence
        Bits = [int]$attack.stream_attack.channel_flipped_bits
        PayloadBER = [double]$attack.stream_attack.payload_BER
        CleanPSNR = [double]$clean.PSNR_dB
        AttackPSNR = [double]$attack.PSNR_dB
        DeltaPSNR = [math]::Round([double]$attack.PSNR_dB - [double]$clean.PSNR_dB, 4)
        CleanLPIPS = [double]$clean.LPIPS
        AttackLPIPS = [double]$attack.LPIPS
        DeltaLPIPS = [math]::Round([double]$attack.LPIPS - [double]$clean.LPIPS, 4)
    }
}
$summary | Format-Table -AutoSize
$summary | ConvertTo-Json | Set-Content -Encoding UTF8 'exp_flf2v/results_720p/cross_sequence_burst_summary.json'
