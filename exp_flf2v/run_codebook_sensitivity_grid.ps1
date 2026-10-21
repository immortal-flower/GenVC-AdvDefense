param(
    [ValidateSet('sign','index','both')]
    [string]$Field = 'sign',
    [int]$Count = 16,
    [int[]]$Steps = (1..17),
    [int[]]$Frames = (0..8),
    [int[]]$AttackSeeds = @(42),
    [int]$ModelSeed = 42,
    [string]$Sequence = 'Jockey',
    [string]$CondaExe = 'D:\anaconda\Scripts\conda.exe',
    [string]$Data = 'D:\yzb and lmk\dataset-720P\UVG\Jockey_720p.yuv',
    [string]$ResultsRoot = 'exp_flf2v/results_720p',
    [string]$CleanCodebook = 'exp_flf2v/results_720p/codebook_sign_bitflip_rate0p02_step1/Jockey/gop0/codebook_clean.tdcm',
    [string]$CleanMetrics = 'exp_flf2v/results_720p/sensitivity_clean_redecode/Jockey/gop0/metrics.json',
    [switch]$PlanOnly
)

# Uniform 17 x 9 sensitivity grid over the real serialized codebook payload.
# Each job changes the same number of physical bits in exactly one
# (SDE step, latent position) cell. Completed jobs are skipped, so the sweep
# can be stopped and resumed safely.

Set-Location (Split-Path -Parent $PSScriptRoot)
$env:CUDA_VISIBLE_DEVICES = '0,1'

if($Count -lt 1 -or $Count -gt 64){
    throw 'Count must be in [1,64] for one latent-position record.'
}
if($Steps | Where-Object { $_ -lt 1 -or $_ -gt 17 }){
    throw 'Steps must use 1-based values in [1,17].'
}
if($Frames | Where-Object { $_ -lt 0 -or $_ -gt 8 }){
    throw 'Frames must use latent positions in [0,8].'
}
if(-not (Test-Path -LiteralPath $ResultsRoot)){
    New-Item -ItemType Directory -Path $ResultsRoot -Force | Out-Null
}
if(-not $PlanOnly -and -not (Test-Path -LiteralPath $CleanCodebook)){
    throw "Clean codebook not found: $CleanCodebook"
}
if($PlanOnly -and -not (Test-Path -LiteralPath $CleanCodebook)){
    Write-Warning "Clean codebook is absent in this checkout; execution would require: $CleanCodebook"
}

$fieldNames = if($Field -eq 'both'){ @('sign','index') } else { @($Field) }
$sequenceTag = ($Sequence -replace '[^A-Za-z0-9_-]', '_').ToLowerInvariant()
$jobs = foreach($fieldName in $fieldNames){
    $attackName = if($fieldName -eq 'sign'){ 'sign-bitflip' } else { 'index-bitflip' }
    foreach($attackSeed in $AttackSeeds){
        foreach($step in $Steps){
            foreach($frame in $Frames){
                $runName = "sensitivity_grid_${fieldName}${Count}_step${step}_frame${frame}_aseed${attackSeed}_${sequenceTag}"
                $metricsPath = Join-Path $ResultsRoot "$runName/$Sequence/gop0/metrics.json"
                [pscustomobject]@{
                    RunName = $runName
                    Field = $fieldName
                    Attack = $attackName
                    Count = $Count
                    Step = $step
                    LatentPosition = $frame
                    AttackSeed = $attackSeed
                    ModelSeed = $ModelSeed
                    Sequence = $Sequence
                    MetricsPath = $metricsPath
                    Status = if(Test-Path -LiteralPath $metricsPath){ 'complete' } else { 'pending' }
                }
            }
        }
    }
}

$manifestPath = Join-Path $ResultsRoot "sensitivity_grid_manifest_${sequenceTag}.csv"
$jobs | Export-Csv -LiteralPath $manifestPath -NoTypeInformation -Encoding UTF8
Write-Host "Grid manifest: $manifestPath"
Write-Host "Jobs: $($jobs.Count); complete: $(($jobs | Where-Object Status -eq 'complete').Count); pending: $(($jobs | Where-Object Status -eq 'pending').Count)"

if($PlanOnly){
    Write-Host '[PLAN ONLY] No model inference was started.'
    exit 0
}

foreach($job in $jobs){
    if($job.Status -eq 'complete'){
        Write-Host "[SKIP] $($job.RunName)"
        continue
    }
    Write-Host ('=' * 88)
    Write-Host "[RUN] field=$($job.Field) step=$($job.Step) latent=$($job.LatentPosition) count=$($job.Count) attack_seed=$($job.AttackSeed)"
    Write-Host ('=' * 88)
    $experimentArgs = @(
        'run','--no-capture-output','-n','GVCC-5090','python',
        'exp_flf2v/run_flf2v_experiment.py',
        '--wan_ckpt','exp_flf2v/Wan2.1-FLF2V-14B-720P',
        '--data_dir',$Data,'--output_dir',$ResultsRoot,
        '--num_frames_per_gop','33','--num_gops','1',
        '--height','720','--width','1280','--M','64','--K','16384',
        '--steps','20','--ddim_tail','3','--g_scale','3.0',
        '--ref_codec','compressai','--ref_quality','4','--seed',[string]$ModelSeed,
        '--sequences',$Sequence,'--attack','none','--defense','none',
        '--reuse_codebook',$CleanCodebook,
        '--stream_attack',$job.Attack,
        '--stream_attack_transport','serialized',
        '--stream_attack_count',[string]$job.Count,
        '--stream_attack_target_steps',[string]$job.Step,
        '--stream_attack_target_frames',[string]$job.LatentPosition,
        '--stream_attack_seed',[string]$job.AttackSeed,
        '--run_name',$job.RunName
    )
    & $CondaExe @experimentArgs
    if($LASTEXITCODE -ne 0){
        throw "Sensitivity-grid job failed: $($job.RunName)"
    }
}

& $CondaExe run --no-capture-output -n GVCC-5090 python `
    'exp_flf2v/summarize_codebook_sensitivity_grid.py' `
    '--results_root' $ResultsRoot `
    '--sequence' $Sequence `
    '--clean_metrics' $CleanMetrics
if($LASTEXITCODE -ne 0){ throw 'Sensitivity-grid summary failed.' }

