param(
    [int[]]$Counts = @(16),
    [int[]]$AttackSeeds = @(42,123,2026),
    [string]$Sequence = 'Jockey',
    [string]$CondaExe = 'D:\anaconda\Scripts\conda.exe',
    [string]$Data = 'D:\yzb and lmk\dataset-720P\UVG\Jockey_720p.yuv',
    [string]$ResultsRoot = 'exp_flf2v/results_720p',
    [string]$CleanCodebook = 'exp_flf2v/results_720p/codebook_sign_bitflip_rate0p02_step1/Jockey/gop0/codebook_clean.tdcm',
    [string]$CleanMetrics = 'exp_flf2v/results_720p/sensitivity_clean_redecode/Jockey/gop0/metrics.json',
    [string]$Plan = 'exp_flf2v/validation_cells_jockey.csv',
    [int]$SourceGop = 0,
    [switch]$SaveVideo,
    [switch]$PlanOnly
)

Set-Location (Split-Path -Parent $PSScriptRoot)
$env:CUDA_VISIBLE_DEVICES = '0,1'

$runnerArgs = @(
    'run','--no-capture-output','-n','GVCC-5090','python',
    'exp_flf2v/run_codebook_validation_batch.py',
    '--plan',$Plan,
    '--data',$Data,
    '--sequence',$Sequence,
    '--wan_ckpt','exp_flf2v/Wan2.1-FLF2V-14B-720P',
    '--output_dir',$ResultsRoot,
    '--clean_codebook',$CleanCodebook,
    '--clean_metrics',$CleanMetrics,
    '--gop_index',[string]$SourceGop,
    '--counts'
)
$runnerArgs += $Counts | ForEach-Object { [string]$_ }
$runnerArgs += '--attack_seeds'
$runnerArgs += $AttackSeeds | ForEach-Object { [string]$_ }
if($SaveVideo){ $runnerArgs += '--save_video' }
if($PlanOnly){ $runnerArgs += '--plan_only' }

& $CondaExe @runnerArgs
if($LASTEXITCODE -ne 0){
    throw "Persistent validation batch failed with exit code $LASTEXITCODE"
}

if(-not $PlanOnly){
    & $CondaExe run --no-capture-output -n GVCC-5090 python `
        'exp_flf2v/summarize_codebook_sensitivity_grid.py' `
        '--results_root' $ResultsRoot `
        '--sequence' $Sequence `
        '--gop_index' $SourceGop `
        '--clean_metrics' $CleanMetrics
    if($LASTEXITCODE -ne 0){
        throw "Validation summary failed with exit code $LASTEXITCODE"
    }
}
