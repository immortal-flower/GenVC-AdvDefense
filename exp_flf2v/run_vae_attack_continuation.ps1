param(
    [string]$Python = 'C:\Users\Admin\.conda\envs\GVCC-5090\python.exe',
    [string]$Data = 'D:\yzb and lmk\dataset-720P\UVG\Jockey_720p.yuv',
    [string]$WanCkpt = '.\exp_flf2v\Wan2.1-FLF2V-14B-720P',
    [string]$Initial = '.\exp_flf2v\results_720p\input_direction_oldvae_eps2_seed42_assets\full_positive.npz',
    [string]$ResultsRoot = '.\exp_flf2v\results_720p',
    [string]$SourceTag = 'vae_warmstart_jockey_gop0_v1',
    [string]$Tag = 'vae_continue_eps4_r12_jockey_gop0_v1',
    [int[]]$TargetRounds = @(8,12),
    [switch]$PlanOnly
)
$ErrorActionPreference = 'Stop'
$env:CUDA_VISIBLE_DEVICES = '0,1'
$arguments = @('--data',$Data,'--wan_ckpt',$WanCkpt,'--initial',$Initial,
    '--results_root',$ResultsRoot,'--extend_tag',$SourceTag,'--tag',$Tag)
$arguments += '--extension_rounds'
foreach ($round in $TargetRounds) { $arguments += [string]$round }
if ($PlanOnly) { $arguments += '--plan_only' }
Start-Transcript -Path '.\vae_attack_continuation.log' -Append
try {
    & $Python (Join-Path $PSScriptRoot 'run_vae_warmstart_batch.py') @arguments
    if ($LASTEXITCODE -ne 0) { throw 'Continuation stopped; fix the error and rerun this command to resume.' }
} finally {
    Stop-Transcript
}
