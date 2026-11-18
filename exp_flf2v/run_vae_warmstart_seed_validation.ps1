param(
    [string]$Python = 'C:\Users\Admin\.conda\envs\GVCC-5090\python.exe',
    [string]$Data = 'D:\yzb and lmk\dataset-720P\UVG\Jockey_720p.yuv',
    [string]$WanCkpt = '.\exp_flf2v\Wan2.1-FLF2V-14B-720P',
    [string]$ResultsRoot = '.\exp_flf2v\results_720p',
    [string]$SourceTag = 'vae_warmstart_jockey_gop0_v1',
    [string]$Tag = 'vae_warmstart_eps4_r4_heldout_v1',
    [switch]$PlanOnly
)
$ErrorActionPreference = 'Stop'
$env:CUDA_VISIBLE_DEVICES = '0,1'
$arguments = @('--data', $Data, '--wan_ckpt', $WanCkpt, '--results_root', $ResultsRoot,
               '--source_tag', $SourceTag, '--tag', $Tag)
if ($PlanOnly) { $arguments += '--plan_only' }
Start-Transcript -Path '.\vae_warmstart_seed_validation.log' -Append
try {
    & $Python (Join-Path $PSScriptRoot 'validate_vae_warmstart_seeds.py') @arguments
    if ($LASTEXITCODE -ne 0) { throw 'Validation stopped. Fix the reported error, then rerun the same command to resume.' }
} finally {
    Stop-Transcript
}
