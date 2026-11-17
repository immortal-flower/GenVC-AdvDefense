param(
    [string]$Python = 'C:\Users\Admin\.conda\envs\GVCC-5090\python.exe',
    [string]$Data = 'D:\yzb and lmk\dataset-720P\UVG\Jockey_720p.yuv',
    [string]$WanCkpt = '.\exp_flf2v\Wan2.1-FLF2V-14B-720P',
    [string]$InitialAttack = '.\exp_flf2v\results_720p\input_direction_oldvae_eps2_seed42_assets\full_positive.npz',
    [string]$OutputRoot = '.\exp_flf2v\results_720p',
    [string]$Tag = 'native_query_jockey_gop0_energy_matched_v1',
    [switch]$PlanOnly
)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:CUDA_VISIBLE_DEVICES = '0,1'
$extra = @()
if ($PlanOnly) { $extra += '--plan_only' }
Start-Transcript '.\native_rgb_query_search.log' -Append | Out-Host
try {
    & $Python -m unittest discover -s tests -p test_native_rgb_search.py -v
    if ($LASTEXITCODE -ne 0) { throw 'CPU search tests failed; no GPU experiment started.' }
    & $Python '.\exp_flf2v\search_native_rgb_input.py' `
        --data $Data --wan_ckpt $WanCkpt --initial_attack $InitialAttack `
        --results_root $OutputRoot --tag $Tag --rounds 4 --epsilon 4 --step_pixel 1 `
        --search_seed 20261117 @extra
    if ($LASTEXITCODE -ne 0) { throw 'Native search stopped; inspect log and resume the same command.' }
} finally {
    Stop-Transcript | Out-Host
}
