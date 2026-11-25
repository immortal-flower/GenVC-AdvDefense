param([switch]$PlanOnly)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:CUDA_VISIBLE_DEVICES = '0,1'
$env:GVCC_REF_CODEC_DEVICE = 'cpu'
$python = 'C:\Users\Admin\.conda\envs\GVCC-5090\python.exe'
$extra = @()
if ($PlanOnly) { $extra += '--plan_only' }
& $python -u '.\exp_flf2v\run_python_logged.py' '.\exp_flf2v\run_vae_content_batch.py' --data_root 'D:\yzb and lmk\dataset-720P\UVG' --wan_ckpt '.\exp_flf2v\Wan2.1-FLF2V-14B-720P' @extra
if ($LASTEXITCODE -ne 0) { throw 'VAE content batch failed. Resume the same command; verified codec cases are reused.' }
