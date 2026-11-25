$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:CUDA_VISIBLE_DEVICES = '0,1'
$python = 'C:\Users\Admin\.conda\envs\GVCC-5090\python.exe'
& $python -u '.\exp_flf2v\run_python_logged.py' '.\exp_flf2v\analyze_vae_error_inheritance.py' --data_root 'D:\yzb and lmk\dataset-720P\UVG' --vae '.\exp_flf2v\Wan2.1-FLF2V-14B-720P\Wan2.1_VAE.pth' 2>&1 | Tee-Object -FilePath '.\vae_error_inheritance.log'
if ($LASTEXITCODE -ne 0) { throw 'VAE audit failed. No GVCC runs were started; completed VAE cases are cached.' }
Import-Csv '.\exp_flf2v\results_720p\vae_content_rgb_v1_inheritance\summary.csv' | Format-Table Sequence,Rounds,Reference,VAE_PSNRDrop,Native_PSNRDrop,Pearson,BothPositiveFrames -AutoSize
