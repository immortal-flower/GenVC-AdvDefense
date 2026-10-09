param(
    [string]$CondaExe = 'D:\anaconda\Scripts\conda.exe',
    [string]$Data = 'D:\yzb and lmk\dataset-720P\UVG\Jockey_720p.yuv',
    [ValidateSet('float32','bfloat16')][string]$Dtype = 'float32'
)
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:CUDA_VISIBLE_DEVICES = '0,1'
$commandArgs=@('run','--no-capture-output','-n','GVCC-5090','python',
    'exp_flf2v/run_vae_only_control.py','--data',$Data,
    '--vae','exp_flf2v/Wan2.1-FLF2V-14B-720P/Wan2.1_VAE.pth',
    '--attack_file','exp_flf2v/attack_assets/jockey_vae_pgd_grid_deviation_eps2.npz',
    '--epsilon','2','--dtype',$Dtype,
    '--output',"exp_flf2v/results_720p/vae_only_grid_eps2_${Dtype}/Jockey")
& $CondaExe @commandArgs
if($LASTEXITCODE -ne 0){throw 'VAE-only control failed.'}
