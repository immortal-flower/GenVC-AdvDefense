param(
    [string]$CondaExe = 'D:\anaconda\Scripts\conda.exe',
    [string]$Data = 'D:\yzb and lmk\dataset-720P\UVG\Jockey_720p.yuv',
    [string]$Gpu0Memory = '20GiB',
    [string]$Gpu1Memory = '20GiB'
)
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:CUDA_VISIBLE_DEVICES = '0,1'
$commandArgs=@('run','--no-capture-output','-n','GVCC-5090','python',
    'exp_flf2v/probe_wan_condition_grad.py','--data',$Data,
    '--gpu0_memory',$Gpu0Memory,'--gpu1_memory',$Gpu1Memory)
& $CondaExe @commandArgs
if($LASTEXITCODE -ne 0){throw 'Wan condition-gradient probe failed; see probe.json and terminal traceback.'}
