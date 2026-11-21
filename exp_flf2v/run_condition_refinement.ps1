param([string]$Python='C:\Users\Admin\.conda\envs\GVCC-5090\python.exe')
$ErrorActionPreference='Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:CUDA_VISIBLE_DEVICES='0,1'
$arguments=@('.\exp_flf2v\run_condition_refinement.py','--data','D:\yzb and lmk\dataset-720P\UVG\Jockey_720p.yuv','--wan_ckpt','.\exp_flf2v\Wan2.1-FLF2V-14B-720P')
Start-Transcript '.\condition_refinement.log' -Append
try {
    & $Python @arguments --generate_only
    if($LASTEXITCODE -ne 0){throw 'Candidate generation failed; no native jobs started'}
    # A separate process frees all gradient graphs and CUDA model allocations.
    & $Python @arguments
    if($LASTEXITCODE -ne 0){throw 'Native acceptance failed'}
} finally {Stop-Transcript}
