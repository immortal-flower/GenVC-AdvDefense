param([string]$Python='C:\Users\Admin\.conda\envs\GVCC-5090\python.exe')
$ErrorActionPreference='Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:CUDA_VISIBLE_DEVICES='0,1'
$root='.\exp_flf2v\results_720p'
$tag='trajectory_rgb_jockey_gop0_eps4_seed42_b6_p1_a0p5_h720w1280_grid128x224_m20GiB_16GiB_vaecpu_stepvjp_decodecpu_r44_native_v4'
$probe=Join-Path $root "${tag}_assets\decoder_probe.npz"
if(-not(Test-Path -LiteralPath $probe)){throw 'Missing saved decoder_probe.npz'}
$out=Join-Path $root 'native720_decoder_gpu_isolated_v1'
New-Item -ItemType Directory -Path $out -Force | Out-Null
# Merge warnings inside Python rather than suppressing stderr in PowerShell.
$ErrorActionPreference='Continue'
& $Python -u '.\exp_flf2v\run_python_logged.py' '.\exp_flf2v\probe_saved_rgb_decoder.py' `
    --probe $probe --vae '.\exp_flf2v\Wan2.1-FLF2V-14B-720P\Wan2.1_VAE.pth' `
    --device cuda:0 --output $out 2>&1 | Tee-Object -FilePath "$out\console.log"
$code=$LASTEXITCODE
@{exit_code=$code;exit_hex=('0x{0:X8}' -f [BitConverter]::ToUInt32([BitConverter]::GetBytes([int]$code),0))} |
    ConvertTo-Json | Set-Content "$out\process_exit.json" -Encoding UTF8
if($code -ne 0){throw "Isolated decoder probe failed: $code. Do not restart the full Wan queue."}
Get-Content "$out\status.json"
