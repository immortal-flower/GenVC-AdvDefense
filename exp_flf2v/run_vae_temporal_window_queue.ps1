$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:CUDA_VISIBLE_DEVICES = '0,1'
$env:GVCC_REF_CODEC_DEVICE = 'cpu'
$python = 'C:\Users\Admin\.conda\envs\GVCC-5090\python.exe'
$root = '.\exp_flf2v\results_720p'
$baseline = Join-Path $root 'vae_content_rgb_v1_paired.csv'
if (-not (Test-Path -LiteralPath $baseline)) { throw 'Missing completed five-frame baseline CSV' }
Start-Transcript -Path '.\vae_temporal_window_queue.log' -Append
try {
    & $python -m unittest tests.test_vae_temporal_schedule -v
    if ($LASTEXITCODE -ne 0) { throw 'Temporal schedule tests failed' }
    foreach ($window in @(9,17)) {
        & $python -u '.\exp_flf2v\run_python_logged.py' '.\exp_flf2v\run_vae_content_batch.py' --data_root 'D:\yzb and lmk\dataset-720P\UVG' --wan_ckpt '.\exp_flf2v\Wan2.1-FLF2V-14B-720P' --rounds 20 --temporal_window $window
        if ($LASTEXITCODE -ne 0) { throw "Window $window failed. Resume the same command; no silent shorter-window fallback." }
    }
    $rows = @()
    foreach ($w in @(5,9,17)) {
        $file = if ($w -eq 5) { $baseline } else { Join-Path $root "vae_content_rgb_t${w}_v1_paired.csv" }
        foreach ($r in (Import-Csv -LiteralPath $file | Where-Object { [int]$_.Rounds -eq 20 })) {
            $rows += [pscustomobject]@{Sequence=$r.Sequence; Window=$w; Rounds=20; CandidatePSNR=$r.CandidatePSNR; RandomPSNR=$r.RandomPSNR; ExtraDropVsRandom=$r.ExtraDropVsRandom; ExtraLPIPSVsRandom=$r.ExtraLPIPSVsRandom; InputMSE=$r.InputMSE; InputLinfPixel=$r.InputLinfPixel; CandidateBPP=$r.CandidateBPP; RandomBPP=$r.RandomBPP}
        }
    }
    $rows | Export-Csv -LiteralPath (Join-Path $root 'vae_temporal_window_comparison.csv') -NoTypeInformation -Encoding UTF8
    $rows | Format-Table Sequence,Window,CandidatePSNR,ExtraDropVsRandom,ExtraLPIPSVsRandom -AutoSize
    Write-Host 'Complete. Fixed sweeps, NOT equal compute; windows use different temporal crop start ranges.'
} finally { Stop-Transcript }
