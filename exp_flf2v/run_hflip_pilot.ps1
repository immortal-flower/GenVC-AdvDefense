# Paper-migration pilot for GVCC / FLF2V.
#
# This is deliberately a *single-route* reversible horizontal-flip baseline:
# the full GOP is mirrored before encoding and mirrored back after decoding.
# It is inspired by the input-randomization component of Song et al. (2024),
# not their full two-way selection algorithm.  The two-way variant would run
# each GOP twice and choose the lower self-supervised reconstruction loss.

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$CondaExe = "D:\anaconda\Scripts\conda.exe"
$ScriptPath = "exp_flf2v/run_flf2v_experiment.py"

if (-not (Test-Path -LiteralPath $CondaExe)) {
    throw "Cannot find conda.exe: $CondaExe"
}

$env:CUDA_VISIBLE_DEVICES = "0,1"

# Keep this identical to the prior Jockey 720p, 33-frame, one-GOP experiment.
$CommonArgs = @(
    "--wan_ckpt", "exp_flf2v/Wan2.1-FLF2V-14B-720P",
    "--data_dir", "D:\yzb and lmk\dataset-720P\UVG\Jockey_720p.yuv",
    "--output_dir", "exp_flf2v/results_720p",
    "--num_frames_per_gop", "33",
    "--num_gops", "1",
    "--height", "720",
    "--width", "1280",
    "--M", "64",
    "--K", "16384",
    "--steps", "20",
    "--ddim_tail", "3",
    "--g_scale", "3.0",
    "--ref_codec", "compressai",
    "--ref_quality", "4",
    "--seed", "42",
    "--sequences", "Jockey"
)

$Runs = @(
    @{
        Name = "clean_hflip_pilot"
        Args = @("--attack", "none", "--defense", "hflip")
    },
    @{
        Name = "ftuap_eps4_hflip_pilot"
        Args = @("--attack", "ftuap", "--defense", "hflip", "--epsilon", "4")
    }
)

foreach ($Run in $Runs) {
    Write-Host ""
    Write-Host ("=" * 86)
    Write-Host "[RUN] $($Run.Name)  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    Write-Host ("=" * 86)

    & $CondaExe run --no-capture-output -n GVCC-5090 python $ScriptPath @CommonArgs @($Run.Args) "--run_name" $Run.Name
    if ($LASTEXITCODE -ne 0) {
        throw "Experiment failed: $($Run.Name) (exit code $LASTEXITCODE)"
    }
}

Write-Host ""
Write-Host "[DONE] HFlip pilot completed. Read metrics from:"
Write-Host "  exp_flf2v/results_720p/clean_hflip_pilot/Jockey/gop0/metrics.json"
Write-Host "  exp_flf2v/results_720p/ftuap_eps4_hflip_pilot/Jockey/gop0/metrics.json"
