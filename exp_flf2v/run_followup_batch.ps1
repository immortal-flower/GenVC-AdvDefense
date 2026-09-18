param(
    [string]$CondaExe = "D:\anaconda\Scripts\conda.exe",
    [string]$CondaEnv = "GVCC-5090",
    [string]$DataFile = "D:\yzb and lmk\dataset-720P\UVG\Jockey_720p.yuv",
    [string]$CudaDevices = "0,1",
    [switch]$Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$experimentScript = Join-Path $PSScriptRoot "run_flf2v_experiment.py"
$checkpoint = Join-Path $PSScriptRoot "Wan2.1-FLF2V-14B-720P"
$outputDir = Join-Path $PSScriptRoot "results_720p"
$logDir = Join-Path $PSScriptRoot "batch_logs"

foreach ($requiredPath in @($CondaExe, $DataFile, $experimentScript, $checkpoint)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "Required path not found: $requiredPath"
    }
}

New-Item -ItemType Directory -Force -Path $outputDir, $logDir | Out-Null
$env:CUDA_VISIBLE_DEVICES = $CudaDevices

$common = @(
    "--wan_ckpt", $checkpoint,
    "--data_dir", $DataFile,
    "--output_dir", $outputDir,
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

$tasks = @(
    [pscustomobject]@{
        Name = "gop_shared_eps4_fixed"
        Args = @("--attack", "gop-shared", "--defense", "none", "--epsilon", "4")
    },
    [pscustomobject]@{
        Name = "gop_shared_eps4_jpeg_q85_fixed"
        Args = @("--attack", "gop-shared", "--defense", "jpeg", "--epsilon", "4", "--jpeg_quality", "85")
    },
    [pscustomobject]@{
        Name = "ftuap_eps4_median_k3"
        Args = @("--attack", "ftuap", "--defense", "median", "--epsilon", "4", "--median_size", "3")
    },
    [pscustomobject]@{
        Name = "ftuap_eps4_jpeg_q85_median_k3"
        Args = @("--attack", "ftuap", "--defense", "jpeg-median", "--epsilon", "4", "--jpeg_quality", "85", "--median_size", "3")
    },
    [pscustomobject]@{
        Name = "uap_eps4_median_k3"
        Args = @("--attack", "uap", "--defense", "median", "--epsilon", "4", "--median_size", "3")
    },
    [pscustomobject]@{
        Name = "uap_eps4_jpeg_q85_median_k3"
        Args = @("--attack", "uap", "--defense", "jpeg-median", "--epsilon", "4", "--jpeg_quality", "85", "--median_size", "3")
    },
    [pscustomobject]@{
        Name = "keyframe_eps4_perframe"
        Args = @("--attack", "keyframe", "--defense", "none", "--epsilon", "4")
    }
)

$failures = @()
Push-Location $repoRoot
try {
    foreach ($task in $tasks) {
        $metricsPath = Join-Path $outputDir "$($task.Name)\Jockey\gop0\metrics.json"
        if ((Test-Path -LiteralPath $metricsPath) -and -not $Force) {
            Write-Host "[SKIP] $($task.Name): metrics.json already exists."
            continue
        }

        $logPath = Join-Path $logDir "$($task.Name).log"
        $commandArgs = @(
            "run", "--no-capture-output", "-n", $CondaEnv,
            "python", $experimentScript
        ) + $common + $task.Args + @("--run_name", $task.Name)

        Write-Host ""
        Write-Host ("=" * 78)
        Write-Host "[RUN] $($task.Name)  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
        Write-Host "[LOG] $logPath"
        Write-Host ("=" * 78)

        & $CondaExe @commandArgs 2>&1 | Tee-Object -FilePath $logPath
        $exitCode = $LASTEXITCODE
        if ($exitCode -ne 0 -or -not (Test-Path -LiteralPath $metricsPath)) {
            $failures += $task.Name
            Write-Warning "Task failed or produced no metrics: $($task.Name) (exit=$exitCode)"
            continue
        }

        $metrics = Get-Content -LiteralPath $metricsPath -Raw | ConvertFrom-Json
        Write-Host "[DONE] $($task.Name): PSNR=$($metrics.PSNR_dB) dB, LPIPS=$($metrics.LPIPS), BPP=$($metrics.BPP)"
    }
}
finally {
    Pop-Location
}

Write-Host ""
Write-Host "Batch summary"
Write-Host "-------------"
$summary = foreach ($task in $tasks) {
    $metricsPath = Join-Path $outputDir "$($task.Name)\Jockey\gop0\metrics.json"
    if (Test-Path -LiteralPath $metricsPath) {
        $metrics = Get-Content -LiteralPath $metricsPath -Raw | ConvertFrom-Json
        [pscustomobject]@{
            Run = $task.Name
            PSNR = $metrics.PSNR_dB
            LPIPS = $metrics.LPIPS
            BPP = $metrics.BPP
            Status = "complete"
        }
    }
    else {
        [pscustomobject]@{
            Run = $task.Name
            PSNR = $null
            LPIPS = $null
            BPP = $null
            Status = "missing"
        }
    }
}
$summary | Format-Table -AutoSize

if ($failures.Count -gt 0) {
    throw "Batch completed with failures: $($failures -join ', ')"
}

