param(
    [string]$LogPath = '.\trajectory_rgb_full_chain_overnight.log',
    [switch]$IncludeFullBackward,
    [switch]$PlanOnly
)

$ErrorActionPreference = 'Stop'
$runner = Join-Path $PSScriptRoot 'run_trajectory_input_pilot.ps1'
$common = @{
    Objective = 'final-rgb'
    Epsilon = 4
    Alpha = 1
    Iterations = 1
    Gpu0Memory = '23GiB'
    Gpu1Memory = '15GiB'
    VaeCpuActivations = $true
    GridRandom = $true
    PairedOnly = $true
}
if ($PlanOnly) {
    Write-Host 'Jockey GOP0: 192x336 RGB candidate, matched grid random, then two full 720p codec runs.'
    Write-Host '6-step backward: one retry at 160x288 only if candidate generation reports CUDA OOM.'
    if ($IncludeFullBackward) {
        Write-Host '20-step backward follows at the same resolved size; no geometry fallback for this stage.'
    }
    return
}

function Show-StageResult([int]$Horizon, [string]$Geometry) {
    $tag = "trajectory_rgb_jockey_gop0_eps4_seed42_b${Horizon}_p1_a1_h" + $Geometry.Replace('x','w')
    $tag += '_grid128x224_m23GiB_15GiB_vaecpu'
    $root = '.\exp_flf2v\results_720p'
    $report = Join-Path $root "${tag}_assets\generation.json"
    $summary = Join-Path $root "${tag}_summary.csv"
    if (Test-Path -LiteralPath $summary -PathType Leaf) {
        Write-Host "[OVERNIGHT DONE] b$Horizon $Geometry completed two full 720p codec runs: $summary"
    } else {
        $gen = Get-Content -LiteralPath $report -Raw | ConvertFrom-Json
        if ($gen.status -ne 'completed' -or $gen.proxy_improved) {
            throw "Expected 720p summary missing: $summary"
        }
        Write-Warning "[OVERNIGHT NO PROXY GAIN] b$Horizon $Geometry produced no accepted update; no 720p run. $report"
    }
}

Push-Location (Split-Path -Parent $PSScriptRoot)
$transcribing = $false
try {
    Start-Transcript -Path $LogPath -Append | Out-Host
    $transcribing = $true
    $h = 192
    $w = 336
    Write-Host '[OVERNIGHT START] b6 192x336'
    try {
        & $runner @common -BackwardSteps 6 -ProxyHeight $h -ProxyWidth $w
    } catch [System.OutOfMemoryException] {
        # Only switch geometry for the explicitly classified generation failure.
        if ($_.Exception.Message -notlike 'Candidate CUDA OOM:*') { throw }
        Write-Warning '192x336 candidate generation ran out of CUDA memory. Retrying once at 160x288.'
        $h = 160
        $w = 288
        & $runner @common -BackwardSteps 6 -ProxyHeight $h -ProxyWidth $w
    }
    $size = "${h}x${w}"
    Show-StageResult 6 $size

    if ($IncludeFullBackward) {
        Write-Host "[OVERNIGHT START] b20 $size"
        try {
            & $runner @common -BackwardSteps 20 -ProxyHeight $h -ProxyWidth $w
            Show-StageResult 20 $size
        } catch [System.OutOfMemoryException] {
            if ($_.Exception.Message -notlike 'Candidate CUDA OOM:*') { throw }
            Write-Warning "[OVERNIGHT FULL BACKWARD OOM] b20 $size; b6 results retained. $($_.Exception.Message)"
        }
    }
} finally {
    try {
        if ($transcribing) { Stop-Transcript | Out-Host }
    } finally {
        Pop-Location
    }
}
