param(
    [string]$LogPath = '.\trajectory_rgb_full_chain_overnight.log',
    [switch]$PlanOnly
)

$ErrorActionPreference = 'Stop'
$runner = Join-Path $PSScriptRoot 'run_trajectory_input_pilot.ps1'
$common = @{
    Objective = 'final-rgb'
    Epsilon = 4
    Alpha = 1
    BackwardSteps = 6
    Iterations = 1
    Gpu0Memory = '23GiB'
    Gpu1Memory = '15GiB'
    VaeCpuActivations = $true
    GridRandom = $true
    PairedOnly = $true
}
if ($PlanOnly) {
    Write-Host 'Jockey GOP0: 192x336 RGB candidate, matched grid random, then two full 720p codec runs.'
    Write-Host 'One retry at 160x288 only if candidate generation reports CUDA OOM.'
    return
}

Push-Location (Split-Path -Parent $PSScriptRoot)
$transcribing = $false
try {
    Start-Transcript -Path $LogPath -Append | Out-Host
    $transcribing = $true
    $size = '192x336'
    try {
        & $runner @common -ProxyHeight 192 -ProxyWidth 336
    } catch [System.OutOfMemoryException] {
        # Only switch geometry for the explicitly classified generation failure.
        if ($_.Exception.Message -notlike 'Candidate CUDA OOM:*') { throw }
        Write-Warning '192x336 candidate generation ran out of CUDA memory. Retrying once at 160x288.'
        $size = '160x288'
        & $runner @common -ProxyHeight 160 -ProxyWidth 288
    }

    $tag = 'trajectory_rgb_jockey_gop0_eps4_seed42_b6_p1_a1_h' + $size.Replace('x','w')
    $tag += '_grid128x224_m23GiB_15GiB_vaecpu'
    $root = '.\exp_flf2v\results_720p'
    $report = Join-Path $root "${tag}_assets\generation.json"
    $summary = Join-Path $root "${tag}_summary.csv"
    if (Test-Path -LiteralPath $summary -PathType Leaf) {
        Write-Host "[OVERNIGHT DONE] $size completed two full 720p codec runs: $summary"
    } else {
        $gen = Get-Content -LiteralPath $report -Raw | ConvertFrom-Json
        if ($gen.status -ne 'completed' -or $gen.proxy_improved) {
            throw "Expected 720p summary missing: $summary"
        }
        Write-Warning "[OVERNIGHT NO PROXY GAIN] $size produced no accepted update; no 720p run. $report"
    }
} finally {
    try {
        if ($transcribing) { Stop-Transcript | Out-Host }
    } finally {
        Pop-Location
    }
}
