param(
    [string]$Python = 'C:\Users\Admin\.conda\envs\GVCC-5090\python.exe',
    [string]$DataRoot = 'D:\data\UVG',
    [string]$Checkpoint = '.\exp_flf2v\Wan2.1-FLF2V-14B-720P',
    [string]$OutputRoot = '.\exp_flf2v\results_720p',
    [string]$RunName = 'flf2v_combinatorial_rate_uvg'
)

$ErrorActionPreference = 'Stop'
$runRoot = Join-Path $OutputRoot $RunName
New-Item -ItemType Directory -Force -Path $runRoot | Out-Null
$transcript = Join-Path $runRoot 'console.log'
$exitCode = 1

foreach ($path in @($Python, $DataRoot, $Checkpoint)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Missing required path: $path"
    }
}

$env:CUDA_VISIBLE_DEVICES = '0,1'
Start-Transcript -LiteralPath $transcript -Append
try {
    $commit = git rev-parse HEAD
    $branch = git branch --show-current
    $gpu = nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
    [ordered]@{
        started_at = (Get-Date).ToString('o')
        git_commit = $commit
        git_branch = $branch
        python = $Python
        data_root = $DataRoot
        checkpoint = $Checkpoint
        cuda_visible_devices = $env:CUDA_VISIBLE_DEVICES
        gpu = @($gpu)
        protocol = [ordered]@{
            sequences = 7
            gops_per_sequence = 3
            frames_per_gop = 33
            resolution = '1280x720'
            M = 64
            K = 16384
            steps = 20
            ddim_tail = 3
            g_scale = 3.0
            ref_codec = 'compressai'
            ref_quality = 4
            seed = 42
        }
    } | ConvertTo-Json -Depth 5 |
        Set-Content -LiteralPath (Join-Path $runRoot 'run_manifest.json') -Encoding UTF8

    & $Python '.\exp_flf2v\run_flf2v_experiment.py' `
        --wan_ckpt $Checkpoint `
        --data_dir $DataRoot `
        --output_dir $OutputRoot `
        --run_name $RunName `
        --num_frames_per_gop 33 `
        --num_gops 3 `
        --height 720 --width 1280 `
        --M 64 --K 16384 `
        --steps 20 --ddim_tail 3 `
        --g_scale 3.0 `
        --ref_codec compressai --ref_quality 4 `
        --seed 42 `
        --attack none --defense none `
        --stream_attack none --stream_defense none
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "FLF2V rate experiment failed with exit code $exitCode"
    }
}
finally {
    & $Python '.\exp_flf2v\summarize_combinatorial_rate.py' `
        --results $runRoot --expected_gops 21
    [ordered]@{
        finished_at = (Get-Date).ToString('o')
        codec_exit_code = $exitCode
        summary_exit_code = $LASTEXITCODE
    } | ConvertTo-Json |
        Set-Content -LiteralPath (Join-Path $runRoot 'process_exit.json') -Encoding UTF8
    Stop-Transcript
}
