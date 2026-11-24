param([string]$Python='C:\Users\Admin\.conda\envs\GVCC-5090\python.exe')
$ErrorActionPreference='Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:CUDA_VISIBLE_DEVICES='0,1'
$root='.\exp_flf2v\results_720p'
$clean="$root\input_pilot_v2_eps2_seed42_steps20_gop0_matched_Bosphorus_clean\Bosphorus\gop0"
Start-Transcript '.\rank_budget_quick_bosphorus.log' -Append
try {
    & $Python '.\exp_flf2v\run_combinatorial_rank_validation_batch.py' `
        --plan '.\exp_flf2v\validation_rank_bosphorus_quick.csv' `
        --data 'D:\yzb and lmk\dataset-720P\UVG\Bosphorus_720p.yuv' `
        --sequence Bosphorus --wan_ckpt '.\exp_flf2v\Wan2.1-FLF2V-14B-720P' `
        --output_dir "$root\combination_bosphorus_budget_quick" `
        --clean_codebook "$clean\codebook.tdcm" --clean_metrics "$clean\metrics.json" `
        --attack_mode rank-bitflip --attack_counts 2 4 8 `
        --gop_index 0 --rank_regions high --attack_seeds 42 123 2026
    if($LASTEXITCODE -ne 0){throw 'Rank budget experiment failed'}
    Get-ChildItem "$root\combination_bosphorus_budget_quick" -Filter '*summary*.csv' |
        ForEach-Object {Import-Csv $_.FullName} | Format-Table -AutoSize
} finally {Stop-Transcript}
