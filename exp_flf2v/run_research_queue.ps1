param(
    [string]$Python = 'C:\Users\Admin\.conda\envs\GVCC-5090\python.exe',
    [string]$DataRoot = 'D:\yzb and lmk\dataset-720P\UVG'
)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:CUDA_VISIBLE_DEVICES = '0,1'
$root = '.\exp_flf2v\results_720p'
$ckpt = '.\exp_flf2v\Wan2.1-FLF2V-14B-720P'
$source = "$root\trajectory_rgb_jockey_gop0_eps4_seed42_b20_p1_a1_h192w336_grid128x224_m20GiB_18GiB_vaecpu_stepvjp_assets"
$bclean = "$root\input_pilot_v2_eps2_seed42_steps20_gop0_matched_Bosphorus_clean\Bosphorus\gop0"
$jobs = @(
    @{Name='hard-direction'; Args=@('exp_flf2v/run_hard_direction_probe.py','--source',$source,'--random_directions','2')},
    @{Name='vae-r44-heldout'; Args=@('exp_flf2v/validate_vae_warmstart_seeds.py','--data',"$DataRoot\Jockey_720p.yuv",'--wan_ckpt',$ckpt,'--results_root',$root,'--source_tag','vae_source_rgb_eps4_r36_jockey_gop0_v1','--source_round','44','--tag','vae_r44_heldout_v1')}
)
foreach ($seed in @(42,123,2026)) {
    $jobs += @{Name="combination-cpu-$seed"; Args=@('exp_flf2v/analyze_combinatorial_index_robustness.py','--bitstream',"$bclean\codebook.tdcm",'--samples','999','--bits_per_region','16','--seed',"$seed",'--output_prefix',"$root\bosphorus_combination_seed$seed")}
}
foreach ($mode in @('rank-bitflip','index-bitflip')) {
    $regions = if ($mode -eq 'rank-bitflip') { @('low','middle','high') } else { @('low') }
    $jobs += @{Name="combination-gpu-$mode"; Args=@('exp_flf2v/run_combinatorial_rank_validation_batch.py','--plan','exp_flf2v/validation_rank_cells_jockey.csv','--data',"$DataRoot\Bosphorus_720p.yuv",'--sequence','Bosphorus','--wan_ckpt',$ckpt,'--output_dir',"$root\combination_bosphorus_validation",'--clean_codebook',"$bclean\codebook.tdcm",'--clean_metrics',"$bclean\metrics.json",'--attack_mode',$mode,'--attack_count','1','--gop_index','0','--rank_regions') + $regions + @('--attack_seeds','42','123','2026')}
}
Start-Transcript '.\research_queue.log' -Append
$status = @()
try {
    foreach ($job in $jobs) {
        Write-Host "[START] $($job.Name)" -ForegroundColor Cyan
        $arguments = $job.Args
        try {
            & $Python @arguments
            if ($LASTEXITCODE -ne 0) { throw "Exit code $LASTEXITCODE" }
            $status += [pscustomobject]@{Job=$job.Name;Status='complete';Error=''}
        } catch {
            $status += [pscustomobject]@{Job=$job.Name;Status='failed';Error=$_.Exception.Message}
            Write-Warning "$($job.Name) failed; continuing independent jobs."
        }
        $status | Export-Csv "$root\research_queue_status.csv" -NoTypeInformation -Encoding UTF8
    }
    $status | Format-Table -AutoSize
} finally { Stop-Transcript }
