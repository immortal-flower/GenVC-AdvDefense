param(
    [string]$CondaExe = 'D:\anaconda\Scripts\conda.exe',
    [string]$Data = 'D:\yzb and lmk\dataset-720P\UVG\Jockey_720p.yuv',
    [string]$AttackFile = 'exp_flf2v/attack_assets/jockey_vae_pgd_grid_deviation_eps2.npz'
)
$ErrorActionPreference = 'Continue'
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:CUDA_VISIBLE_DEVICES = '0,1'
if (-not (Test-Path -LiteralPath $AttackFile)) {
    throw "Missing saved perturbation: $AttackFile. Reuse the existing eps2 grid NPZ; do not regenerate it for this comparison."
}
$common = @('run','--no-capture-output','-n','GVCC-5090','python',
    'exp_flf2v/run_flf2v_experiment.py','--wan_ckpt','exp_flf2v/Wan2.1-FLF2V-14B-720P',
    '--data_dir',$Data,'--output_dir','exp_flf2v/results_720p','--num_frames_per_gop','33',
    '--num_gops','1','--height','720','--width','1280','--M','64','--K','16384',
    '--steps','20','--ddim_tail','3','--g_scale','3.0','--ref_codec','compressai',
    '--ref_quality','4','--seed','42','--sequences','Jockey','--attack','vae-pgd',
    '--attack_file',$AttackFile,'--epsilon','2','--defense','none')
foreach ($scope in @('target-only','condition-only')) {
    $suffix = $scope.Replace('-','_')
    $runName = "vae_grid_eps2_${suffix}"
    Write-Host "Running attribution: $runName"
    & $CondaExe @common --attack_scope $scope --run_name $runName
    if ($LASTEXITCODE -ne 0) { throw "Attribution run failed: $runName" }
}
