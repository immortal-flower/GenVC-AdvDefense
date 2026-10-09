param(
    [string]$CondaExe = 'D:\anaconda\Scripts\conda.exe',
    [string]$Data = 'D:\yzb and lmk\dataset-720P\UVG\Jockey_720p.yuv',
    [string]$InitialAttack = 'exp_flf2v/attack_assets/jockey_vae_temporal_pgd_eps2.npz',
    [switch]$GenerateOnly
)
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:CUDA_VISIBLE_DEVICES = '0,1'
if((-not (Test-Path -LiteralPath $InitialAttack)) -and
   ($InitialAttack -eq 'exp_flf2v/attack_assets/jockey_vae_temporal_pgd_eps2.npz')){
    $fallback='exp_flf2v/attack_assets/jockey_vae_temporal_eps2.npz'
    if(Test-Path -LiteralPath $fallback){$InitialAttack=$fallback}
}
if(-not (Test-Path -LiteralPath $InitialAttack)){
    throw "Initial VAE perturbation not found: $InitialAttack"
}
$asset='exp_flf2v/attack_assets/jockey_joint_vae_wan_multitime_eps2.npz'
$generateArgs=@('run','--no-capture-output','-n','GVCC-5090','python',
    'exp_flf2v/generate_wan_boundary_pgd.py','--data',$Data,'--output',$asset,
    '--initial_attack_file',$InitialAttack,'--frames','33','--height','128','--width','224',
    '--times','0.25','0.5','0.75','--steps','8','--epsilon','2','--alpha','0.5')
& $CondaExe @generateArgs
if($LASTEXITCODE -ne 0){throw 'Joint VAE + Wan perturbation generation failed.'}
if($GenerateOnly){exit 0}
$experimentArgs=@('run','--no-capture-output','-n','GVCC-5090','python',
    'exp_flf2v/run_flf2v_experiment.py','--wan_ckpt','exp_flf2v/Wan2.1-FLF2V-14B-720P',
    '--data_dir',$Data,'--output_dir','exp_flf2v/results_720p','--num_frames_per_gop','33',
    '--num_gops','1','--height','720','--width','1280','--M','64','--K','16384',
    '--steps','20','--ddim_tail','3','--g_scale','3.0','--ref_codec','compressai',
    '--ref_quality','4','--seed','42','--sequences','Jockey','--attack','vae-pgd',
    '--attack_file',$asset,'--epsilon','2','--attack_scope','all',
    '--defense','none','--run_name','joint_vae_wan_multitime_eps2')
& $CondaExe @experimentArgs
if($LASTEXITCODE -ne 0){throw 'Joint VAE + Wan GVCC evaluation failed.'}
