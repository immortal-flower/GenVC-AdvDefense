param(
    [string]$CondaExe = 'D:\anaconda\Scripts\conda.exe',
    [string]$Data = 'D:\yzb and lmk\dataset-720P\UVG\Jockey_720p.yuv',
    [switch]$GenerateOnly
)
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:CUDA_VISIBLE_DEVICES = '0,1'
$asset='exp_flf2v/attack_assets/jockey_wan_boundary_eps2.npz'
$generateArgs=@('run','--no-capture-output','-n','GVCC-5090','python',
    'exp_flf2v/generate_wan_boundary_pgd.py','--data',$Data,'--output',$asset)
& $CondaExe @generateArgs
if($LASTEXITCODE -ne 0){throw 'Wan boundary pixel-gradient generation failed.'}
if($GenerateOnly){exit 0}
$experimentArgs=@('run','--no-capture-output','-n','GVCC-5090','python',
    'exp_flf2v/run_flf2v_experiment.py','--wan_ckpt','exp_flf2v/Wan2.1-FLF2V-14B-720P',
    '--data_dir',$Data,'--output_dir','exp_flf2v/results_720p','--num_frames_per_gop','33',
    '--num_gops','1','--height','720','--width','1280','--M','64','--K','16384',
    '--steps','20','--ddim_tail','3','--g_scale','3.0','--ref_codec','compressai',
    '--ref_quality','4','--seed','42','--sequences','Jockey','--attack','vae-pgd',
    '--attack_file',$asset,'--epsilon','2','--attack_scope','condition-only',
    '--defense','none','--run_name','wan_boundary_pgd_eps2_condition_only')
& $CondaExe @experimentArgs
if($LASTEXITCODE -ne 0){throw 'Wan boundary GVCC evaluation failed.'}
