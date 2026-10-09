param(
    [string]$CondaExe = 'D:\anaconda\Scripts\conda.exe',
    [string]$Data = 'D:\yzb and lmk\dataset-720P\UVG\Jockey_720p.yuv',
    [int]$OptHeight = 192,
    [int]$OptWidth = 336,
    [int]$AttackSteps = 20,
    [switch]$GenerateOnly
)
$ErrorActionPreference = 'Continue'
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:CUDA_VISIBLE_DEVICES = '0,1'
$runName = 'vae_fullcontext_latent_eps2'
$asset = "exp_flf2v/attack_assets/jockey_${runName}.npz"
$generateArgs = @('run','--no-capture-output','-n','GVCC-5090','python',
    'exp_flf2v/generate_vae_fullcontext_pgd.py','--data',$Data,
    '--vae','exp_flf2v/Wan2.1-FLF2V-14B-720P/Wan2.1_VAE.pth',
    '--output',$asset,'--epsilon','2','--alpha','0.25','--anchors','1',
    '--opt_height',"$OptHeight",'--opt_width',"$OptWidth",'--steps',"$AttackSteps")
& $CondaExe @generateArgs
if ($LASTEXITCODE -ne 0) { throw 'Full-context attack generation failed.' }
if ($GenerateOnly) { exit 0 }
$experimentArgs = @('run','--no-capture-output','-n','GVCC-5090','python',
    'exp_flf2v/run_flf2v_experiment.py','--wan_ckpt','exp_flf2v/Wan2.1-FLF2V-14B-720P',
    '--data_dir',$Data,'--output_dir','exp_flf2v/results_720p','--num_frames_per_gop','33',
    '--num_gops','1','--height','720','--width','1280','--M','64','--K','16384',
    '--steps','20','--ddim_tail','3','--g_scale','3.0','--ref_codec','compressai',
    '--ref_quality','4','--seed','42','--sequences','Jockey','--attack','vae-pgd',
    '--attack_file',$asset,'--epsilon','2','--defense','none','--run_name',$runName)
& $CondaExe @experimentArgs
if ($LASTEXITCODE -ne 0) { throw 'Full-context GVCC evaluation failed.' }
