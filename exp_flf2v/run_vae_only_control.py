"""Native-resolution VAE-only clean/attack control, without loading Wan DiT.

Metrics use in-memory 8-bit reconstructions with the same conversion as the
GVCC wrapper; saved MP4 files are viewing copies only.
"""
import argparse
import gc
import json
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from generate_vae_pgd import load_yuv420_frames, resize_frames, WanVAE
from attacks import apply_attack
from run_flf2v_experiment import frames_to_tensor, compute_psnr, compute_lpips, compute_msssim, save_video_mp4


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True);p.add_argument('--vae',required=True)
    p.add_argument('--attack_file',default='exp_flf2v/attack_assets/jockey_vae_pgd_grid_deviation_eps2.npz')
    p.add_argument('--output',default='exp_flf2v/results_720p/vae_only_grid_eps2/Jockey')
    p.add_argument('--height',type=int,default=720);p.add_argument('--width',type=int,default=1280)
    p.add_argument('--frames',type=int,default=33);p.add_argument('--epsilon',type=float,default=2)
    p.add_argument('--dtype',choices=['float32','bfloat16'],default='float32',
                   help='float32 matches the VAE constructed by the current GVCC wrapper')
    p.add_argument('--gvcc_clean',default='exp_flf2v/results_720p/clean_smoke/Jockey/gop0/metrics.json')
    p.add_argument('--gvcc_attack',default='exp_flf2v/results_720p/vae_grid_eps2_target_only/Jockey/gop0/metrics.json')
    a=p.parse_args()
    if a.frames<5 or (a.frames-1)%4 or a.height%8 or a.width%8:
        p.error('Use 4n+1 frames and spatial dimensions divisible by 8')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    original=resize_frames(load_yuv420_frames(a.data,a.frames),a.width,a.height)
    if len(original)!=a.frames:raise ValueError('Not enough frames')
    attacked,metadata=apply_attack(original,'vae-pgd',epsilon=a.epsilon,attack_file=a.attack_file)
    dtype=getattr(torch,a.dtype)
    print(f'Loading VAE only: native {a.frames}x{a.height}x{a.width}, dtype={a.dtype}',flush=True)
    vae=WanVAE(vae_pth=a.vae,device='cuda:0',dtype=dtype)
    latents={};reconstructions={};timings={}
    for name,frames in [('clean',original),('attacked',attacked)]:
        print(f'VAE {name}: encode...',flush=True);start=time.time()
        pixels=frames_to_tensor(frames).permute(1,0,2,3).to('cuda:0')*2-1
        latent=vae.encode([pixels])[0]
        latents[name]=latent.cpu()
        del pixels
        print(f'VAE {name}: decode...',flush=True)
        video=(vae.decode([latent.float()])[0]/2+0.5).clamp(0,1).permute(1,2,3,0).cpu().numpy()
        reconstructions[name]=[Image.fromarray((f*255).astype(np.uint8)) for f in video]
        timings[name]=time.time()-start
        del latent,video
        gc.collect();torch.cuda.empty_cache()
    del vae;gc.collect();torch.cuda.empty_cache()
    target=frames_to_tensor(original)
    results={}
    for name in ['clean','attacked']:
        frames=reconstructions[name];rec=frames_to_tensor(frames)
        psnr,perframe=compute_psnr(target,rec)
        metric_device='cuda:1' if torch.cuda.device_count()>1 else 'cuda:0'
        results[name]=dict(PSNR_dB=psnr,LPIPS=compute_lpips(target,rec,device=metric_device),
                           MS_SSIM=compute_msssim(target,rec),per_frame_PSNR_dB=perframe.tolist(),
                           vae_seconds=timings[name])
        save_video_mp4(frames,out/f'{name}_vae_reconstructed.mp4')
        for f in [0,a.frames//2,a.frames-1]:frames[f].save(out/f'frame{f:02d}_{name}_vae_reconstructed.png')
        print(f'VAE {name}: {json.dumps(results[name])}',flush=True)
        del rec
    attack_tensor=frames_to_tensor(attacked)
    input_psnr,_=compute_psnr(target,attack_tensor)
    clean_rec=frames_to_tensor(reconstructions['clean']);adv_rec=frames_to_tensor(reconstructions['attacked'])
    shift_psnr,_=compute_psnr(clean_rec,adv_rec)
    report=vars(a)|dict(attack_metadata=metadata,metrics=results,input_PSNR_dB=input_psnr,
        native_latent_shift_MSE=float((latents['attacked']-latents['clean']).square().mean()),
        reconstruction_shift_PSNR_dB=shift_psnr,
        reconstruction_shift_MSE=float((adv_rec-clean_rec).square().mean()),
        attack_effect=dict(PSNR_drop_dB=results['clean']['PSNR_dB']-results['attacked']['PSNR_dB'],
                           LPIPS_increase=results['attacked']['LPIPS']-results['clean']['LPIPS']),
        note='VAE-only diagnostic: no codebook, Wan DiT or boundary codec; no comparable bitrate is assigned.')
    gvcc={}
    for name,file in [('clean',a.gvcc_clean),('attacked',a.gvcc_attack)]:
        if Path(file).exists():gvcc[name]=json.loads(Path(file).read_text(encoding='utf-8'))
    if len(gvcc)==2:
        report['gvcc_reference']=dict(metrics=gvcc,PSNR_drop_dB=gvcc['clean']['PSNR_dB']-gvcc['attacked']['PSNR_dB'],
                                      LPIPS_increase=gvcc['attacked']['LPIPS']-gvcc['clean']['LPIPS'])
    (out/'vae_only_metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(f'Saved {out / "vae_only_metrics.json"}; effect={report["attack_effect"]}',flush=True)


if __name__=='__main__':main()
