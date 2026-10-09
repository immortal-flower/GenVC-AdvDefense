"""Attack the VAE latent of an entire 33-frame GOP at reduced resolution.

Optimize low-resolution spatial controls, lift them to native-resolution RGB,
then downsample the actual perturbed frames for differentiable VAE encoding.
This preserves temporal context but still tests cross-resolution transfer.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from generate_vae_pgd import load_yuv420_frames, resize_frames, WanVAE
from generate_vae_temporal_pgd import temporal_weights


def resize_video(x, size):
    return F.interpolate(x.permute(1,0,2,3),size=size,mode='bilinear',
                         align_corners=False,antialias=True).permute(1,0,2,3)


def lift_controls(controls, weights, size):
    spatial=F.interpolate(controls,size=size,mode='bilinear',align_corners=False)
    return torch.einsum('fa,achw->cfhw',weights,spatial)


@torch.enable_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True);p.add_argument('--vae',required=True)
    p.add_argument('--output',default='exp_flf2v/attack_assets/jockey_vae_fullcontext_latent_eps2.npz')
    p.add_argument('--height',type=int,default=720);p.add_argument('--width',type=int,default=1280)
    p.add_argument('--opt_height',type=int,default=192);p.add_argument('--opt_width',type=int,default=336)
    p.add_argument('--frames',type=int,default=33);p.add_argument('--anchors',type=int,default=1)
    p.add_argument('--steps',type=int,default=20);p.add_argument('--epsilon',type=float,default=2)
    p.add_argument('--alpha',type=float,default=0.25);p.add_argument('--momentum',type=float,default=0.8)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--eval_every',type=int,default=5)
    a=p.parse_args()
    if not (a.frames>=5 and (a.frames-1)%4==0 and 1<=a.anchors<=a.frames and a.steps>0
            and min(a.height,a.width,a.opt_height,a.opt_width)>0
            and a.opt_height%16==0 and a.opt_width%16==0 and a.epsilon>0 and a.alpha>0
            and 0<=a.momentum<1 and a.eval_every>0):
        p.error('Invalid dimensions, optimizer parameters or 4n+1 frame count')
    torch.manual_seed(a.seed)
    frames=resize_frames(load_yuv420_frames(a.data,a.frames),a.width,a.height)
    if len(frames)!=a.frames:raise ValueError('Not enough frames')
    device=torch.device('cuda:0');eps=a.epsilon/255;alpha=a.alpha/255
    original=np.stack([np.asarray(f) for f in frames]).copy()
    target=torch.from_numpy(original).permute(3,0,1,2).float().to(device)/255
    weights=torch.from_numpy(temporal_weights(a.frames,a.anchors)).to(device)
    controls=torch.empty(a.anchors,3,a.opt_height,a.opt_width,device=device).uniform_(-eps,eps)
    initial=controls.clone();velocity=torch.zeros_like(controls)
    print(f'Full temporal context: {a.frames} frames; VAE resolution {a.opt_height}x{a.opt_width}; anchors={a.anchors}',flush=True)
    vae=WanVAE(vae_pth=a.vae,dtype=torch.bfloat16,device=device)
    with torch.no_grad():
        clean_small=resize_video(target,(a.opt_height,a.opt_width))
        z0=vae.encode([clean_small*2-1])[0]
        r0=(vae.decode([z0])[0]+1)/2
    evaluations=[];history=[]
    @torch.no_grad()
    def evaluate(step):
        rows={}
        for name,c in [('random_initial',initial),('optimized',controls)]:
            delta=lift_controls(c,weights,(a.height,a.width))
            # Exact native-resolution quantization before VAE downsampling.
            pixels=((target+delta).clamp(0,1)*255).round()/255
            small=resize_video(pixels,(a.opt_height,a.opt_width))
            z=vae.encode([small*2-1])[0];r=(vae.decode([z])[0]+1)/2
            rows[name]=dict(latent_shift_MSE=float((z-z0).square().mean()),
                            lowres_reconstruction_shift_MSE=float((r-r0).square().mean()),
                            lowres_reconstruction_MSE=float((r-clean_small).square().mean()),
                            native_input_PSNR_dB=float(-10*torch.log10((pixels-target).square().mean().clamp_min(1e-12))))
        evaluations.append(dict(step=step,variants=rows))
        print(f'Full-GOP quantized evaluation {step}: {json.dumps(rows)}',flush=True)
    evaluate(0)
    for step in range(a.steps):
        c=controls.detach().requires_grad_(True)
        delta=lift_controls(c,weights,(a.height,a.width))
        small=resize_video((target+delta).clamp(0,1),(a.opt_height,a.opt_width))
        z=vae.encode([small*2-1])[0]
        loss=(z-z0).square().mean()
        grad,=torch.autograd.grad(loss,c)
        if not torch.isfinite(grad).all() or grad.abs().max().item()==0:
            raise RuntimeError('Invalid/zero full-context VAE gradient')
        with torch.no_grad():
            normalized=grad/grad.abs().mean(dim=(1,2,3),keepdim=True).clamp_min(1e-12)
            velocity=a.momentum*velocity+normalized
            controls=(c+alpha*velocity.sign()).clamp(-eps,eps).detach()
        history.append(dict(step=step+1,latent_shift_MSE=float(loss.detach())))
        print(f'Full-context PGD {step+1}/{a.steps}: loss={history[-1]["latent_shift_MSE"]:.6f}',flush=True)
        del c,delta,small,z,loss,grad,normalized
        if (step+1)%a.eval_every==0 or step+1==a.steps:evaluate(step+1)
    with torch.no_grad():delta=lift_controls(controls,weights,(a.height,a.width)).permute(1,2,3,0).cpu().numpy()
    pixels=np.floor(np.clip(original.astype(np.float32)/255+delta,0,1)*255+0.5).astype(np.uint8)
    actual=(pixels.astype(np.float32)-original.astype(np.float32))/255
    report=vars(a)|dict(scope='full temporal-context, reduced-resolution VAE latent surrogate; cross-resolution transfer',
        input_PSNR_dB=float(-10*np.log10(max(float(np.mean(actual**2)),1e-12))),
        measured_Linf_pixel=float(np.abs(actual).max()*255),history=history,fixed_evaluations=evaluations)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    # Shared mode uses the existing compact HWC file format.
    np.savez_compressed(out,delta=delta[0] if a.anchors==1 else delta)
    previews=out.parent/(out.stem+'_input_png');previews.mkdir(parents=True,exist_ok=True)
    for f in [0,a.frames//2,a.frames-1]:
        Image.fromarray(original[f]).save(previews/f'frame{f:02d}_original.png')
        Image.fromarray(pixels[f]).save(previews/f'frame{f:02d}_attacked.png')
    report['lossless_input_png_dir']=str(previews)
    out.with_suffix('.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(f'Saved {out}; input PSNR={report["input_PSNR_dB"]:.2f} dB',flush=True)


if __name__=='__main__':main()
