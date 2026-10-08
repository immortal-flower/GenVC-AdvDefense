"""Overlapping crop gradient accumulation with smooth temporal controls.

One optimizer update follows a complete spatial sweep. VAE parameters remain
frozen. This is a single-video VAE surrogate attack, not full GVCC/UAP training.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from generate_vae_pgd import load_yuv420_frames, resize_frames, WanVAE, reconstruct


def overlapping_starts(length, crop, stride):
    result = list(range(0, length-crop+1, stride))
    if result[-1] != length-crop:
        result.append(length-crop)
    return result


def temporal_weights(frames, anchors):
    positions = np.linspace(0, anchors-1, frames)
    weights = np.zeros((frames, anchors), dtype=np.float32)
    for f, position in enumerate(positions):
        left = int(np.floor(position)); right = min(left+1, anchors-1)
        fraction = position-left
        weights[f, left] += 1-fraction
        weights[f, right] += fraction
    return weights


@torch.enable_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True); p.add_argument('--vae', required=True)
    p.add_argument('--output', default='exp_flf2v/attack_assets/jockey_vae_temporal_eps2.npz')
    p.add_argument('--height', type=int, default=720); p.add_argument('--width', type=int, default=1280)
    p.add_argument('--frames', type=int, default=33); p.add_argument('--anchors', type=int, default=3)
    p.add_argument('--crop', type=int, default=192); p.add_argument('--stride', type=int, default=144)
    p.add_argument('--rounds', type=int, default=8); p.add_argument('--alpha', type=float, default=0.5)
    p.add_argument('--epsilon', type=float, default=2); p.add_argument('--momentum', type=float, default=0.8)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    if not (a.frames >= 5 and 2 <= a.anchors <= a.frames and 0 < a.crop <= min(a.height,a.width)
            and a.crop % 16 == 0 and 0 < a.stride <= a.crop and a.rounds >= 1
            and a.epsilon > 0 and a.alpha > 0 and 0 <= a.momentum < 1):
        p.error('Invalid dimensions, budget, rounds or momentum')
    torch.manual_seed(a.seed)
    frames = resize_frames(load_yuv420_frames(a.data,a.frames),a.width,a.height)
    if len(frames) != a.frames: raise ValueError('Not enough frames')
    clean = torch.from_numpy(np.stack([np.asarray(f) for f in frames]).copy()).permute(3,0,1,2).float()/255
    eps = a.epsilon/255; alpha = a.alpha/255
    weights = torch.from_numpy(temporal_weights(a.frames,a.anchors))
    controls = torch.empty(a.anchors,3,a.height,a.width).uniform_(-eps,eps)
    initial = controls.clone(); velocity = torch.zeros_like(controls)
    tiles = [(y,x) for y in overlapping_starts(a.height,a.crop,a.stride)
             for x in overlapping_starts(a.width,a.crop,a.stride)]
    device = torch.device('cuda:0')
    print(f'Loading frozen VAE; {len(tiles)} overlapping crops per round, {a.rounds} rounds',flush=True)
    vae = WanVAE(vae_pth=a.vae,dtype=torch.bfloat16,device=device)
    specs = [(0,0,0),((a.frames-5)//2,(a.height-a.crop)//2,(a.width-a.crop)//2),
             (a.frames-5,a.height-a.crop,a.width-a.crop)]
    evaluations = []
    @torch.no_grad()
    def evaluate(round_id):
        values = {n:[] for n in ['clean','random_initial','optimized']}
        for t,y,x in specs:
            target=clean[:,t:t+5,y:y+a.crop,x:x+a.crop].to(device)
            z0,r0=reconstruct(vae,target)
            values['clean'].append(dict(reconstruction_MSE=float((r0-target).square().mean()),shift_MSE=0.,latent_shift_MSE=0.))
            for name,c in [('random_initial',initial),('optimized',controls)]:
                d=torch.einsum('fa,achw->cfhw',weights[t:t+5].to(device),c[:,:,y:y+a.crop,x:x+a.crop].to(device))
                adv=((target+d).clamp(0,1)*255).round()/255
                z,r=reconstruct(vae,adv)
                values[name].append(dict(reconstruction_MSE=float((r-target).square().mean()),
                                         shift_MSE=float((r-r0).square().mean()),latent_shift_MSE=float((z-z0).square().mean())))
        means={name:{k:float(np.mean([row[k] for row in rows])) for k in rows[0]} for name,rows in values.items()}
        evaluations.append(dict(round=round_id,means=means,crops=values))
        print(f'Fixed quantized evaluation round {round_id}: {json.dumps(means)}',flush=True)
    evaluate(0)
    history=[]
    for round_id in range(a.rounds):
        gradients=torch.zeros_like(controls); counts=torch.zeros(a.anchors,1,a.height,a.width)
        losses=[]
        # Each round samples a different temporal window at each tile. The
        # gradient is accumulated before any control is changed.
        for tile_id,(y,x) in enumerate(tiles):
            t=(tile_id*7+round_id*5) % (a.frames-4)
            target=clean[:,t:t+5,y:y+a.crop,x:x+a.crop].to(device)
            with torch.no_grad(): _,r0=reconstruct(vae,target)
            c=controls[:,:,y:y+a.crop,x:x+a.crop].to(device).detach().requires_grad_(True)
            w=weights[t:t+5].to(device)
            d=torch.einsum('fa,achw->cfhw',w,c)
            _,r=reconstruct(vae,(target+d).clamp(0,1))
            loss=(r-r0).square().mean()
            grad,=torch.autograd.grad(loss,c)
            if not torch.isfinite(grad).all() or grad.abs().max().item()==0:
                raise RuntimeError('Invalid/zero VAE input gradient')
            gradients[:,:,y:y+a.crop,x:x+a.crop]+=grad.detach().cpu()
            counts[:,:,y:y+a.crop,x:x+a.crop]+=(w.sum(0).cpu()/5)[:,None,None,None]
            losses.append(float(loss.detach()))
            del target,r0,c,w,d,r,loss,grad
            if (tile_id+1)%10==0 or tile_id+1==len(tiles):
                print(f'Round {round_id+1}/{a.rounds}: crops {tile_id+1}/{len(tiles)}',flush=True)
        gradients/=counts.clamp_min(1e-8)
        gradients/=gradients.abs().mean(dim=(1,2,3),keepdim=True).clamp_min(1e-12)
        velocity=a.momentum*velocity+gradients
        controls=(controls+alpha*velocity.sign()).clamp(-eps,eps)
        history.append(dict(round=round_id+1,mean_crop_loss=float(np.mean(losses)),
                            anchor_spatial_coverage=float((counts>0).float().mean())))
        evaluate(round_id+1)
    delta=torch.einsum('fa,achw->fchw',weights,controls).permute(0,2,3,1).numpy()
    original=clean.permute(1,2,3,0).numpy()
    pixels=np.floor(np.clip(original+delta,0,1)*255+0.5)/255
    actual=pixels-original; mse=float(np.mean(actual**2))
    report=vars(a)|dict(scope='single-video temporal VAE surrogate attack',fixed_specs=specs,
        input_PSNR_dB=float(-10*np.log10(max(mse,1e-12))),measured_Linf_pixel=float(np.abs(actual).max()*255),
        delta_adjacent_RMS_pixel=float(np.sqrt(np.mean(np.diff(delta,axis=0)**2))*255),
        delta_adjacent_max_pixel=float(np.abs(np.diff(delta,axis=0)).max()*255),
        quantized_delta_adjacent_RMS_pixel=float(np.sqrt(np.mean(np.diff(actual,axis=0)**2))*255),
        history=history,fixed_evaluations=evaluations)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,delta=delta)
    # Lossless pre-MP4 inputs for judging visibility without codec artifacts.
    preview_dir=out.parent/(out.stem+'_input_png')
    preview_dir.mkdir(parents=True,exist_ok=True)
    for f in [0,a.frames//2,a.frames-1]:
        Image.fromarray(np.rint(original[f]*255).astype(np.uint8)).save(preview_dir/f'frame{f:02d}_original.png')
        Image.fromarray(np.rint(pixels[f]*255).astype(np.uint8)).save(preview_dir/f'frame{f:02d}_attacked.png')
    report['lossless_input_png_dir']=str(preview_dir)
    out.with_suffix('.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(f'Saved {out}; input PSNR={report["input_PSNR_dB"]:.2f} dB',flush=True)


if __name__=='__main__': main()
