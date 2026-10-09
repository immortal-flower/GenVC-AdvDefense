"""Single-step Wan velocity attack on first/last RGB frames.

Two-stage vector-Jacobian products propagate the exact condition gradient
back to pixels without retaining VAE/CLIP and DiT graphs simultaneously.
Reference compression is evaluated afterwards, not differentiated through.
The optimization is a low-resolution RF-state surrogate, not full GVCC PGD.
"""
import argparse
import gc
import json
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from accelerate import dispatch_model, infer_auto_device_map

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from run_flf2v_experiment import load_yuv420_frames, resize_frames, compress_boundary_frame
from sde_rf_wan.wan_flf2v_wrapper import WanFLF2VWrapper
from probe_wan_condition_grad import checkpoint_blocks


def tensor_conditions(model, pixels, frames, branch='both'):
    """RGB [2,3,H,W] in [0,1] -> native FLF2V condition tensors."""
    first,last=(pixels*2-1).unbind(0)
    h,w=first.shape[-2:];lh,lw=h//8,w//8
    clip=model.clip.visual([first[:,None],last[:,None]]) if branch in ['both','clip'] else None
    y=None
    if branch in ['both','vae']:
        video=torch.cat([first[:,None],first.new_zeros(3,frames-2,h,w),last[:,None]],dim=1)
        latent=model.vae.encode([video])[0]
        mask=torch.ones(1,frames,lh,lw,device=pixels.device)
        mask[:,1:-1]=0
        mask=torch.cat([mask[:,:1].repeat_interleave(4,dim=1),mask[:,1:]],dim=1)
        mask=mask.view(1,(frames+3)//4,4,lh,lw).transpose(1,2)[0]
        y=torch.cat([mask,latent],dim=0)
    return dict(y=y,clip_fea=clip)


@torch.enable_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True)
    p.add_argument('--wan_ckpt',default='exp_flf2v/Wan2.1-FLF2V-14B-720P')
    p.add_argument('--output',default='exp_flf2v/attack_assets/jockey_wan_boundary_eps2.npz')
    p.add_argument('--height',type=int,default=128);p.add_argument('--width',type=int,default=224)
    p.add_argument('--source_height',type=int,default=720);p.add_argument('--source_width',type=int,default=1280)
    p.add_argument('--frames',type=int,default=9);p.add_argument('--source_frames',type=int,default=33)
    p.add_argument('--steps',type=int,default=8);p.add_argument('--epsilon',type=float,default=2)
    p.add_argument('--alpha',type=float,default=0.5);p.add_argument('--time',type=float,default=0.5)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--gpu0_memory',default='20GiB');p.add_argument('--gpu1_memory',default='20GiB')
    a=p.parse_args()
    if a.height%16 or a.width%16 or (a.frames-1)%4 or a.frames<5 or a.source_frames<a.frames or a.steps<1 or a.epsilon<=0 or a.alpha<=0 or not 0<a.time<1:
        p.error('Invalid dimensions or optimization parameters')
    if torch.cuda.device_count()<2:raise RuntimeError('Requires two GPUs')
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    report=vars(a)|dict(status='started',scope='boundary pixels, frozen Wan single-step low-resolution surrogate',
                        reference_codec_in_optimization=False,history=[])
    def save():out.with_suffix('.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    save()
    try:
        source=resize_frames(load_yuv420_frames(a.data,a.source_frames),a.source_width,a.source_height)
        if len(source)!=a.source_frames:raise ValueError('Not enough source frames')
        indices=np.linspace(0,a.source_frames-1,a.frames).round().astype(int).tolist()
        report['source_frame_indices']=indices
        small=resize_frames([source[i] for i in indices],a.width,a.height)
        original=torch.from_numpy(np.stack([np.asarray(small[0]),np.asarray(small[-1])]).copy()).permute(0,3,1,2).float()/255
        model=WanFLF2VWrapper(a.wan_ckpt);model.load('cpu',torch.bfloat16)
        model.device=torch.device('cuda:1')
        with torch.no_grad():
            model.text_encoder.model.to('cuda:1')
            embeds=[u.cpu() for u in model.encode_prompt('')['prompt_embeds']]
            model.text_encoder.model.to('cpu')
            model.vae.model.to('cuda:1');model.clip.model.to('cuda:1')
            clean_latent=model.encode_video(small,a.height,a.width).cpu()
            base=tensor_conditions(model,original.to('cuda:1'),a.frames)
            base={k:v.cpu() for k,v in base.items()}
            model.vae.model.to('cpu');model.clip.model.to('cpu')
        gc.collect();torch.cuda.empty_cache()
        checkpoint_blocks(model.model)
        mapping=infer_auto_device_map(
            model.model,
            max_memory={0:a.gpu0_memory,1:a.gpu1_memory},
            no_split_module_classes=['WanAttentionBlock'],
        )
        if any(v not in [0,1,'cuda:0','cuda:1'] for v in mapping.values()):raise RuntimeError(f'Non-GPU DiT mapping: {mapping}')
        model.model=dispatch_model(model.model,device_map=mapping)
        model.device=torch.device('cuda:0')
        model.model.requires_grad_(False)
        report['device_map']={k:str(v) for k,v in mapping.items()}
        model.vae.model.to('cuda:1').requires_grad_(False)
        model.clip.model.to('cuda:1').requires_grad_(False)
        original=original.to('cuda:1')
        embeds=[u.to('cuda:0') for u in embeds]
        gen=torch.Generator(device='cpu').manual_seed(a.seed)
        noise=torch.randn(clean_latent.shape,generator=gen)
        x_t=((1-a.time)*clean_latent+a.time*noise).to('cuda:0')
        with torch.no_grad():
            baseline=model.predict_velocity(x_t,a.time,embeds,{k:v.to('cuda:0') for k,v in base.items()})
        torch.manual_seed(a.seed)
        eps=a.epsilon/255;alpha=a.alpha/255
        delta=torch.empty_like(original).uniform_(-eps,eps)
        initial=delta.clone()
        @torch.no_grad()
        def evaluate(pixels):
            cond=tensor_conditions(model,pixels,a.frames)
            velocity=model.predict_velocity(x_t,a.time,embeds,cond)
            return float((velocity-baseline).square().mean())
        report['random_velocity_shift_MSE']=evaluate(((original+initial).clamp(0,1)*255).round()/255)
        for step in range(a.steps):
            pixels=(original+delta).clamp(0,1)
            with torch.no_grad():cond=tensor_conditions(model,pixels,a.frames)
            y=cond['y'].detach().to('cuda:0').requires_grad_(True)
            clip=cond['clip_fea'].detach().to('cuda:0').requires_grad_(True)
            velocity=model.predict_velocity_with_grad(x_t,a.time,embeds,dict(y=y,clip_fea=clip))
            loss=(velocity-baseline).square().mean()
            gy,gclip=torch.autograd.grad(loss,(y,clip))
            value=float(loss.detach())
            gy=gy.detach().to('cuda:1');gclip=gclip.detach().to('cuda:1')
            del velocity,loss,y,clip,cond
            # Rebuild each smaller encoder graph separately and apply its VJP.
            leaf=pixels.detach().requires_grad_(True)
            vae_cond=tensor_conditions(model,leaf,a.frames,'vae')['y']
            g_vae,=torch.autograd.grad(vae_cond,leaf,grad_outputs=gy)
            del vae_cond,gy
            clip_cond=tensor_conditions(model,leaf,a.frames,'clip')['clip_fea']
            g_clip,=torch.autograd.grad(clip_cond,leaf,grad_outputs=gclip)
            grad=g_vae+g_clip
            if not torch.isfinite(grad).all() or grad.abs().max().item()==0:raise RuntimeError('Invalid/zero RGB input gradient')
            delta=(delta+alpha*grad.sign()).clamp(-eps,eps).detach()
            row=dict(step=step+1,velocity_shift_MSE=value,pixel_grad_max=float(grad.abs().max()),
                     vae_pixel_grad_mean=float(g_vae.abs().mean()),clip_pixel_grad_mean=float(g_clip.abs().mean()))
            report['history'].append(row);save()
            print(f'Boundary PGD {step+1}/{a.steps}: {json.dumps(row)}',flush=True)
            del leaf,clip_cond,gclip,g_vae,g_clip,grad,pixels
        report['optimized_velocity_shift_MSE']=evaluate(((original+delta).clamp(0,1)*255).round()/255)
        with torch.no_grad():lifted=F.interpolate(delta,size=(a.source_height,a.source_width),mode='bilinear',align_corners=False).permute(0,2,3,1).cpu().numpy()
        # Only boundary frames are changed; middle-frame delta is exactly zero.
        full_delta=np.zeros((a.source_frames,a.source_height,a.source_width,3),np.float32)
        full_delta[0]=lifted[0];full_delta[-1]=lifted[1]
        native=np.stack([np.asarray(source[0]),np.asarray(source[-1])]).astype(np.float32)/255
        quantized=np.floor(np.clip(native+lifted,0,1)*255+0.5)/255
        mse=float(np.mean((quantized-native)**2))
        report['boundary_input_PSNR_dB']=float(-10*np.log10(max(mse,1e-12)))
        report['boundary_Linf_pixel']=float(np.abs(quantized-native).max()*255)
        previews=out.parent/(out.stem+'_input_png');previews.mkdir(parents=True,exist_ok=True)
        attacked=[Image.fromarray(np.rint(f*255).astype(np.uint8)) for f in quantized]
        for n,f in enumerate([0,a.source_frames-1]):
            source[f].save(previews/f'frame{f:02d}_original.png');attacked[n].save(previews/f'frame{f:02d}_attacked.png')
        # True entropy coding is a post-optimization transfer test, not a
        # straight-through claim of differentiating discrete compression.
        codec_clean=[];codec_adv=[];codec_stats=[]
        for clean,adv in zip([source[0],source[-1]],attacked):
            c,cb=compress_boundary_frame(clean,ref_codec='compressai',ref_quality=4)
            d,db=compress_boundary_frame(adv,ref_codec='compressai',ref_quality=4)
            codec_clean.append(c);codec_adv.append(d)
            codec_stats.append(dict(clean_bytes=cb,attacked_bytes=db))
        def pair_tensor(pair):
            return torch.from_numpy(np.stack([np.asarray(f) for f in resize_frames(pair,a.width,a.height)]).copy()).permute(0,3,1,2).float().to('cuda:1')/255
        with torch.no_grad():
            cc=tensor_conditions(model,pair_tensor(codec_clean),a.frames)
            ac=tensor_conditions(model,pair_tensor(codec_adv),a.frames)
            vc=model.predict_velocity(x_t,a.time,embeds,cc)
            va=model.predict_velocity(x_t,a.time,embeds,ac)
            report['after_real_codec_velocity_shift_MSE']=float((va-vc).square().mean())
        report['reference_codec_bytes']=codec_stats
        report['status']='completed';report['lossless_input_png_dir']=str(previews)
        np.savez_compressed(out,delta=full_delta);save()
        print(f'Saved {out}; {json.dumps({k:v for k,v in report.items() if k!="history"})}',flush=True)
    except Exception as e:
        report['status']='failed';report['error']=str(e);save();raise


if __name__=='__main__':main()
