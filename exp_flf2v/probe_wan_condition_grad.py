"""One-step frozen-Wan gradient feasibility probe on detached conditions.

This measures d(velocity projection)/d(y, clip_fea), not yet pixel gradients
or an adversarial attack. x_t is an RF-interpolated diagnostic state rather
than a stored GVCC codebook trajectory.
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from accelerate import dispatch_model, infer_auto_device_map

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from run_flf2v_experiment import load_yuv420_frames, resize_frames, compress_boundary_frame
from sde_rf_wan.wan_flf2v_wrapper import WanFLF2VWrapper


def checkpoint_blocks(model):
    for block in model.blocks:
        original=block.forward
        def wrapped(x, _original=original, **kwargs):
            return checkpoint(_original,x,use_reentrant=False,**kwargs)
        block.forward=wrapped


@torch.enable_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True)
    p.add_argument('--wan_ckpt',default='exp_flf2v/Wan2.1-FLF2V-14B-720P')
    p.add_argument('--output',default='exp_flf2v/results_720p/wan_condition_grad_probe/probe.json')
    p.add_argument('--height',type=int,default=128);p.add_argument('--width',type=int,default=224)
    p.add_argument('--frames',type=int,default=9);p.add_argument('--source_frames',type=int,default=33)
    p.add_argument('--time',type=float,default=0.5);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--gpu0_memory',default='20GiB');p.add_argument('--gpu1_memory',default='20GiB')
    p.add_argument('--no_checkpoint',action='store_true')
    a=p.parse_args()
    if a.frames<5 or (a.frames-1)%4 or a.source_frames<a.frames or a.height%16 or a.width%16 or not 0<a.time<1:
        p.error('Use 4n+1 frames, dimensions divisible by 16, and time strictly between 0 and 1')
    if torch.cuda.device_count()<2:raise RuntimeError('This probe requires two visible GPUs')
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    report=vars(a)|dict(scope='condition tensor gradient feasibility only; not pixel attack',status='started')
    def save():out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    save()
    try:
        frames=load_yuv420_frames(a.data,a.source_frames)
        if len(frames)!=a.source_frames:raise ValueError('Not enough source frames')
        indices=np.linspace(0,a.source_frames-1,a.frames).round().astype(int).tolist()
        frames=resize_frames([frames[i] for i in indices],a.width,a.height)
        report['source_frame_indices']=indices
        model=WanFLF2VWrapper(a.wan_ckpt)
        model.load('cpu',torch.bfloat16)
        # Preprocessing occurs before DiT dispatch, with no retained graph.
        model.device=torch.device('cuda:1')
        with torch.no_grad():
            model.text_encoder.model.to('cuda:1')
            embeds=model.encode_prompt('')['prompt_embeds']
            embeds=[u.cpu() for u in embeds]
            model.text_encoder.model.to('cpu')
            model.vae.model.to('cuda:1');model.clip.model.to('cuda:1')
            first,_=compress_boundary_frame(frames[0],ref_codec='compressai',ref_quality=4)
            last,_=compress_boundary_frame(frames[-1],ref_codec='compressai',ref_quality=4)
            cond=model.encode_first_last_frames(first,last,a.frames,a.height,a.width)
            cond={k:v.cpu() for k,v in cond.items()}
            clean_latent=model.encode_video(frames,a.height,a.width).cpu()
            model.vae.model.to('cpu');model.clip.model.to('cpu')
        gc.collect();torch.cuda.empty_cache()
        report['gpu_memory_before_dispatch']={str(i):dict(
            total_GiB=torch.cuda.get_device_properties(i).total_memory/2**30,
            free_GiB=torch.cuda.mem_get_info(i)[0]/2**30) for i in range(2)}
        report['dit_parameter_GiB']=sum(p.numel()*p.element_size() for p in model.model.parameters())/2**30
        print(f'DiT weights: {report["dit_parameter_GiB"]:.2f} GiB; GPU weight budgets: {a.gpu0_memory}, {a.gpu1_memory}',flush=True)
        if not a.no_checkpoint:checkpoint_blocks(model.model)
        device_map=infer_auto_device_map(model.model,max_memory={0:a.gpu0_memory,1:a.gpu1_memory},
                                         no_split_module_classes=['WanAttentionBlock'])
        if any(v not in [0,1,'cuda:0','cuda:1'] for v in device_map.values()):
            report['device_map']={k:str(v) for k,v in device_map.items()}
            raise RuntimeError(f'DiT mapping includes CPU/disk; adjust GPU memory limits: {device_map}')
        model.model=dispatch_model(model.model,device_map=device_map)
        model.device=torch.device('cuda:0')
        model.model.requires_grad_(False)
        report['device_map']={k:str(v) for k,v in device_map.items()}
        if any(v.requires_grad for v in model.model.parameters()):raise RuntimeError('DiT weights were not frozen')
        # Mask channels stay fixed; only the VAE condition and CLIP leaves vary.
        condition_latent=cond['y'][4:].to('cuda:0').detach().requires_grad_(True)
        clip=cond['clip_fea'].to('cuda:0').detach().requires_grad_(True)
        mask=cond['y'][:4].to('cuda:0')
        gen=torch.Generator(device='cpu').manual_seed(a.seed)
        noise=torch.randn(clean_latent.shape,generator=gen)
        x_t=((1-a.time)*clean_latent+a.time*noise).to('cuda:0')
        embeds=[u.to('cuda:0') for u in embeds]
        for i in range(2):torch.cuda.reset_peak_memory_stats(i)
        start=time.time()
        print('Wan single-step forward with condition gradients enabled...',flush=True)
        velocity=model.predict_velocity_with_grad(x_t,a.time,embeds,dict(y=torch.cat([mask,condition_latent]),clip_fea=clip))
        # A deterministic linear probe gives nonzero gradient at the clean
        # condition, unlike squared clean-vs-clean velocity difference.
        direction=torch.randn(velocity.shape,generator=gen).to(velocity.device)
        loss=(velocity*direction).mean()
        print('Wan backward to VAE-condition and CLIP tensors...',flush=True)
        grads=torch.autograd.grad(loss,(condition_latent,clip),allow_unused=False)
        torch.cuda.synchronize(0);torch.cuda.synchronize(1)
        report['seconds']=time.time()-start
        report['velocity_shape']=list(velocity.shape)
        report['gradients']={}
        for name,g in zip(['vae_condition','clip_condition'],grads):
            stats=dict(shape=list(g.shape),finite=bool(torch.isfinite(g).all()),
                       max_abs=float(g.abs().max()),mean_abs=float(g.abs().mean()),norm=float(g.norm()))
            report['gradients'][name]=stats
            if not stats['finite'] or stats['max_abs']==0:raise RuntimeError(f'Invalid gradient: {name}: {stats}')
        report['peak_memory_GiB']={str(i):dict(allocated=torch.cuda.max_memory_allocated(i)/2**30,
                                              reserved=torch.cuda.max_memory_reserved(i)/2**30) for i in range(2)}
        report['status']='passed';save()
        print(json.dumps(report,indent=2),flush=True)
    except Exception as e:
        report['status']='failed';report['error']=str(e);save();raise


if __name__=='__main__':main()
