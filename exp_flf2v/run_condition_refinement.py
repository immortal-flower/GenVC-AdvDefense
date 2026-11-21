"""Real boundary-codec forward / identity-BPDA condition refinement, then native acceptance.

This optimizes condition feature displacement, NOT full GVCC RGB gradients.
"""
import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exp_flf2v.search_native_rgb_input import load_source, sha256, atomic_json
from exp_flf2v.run_vae_warmstart_batch import codec_command, write_csv
from exp_flf2v.generate_wan_boundary_pgd import tensor_conditions
from exp_flf2v.run_flf2v_experiment import compress_boundary_frame
from wan.configs import WAN_CONFIGS
from wan.modules.vae import WanVAE
from wan.modules.clip import CLIPModel


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--wan_ckpt', type=Path, required=True)
    p.add_argument('--results_root', type=Path, default=Path('exp_flf2v/results_720p'))
    p.add_argument('--source_tag', default='vae_source_rgb_eps4_r36_jockey_gop0_v1')
    p.add_argument('--round', type=int, default=44)
    p.add_argument('--steps', type=int, default=8)
    p.add_argument('--tag', default='condition_refine_r44_v1')
    p.add_argument('--generate_only', action='store_true')
    a = p.parse_args()
    if a.steps < 1: p.error('steps must be positive')
    source = a.results_root / (a.source_tag + '_assets')
    report = json.loads((source/'generation.json').read_text(encoding='utf-8'))
    item, = [r for r in report['assets'] if r['name'] == f'eps4_r{a.round}_candidate']
    asset = source/item['file']
    if report['status'] != 'completed' or sha256(asset) != item['sha256']:
        raise ValueError('Invalid source asset')
    if sha256(a.data) != report['configuration']['data_sha256']:
        raise ValueError('Source data mismatch')
    raw = load_source(a.data)
    with np.load(asset) as z: base_np = z['delta'].copy()
    out = a.results_root/(a.tag+'_assets'); out.mkdir(parents=True, exist_ok=True)
    config = dict(data=sha256(a.data), source=sha256(asset), steps=a.steps,
                  generator=sha256(__file__), epsilon=4, alpha=.25,
                  objective='relative VAE-condition and CLIP displacement; joint adds normalized VAE RGB MSE',
                  compression='true CompressAI q4 every forward, identity BPDA backward',
                  proxy=[192,336], native=[720,1280])
    config_path = out/'configuration.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Configuration changed; use new tag')
    atomic_json(config_path, config)
    x = torch.from_numpy(raw.copy()).permute(3,0,1,2).float().cuda()/255
    base = torch.from_numpy(base_np).permute(3,0,1,2).cuda()
    cfg = WAN_CONFIGS['flf2v-14B']
    model = None
    def codec(pixels):
        frames = []
        for t in (0,32):
            im = Image.fromarray(torch.floor(pixels[:,t].detach().clamp(0,1)*255+.5).byte().permute(1,2,0).cpu().numpy())
            dec,_ = compress_boundary_frame(im, ref_codec='compressai', ref_quality=4)
            frames.append(torch.from_numpy(np.asarray(dec).copy()).permute(2,0,1).to(pixels)/255)
        actual = torch.stack(frames)
        boundary = pixels[:,[0,32]].permute(1,0,2,3)
        return F.interpolate(boundary+(actual-boundary).detach(), (192,336), mode='bilinear', align_corners=False)
    for branch in ('boundary','joint'):
        dest = out/(branch+'.npz')
        if dest.exists():
            saved = json.loads((out/(branch+'.json')).read_text())
            if saved['configuration'] != config or saved['sha256'] != sha256(dest):
                raise ValueError('Saved candidate changed')
            continue
        if model is None:
            model = SimpleNamespace(vae=WanVAE(vae_pth=str(a.wan_ckpt/cfg.vae_checkpoint), device='cuda:0', dtype=torch.bfloat16),
                clip=CLIPModel(dtype=cfg.clip_dtype,device=torch.device('cuda:0'),checkpoint_path=str(a.wan_ckpt/cfg.clip_checkpoint),tokenizer_path=str(a.wan_ckpt/cfg.clip_tokenizer)))
            model.vae.model.eval().requires_grad_(False)
            model.clip.model.eval().requires_grad_(False)
        with torch.no_grad(): reference = {k:v.detach() for k,v in tensor_conditions(model,codec(x),33).items()}
        d = base.clone()
        history = []
        for step in range(a.steps):
            d = d.detach().requires_grad_(True)
            pixels = (x+d).clamp(0,1)
            cond = tensor_conditions(model,codec(pixels),33)
            terms = [(cond[k]-reference[k]).float().square().mean()/reference[k].float().square().mean().clamp_min(1e-8) for k in reference]
            loss = sum(terms)
            if branch == 'joint':
                # Short spatial crops retain the effective native VAE RGB attack path.
                yy = (step*144) % (720-192+1); xx = (step*144) % (1280-192+1)
                target = x[:,:,yy:yy+192,xx:xx+192]
                tile = pixels[:,:,yy:yy+192,xx:xx+192]
                rec = (model.vae.decode([model.vae.encode([tile*2-1])[0]])[0]+1)/2
                rgb_loss = (rec-target).square().mean()
                loss = loss + rgb_loss/rgb_loss.detach().clamp_min(1e-8)
            grad, = torch.autograd.grad(loss,d)
            if not torch.isfinite(grad).all() or grad.abs().max()==0: raise ValueError('Invalid gradient')
            if branch == 'boundary': grad[:,1:-1] = 0
            with torch.no_grad():
                d = (d+.25/255*grad.sign()).clamp(-4/255,4/255)
                if branch == 'boundary': d[:,1:-1] = base[:,1:-1]
            history.append(dict(step=step+1, loss=float(loss.detach())))
            print(branch,history[-1],flush=True)
        actual = (torch.floor((x+d).clamp(0,1)*255+.5)/255-x).detach().permute(1,2,3,0).cpu().numpy()
        if np.abs(actual).max()*255 > 4.001: raise ValueError('Bound exceeded')
        if branch == 'boundary' and not np.allclose(actual[1:-1],base_np[1:-1],atol=1e-7): raise ValueError('Interior changed')
        np.savez_compressed(dest,delta=actual)
        atomic_json(out/(branch+'.json'),dict(history=history,sha256=sha256(dest),configuration=config))
    if a.generate_only:
        print('Candidates ready; exit this process before native codec acceptance.',flush=True)
        return
    del model,x,base
    gc.collect(); torch.cuda.empty_cache()
    rows = []
    for name, path in [('initial',asset),('boundary',out/'boundary.npz'),('joint',out/'joint.npz')]:
        run = a.tag+'_'+name
        metric = a.results_root/run/'Jockey/gop0/metrics.json'
        receipt = out/(name+'_accepted.json')
        signature = dict(asset=sha256(path), data=sha256(a.data), checkpoint=str(a.wan_ckpt.resolve()), seed=42, codec='720p33-M64-K16384-20-q4')
        valid = False
        if receipt.exists() and metric.exists():
            saved = json.loads(receipt.read_text()); valid = saved.get('signature') == signature and saved.get('metrics') == sha256(metric)
        if not valid:
            cmd = codec_command(sys.executable,Path(__file__).with_name('run_flf2v_experiment.py'),a.data,a.wan_ckpt,a.results_root,run,path,4)
            subprocess.run(cmd,check=True)
            atomic_json(receipt,dict(signature=signature,metrics=sha256(metric)))
        m = json.loads(metric.read_text())
        rows.append(dict(Variant=name,PSNR=m['PSNR_dB'],LPIPS=m['LPIPS'],BPP=m['BPP']))
    write_csv(a.results_root/(a.tag+'_summary.csv'),rows)
    print(rows)

if __name__ == '__main__': main()
