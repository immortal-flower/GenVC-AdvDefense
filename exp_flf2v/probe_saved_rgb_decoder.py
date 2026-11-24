"""Replay saved native decoder inputs with VAE alone, without loading Wan DiT."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from wan.modules.vae import WanVAE
from exp_flf2v.trajectory_input_objective import rgb_distortion


@torch.enable_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probe',type=Path,required=True)
    p.add_argument('--vae',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    status=a.output/'status.json'
    def record(**kw):
        status.write_text(json.dumps(kw,indent=2),encoding='utf-8')
        print(kw,flush=True)
    with np.load(a.probe,allow_pickle=False) as z:
        latent=torch.from_numpy(z['latent'].copy()).float()
        reference=torch.from_numpy(z['reference'].copy()).float()
    record(stage='loading VAE only',latent_shape=list(latent.shape),reference_shape=list(reference.shape))
    model=WanVAE(vae_pth=str(a.vae),device=a.device,dtype=torch.float32)
    model.model.eval().requires_grad_(False)
    leaf=latent.to(a.device).requires_grad_(True)
    context=torch.autograd.graph.save_on_cpu(pin_memory=False) if a.device.startswith('cuda') else nullcontext()
    try:
        started=time.monotonic()
        record(stage='grad-enabled decoder forward',device=a.device)
        with context:
            video=model.decode([leaf[0]])[0]
            rgb=(video.permute(1,0,2,3)/2+.5).clamp(0,1)
            loss=rgb_distortion(rgb,reference.to(a.device))
        record(stage='decoder forward complete; backward starting',score=float(loss.detach()),seconds=time.monotonic()-started)
        grad,=torch.autograd.grad(loss,leaf)
        if not torch.isfinite(grad).all() or grad.abs().max()==0:
            raise RuntimeError('Invalid terminal latent gradient')
        np.savez_compressed(a.output/'terminal_gradient.npz',gradient=grad.detach().cpu().numpy())
        record(stage='completed',score=float(loss.detach()),grad_max=float(grad.abs().max()),seconds=time.monotonic()-started,
               peak_gpu_GB=torch.cuda.max_memory_allocated(a.device)/2**30 if a.device.startswith('cuda') else None)
    except Exception as e:
        record(stage='failed',error=repr(e))
        raise


if __name__=='__main__':main()
