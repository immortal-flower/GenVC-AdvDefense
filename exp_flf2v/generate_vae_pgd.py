"""Optimize one GOP-shared RGB perturbation through frozen Wan VAE.

Random 5-frame spatial crops keep backward memory bounded. This is a VAE
surrogate attack, not end-to-end GVCC PGD and not a dataset-universal UAP.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_flf2v_experiment import load_yuv420_frames, resize_frames
from wan.modules.vae import WanVAE


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--vae', required=True)
    p.add_argument('--output', default='exp_flf2v/attack_assets/jockey_vae_pgd_eps4.npz')
    p.add_argument('--height', type=int, default=720)
    p.add_argument('--width', type=int, default=1280)
    p.add_argument('--frames', type=int, default=33)
    p.add_argument('--epsilon', type=float, default=4.0, help='Pixel levels /255')
    p.add_argument('--alpha', type=float, default=0.5, help='Pixel levels /255')
    p.add_argument('--steps', type=int, default=40)
    p.add_argument('--crop', type=int, default=192)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--objective', choices=['reconstruction', 'latent'], default='reconstruction')
    a = p.parse_args()
    if a.frames < 5 or a.crop % 16 or a.crop > min(a.height, a.width):
        p.error('Need >=5 frames, crop multiple of 16 and within image')
    if a.epsilon <= 0 or a.alpha <= 0 or a.steps < 1:
        p.error('epsilon, alpha and steps must be positive')
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    frames = resize_frames(load_yuv420_frames(a.data, a.frames), a.width, a.height)
    if len(frames) != a.frames:
        raise ValueError('Not enough input frames')
    clean = torch.from_numpy(np.stack([np.asarray(f) for f in frames]).copy()).permute(3, 0, 1, 2).float() / 255
    eps, alpha = a.epsilon / 255, a.alpha / 255
    delta = torch.empty(3, a.height, a.width).uniform_(-eps, eps)
    device = torch.device('cuda:0')
    print('Loading frozen VAE only (no Wan DiT)...', flush=True)
    vae = WanVAE(vae_pth=a.vae, dtype=torch.bfloat16, device=device)
    history = []
    for step in range(a.steps):
        t = int(rng.integers(0, a.frames - 4))
        y = int(rng.integers(0, a.height - a.crop + 1))
        x = int(rng.integers(0, a.width - a.crop + 1))
        target = clean[:, t:t + 5, y:y + a.crop, x:x + a.crop].to(device)
        d = delta[:, y:y + a.crop, x:x + a.crop].to(device).detach().requires_grad_(True)
        # Direct tensor path preserves gradients. Parameters remain frozen.
        adv = (target + d[:, None]).clamp(0, 1)
        z = vae.encode([adv * 2 - 1])[0]
        if a.objective == 'latent':
            with torch.no_grad():
                z0 = vae.encode([target * 2 - 1])[0]
            loss = (z - z0).square().mean()
        else:
            recon = (vae.decode([z])[0] + 1) / 2
            loss = (recon - target).square().mean()
        grad, = torch.autograd.grad(loss, d)
        if not torch.isfinite(grad).all() or grad.abs().max().item() == 0:
            raise RuntimeError('Invalid/zero input gradient; aborting rather than saving a fake attack')
        delta[:, y:y + a.crop, x:x + a.crop] = (d.detach().cpu() + alpha * grad.detach().cpu().sign()).clamp(-eps, eps)
        history.append(dict(step=step + 1, loss=float(loss.detach()), frame_start=t, top=y, left=x))
        print(f'PGD {step+1}/{a.steps}: {a.objective} loss={history[-1]["loss"]:.6f}', flush=True)
        del target, d, adv, z, loss, grad
    original = clean.permute(1, 2, 3, 0).numpy()
    attack = np.clip(original + delta.permute(1, 2, 0).numpy()[None], 0, 1)
    # Match the real runner's 8-bit rounding; report input perturbation quality.
    quantized = np.floor(attack * 255 + 0.5) / 255
    mse = float(np.mean((quantized - original) ** 2))
    report = vars(a) | dict(input_PSNR_dB=float(-10*np.log10(max(mse, 1e-12))),
                           measured_Linf_pixel=float(np.abs(quantized-original).max()*255),
                           scope='single-GOP shared spatial perturbation optimized on random VAE crops', history=history)
    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, delta=delta.permute(1, 2, 0).numpy())
    output.with_suffix('.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Saved {output}; input PSNR={report["input_PSNR_dB"]:.2f} dB', flush=True)


if __name__ == '__main__':
    main()
