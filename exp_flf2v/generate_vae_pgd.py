"""Optimize one GOP-shared RGB perturbation through frozen Wan VAE.

Grid or random 5-frame spatial crops keep backward memory bounded. This is a VAE
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


def grid_starts(length, crop):
    """Cover every pixel, including the final partial tile, with full crops."""
    starts = list(range(0, length - crop + 1, crop))
    if starts[-1] != length - crop:
        starts.append(length - crop)
    return starts


def crop_schedule(height, width, crop, frames, mode, steps, tile_steps, rng):
    if mode == 'random':
        return [(int(rng.integers(frames - 4)), int(rng.integers(height - crop + 1)),
                 int(rng.integers(width - crop + 1))) for _ in range(steps)]
    schedule = []
    for y in grid_starts(height, crop):
        for x in grid_starts(width, crop):
            t = int(rng.integers(frames - 4))
            schedule.extend([(t, y, x)] * tile_steps)
    return schedule


def reconstruct(vae, pixels):
    z = vae.encode([pixels * 2 - 1])[0]
    return z, (vae.decode([z])[0] + 1) / 2


@torch.no_grad()
def evaluate_fixed(vae, clean, deltas, specs, device, crop):
    """Evaluate identical crops after 8-bit rounding, like the GVCC runner."""
    rows = []
    for t, y, x in specs:
        target = clean[:, t:t+5, y:y+crop, x:x+crop].to(device)
        z0, r0 = reconstruct(vae, target)
        row = dict(frame_start=t, top=y, left=x, variants={})
        row['variants']['clean'] = dict(reconstruction_MSE=float((r0-target).square().mean()),
                                       reconstruction_shift_MSE=0.0, latent_shift_MSE=0.0)
        for name, delta in deltas.items():
            d = delta[:, y:y+crop, x:x+crop].to(device)
            adv = ((target + d[:, None]).clamp(0, 1) * 255).round() / 255
            z, r = reconstruct(vae, adv)
            row['variants'][name] = dict(reconstruction_MSE=float((r-target).square().mean()),
                                        reconstruction_shift_MSE=float((r-r0).square().mean()),
                                        latent_shift_MSE=float((z-z0).square().mean()))
        rows.append(row)
    means = {name: {key: float(np.mean([row['variants'][name][key] for row in rows]))
                    for key in rows[0]['variants'][name]} for name in rows[0]['variants']}
    return dict(means=means, crops=rows)


@torch.enable_grad()
def main():
    # Importing the inference runner above disables autograd globally.
    # Explicitly enable it for this attack only; VAE weights stay frozen.
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
    p.add_argument('--schedule', choices=['grid', 'random'], default='grid')
    p.add_argument('--tile_steps', type=int, default=10, help='Consecutive updates per grid crop; grid ignores --steps')
    p.add_argument('--eval_every', type=int, default=70, help='Evaluate fixed crops every N updates; 0 means final only')
    p.add_argument('--crop', type=int, default=192)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--objective', choices=['reconstruction', 'deviation', 'latent'], default='reconstruction')
    a = p.parse_args()
    if a.frames < 5 or a.crop <= 0 or a.crop % 16 or a.crop > min(a.height, a.width):
        p.error('Need >=5 frames, crop multiple of 16 and within image')
    if a.epsilon <= 0 or a.alpha <= 0 or a.steps < 1 or a.tile_steps < 1 or a.eval_every < 0:
        p.error('epsilon, alpha and steps must be positive')
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    frames = resize_frames(load_yuv420_frames(a.data, a.frames), a.width, a.height)
    if len(frames) != a.frames:
        raise ValueError('Not enough input frames')
    clean = torch.from_numpy(np.stack([np.asarray(f) for f in frames]).copy()).permute(3, 0, 1, 2).float() / 255
    eps, alpha = a.epsilon / 255, a.alpha / 255
    delta = torch.empty(3, a.height, a.width).uniform_(-eps, eps)
    initial_delta = delta.clone()
    sign_delta = torch.where(initial_delta >= 0, eps, -eps)
    schedule = crop_schedule(a.height, a.width, a.crop, a.frames, a.schedule, a.steps, a.tile_steps, rng)
    # The same locations and frames are reused throughout. They diagnose the
    # proxy objective; they are not a held-out dataset/generalization test.
    specs = [(0, 0, 0), ((a.frames-5)//2, (a.height-a.crop)//2, (a.width-a.crop)//2),
             (a.frames-5, a.height-a.crop, a.width-a.crop)]
    coverage = np.zeros((a.height, a.width), dtype=np.int32)
    device = torch.device('cuda:0')
    print('Loading frozen VAE only (no Wan DiT)...', flush=True)
    vae = WanVAE(vae_pth=a.vae, dtype=torch.bfloat16, device=device)
    history = []
    evaluations = []
    print(f'{a.schedule}: {len(schedule)} updates; objective={a.objective}', flush=True)
    def evaluate(step):
        result = evaluate_fixed(vae, clean, dict(random_uniform=initial_delta, random_sign=sign_delta,
                                                optimized=delta), specs, device, a.crop)
        evaluations.append(dict(step=step, **result))
        print(f'Fixed-crop evaluation step {step}: {json.dumps(result["means"])}', flush=True)
    evaluate(0)
    baseline_spec, z0, r0 = None, None, None
    for step, (t, y, x) in enumerate(schedule):
        target = clean[:, t:t + 5, y:y + a.crop, x:x + a.crop].to(device)
        d = delta[:, y:y + a.crop, x:x + a.crop].to(device).detach().requires_grad_(True)
        if a.objective != 'reconstruction' and baseline_spec != (t, y, x):
            with torch.no_grad():
                z0 = vae.encode([target * 2 - 1])[0]
                r0 = (vae.decode([z0])[0] + 1) / 2 if a.objective == 'deviation' else None
            baseline_spec = (t, y, x)
        # Direct tensor path preserves gradients. Parameters remain frozen.
        adv = (target + d[:, None]).clamp(0, 1)
        z = vae.encode([adv * 2 - 1])[0]
        if a.objective == 'latent':
            loss = (z - z0).square().mean()
        else:
            recon = (vae.decode([z])[0] + 1) / 2
            loss = (recon - (r0 if a.objective == 'deviation' else target)).square().mean()
        if not loss.requires_grad:
            raise RuntimeError('VAE loss has no gradient graph: check no_grad/inference_mode in the VAE path')
        grad, = torch.autograd.grad(loss, d)
        if not torch.isfinite(grad).all() or grad.abs().max().item() == 0:
            raise RuntimeError('Invalid/zero input gradient; aborting rather than saving a fake attack')
        delta[:, y:y + a.crop, x:x + a.crop] = (d.detach().cpu() + alpha * grad.detach().cpu().sign()).clamp(-eps, eps)
        coverage[y:y+a.crop, x:x+a.crop] += 1
        history.append(dict(step=step + 1, loss=float(loss.detach()), frame_start=t, top=y, left=x))
        print(f'PGD {step+1}/{len(schedule)}: {a.objective} loss={history[-1]["loss"]:.6f}', flush=True)
        del target, d, adv, z, loss, grad
        if a.objective != 'latent':
            del recon
        if a.eval_every and (step+1) % a.eval_every == 0 and step+1 != len(schedule):
            evaluate(step+1)
    evaluate(len(schedule))
    original = clean.permute(1, 2, 3, 0).numpy()
    attack = np.clip(original + delta.permute(1, 2, 0).numpy()[None], 0, 1)
    # Match the real runner's 8-bit rounding; report input perturbation quality.
    quantized = np.floor(attack * 255 + 0.5) / 255
    mse = float(np.mean((quantized - original) ** 2))
    input_controls = {}
    for name, control in [('random_uniform', initial_delta), ('random_sign', sign_delta)]:
        pixels = np.floor(np.clip(original + control.permute(1, 2, 0).numpy()[None], 0, 1)*255 + 0.5)/255
        control_mse = float(np.mean((pixels-original)**2))
        input_controls[name] = dict(PSNR_dB=float(-10*np.log10(max(control_mse, 1e-12))),
                                    Linf_pixel=float(np.abs(pixels-original).max()*255))
    report = vars(a) | dict(input_PSNR_dB=float(-10*np.log10(max(mse, 1e-12))),
                           measured_Linf_pixel=float(np.abs(quantized-original).max()*255),
                           scope='single-GOP shared spatial perturbation optimized on VAE crops', history=history,
                           actual_updates=len(schedule), coverage_fraction=float((coverage>0).mean()),
                           minimum_pixel_updates=int(coverage.min()), mean_pixel_updates=float(coverage.mean()),
                           fixed_evaluations=evaluations, input_random_controls=input_controls)
    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, delta=delta.permute(1, 2, 0).numpy())
    for name, baseline in [('random_uniform', initial_delta), ('random_sign', sign_delta)]:
        np.savez_compressed(output.with_name(output.stem + '_' + name + '.npz'),
                            delta=baseline.permute(1, 2, 0).numpy())
    output.with_suffix('.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Saved {output}; input PSNR={report["input_PSNR_dB"]:.2f} dB', flush=True)


if __name__ == '__main__':
    main()
