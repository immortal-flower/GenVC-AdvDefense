"""Short latent rollout or full-forward RGB robustness pilot; not exact codec AD.

Hard full-K selection is recomputed at EVERY rollout step. Backward uses a
local soft top-N bank. VAE and DiT graphs are separated by an exact latent VJP.
Native boundary pixels are held at their initialization throughout optimization.
"""
import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import dispatch_model, infer_auto_device_map

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_flf2v_experiment import load_yuv420_frames, resize_frames, compress_boundary_frame
from generate_wan_boundary_pgd import tensor_conditions
from probe_wan_condition_grad import checkpoint_blocks
from trajectory_input_objective import correction, full_trajectory, rgb_distortion, rollout_loss
from sde_rf_wan.sde_convert import shifted_timesteps
from sde_rf_wan.turbo_codebook import TurboPerFrameCodebook
from sde_rf_wan.wan_flf2v_wrapper import WanFLF2VWrapper


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    tmp = path.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def matched_random(original, candidate, seed):
    """Match quantized per-pixel magnitude; preserve the two boundary frames."""
    magnitude = np.abs(candidate)
    signs = np.random.default_rng(seed).choice([-1., 1.], size=(1, *original.shape[1:]))
    result = (magnitude * signs).astype(np.float32)
    invalid = (original + result < 0) | (original + result > 1)
    result[invalid] *= -1
    result[[0, -1]] = candidate[[0, -1]]
    return result


@torch.enable_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--wan_ckpt', required=True)
    p.add_argument('--initial_attack_file', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--iterations', type=int, default=4)
    p.add_argument('--rollout_steps', type=int, default=2)
    p.add_argument('--objective', choices=['early-latent', 'final-rgb'], default='early-latent')
    p.add_argument('--backward_steps', type=int, default=6, help='Final RGB mode only')
    p.add_argument('--height', type=int, default=128)
    p.add_argument('--width', type=int, default=224)
    p.add_argument('--epsilon', type=float, default=2.)
    p.add_argument('--alpha', type=float, default=.5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--pool_size', type=int, default=80)
    p.add_argument('--gpu0_memory', default='20GiB')
    p.add_argument('--gpu1_memory', default='20GiB')
    a = p.parse_args()
    final_rgb = a.objective == 'final-rgb'
    if final_rgb and not 4 <= a.backward_steps <= 20:
        p.error('Final RGB backward suffix must include at least one SDE step (4–20 steps)')
    if (a.iterations < 1 or not 1 <= a.rollout_steps <= 16 or
            min(a.height, a.width) < 32 or a.height % 16 or a.width % 16 or
            a.epsilon <= 0 or a.alpha <= 0 or not 64 <= a.pool_size <= 16384):
        p.error('Invalid rollout, geometry, bound, or pool size')
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    report_path = out / 'generation.json'
    checkpoint = out / 'optimizer_checkpoint.npz'
    config = vars(a) | dict(data_sha256=digest(a.data), init_sha256=digest(a.initial_attack_file), version=1)
    if not final_rgb:  # retain compatibility with existing short-rollout checkpoints
        del config['objective'], config['backward_steps']
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if report_path.exists():
        previous = json.loads(report_path.read_text(encoding='utf-8'))
        if previous['fingerprint'] != fingerprint:
            raise ValueError('Configuration/source changed: use a new output directory')
        if previous['status'] == 'completed':
            for name, sha in previous['asset_sha256'].items():
                if digest(out / name) != sha:
                    raise ValueError(f'Asset changed: {name}')
            print('[SKIP] Verified completed candidate generation', flush=True)
            return
    if torch.cuda.device_count() < 2:
        raise RuntimeError('Two visible CUDA GPUs are required')
    report = dict(fingerprint=fingerprint, configuration=config, status='running', history=[],
                  limitation='Reduced-resolution short rollout, full-K hard forward/local soft backward; not exact native codec gradient',
                  objective='Mean post-update predicted-x0 MSE against clean latent across rollout checkpoints and all latent positions',
                  boundary='Fixed initialized first/last RGB, compressed once using compressai q4 before resizing')
    if final_rgb:
        report.update(objective='Negative mean per-frame PSNR, float RGB decoder output vs clean source',
                      limitation='Low-resolution 20-step hard forward, truncated backward and soft codebook gradients',
                      tracked_steps=list(range(21 - a.backward_steps, 21)),
                      tracked_sde_steps=list(range(21 - a.backward_steps, 18)))
    atomic_json(report_path, report)
    start = time.monotonic()
    try:
        frames = load_yuv420_frames(a.data, 33)
        if len(frames) != 33 or any(f.size != (1280, 720) for f in frames):
            raise ValueError('This pilot requires 33 native 1280x720 GOP0 frames')
        original = np.stack([np.asarray(f) for f in frames]).astype(np.float32) / 255
        with np.load(a.initial_attack_file, allow_pickle=False) as z:
            init = z['delta'].astype(np.float32)
        if init.shape == original.shape[1:]:
            init = np.broadcast_to(init, original.shape).copy()
        eps = a.epsilon / 255
        if init.shape != original.shape or not np.isfinite(init).all() or abs(init).max() > eps + 1e-6:
            raise ValueError('Invalid initial perturbation or epsilon violation')
        init = np.floor(np.clip(original + init, 0, 1) * 255 + .5) / 255 - original
        from PIL import Image
        boundaries = [Image.fromarray(np.rint((original[i] + init[i]) * 255).astype(np.uint8)) for i in [0, -1]]
        boundaries = [compress_boundary_frame(f, ref_codec='compressai', ref_quality=4)[0] for f in boundaries]
        # The reference codec is no longer needed in this generator process.
        # Release its cached GPU weights before loading/dispatching the 14B DiT.
        from sde_rf_wan.ref_codec import _compressai_cache
        _compressai_cache.clear()
        gc.collect(); torch.cuda.empty_cache()
        model = WanFLF2VWrapper(a.wan_ckpt)
        model.load('cpu', torch.bfloat16)
        model.model.eval().requires_grad_(False)
        model.device = torch.device('cuda:1')
        with torch.no_grad():
            model.text_encoder.model.to('cuda:1')
            embeds = [v.cpu() for v in model.encode_prompt('')['prompt_embeds']]
            model.text_encoder.model.to('cpu')
            model.vae.model.to('cuda:1').eval().requires_grad_(False)
            model.clip.model.to('cuda:1').eval().requires_grad_(False)
            boundary_tensor = torch.from_numpy(np.stack([np.asarray(f) for f in resize_frames(boundaries, a.width, a.height)]).copy()).permute(0, 3, 1, 2).float().to('cuda:1') / 255
            cond = {k: v.cpu() for k, v in tensor_conditions(model, boundary_tensor, 33).items()}
            model.clip.model.to('cpu')
        del boundary_tensor
        gc.collect(); torch.cuda.empty_cache()
        checkpoint_blocks(model.model)
        mapping = infer_auto_device_map(model.model, max_memory={0: a.gpu0_memory, 1: a.gpu1_memory}, no_split_module_classes=['WanAttentionBlock'])
        if any(v not in [0, 1, 'cuda:0', 'cuda:1'] for v in mapping.values()):
            raise RuntimeError(f'DiT does not fit on two GPUs: {mapping}')
        model.model = dispatch_model(model.model, device_map=mapping)
        model.device = torch.device('cuda:0')
        embeds = [v.to('cuda:0') for v in embeds]
        cond = {k: v.to('cuda:0') for k, v in cond.items()}
        rgb = torch.from_numpy(original).permute(0, 3, 1, 2).to('cuda:1')
        base = torch.from_numpy(init).permute(0, 3, 1, 2).to('cuda:1')
        mask = torch.ones(33, 1, 1, 1, device='cuda:1'); mask[[0, -1]] = 0
        controls = torch.zeros(33, 3, a.height, a.width, device='cuda:1')
        def pixels(c):
            delta = (base + mask * F.interpolate(c, (720, 1280), mode='bilinear', align_corners=False)).clamp(-eps, eps)
            value = (rgb + delta).clamp(0, 1)
            rounded = torch.floor(value * 255 + .5) / 255
            return value + (rounded - value).detach()
        def encode(value):
            small = F.interpolate(value, (a.height, a.width), mode='bilinear', align_corners=False, antialias=True)
            return model.vae.encode([(small * 2 - 1).permute(1, 0, 2, 3)])[0].float().unsqueeze(0)
        with torch.no_grad():
            reference = encode(rgb).to('cuda:0')
        initial_state = torch.randn(reference.shape, generator=torch.Generator(device='cpu').manual_seed(a.seed)).to('cuda:0')
        schedule = shifted_timesteps(20, shift=model.flow_shift, device='cpu').tolist()
        times = schedule if final_rgb else schedule[:a.rollout_steps + 1]
        book = TurboPerFrameCodebook(K=16384, M=64, frame_shape=(reference.shape[1], *reference.shape[-2:]), seed=a.seed, device=torch.device('cuda:0'))
        report['times'] = times
        report['device_map'] = {k: str(v) for k, v in mapping.items()}
        with torch.no_grad():
            rgb_reference = (F.interpolate(rgb, (a.height, a.width), mode='bilinear',
                                          align_corners=False, antialias=True)
                             if final_rgb else None)

        def terminal(latent, grad=False):
            return full_trajectory(
                latent, reference, initial_state, times, 17,
                lambda x, t: model.predict_velocity(x, t, embeds, cond),
                lambda r, s, f: book.select_atoms(r, s, f)[2],
                backward_steps=a.backward_steps if grad else 0,
                predict_grad=lambda x, t: model.predict_velocity_with_grad(x, t, embeds, cond),
                correct_grad=lambda r, s, f: correction(book, r, s, f, a.pool_size),
            )

        def decoded_score(state):
            video = model.vae.decode([state[0].to('cuda:1')])[0]
            reconstructed = (video.permute(1, 0, 2, 3) / 2 + .5).clamp(0, 1)
            return rgb_distortion(reconstructed, rgb_reference)

        def rgb_target_gradient(latent_value):
            # Never hold the VAE decoder and the DiT suffix graphs at once.
            with torch.no_grad():
                hard_end, _ = terminal(latent_value)
            end_leaf = hard_end.detach().requires_grad_(True)
            score = decoded_score(end_leaf)
            end_grad, = torch.autograd.grad(score, end_leaf)
            del score, end_leaf
            if not torch.isfinite(end_grad).all() or not torch.any(end_grad):
                raise RuntimeError('Final RGB gradient is zero or non-finite')
            target_leaf = latent_value.detach().requires_grad_(True)
            replay, _ = terminal(target_leaf, grad=True)
            if not torch.allclose(replay.detach(), hard_end, atol=1e-5, rtol=1e-4):
                raise RuntimeError('Tracked replay changed the hard-forward final state')
            latent_grad, = torch.autograd.grad(replay, target_leaf, grad_outputs=end_grad.detach())
            return latent_grad.detach()

        def objective(latent, grad):
            predictor = model.predict_velocity_with_grad if grad else model.predict_velocity
            return rollout_loss(latent, reference, initial_state, times,
                                lambda x, t: predictor(x, t, embeds, cond),
                                lambda r, s, f: correction(book, r, s, f, a.pool_size, grad))
        @torch.no_grad()
        def evaluate(c):
            latent = encode(pixels(c)).to('cuda:0')
            if final_rgb:
                state, early = terminal(latent)
                score = decoded_score(state)
                if not torch.isfinite(score):
                    raise RuntimeError('Non-finite final RGB score')
                return float(score), early
            score, terms = objective(latent, False)
            if not torch.isfinite(score):
                raise RuntimeError('Non-finite hard rollout score')
            return float(score), terms.cpu().tolist()
        done = 0
        if checkpoint.exists():
            with np.load(checkpoint, allow_pickle=False) as z:
                if str(z['fingerprint'].item()) != fingerprint:
                    raise ValueError('Checkpoint configuration mismatch')
                controls = torch.from_numpy(z['controls'].copy()).to('cuda:1')
                done = int(z['done'])
                report['history'] = json.loads(str(z['history'].item()))
                report['initial_score'] = float(z['initial_score'])
        score, terms = evaluate(controls)
        report.setdefault('initial_score', score)
        print(f'[ROLLOUT] initial/resumed hard score={score:.8g}; completed={done}', flush=True)
        for iteration in range(done, a.iterations):
            tick = time.monotonic()
            if final_rgb:
                print(f'[BACKWARD] iteration {iteration+1}/{a.iterations}, final {a.backward_steps} steps', flush=True)
            with torch.no_grad():
                latent_value = encode(pixels(controls)).to('cuda:0')
            if final_rgb:
                latent_grad = rgb_target_gradient(latent_value)
            else:
                leaf = latent_value.detach().requires_grad_(True)
                loss, terms_tensor = objective(leaf, True)
                latent_grad, = torch.autograd.grad(loss, leaf)
                del loss, terms_tensor, leaf
            latent_grad = latent_grad.detach().to('cuda:1')
            del latent_value
            c = controls.detach().requires_grad_(True)
            encoded = encode(pixels(c))
            grad, = torch.autograd.grad(encoded, c, grad_outputs=latent_grad)
            if not torch.isfinite(grad).all() or grad.abs().max().item() == 0:
                raise RuntimeError('Pixel-control gradient is zero or non-finite; no candidate accepted')
            grad_max = float(grad.abs().max())
            direction = grad.sign().detach()
            del encoded, c, grad, latent_grad
            accepted_scale = 0.
            trials = []
            for scale in [1., .5]:
                proposal = (controls + scale * a.alpha / 255 * direction).clamp(-2 * eps, 2 * eps)
                new_score, new_terms = evaluate(proposal)
                trials.append(dict(scale=scale, score=new_score))
                if new_score > score:
                    controls, score, terms = proposal.detach(), new_score, new_terms
                    accepted_scale = scale
                    break
            row = dict(iteration=iteration + 1, score=score, checkpoint_MSE=terms,
                       accepted_scale=accepted_scale, trials=trials, grad_max=grad_max, seconds=time.monotonic()-tick)
            report['history'].append(row)
            temp = checkpoint.with_suffix('.tmp.npz')
            np.savez_compressed(temp, controls=controls.cpu().numpy(), done=iteration+1,
                                fingerprint=fingerprint, initial_score=report['initial_score'], history=json.dumps(report['history']))
            os.replace(temp, checkpoint)
            atomic_json(report_path, report)
            print('[PGD] ' + json.dumps(row), flush=True)
        with torch.no_grad():
            candidate = (pixels(controls)-rgb).permute(0, 2, 3, 1).cpu().numpy()
        random = matched_random(original, candidate, a.seed)
        if not np.allclose(candidate[[0, -1]], init[[0, -1]], atol=1e-7):
            raise RuntimeError('Boundary invariant failed')
        for name, delta in [('initial', init), ('candidate', candidate), ('random', random)]:
            if not np.isfinite(delta).all() or abs(delta).max()*255 > a.epsilon+1e-3:
                raise RuntimeError('Output perturbation invalid')
            np.savez_compressed(out / (name+'.npz'), delta=delta)
        report.update(status='completed', final_score=score, proxy_improved=score>report['initial_score'],
                      seconds=time.monotonic()-start, candidate_input_MSE=float(np.mean(candidate**2)),
                      random_input_MSE=float(np.mean(random**2)), candidate_Linf_pixel=float(abs(candidate).max()*255),
                      asset_sha256={n+'.npz': digest(out/(n+'.npz')) for n in ['initial', 'candidate', 'random']})
        if final_rgb:
            report.update(initial_proxy_rgb_psnr_db=-report['initial_score'],
                          final_proxy_rgb_psnr_db=-score)
        atomic_json(report_path, report)
        print(f'[READY] {out}; proxy_improved={report["proxy_improved"]}. Full codec acceptance still required.', flush=True)
    except Exception as exc:
        report.update(status='failed', error=str(exc), seconds=time.monotonic()-start)
        atomic_json(report_path, report)
        raise


if __name__ == '__main__':
    main()
