"""Short latent rollout or full-forward RGB robustness pilot; not exact codec AD.

Hard full-K selection is recomputed at EVERY rollout step. Backward uses a
local soft top-N bank. VAE and DiT graphs are separated by an exact latent VJP.
Native boundary pixels are held at their initialization throughout optimization.
"""
import argparse
from contextlib import nullcontext
import gc
import hashlib
import json
import os
import shutil
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
from trajectory_input_objective import (
    correction, full_trajectory, match_grid_amplitude, rgb_distortion, rollout_loss,
    uniform_grid_direction, stepwise_trajectory_vjp,
)
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
    p.add_argument('--control_height', type=int)
    p.add_argument('--control_width', type=int)
    p.add_argument('--grid_random', action='store_true')
    p.add_argument('--epsilon', type=float, default=2.)
    p.add_argument('--alpha', type=float, default=.5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--pool_size', type=int, default=80)
    p.add_argument('--gpu0_memory', default='20GiB')
    p.add_argument('--gpu1_memory', default='20GiB')
    p.add_argument('--vae_cpu_activations', action='store_true',
                   help='Keep VAE autograd saved tensors in host RAM, not on the GPU')
    p.add_argument('--stepwise_backward', action='store_true',
                   help='Replay one trajectory step at a time for the RGB VJP')
    p.add_argument('--vae_decode_cpu', action='store_true',
                   help='Full-resolution VAE RGB decoder forward/backward on CPU; no tiling')
    p.add_argument('--direction_audit_from', type=str,
                   help='Replay +/-0.5 and +/-1 pixel steps from a completed one-update checkpoint')
    p.add_argument('--audit_random_directions', type=int, default=0)
    a = p.parse_args()
    if a.audit_random_directions < 0 or (a.audit_random_directions and not a.direction_audit_from):
        p.error('Random audit directions require direction_audit_from and a nonnegative count')
    final_rgb = a.objective == 'final-rgb'
    if a.stepwise_backward and not final_rgb:
        p.error('Stepwise backward requires final-rgb')
    if a.direction_audit_from and not final_rgb:
        p.error('Direction audit requires final-rgb')
    control_h = a.height if a.control_height is None else a.control_height
    control_w = a.width if a.control_width is None else a.control_width
    if min(control_h, control_w) < 1:
        p.error('Control grid dimensions must be positive')
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
    if os.environ.get('GVCC_REF_CODEC_DEVICE'):
        config['reference_codec_device'] = os.environ['GVCC_REF_CODEC_DEVICE']
    if not a.vae_decode_cpu:
        del config['vae_decode_cpu']
    if not a.audit_random_directions:
        del config['audit_random_directions']
    if not a.stepwise_backward:
        del config['stepwise_backward']
    audit_source = None
    if a.direction_audit_from:
        audit_source = Path(a.direction_audit_from)
        source_report = json.loads((audit_source / 'generation.json').read_text(encoding='utf-8'))
        source_config = source_report['configuration']
        for key in ('data_sha256', 'init_sha256', 'wan_ckpt', 'height', 'width',
                    'control_height', 'control_width', 'epsilon', 'alpha', 'seed',
                    'pool_size', 'backward_steps', 'gpu0_memory', 'gpu1_memory'):
            if config.get(key) != source_config.get(key):
                raise ValueError(f'Direction audit source configuration mismatch: {key}')
        config['direction_report_sha256'] = digest(audit_source / 'generation.json')
        config['direction_checkpoint_sha256'] = digest(audit_source / 'optimizer_checkpoint.npz')
    else:
        del config['direction_audit_from']
    if not a.vae_cpu_activations:
        del config['vae_cpu_activations']
    if a.control_height is None and a.control_width is None and not a.grid_random:
        del config['control_height'], config['control_width'], config['grid_random']
    if not final_rgb:  # retain compatibility with existing short-rollout checkpoints
        del config['objective'], config['backward_steps']
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if report_path.exists():
        previous = json.loads(report_path.read_text(encoding='utf-8'))
        if previous['fingerprint'] != fingerprint:
            raise ValueError('Configuration/source changed: use a new output directory')
        if previous['status'] == 'completed':
            if a.grid_random and previous.get('grid_random_distribution') != 'uniform_continuous':
                raise ValueError('Completed run used the old grid-random distribution; use a new output directory')
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
        report['backward_engine'] = 'stepwise-vjp-v1' if a.stepwise_backward else 'retained-graph'
    if a.grid_random:
        report.update(control_grid=[control_h, control_w], grid_random_mse_rtol=.01,
                      grid_random_distribution='uniform_continuous',
                      grid_random_rule='Fixed uniform [-1,1] control-grid direction centered on the shared initialization; only amplitude fitted to input MSE')
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
        print(f'[MEMORY] DiT budgets: GPU0={a.gpu0_memory}, GPU1={a.gpu1_memory}; '
              f'VAE saved tensors on CPU: {a.vae_cpu_activations}', flush=True)
        embeds = [v.to('cuda:0') for v in embeds]
        cond = {k: v.to('cuda:0') for k, v in cond.items()}
        rgb = torch.from_numpy(original).permute(0, 3, 1, 2).to('cuda:1')
        base = torch.from_numpy(init).permute(0, 3, 1, 2).to('cuda:1')
        mask = torch.ones(33, 1, 1, 1, device='cuda:1'); mask[[0, -1]] = 0
        controls = torch.zeros(33, 3, control_h, control_w, device='cuda:1')
        def lift(c):
            return mask * F.interpolate(c, (720, 1280), mode='bilinear', align_corners=False)

        def pixels_from_lift(lifted):
            delta = (base + lifted).clamp(-eps, eps)
            value = (rgb + delta).clamp(0, 1)
            rounded = torch.floor(value * 255 + .5) / 255
            return value + (rounded - value).detach()

        def pixels(c):
            return pixels_from_lift(lift(c))
        def encode(value):
            model.vae.model.to('cuda:1')
            model.vae.scale = [v.to('cuda:1') for v in model.vae.scale]
            small = F.interpolate(value, (a.height, a.width), mode='bilinear', align_corners=False, antialias=True)
            return model.vae.encode([(small * 2 - 1).permute(1, 0, 2, 3)])[0].float().unsqueeze(0)
        with torch.no_grad():
            print('[INIT] Encoding full input video with VAE', flush=True)
            reference = encode(rgb).to('cuda:0')
            print('[INIT] Input latent ready', flush=True)
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
                state_callback=lambda step, _: print(f'[FORWARD] completed step {step}/20', flush=True)
                    if step % 5 == 0 else None,
            )

        def decoded_score(state):
            decoder_device = 'cpu' if a.vae_decode_cpu else 'cuda:1'
            model.vae.model.to(decoder_device)
            model.vae.scale = [v.to(decoder_device) for v in model.vae.scale]
            if a.vae_decode_cpu:
                gc.collect(); torch.cuda.empty_cache()
                print('[RGB] Full VAE decode on CPU (no spatial/temporal tiling)', flush=True)
            video = model.vae.decode([state[0].to(decoder_device)])[0]
            reconstructed = (video.permute(1, 0, 2, 3) / 2 + .5).clamp(0, 1)
            return rgb_distortion(reconstructed, rgb_reference.to(decoder_device))

        def vae_saved_tensor_context():
            # Scope it to the VAE VJPs; the no-grad forward and DiT are unchanged.
            return (torch.autograd.graph.save_on_cpu(pin_memory=False)
                    if a.vae_cpu_activations else nullcontext())

        def rgb_target_gradient(latent_value):
            # Never hold the VAE decoder and the DiT suffix graphs at once.
            with torch.no_grad():
                hard_end, _ = terminal(latent_value)
            # Keep CPU decoder autograd entirely on CPU, then transfer only its
            # small latent gradient back to the GPU trajectory. No CPU<->CUDA
            # CopyBackward or save_on_cpu hook is needed for this decoder graph.
            end_leaf = hard_end.detach().to('cpu' if a.vae_decode_cpu else hard_end.device).requires_grad_(True)
            print('[BACKWARD] VAE decoder RGB gradient', flush=True)
            if a.vae_decode_cpu:
                np.savez_compressed(out/'decoder_probe.npz',
                    latent=end_leaf.detach().cpu().numpy(),
                    reference=rgb_reference.detach().cpu().numpy())
                print('[BACKWARD] Saved decoder_probe.npz before CPU backward', flush=True)
            with (nullcontext() if a.vae_decode_cpu else vae_saved_tensor_context()):
                score = decoded_score(end_leaf)
            end_grad, = torch.autograd.grad(score, end_leaf)
            end_grad = end_grad.to(hard_end.device)
            del score, end_leaf
            if not torch.isfinite(end_grad).all() or not torch.any(end_grad):
                raise RuntimeError('Final RGB gradient is zero or non-finite')
            if a.stepwise_backward:
                print('[BACKWARD] One-step-at-a-time trajectory VJP', flush=True)
                return stepwise_trajectory_vjp(
                    latent_value, reference, initial_state, times, 17,
                    lambda x, t: model.predict_velocity(x, t, embeds, cond),
                    lambda r, s, f: book.select_atoms(r, s, f)[2],
                    lambda x, t: model.predict_velocity_with_grad(x, t, embeds, cond),
                    lambda r, s, f: correction(book, r, s, f, a.pool_size),
                    end_grad, backward_steps=a.backward_steps, expected_end=hard_end,
                    progress=lambda step: print(f'[VJP] completed step {step}/20', flush=True),
                )
            target_leaf = latent_value.detach().requires_grad_(True)
            print('[BACKWARD] DiT trajectory suffix gradient', flush=True)
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
        if audit_source is not None:
            from rgb_direction_audit import recover_first_direction, assert_same_submitted_pixels
            with np.load(audit_source / 'optimizer_checkpoint.npz', allow_pickle=False) as saved:
                direction_np, accepted_step = recover_first_direction(source_report, saved)
            direction = torch.from_numpy(direction_np).to(controls)
            if direction.shape != controls.shape:
                raise ValueError('Direction control-grid shape mismatch')
            for name, sha in source_report['asset_sha256'].items():
                if digest(audit_source / name) != sha:
                    raise ValueError(f'Source asset changed: {name}')
            with torch.no_grad():
                reproduced = (pixels(direction * (accepted_step / 255)) - rgb).permute(0, 2, 3, 1).cpu().numpy()
            with np.load(audit_source / 'candidate.npz', allow_pickle=False) as saved:
                replay_check = assert_same_submitted_pixels(
                    original, reproduced, saved['delta'], 'Recovered candidate')
            cases = [('initial', 0.), ('positive_half', .5), ('negative_half', -.5),
                     ('positive_one', 1.), ('negative_one', -1.)]
            directions = {name: direction for name, _ in cases}
            if a.audit_random_directions:
                for magnitude in (.25, 2.):
                    for sign, label in ((1, 'positive'), (-1, 'negative')):
                        name = f'{label}_step{magnitude}'
                        cases.append((name, sign * magnitude))
                        directions[name] = direction
                for k in range(a.audit_random_directions):
                    rng = torch.Generator(device='cpu').manual_seed(a.seed + 20000 + k)
                    random_direction = (torch.randint(0, 2, direction.shape, generator=rng) * 2 - 1).to(direction)
                    for magnitude in (.25, .5, 1., 2.):
                        for sign, label in ((1, 'plus'), (-1, 'minus')):
                            name = f'random{k}_{label}_step{magnitude}'
                            cases.append((name, sign * magnitude))
                            directions[name] = random_direction
            report.update(kind='fixed_initialization_direction_audit', direction_reproduces_candidate=True,
                          source_accepted_step_pixel=accepted_step, rows=[], asset_sha256={},
                          input_equality_domain='codec-submitted uint8 RGB', replay_check=replay_check)
            for name, step in cases:
                tick = time.monotonic()
                c = directions[name] * (step / 255)
                with torch.no_grad():
                    delta = (pixels(c) - rgb).permute(0, 2, 3, 1).cpu().numpy()
                boundary_check = assert_same_submitted_pixels(
                    original[[0, -1]], delta[[0, -1]], init[[0, -1]], 'Boundary frames')
                if not np.isfinite(delta).all() or np.abs(delta).max() * 255 > a.epsilon + 1e-3:
                    raise RuntimeError('Direction audit exceeded perturbation bound')
                path = out / (name + '.npz')
                reuse_variant = None
                if step == 0:
                    with np.load(audit_source / 'initial.npz', allow_pickle=False) as saved:
                        assert_same_submitted_pixels(original, delta, saved['delta'], 'Initial frames')
                    reuse_variant = 'initial'
                elif step == accepted_step and not name.startswith('random'):
                    reuse_variant = 'candidate'
                if reuse_variant:
                    shutil.copyfile(audit_source / (reuse_variant + '.npz'), path)
                else:
                    np.savez_compressed(path, delta=delta)
                if step == 0:
                    score = source_report['initial_score']
                elif step == accepted_step and not name.startswith('random'):
                    score = source_report['final_score']
                else:
                    score, _ = evaluate(c)
                row = dict(variant=name, step_pixel=step, proxy_rgb_psnr_db=-score,
                           input_MSE=float(np.mean(delta**2)),
                           input_Linf_pixel=float(np.abs(delta).max() * 255),
                           seconds=time.monotonic()-tick, boundary_check=boundary_check)
                report['rows'].append(row)
                report['asset_sha256'][name + '.npz'] = digest(path)
                atomic_json(report_path, report)
                print('[DIRECTION] ' + json.dumps(row), flush=True)
            report.update(status='completed', seconds=time.monotonic()-start,
                          note='Signed PGD direction diagnostic; step pairs are not input-MSE matched. No proxy gate.')
            atomic_json(report_path, report)
            return
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
            if a.vae_cpu_activations:
                print('[BACKWARD] VAE encoder/input gradient', flush=True)
            with vae_saved_tensor_context():
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
            candidate_d = pixels(controls) - rgb
            candidate = candidate_d.permute(0, 2, 3, 1).cpu().numpy()
            if a.grid_random:
                target_mse = float(candidate_d.square().mean())
                grid = uniform_grid_direction(controls.shape, a.seed + 10001)
                lifted = lift(grid.to(device=controls.device, dtype=controls.dtype))
                amp, grid_mse = match_grid_amplitude(
                    lambda v: float((pixels_from_lift(lifted * (v / 255)) - rgb).square().mean()),
                    target_mse, 2 * a.epsilon,
                )
                random = (pixels_from_lift(lifted * (amp / 255))-rgb).permute(0, 2, 3, 1).cpu().numpy()
                report.update(grid_random_amplitude_pixel=amp, grid_random_mse_relative_gap=abs(grid_mse-target_mse)/target_mse)
            else:
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
