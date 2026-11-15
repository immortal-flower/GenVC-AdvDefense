"""Evaluate already-saved GOP0 inputs through a complete low-resolution codec trajectory.

One model load; no gradients, input optimization, or new perturbation assets.
The paired native 720p runs are separate evidence, not reproduced here.
"""
import argparse
import gc
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from accelerate import dispatch_model, infer_auto_device_map

from generate_trajectory_input_pilot import (
    WanFLF2VWrapper, atomic_json, compress_boundary_frame, load_yuv420_frames,
    resize_frames, tensor_conditions,
)
from sde_rf_wan.ref_codec import _compressai_cache
from sde_rf_wan.sde_convert import shifted_timesteps
from sde_rf_wan.turbo_codebook import TurboPerFrameCodebook
from trajectory_input_objective import full_hard_forward


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--wan_ckpt', required=True, type=Path)
    parser.add_argument('--assets', required=True, type=Path,
                        help='Existing directory containing initial/candidate/random.npz')
    parser.add_argument('--output', required=True, type=Path)
    a = parser.parse_args()
    names = ('initial', 'candidate', 'random')
    files = [a.data, a.wan_ckpt / 'Wan2.1_VAE.pth',
             *(a.assets / (name + '.npz') for name in names)]
    for path in files:
        if not path.is_file():
            parser.error(f'Missing file: {path}')
    if torch.cuda.device_count() != 2:
        parser.error('Expose exactly two lab CUDA GPUs via CUDA_VISIBLE_DEVICES')
    print('[GPU] ' + ' / '.join(torch.cuda.get_device_name(i) for i in range(2)), flush=True)
    start = time.monotonic()
    frames = load_yuv420_frames(str(a.data), 33)
    if len(frames) != 33 or any(f.size != (1280, 720) for f in frames):
        raise ValueError('Expected Jockey GOP0: 33 native 1280x720 frames')
    raw = np.stack([np.asarray(f) for f in frames])
    source = raw.astype(np.float32) / 255
    small_size = (128, 224)

    def prepare(name):
        with np.load(a.assets / (name + '.npz'), allow_pickle=False) as saved:
            delta = saved['delta'].astype(np.float32)
        if delta.shape != source.shape or not np.isfinite(delta).all() or np.abs(delta).max() > 4/255 + 1e-6:
            raise ValueError(f'{name}: expected finite GOP0 RGB delta within 4/255')
        submitted = np.floor(np.clip(source + delta, 0, 1) * 255 + .5).astype(np.uint8)
        actual = submitted.astype(np.int16) - raw
        return submitted, dict(input_mse=float(np.mean(actual**2, dtype=np.float64)) / 255**2,
                               linf_pixel=int(np.abs(actual).max()))

    def downsample(pixels):
        x = torch.from_numpy(pixels).permute(0, 3, 1, 2).to('cuda:1', dtype=torch.float32) / 255
        return F.interpolate(x, small_size, mode='bilinear', align_corners=False, antialias=True)

    with torch.no_grad():
        scores = {}
        inputs = {}
        boundaries = None
        for name in names:
            submitted, scores[name] = prepare(name)
            current = submitted[[0, -1]]
            if boundaries is None:
                boundaries = current
            elif not np.array_equal(boundaries, current):
                raise ValueError(f'{name}: first/last submitted frames differ')
            inputs[name] = downsample(submitted)
        if abs(scores['candidate']['input_mse'] - scores['random']['input_mse']) > 1e-9:
            raise ValueError('Candidate and random native input MSE does not match')
        reference_rgb = downsample(raw)
        ref_frames = [compress_boundary_frame(Image.fromarray(f), ref_quality=4)[0]
                      for f in boundaries]
        _compressai_cache.clear()
        gc.collect()
        torch.cuda.empty_cache()
        model = WanFLF2VWrapper(str(a.wan_ckpt))
        model.load('cpu', torch.bfloat16)
        model.model.eval().requires_grad_(False)
        model.device = torch.device('cuda:1')
        model.text_encoder.model.to('cuda:1')
        embeds = [v.cpu() for v in model.encode_prompt('')['prompt_embeds']]
        model.text_encoder.model.to('cpu')
        model.vae.model.to('cuda:1').eval()
        model.clip.model.to('cuda:1').eval()
        ref_frames = resize_frames(ref_frames, small_size[1], small_size[0])
        boundary_tensor = torch.from_numpy(np.stack([np.asarray(f) for f in ref_frames])).permute(0, 3, 1, 2)
        cond = tensor_conditions(model, boundary_tensor.to('cuda:1', dtype=torch.float32) / 255, 33)
        model.clip.model.to('cpu')
        torch.cuda.empty_cache()
        mapping = infer_auto_device_map(model.model, max_memory={0: '20GiB', 1: '20GiB'},
                                        no_split_module_classes=['WanAttentionBlock'])
        if any(v not in (0, 1, 'cuda:0', 'cuda:1') for v in mapping.values()):
            raise RuntimeError(f'DiT does not fit on the two GPUs: {mapping}')
        model.model = dispatch_model(model.model, device_map=mapping)
        model.device = torch.device('cuda:0')
        embeds = [v.to(model.device) for v in embeds]
        cond = {k: v.to(model.device) for k, v in cond.items()}

        def encode(rgb):
            video = (2 * rgb - 1).permute(1, 0, 2, 3)
            return model.vae.encode([video])[0].unsqueeze(0).to(model.device)

        ref_latent = encode(reference_rgb)
        noise = torch.randn(ref_latent.shape, generator=torch.Generator().manual_seed(42)).to(model.device)
        times = shifted_timesteps(20, shift=model.flow_shift, device='cpu').tolist()
        book = TurboPerFrameCodebook(K=16384, M=64, seed=42, device=model.device,
                                    frame_shape=(ref_latent.shape[1], *ref_latent.shape[-2:]))
        setup_seconds = time.monotonic() - start
        print(f'[READY] Shared model and inputs: {setup_seconds:.1f}s', flush=True)
        a.output.parent.mkdir(parents=True, exist_ok=True)
        report = dict(status='in_progress', resolution=list(small_size), seed=42,
                      schedule=times, sde_steps=17, deterministic_steps=3,
                      setup_seconds=setup_seconds, runs={},
                      note='Forward-only reduced geometry; local codebook differs from native 720p. RGB metrics use float decoder output.')
        atomic_json(a.output, report)
        # Quantize/dequantize is not needed for the forward-only float RGB
        # diagnostic; compare like for like across all three variants.
        for name in names:
            tick = time.monotonic()
            print(f'[RUN] {name}', flush=True)
            target = encode(inputs[name])
            state, early = full_hard_forward(
                target, ref_latent, noise, times, 17,
                lambda x, t: model.predict_velocity(x, t, embeds, cond),
                lambda r, step, pos: book.select_atoms(r, step, pos)[2],
                progress=lambda i: print(f'  step {i}/20', flush=True) if i % 5 == 0 else None,
            )
            out = model.vae.decode([state[0].to('cuda:1')])[0]
            recon = (out.permute(1, 0, 2, 3) / 2 + .5).clamp(0, 1)
            if recon.shape != reference_rgb.shape or not torch.isfinite(recon).all():
                raise RuntimeError(f'{name}: invalid RGB decoder output')
            frame_mse = (recon - reference_rgb).square().mean(dim=(1, 2, 3))
            result = dict(scores[name], early_latent_mse=early,
                          rgb_mse=float(frame_mse.mean()),
                          rgb_psnr_db=float((-10 * torch.log10(frame_mse + 1e-10)).mean()),
                          seconds=time.monotonic() - tick)
            report['runs'][name] = result
            atomic_json(a.output, report)
            print(f'[DONE] {name}: PSNR={result["rgb_psnr_db"]:.4f} dB; '
                  f'time={result["seconds"]:.1f}s', flush=True)
        runs = report['runs']
        report.update(status='completed', total_seconds=time.monotonic()-start,
                      candidate_psnr_drop_vs_random_db=runs['random']['rgb_psnr_db']-runs['candidate']['rgb_psnr_db'],
                      candidate_psnr_drop_vs_initial_db=runs['initial']['rgb_psnr_db']-runs['candidate']['rgb_psnr_db'])
        atomic_json(a.output, report)
        print(f'[SAVED] {a.output} ({report["total_seconds"]:.1f}s)', flush=True)


if __name__ == '__main__':
    main()
