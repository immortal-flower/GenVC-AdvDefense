"""Generate one bounded input candidate with a clean-trajectory codebook proxy.

Research pilot only: the Wan predictor is frozen at its CLEAN trajectory and
the VAE is optimized at reduced resolution. Normal full-resolution GVCC
re-encoding is mandatory for evaluation; this script never reuses a codebook.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generate_vae_pgd import load_yuv420_frames, resize_frames, WanVAE
from generate_vae_fullcontext_pgd import lift_controls, resize_video
from generate_vae_temporal_pgd import temporal_weights
from input_codebook_objective import input_codebook_loss_from_latent, straight_through_8bit
from sde_rf_wan.turbo_codebook import TurboBitstream, TurboPerFrameCodebook


def candidate_bank(stream, residual, step, position, count, device):
    """Select the strongest clean-residual atoms from the *actual* K-bank."""
    indices, signs = stream["step_data"][step - 1][position]
    if count < len(indices):
        raise ValueError("Candidate count cannot be smaller than transmitted M")
    book = TurboPerFrameCodebook(
        K=stream["K"], M=stream["M"], frame_shape=stream["frame_shape"],
        seed=stream["seed"], device=torch.device(device),
    )
    gen = torch.Generator(device=device).manual_seed(book._sf_seed(step - 1, position))
    scores = torch.empty(book.K, device=device)
    flat = residual.flatten().to(device)
    with torch.no_grad():
        for start in range(0, book.K, book.gen_batch):
            batch = book._generate_atoms_batch(gen, min(book.gen_batch, book.K - start))
            scores[start:start + len(batch)] = batch @ flat
            del batch
        candidates = scores.abs().topk(count).indices.sort().values.tolist()
        if not set(indices).issubset(candidates):
            raise RuntimeError("Clean transmitted indices disagree with scanned bank")
        chosen_signs = torch.where(scores[indices] >= 0, 1, -1).cpu().tolist()
        if chosen_signs != signs:
            raise RuntimeError("Clean transmitted signs disagree with residual; wrong trace/bitstream")
        atoms = book.regenerate_selected_atoms(candidates, step - 1, position)
    return atoms


@torch.enable_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--vae", type=Path, required=True)
    p.add_argument("--bitstream", type=Path, required=True)
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--start_frame", type=int, default=0)
    p.add_argument("--latent_position", type=int, default=3)
    p.add_argument("--candidate_count", type=int, default=80)
    p.add_argument("--noise_coeff", type=float, required=True)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--epsilon", type=float, default=2)
    p.add_argument("--alpha", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--opt_height", type=int, default=192)
    p.add_argument("--opt_width", type=int, default=336)
    a = p.parse_args()
    if min(a.steps, a.epsilon, a.alpha, a.noise_coeff, a.opt_height, a.opt_width) <= 0:
        p.error("Positive steps, epsilon, alpha, noise coefficient and dimensions required")
    if a.start_frame < 0 or a.opt_height % 16 or a.opt_width % 16:
        p.error("Invalid start frame or VAE optimization dimensions")
    for path in (a.data, a.vae, a.bitstream, a.trace):
        if not path.is_file():
            p.error(f"Missing prerequisite: {path}")
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        p.error("CUDA is required for the Wan VAE")
    torch.manual_seed(a.seed)
    stream = TurboBitstream.load(str(a.bitstream))
    if stream["num_frames"] != 33 or stream["height"] != 720 or stream["width"] != 1280:
        p.error("This pilot accepts a 33-frame 720p clean bitstream only")
    if not 0 <= a.latent_position < stream["num_latent_frames"]:
        p.error("Latent position outside bitstream")
    with np.load(a.trace) as trace:
        if "target_latent" not in trace or "residual_step1" not in trace:
            p.error("Clean trace must contain target_latent and residual_step1")
        native_latent = torch.from_numpy(trace["target_latent"][0].copy())
        native_residual = torch.from_numpy(trace["residual_step1"][0].copy())
    expected = (stream["frame_shape"][0], stream["num_latent_frames"],
                *stream["frame_shape"][1:])
    if native_latent.shape != expected or native_residual.shape != expected:
        p.error("Clean trace and bitstream geometry differ")

    frames = resize_frames(load_yuv420_frames(str(a.data), 33, start_frame=a.start_frame), 1280, 720)
    if len(frames) != 33:
        p.error("Input has fewer than 33 frames at requested start")
    original = np.stack([np.asarray(frame) for frame in frames]).copy()
    target = torch.from_numpy(original).permute(3, 0, 1, 2).float().to(device) / 255
    native_atoms = candidate_bank(stream, native_residual[:, a.latent_position],
                                  1, a.latent_position, a.candidate_count, device)
    low_shape = (a.opt_height // 8, a.opt_width // 8)
    atoms = F.interpolate(native_atoms.view(-1, *stream["frame_shape"]),
                          size=low_shape, mode="bilinear", align_corners=False)
    atoms = atoms.flatten(1)
    atoms = atoms / atoms.std(dim=1, keepdim=True).clamp_min(1e-8)
    del native_atoms

    def downsample_latent(z):
        # The VAE has (C,T,H,W); interpolate only spatial dimensions.
        return F.interpolate(z.permute(1, 0, 2, 3), size=low_shape,
                             mode="bilinear", align_corners=False).permute(1, 0, 2, 3)

    residual_small = downsample_latent(native_residual).to(device)
    clean_small = resize_video(target, (a.opt_height, a.opt_width))
    vae = WanVAE(vae_pth=str(a.vae), dtype=torch.bfloat16, device=device)
    with torch.no_grad():
        clean_latent = vae.encode([clean_small * 2 - 1])[0].float()
    if clean_latent.shape != residual_small.shape:
        raise RuntimeError("Low-resolution VAE latent and residual geometry differ")
    weights = torch.from_numpy(temporal_weights(33, 3)).to(device)
    eps, alpha = a.epsilon / 255, a.alpha / 255
    controls = torch.empty(3, 3, a.opt_height, a.opt_width, device=device).uniform_(-eps, eps)
    mask = torch.ones(1, 33, 1, 1, device=device)
    mask[:, 0] = 0
    mask[:, -1] = 0
    history = []
    for step in range(a.steps):
        c = controls.detach().requires_grad_(True)
        delta = lift_controls(c, weights, (720, 1280)) * mask
        quantized = straight_through_8bit(target + delta)
        attacked_small = resize_video(quantized, (a.opt_height, a.opt_width))
        encoded = vae.encode([attacked_small * 2 - 1])[0].float()
        if not encoded.requires_grad:
            raise RuntimeError("VAE encoding detached from the input; check grad mode")
        objective = input_codebook_loss_from_latent(
            encoded, clean_latent, residual_small, atoms,
            a.latent_position, stream["M"], a.noise_coeff,
        )
        if not objective.requires_grad:
            raise RuntimeError("Codebook proxy objective detached from the input")
        grad, = torch.autograd.grad(objective, c)
        if not torch.isfinite(grad).all() or grad.abs().max() == 0:
            raise RuntimeError("Non-finite or zero pixel gradient")
        with torch.no_grad():
            controls = (c + alpha * grad.sign()).clamp(-eps, eps).detach()
        history.append(float(objective.detach()))
        print(f"Proxy step {step+1}/{a.steps}: uncovered ratio={history[-1]:.6f}", flush=True)
        del c, delta, quantized, attacked_small, encoded, objective, grad

    with torch.no_grad():
        delta = (lift_controls(controls, weights, (720, 1280)) * mask)
        delta = delta.permute(1, 2, 3, 0).cpu().numpy()
    pixels = np.floor(np.clip(original.astype(np.float32) / 255 + delta, 0, 1) * 255 + 0.5).astype(np.uint8)
    magnitude = np.abs(pixels.astype(np.int16) - original.astype(np.int16))
    rng = np.random.default_rng(a.seed + 1)
    signs = rng.integers(0, 2, size=original.shape[1:], dtype=np.int16) * 2 - 1
    random_delta = magnitude * signs
    invalid = (original.astype(np.int16) + random_delta < 0) | (original.astype(np.int16) + random_delta > 255)
    random_delta[invalid] *= -1
    if np.any((original.astype(np.int16) + random_delta < 0) |
              (original.astype(np.int16) + random_delta > 255)):
        raise RuntimeError("Matched random control cannot preserve pixel magnitudes")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.output, delta=delta.astype(np.float32))
    random_path = a.output.with_name(a.output.stem + "_matched.npz")
    np.savez_compressed(random_path, delta=(random_delta / 255).astype(np.float32))
    input_error = (pixels.astype(np.float32) - original.astype(np.float32)) / 255.0
    report = dict(vars(a), proxy_only=True, clean_trajectory_fixed=True,
                  full_K_reselection=False, candidate_count=a.candidate_count,
                  objective_history=history, random_asset=str(random_path),
                  quantized_input_MSE=float(np.square(input_error).mean()))
    report = {k: str(v) if isinstance(v, Path) else v for k, v in report.items()}
    a.output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved: {a.output} and {random_path}", flush=True)


if __name__ == "__main__":
    main()
