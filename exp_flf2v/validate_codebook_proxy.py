"""Check a clean GVCC trajectory against its real transmitted codebook atoms.

This is a lightweight prerequisite for input optimization, not an attack run.
It rejects mismatched traces/bitstreams and reports whether the differentiable
soft correction resembles the actual hard codebook correction at a clean cell.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sde_rf_wan.codebook_surrogate import soft_codebook_correction
from sde_rf_wan.turbo_codebook import TurboBitstream, TurboPerFrameCodebook


def check_cell(bitstream, trace, step, position, device):
    data = TurboBitstream.load(str(bitstream))
    if step not in (1, 2) or step > data["num_sde_steps"]:
        raise ValueError("Clean diagnostics contain only SDE steps 1 and 2")
    if not 0 <= position < data["num_latent_frames"]:
        raise ValueError("Latent position out of range")
    with np.load(trace) as saved:
        needed = ("target_latent", f"residual_step{step}", f"noise_step{step}")
        missing = [key for key in needed if key not in saved]
        if missing:
            raise ValueError(f"Missing clean-trace arrays: {missing}")
        target_shape = saved["target_latent"].shape
        residual = torch.from_numpy(saved[f"residual_step{step}"][0, :, position].copy()).to(device)
        actual = torch.from_numpy(saved[f"noise_step{step}"][0, :, position].copy()).to(device)
    if target_shape != (1, data["frame_shape"][0], data["num_latent_frames"],
                        *data["frame_shape"][1:]):
        raise ValueError(f"Trace/bitstream latent geometry mismatch: {target_shape}")
    if tuple(residual.shape) != data["frame_shape"] or actual.shape != residual.shape:
        raise ValueError("Trace residual/noise geometry differs from bitstream")

    indices, signs = data["step_data"][step - 1][position]
    book = TurboPerFrameCodebook(
        K=data["K"], M=data["M"], frame_shape=data["frame_shape"],
        seed=data["seed"], device=torch.device(device),
    )
    with torch.no_grad():
        atoms = book.regenerate_selected_atoms(indices, step - 1, position)
        scores = atoms @ residual.flatten()
        predicted_signs = torch.where(scores >= 0, 1, -1)
        transmitted = torch.tensor(signs, device=device)
        sign_agreement = float((predicted_signs == transmitted).float().mean())
        hard = (transmitted.float() @ atoms)
        hard = hard / hard.std().clamp_min(1e-8)
        actual_flat = actual.flatten()
        hard_cosine = float(F.cosine_similarity(hard, actual_flat, dim=0))
        hard_mse = float((hard - actual_flat).square().mean())
        soft = soft_codebook_correction(residual.flatten(), atoms, len(indices))
        soft_cosine = float(F.cosine_similarity(soft, actual_flat, dim=0))

    return dict(
        bitstream=str(bitstream), trace=str(trace), step=step, latent_position=position,
        atoms=len(indices), sign_agreement=sign_agreement,
        hard_vs_trace_cosine=hard_cosine, hard_vs_trace_MSE=hard_mse,
        soft_vs_trace_cosine=soft_cosine,
        selected_only=True,
        limitation="Selected-only bank validates local correction, not full-K reselection",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bitstream", required=True, type=Path)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--latent_position", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.bitstream.is_file() or not args.trace.is_file():
        parser.error("The clean bitstream and clean input_attack_trace.npz must exist")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; specify --device cpu only for small test files")
    result = check_cell(args.bitstream, args.trace, args.step, args.latent_position, args.device)
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if result["sign_agreement"] < 1 or result["hard_vs_trace_cosine"] < 0.999:
        raise RuntimeError("Clean trace does not match transmitted atoms; do not optimize")


if __name__ == "__main__":
    main()
