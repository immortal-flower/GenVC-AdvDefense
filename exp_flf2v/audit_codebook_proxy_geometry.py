"""Check whether the reduced-resolution proxy preserves clean codebook ranking.

This scans the real deterministic K-bank once but loads no Wan model and does
not decode video. Agreement is measured only within the clean top-N candidate
pool; even high agreement cannot establish full-K attacked-stream agreement.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from generate_codebook_input_pilot import candidate_bank
from sde_rf_wan.turbo_codebook import TurboBitstream


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bitstream", type=Path, required=True)
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--position", type=int, default=3)
    p.add_argument("--candidate_count", type=int, default=80)
    p.add_argument("--opt_height", type=int, default=192)
    p.add_argument("--opt_width", type=int, default=336)
    a = p.parse_args()
    if not torch.cuda.is_available():
        p.error("CUDA is required for the full K-bank scan")
    stream = TurboBitstream.load(str(a.bitstream))
    if a.step != 1 or not 0 <= a.position < stream["num_latent_frames"]:
        p.error("This pilot supports step 1 and a valid latent position")
    with np.load(a.trace, allow_pickle=False) as saved:
        residual = torch.from_numpy(saved["residual_step1"][0, :, a.position].copy()).to("cuda:0")
    selected, signs = stream["step_data"][0][a.position]
    if a.candidate_count < len(selected) or a.candidate_count > stream["K"]:
        p.error("Candidate count must lie between transmitted M and K")
    with torch.no_grad():
        # Same clean candidate pool and low-resolution transform as the attack.
        atoms, indices = candidate_bank(stream, residual, 1, a.position,
                                        a.candidate_count, "cuda:0", return_indices=True)
        low_shape = (a.opt_height // 8, a.opt_width // 8)
        low_atoms = F.interpolate(atoms.view(-1, *stream["frame_shape"]),
                                  size=low_shape, mode="bilinear", align_corners=False).flatten(1)
        low_atoms = low_atoms / low_atoms.std(dim=1, keepdim=True).clamp_min(1e-8)
        low_residual = F.interpolate(residual.unsqueeze(0), size=low_shape,
                                     mode="bilinear", align_corners=False).flatten()
        scores = low_atoms @ low_residual
        proxy_selected_offsets = scores.abs().topk(len(selected)).indices.cpu().tolist()
        proxy_selected = {indices[offset] for offset in proxy_selected_offsets}
        true_selected = set(selected)
        shared = true_selected & proxy_selected
        proxy_sign = {indices[offset]: (1 if float(scores[offset]) >= 0 else -1)
                      for offset in proxy_selected_offsets}
        true_sign = dict(zip(selected, signs))
        sign_matches = sum(proxy_sign[index] == true_sign[index] for index in shared)
        ranked = scores.abs().topk(len(selected) + 1).values
    report = {
        "bitstream": str(a.bitstream),
        "trace": str(a.trace),
        "sde_step": a.step,
        "latent_position": a.position,
        "candidate_pool": a.candidate_count,
        "transmitted_atoms": len(selected),
        "clean_selected_overlap_within_pool": len(shared),
        "clean_selected_overlap_fraction": len(shared) / len(selected),
        "shared_sign_agreement": sign_matches / len(shared) if shared else None,
        "proxy_threshold_gap": float(ranked[-2] - ranked[-1]),
        "limitation": "Only a clean top-N pool is checked; attacked full-K reselection and Wan dynamics are not modeled.",
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved: {a.output}")


if __name__ == "__main__":
    main()
