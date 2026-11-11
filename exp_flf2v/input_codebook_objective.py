"""Experimental input-to-codebook gradient path for one FLF2V latent cell.

The clean Wan trajectory and its residual are held fixed. Only the attacked
video passes through the frozen VAE with autograd; its latent displacement is
added to that clean residual. This first-order surrogate is NOT a full codec
gradient and must be checked by a fresh normal encode/decode after optimization.
"""

import torch

from sde_rf_wan.codebook_surrogate import (
    soft_codebook_correction,
    uncovered_residual_ratio,
)


def straight_through_8bit(video: torch.Tensor) -> torch.Tensor:
    """Use the exact 8-bit forward input while retaining a pixel gradient."""
    clipped = video.clamp(0.0, 1.0)
    quantized = (clipped * 255.0).round() / 255.0
    return clipped + (quantized - clipped).detach()


def input_codebook_loss(
    vae,
    attacked_rgb: torch.Tensor,
    clean_latent: torch.Tensor,
    clean_residual: torch.Tensor,
    atoms: torch.Tensor,
    latent_position: int,
    m: int,
    gain: float,
    *,
    selection_temperature: float = 0.1,
    sign_temperature: float = 0.1,
) -> torch.Tensor:
    """Return an uncovered-residual proxy differentiable to input pixels.

    Inputs are native resolution. ``attacked_rgb`` is ``(F,3,H,W)`` in [0,1];
    clean latent/residual are ``(C,F_lat,H_lat,W_lat)`` from the same clean
    codec trajectory; atoms are the real deterministic bank ``(K,C*H_lat*W_lat)``
    at this step/latent position. Neither the model prediction nor the
    discrete codebook selection is differentiated here.
    """
    if attacked_rgb.ndim != 4 or attacked_rgb.shape[1] != 3:
        raise ValueError("Expected attacked RGB video (F,3,H,W)")
    if clean_latent.ndim != 4 or clean_residual.shape != clean_latent.shape:
        raise ValueError("Clean latent and residual must both be (C,F_lat,H_lat,W_lat)")
    if not 0 <= latent_position < clean_latent.shape[1]:
        raise ValueError("Latent position outside clean latent")

    quantized = straight_through_8bit(attacked_rgb)
    video = quantized.permute(1, 0, 2, 3) * 2.0 - 1.0
    encoded = vae.encode([video])[0].float()
    if encoded.shape != clean_latent.shape:
        raise ValueError(f"Attacked latent shape {encoded.shape} differs from clean {clean_latent.shape}")
    clean_latent = clean_latent.to(device=encoded.device, dtype=encoded.dtype)
    clean_residual = clean_residual.to(device=encoded.device, dtype=encoded.dtype)
    atoms = atoms.to(device=encoded.device, dtype=encoded.dtype)
    residual = (clean_residual[:, latent_position] +
                encoded[:, latent_position] - clean_latent[:, latent_position]).flatten()
    correction = soft_codebook_correction(
        residual, atoms, m,
        selection_temperature=selection_temperature,
        sign_temperature=sign_temperature,
    )
    return uncovered_residual_ratio(residual, correction, gain)
