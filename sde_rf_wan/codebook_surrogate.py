"""Differentiable *surrogate* for one Turbo-DDCM codebook correction.

This is deliberately not the production encoder: its soft threshold and soft
sign approximate the discrete top-M/sign operation. A real attack must still
be evaluated by normal re-encoding and decoding, with no reused clean stream.
"""

import torch


def soft_codebook_correction(
    residual: torch.Tensor,
    atoms: torch.Tensor,
    m: int,
    *,
    selection_temperature: float = 0.1,
    sign_temperature: float = 0.1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Approximate the signed top-M combination for flattened atoms.

    ``residual`` has shape ``(D,)``; ``atoms`` has shape ``(K,D)``. The bank
    must use the actual shared codebook seed and geometry if this is used in
    an experiment. The top-M threshold is detached, but scores and soft signs
    remain differentiable with respect to the residual.
    """
    if residual.ndim != 1 or atoms.ndim != 2 or atoms.shape[1] != residual.numel():
        raise ValueError("Expected residual (D,) and atoms (K,D)")
    if not 1 <= m <= atoms.shape[0]:
        raise ValueError("Expected 1 <= m <= K")
    if selection_temperature <= 0 or sign_temperature <= 0 or eps <= 0:
        raise ValueError("Temperatures and eps must be positive")

    scores = atoms @ residual
    threshold = scores.detach().abs().topk(m).values[-1]
    weights = torch.sigmoid((scores.abs() - threshold) / selection_temperature)
    weights = weights * (m / weights.sum().clamp_min(eps))
    soft_signs = torch.tanh(scores / sign_temperature)
    combined = (weights * soft_signs) @ atoms
    return combined / combined.std().clamp_min(eps)


def uncovered_residual_ratio(
    residual: torch.Tensor,
    correction: torch.Tensor,
    gain: float,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Fractional residual energy after a candidate correction.

    ``gain`` must be the physical SDE noise coefficient for the targeted step.
    This is only a ranking proxy, not a reconstruction-quality metric.
    """
    if residual.shape != correction.shape:
        raise ValueError("Residual and correction shapes must match")
    if gain < 0:
        raise ValueError("Gain must be non-negative")
    return (residual - gain * correction).square().mean() / residual.square().mean().clamp_min(eps)
