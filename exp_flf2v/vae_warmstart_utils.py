"""Small, testable helpers for temporal VAE warm-start refinement."""
import numpy as np
import torch


def vae_attack_loss(reconstruction, original, clean_reconstruction=None, direct_rgb=False):
    reference = original if direct_rgb else clean_reconstruction
    if reference is None or reference.shape != reconstruction.shape:
        raise ValueError('Missing or incompatible objective reference')
    return (reconstruction-reference.detach()).square().mean()


def temporal_weights(frames, anchors=3):
    positions = np.linspace(0, anchors - 1, frames)
    weights = np.zeros((frames, anchors), np.float32)
    for f, pos in enumerate(positions):
        left = int(np.floor(pos))
        right = min(left + 1, anchors - 1)
        weights[f, left] += 1 - (pos - left)
        weights[f, right] += pos - left
    return torch.from_numpy(weights)


def bounded_delta(base, controls, weights, epsilon, straight_through=False):
    # base: C,F,H,W; controls: anchors,C,H,W. Frozen base + smooth update.
    raw = base + torch.einsum('fa,achw->cfhw', weights, controls)
    bounded = raw.clamp(-epsilon, epsilon)
    # Optional surrogate derivative only; forward perturbations stay bounded.
    return bounded.detach() + (raw-raw.detach()) if straight_through else bounded


def clipping_audit(base, controls, weights, epsilon):
    """Count latent control overshoot, not empirical attack effectiveness."""
    total = outside = 0
    excess_sum = excess_max = 0.
    with torch.no_grad():
        for f in range(weights.shape[0]):
            raw = base[:,f:f+1] + torch.einsum('fa,achw->cfhw',weights[f:f+1],controls)
            excess = (raw.abs()-epsilon).clamp_min(0)
            total += raw.numel()
            outside += int((raw.abs()>epsilon).sum())
            excess_sum += float(excess.double().sum())
            excess_max = max(excess_max,float(excess.max()))
    return dict(clamp_zero_derivative_fraction=outside/total,
                mean_overshoot_pixel=255*excess_sum/total,max_overshoot_pixel=255*excess_max,
                note='Strictly outside the epsilon clamp. Does not prove this causes the quality plateau.')


def quantized_pair(original, delta, seed):
    from exp_flf2v.prepare_input_direction_calibration import matched_random
    submitted = (np.clip(original.astype(np.float32)/255 + delta, 0, 1)*255 + .5).astype(np.uint8)
    actual = submitted.astype(np.int16) - original.astype(np.int16)
    random = matched_random(original, actual, seed)
    if not np.array_equal(np.abs(actual), np.abs(random)):
        raise ValueError('Random reference must match every realized pixel magnitude')
    return actual.astype(np.float32)/255, random.astype(np.float32)/255


def crop_starts(length, crop=192, stride=144):
    if not 0 < stride <= crop <= length:
        raise ValueError('Invalid crop geometry')
    starts = list(range(0, length-crop+1, stride))
    if starts[-1] != length-crop:
        starts.append(length-crop)
    return starts
