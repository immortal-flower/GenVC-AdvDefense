"""Small, testable helpers for temporal VAE warm-start refinement."""
import numpy as np
import torch


def temporal_weights(frames, anchors=3):
    positions = np.linspace(0, anchors - 1, frames)
    weights = np.zeros((frames, anchors), np.float32)
    for f, pos in enumerate(positions):
        left = int(np.floor(pos))
        right = min(left + 1, anchors - 1)
        weights[f, left] += 1 - (pos - left)
        weights[f, right] += pos - left
    return torch.from_numpy(weights)


def bounded_delta(base, controls, weights, epsilon):
    # base: C,F,H,W; controls: anchors,C,H,W. Frozen base + smooth update.
    return (base + torch.einsum('fa,achw->cfhw', weights, controls)).clamp(-epsilon, epsilon)


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
