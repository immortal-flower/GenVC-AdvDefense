"""Attack preprocessing utilities for FLF2V experiments.

These attacks operate in RGB image space before GVCC compression. They are
intended for fast robustness sweeps and as extension points for stronger
model-aware attacks.
"""

import numpy as np
from PIL import Image

ATTACK_CHOICES = [
    "none",
    "uap",
    "ftuap",
    "pgd-d",
    "pgd-r",
    "pgd-rd",
    "keyframe",
    "gop-shared",
]


def normalize_epsilon(epsilon):
    return epsilon / 255.0 if epsilon > 1.0 else epsilon


def frames_to_float_np(frames):
    return np.stack([np.array(f).astype(np.float32) / 255.0 for f in frames], axis=0)


def float_np_to_frames(arr):
    arr = np.clip(arr, 0.0, 1.0)
    arr_u8 = (arr * 255.0 + 0.5).astype(np.uint8)
    return [Image.fromarray(x) for x in arr_u8]


def make_ftuap_pattern(height, width, channels, epsilon, seed, frequency=8):
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, height, dtype=np.float32),
        np.linspace(0.0, 1.0, width, dtype=np.float32),
        indexing="ij",
    )
    pattern = np.zeros((height, width, channels), dtype=np.float32)
    for c in range(channels):
        fx = rng.integers(1, frequency + 1)
        fy = rng.integers(1, frequency + 1)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        pattern[..., c] = np.sin(2.0 * np.pi * (fx * xx + fy * yy) + phase)
    return epsilon * np.sign(pattern)


def make_proxy_pgd_delta(shape, epsilon, steps, alpha, seed, mode):
    """Deterministic image-space proxy until model-gradient PGD is wired in."""
    rng = np.random.default_rng(seed)
    delta = rng.uniform(-epsilon, epsilon, size=shape).astype(np.float32)
    if alpha <= 0:
        alpha = epsilon / max(steps, 1)

    for _ in range(max(steps, 1)):
        if mode == "pgd-d":
            grad = rng.choice([-1.0, 1.0], size=shape).astype(np.float32)
        elif mode == "pgd-r":
            base = rng.normal(0.0, 1.0, size=(shape[0], max(1, shape[1] // 16), max(1, shape[2] // 16), shape[3]))
            grad = np.repeat(np.repeat(base, 16, axis=1), 16, axis=2)[:, :shape[1], :shape[2], :]
            grad = np.sign(grad).astype(np.float32)
        else:
            high = rng.choice([-1.0, 1.0], size=shape).astype(np.float32)
            base = rng.normal(0.0, 1.0, size=(shape[0], max(1, shape[1] // 16), max(1, shape[2] // 16), shape[3]))
            low = np.repeat(np.repeat(base, 16, axis=1), 16, axis=2)[:, :shape[1], :shape[2], :]
            grad = np.sign(0.5 * high + 0.5 * low).astype(np.float32)
        delta = np.clip(delta + alpha * grad, -epsilon, epsilon)
    return delta


def apply_attack(frames, attack, epsilon=4.0, attack_steps=8, attack_alpha=0.0, seed=42,
                 frames_per_gop_excl_first=32):
    if attack == "none":
        return list(frames), {"attack": "none"}

    eps = normalize_epsilon(epsilon)
    arr = frames_to_float_np(frames)
    rng = np.random.default_rng(seed)
    metadata = {
        "attack": attack,
        "epsilon_input": epsilon,
        "epsilon_normalized": eps,
        "attack_steps": attack_steps,
        "attack_alpha": attack_alpha,
        "seed": seed,
    }

    if attack == "uap":
        delta = rng.choice([-eps, eps], size=arr.shape[1:]).astype(np.float32)
        arr = arr + delta[None, ...]
    elif attack == "ftuap":
        delta = make_ftuap_pattern(arr.shape[1], arr.shape[2], arr.shape[3], eps, seed)
        arr = arr + delta[None, ...]
    elif attack in ("pgd-d", "pgd-r", "pgd-rd"):
        alpha = normalize_epsilon(attack_alpha)
        delta = make_proxy_pgd_delta(arr.shape, eps, attack_steps, alpha, seed, attack)
        arr = arr + delta
        metadata["note"] = "Image-space proxy PGD; replace with model-gradient objective for end-to-end attacks."
    elif attack == "keyframe":
        key_indices = set(range(0, arr.shape[0], frames_per_gop_excl_first))
        key_indices.add(arr.shape[0] - 1)
        delta = rng.choice([-eps, eps], size=arr.shape[1:]).astype(np.float32)
        for idx in sorted(key_indices):
            arr[idx] = arr[idx] + delta
        metadata["keyframe_indices"] = sorted(int(i) for i in key_indices)
    elif attack == "gop-shared":
        for start in range(0, arr.shape[0], frames_per_gop_excl_first):
            end = min(start + frames_per_gop_excl_first + 1, arr.shape[0])
            delta = rng.choice([-eps, eps], size=arr.shape[1:]).astype(np.float32)
            arr[start:end] = arr[start:end] + delta[None, ...]
    else:
        raise ValueError(f"Unsupported attack: {attack}")

    return float_np_to_frames(arr), metadata
