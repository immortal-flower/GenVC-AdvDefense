"""Prepare signed VAE-PGD directions and exactly magnitude-matched random inputs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def realize(original, delta):
    pixels = np.floor(np.clip(original.astype(np.float32) / 255 + delta, 0, 1) * 255 + 0.5).astype(np.uint8)
    return pixels.astype(np.int16) - original.astype(np.int16)


def matched_random(original, realized, seed):
    magnitude = np.abs(realized)
    rng = np.random.default_rng(seed)
    signs = rng.integers(0, 2, size=original.shape[1:], dtype=np.int16) * 2 - 1
    perturbation = magnitude * signs[None]
    base = original.astype(np.int16)
    invalid = (base + perturbation < 0) | (base + perturbation > 255)
    perturbation[invalid] *= -1
    if np.any((base + perturbation < 0) | (base + perturbation > 255)):
        raise ValueError("Could not preserve the realized magnitude at saturated pixels")
    if not np.array_equal(np.abs(perturbation), magnitude):
        raise ValueError("Random control is not pixel-magnitude matched")
    return perturbation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--old_attack", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    import cv2
    height, width, count = 720, 1280, 33
    frame_size = height * width * 3 // 2
    if a.data.stat().st_size < count * frame_size:
        p.error("Expected at least 33 complete 1280x720 YUV420 frames")
    frames = []
    with a.data.open("rb") as source:
        for _ in range(count):
            raw = np.frombuffer(source.read(frame_size), dtype=np.uint8)
            y = raw[:height * width].reshape(height, width)
            u = raw[height * width:height * width * 5 // 4].reshape(height // 2, width // 2)
            v = raw[height * width * 5 // 4:].reshape(height // 2, width // 2)
            rgb = cv2.cvtColor(np.stack((y, cv2.resize(u, (width, height)),
                                         cv2.resize(v, (width, height))), axis=-1), cv2.COLOR_YUV2RGB)
            frames.append(rgb)
    original = np.stack(frames)
    with np.load(a.old_attack, allow_pickle=False) as saved:
        old = saved["delta"].astype(np.float32)
    if old.shape != original.shape or not np.isfinite(old).all() or np.abs(old).max() > 2 / 255 + 1e-6:
        p.error("Old attack must be a finite 33-frame RGB perturbation within 2/255")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for amplitude, amp_name in ((1.0, "full"), (0.5, "half")):
        for direction, name in ((1, "positive"), (-1, "negative")):
            delta = direction * amplitude * old
            actual = realize(original, delta)
            random_actual = matched_random(original, actual, a.seed + (0 if direction == 1 else 1000) + (0 if amp_name == "full" else 100))
            for variant, values in ((name, actual), (name + "_random", random_actual)):
                path = a.output_dir / f"{amp_name}_{variant}.npz"
                np.savez_compressed(path, delta=(values.astype(np.float32) / 255))
                mse = float(np.mean(np.square(values.astype(np.float32) / 255)))
                rows.append({"name": f"{amp_name}_{variant}", "file": str(path),
                             "input_MSE": mse, "input_Linf_pixel": int(np.abs(values).max()),
                             "input_PSNR_dB": float(-10 * np.log10(max(mse, 1e-12)))})
    report = {"source_attack": str(a.old_attack),
              "source_sha256": hashlib.sha256(a.old_attack.read_bytes()).hexdigest(),
              "data": str(a.data), "rows": rows,
              "note": "Random controls match each realized integer pixel magnitude exactly; direction signs can be forced by saturation."}
    (a.output_dir / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for row in rows:
        print(f"{row['name']:22} input_MSE={row['input_MSE']:.8g} PSNR={row['input_PSNR_dB']:.2f}")
    print(f"Saved: {a.output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
