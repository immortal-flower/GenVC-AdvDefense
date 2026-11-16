"""Recover a single accepted PGD sign direction, not a quantized RGB delta."""
import numpy as np


def assert_same_submitted_pixels(original, actual_delta, expected_delta, label):
    """Compare the uint8 RGB consumed by the codec, not float delta bit patterns.

    Uses the same clipping and round-half-up conversion as attacks.py. Genuine
    changes of even one channel by one pixel level remain an error.
    """
    original = np.asarray(original, dtype=np.float32)
    actual_delta = np.asarray(actual_delta, dtype=np.float32)
    expected_delta = np.asarray(expected_delta, dtype=np.float32)
    if original.shape != actual_delta.shape or original.shape != expected_delta.shape:
        raise ValueError(f'{label}: RGB shapes differ')
    if not all(np.isfinite(v).all() for v in (original, actual_delta, expected_delta)):
        raise ValueError(f'{label}: non-finite RGB/delta')
    def submitted(delta):
        return (np.clip(original + delta, 0, 1) * 255 + .5).astype(np.uint8)
    a, b = submitted(actual_delta), submitted(expected_delta)
    changed = int(np.count_nonzero(a != b))
    max_delta_difference = float(np.max(np.abs(actual_delta - expected_delta)))
    if changed:
        max_pixel_difference = int(np.max(np.abs(a.astype(np.int16) - b.astype(np.int16))))
        raise RuntimeError(f'{label}: {changed} submitted RGB channels changed; '
                           f'max difference={max_pixel_difference} pixel levels')
    return dict(changed_submitted_channels=changed,
                max_float_delta_difference=max_delta_difference)


def recover_first_direction(report, checkpoint):
    config = report['configuration']
    history = report['history']
    if (report['status'] != 'completed' or config['objective'] != 'final-rgb'
            or config['iterations'] != 1 or int(checkpoint['done']) != 1
            or len(history) != 1 or history[0]['iteration'] != 1):
        raise ValueError('Direction audit requires exactly one completed PGD update')
    if str(checkpoint['fingerprint'].item()) != report['fingerprint']:
        raise ValueError('Source checkpoint fingerprint mismatch')
    step = float(history[0]['accepted_scale']) * config['alpha']
    if not 0 < step < 2 * config['epsilon']:
        raise ValueError('Cannot recover a rejected or control-clipped direction')
    controls = np.asarray(checkpoint['controls'], dtype=np.float32)
    if not np.isfinite(controls).all() or not np.any(controls):
        raise ValueError('Invalid/zero controls')
    direction = np.sign(controls)
    if not np.allclose(controls, direction * (step / 255), atol=1e-9, rtol=1e-5):
        raise ValueError('Controls are not a single sign-gradient update from zero')
    return direction, step
