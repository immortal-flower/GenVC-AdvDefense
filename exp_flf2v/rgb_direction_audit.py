"""Recover a single accepted PGD sign direction, not a quantized RGB delta."""
import numpy as np


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
