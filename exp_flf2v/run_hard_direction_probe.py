"""Replay a saved gradient against same-resolution hard rollouts, without native jobs."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True, type=Path)
    p.add_argument('--random_directions', type=int, default=2)
    a = p.parse_args()
    report = json.loads((a.source / 'generation.json').read_text(encoding='utf-8'))
    if report['status'] != 'completed' or report['configuration']['iterations'] != 1:
        raise ValueError('Require completed one-update source')
    cfg = report['configuration']
    out = a.source.with_name(a.source.name + f'_hardprobe_r{a.random_directions}')
    cmd = [sys.executable, str(Path(__file__).with_name('generate_trajectory_input_pilot.py'))]
    for key in ('data', 'wan_ckpt', 'initial_attack_file', 'objective', 'backward_steps',
                'iterations', 'epsilon', 'alpha', 'seed', 'pool_size', 'height', 'width',
                'control_height', 'control_width', 'gpu0_memory', 'gpu1_memory'):
        if cfg.get(key) is not None:
            cmd += ['--' + key, str(cfg[key])]
    for key in ('grid_random', 'vae_cpu_activations', 'stepwise_backward'):
        if cfg.get(key):
            cmd += ['--' + key]
    cmd += ['--output_dir', str(out), '--direction_audit_from', str(a.source),
            '--audit_random_directions', str(a.random_directions)]
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '0,1'
    subprocess.run(cmd, env=env, check=True)
    result = json.loads((out / 'generation.json').read_text(encoding='utf-8'))
    baseline = next(r['proxy_rgb_psnr_db'] for r in result['rows'] if r['variant'] == 'initial')
    rows = []
    for r in result['rows']:
        row = {k: r[k] for k in ('variant', 'step_pixel', 'proxy_rgb_psnr_db', 'input_MSE', 'input_Linf_pixel')}
        row['drop_vs_initial_db'] = baseline - r['proxy_rgb_psnr_db']
        rows.append(row)
        print(f"{row['variant']:32} drop={row['drop_vs_initial_db']:+.4f} dB MSE={row['input_MSE']:.8g}")
    with (out / 'summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved: {out / "summary.csv"}')
    print('Same-resolution diagnostic only. Steps matched in control space, not input energy; inspect MSE. No native 720p success claim.')


if __name__ == '__main__':
    main()
