"""Budgeted offline search using normal 720p GVCC encoding/decoding as oracle.

No proxy gradients or reused clean codebooks. Selection uses source-referenced
mean frame PSNR. This assumes access to codec queries and reconstruction output;
it is not evidence of an attack on a feedback-free deployed service.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp.json')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def sse(delta):
    return int(np.square(delta, dtype=np.int32).sum(dtype=np.int64))


def realize(original, vector, scale, epsilon, boundary):
    pixels = np.floor(np.clip(original.astype(np.float32) +
                             np.clip(vector * scale, -epsilon, epsilon), 0, 255) + .5)
    delta = (pixels.astype(np.int16) - original.astype(np.int16)).astype(np.int8)
    delta[[0, -1]] = boundary
    return delta


def project_energy(original, vector, initial, epsilon, rtol=.01):
    """Match quantized input energy from below, retaining initial boundaries."""
    if original.shape != vector.shape or original.shape != initial.shape:
        raise ValueError('Mismatched RGB geometry')
    if not np.isfinite(vector).all() or not 1 <= epsilon <= 16:
        raise ValueError('Invalid direction/bound')
    target = sse(initial)
    if target <= sse(initial[[0, -1]]):
        raise ValueError('Initialization needs nonzero interior perturbation')
    boundary = initial[[0, -1]]
    lo, hi = 0., 1.
    best = realize(original, vector, lo, epsilon, boundary)
    high = realize(original, vector, hi, epsilon, boundary)
    while sse(high) < target and hi < 128:
        hi *= 2
        high = realize(original, vector, hi, epsilon, boundary)
    if sse(high) <= target:
        best = high
    else:
        for _ in range(30):
            mid = (lo + hi) / 2
            proposed = realize(original, vector, mid, epsilon, boundary)
            energy = sse(proposed)
            if energy <= target:
                lo, best = mid, proposed
                if target - energy <= target * .001:
                    break
            else:
                hi = mid
    if not 0 <= target - sse(best) <= rtol * target:
        raise ValueError('Could not match input energy within 1%; no codec query spent')
    if np.abs(best.astype(np.int16)).max() > epsilon:
        raise ValueError('Perturbation exceeded bound')
    return best


def smooth_direction(shape, seed, grid):
    import cv2
    frames, height, width, channels = shape
    rng = np.random.default_rng(seed)
    knots = rng.standard_normal((5, *grid, channels)).astype(np.float32)
    spatial = [cv2.resize(x, (width, height), interpolation=cv2.INTER_LINEAR) for x in knots]
    result = np.empty(shape, dtype=np.float32)
    for i, t in enumerate(np.linspace(0, 4, frames)):
        k = min(int(t), 3)
        result[i] = spatial[k] * (1 - (t-k)) + spatial[k+1] * (t-k)
    result[[0, -1]] = 0
    rms = float(np.sqrt(np.mean(result[1:-1]**2)))
    if not np.isfinite(rms) or rms == 0:
        raise ValueError('Invalid search direction')
    return result / rms


def load_source(path):
    """Same 8-bit planar YUV420 conversion as run_flf2v_experiment.py."""
    import cv2
    match = re.search(r'(\d+)x(\d+)', Path(path).name)
    if match and tuple(map(int, match.groups())) != (1280, 720):
        raise ValueError('This pilot requires native 1280x720 YUV420')
    h, w = 720, 1280
    size = h*w*3//2
    result = []
    with Path(path).open('rb') as source:
        for _ in range(33):
            raw = source.read(size)
            if len(raw) != size:
                raise ValueError('Expected at least 33 complete YUV frames')
            raw = np.frombuffer(raw, np.uint8)
            y = raw[:h*w].reshape(h, w)
            u = cv2.resize(raw[h*w:h*w*5//4].reshape(h//2, w//2), (w, h))
            v = cv2.resize(raw[h*w*5//4:].reshape(h//2, w//2), (w, h))
            result.append(cv2.cvtColor(np.stack((y, u, v), -1), cv2.COLOR_YUV2RGB))
    return np.stack(result)


def read_metrics(path):
    m = json.loads(Path(path).read_text(encoding='utf-8'))
    frames = m.get('per_frame_PSNR_dB')
    if not isinstance(frames, list) or len(frames) != 33:
        raise ValueError('Expected 33 frame-wise PSNR values for precise selection')
    score = float(np.mean(frames))
    values = [score, m['LPIPS'], m['input_MSE'], m['input_Linf_pixel'], m['BPP'],
              m['reconstruction_vs_submitted_PSNR_dB']]
    if not np.isfinite(values).all():
        raise ValueError('Non-finite codec result')
    return dict(PSNR=score, LPIPS=float(m['LPIPS']), InputMSE=float(m['input_MSE']),
                InputLinfPixel=float(m['input_Linf_pixel']), BPP=float(m['BPP']),
                PSNRvsSubmitted=float(m['reconstruction_vs_submitted_PSNR_dB']))


def choose_best(parent, candidates, min_gain=.001):
    best = min(candidates, key=lambda x: x['PSNR'])
    return best if parent['PSNR'] - best['PSNR'] >= min_gain else parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--wan_ckpt', type=Path, required=True)
    p.add_argument('--initial_attack', type=Path, required=True)
    p.add_argument('--results_root', type=Path, default=Path('exp_flf2v/results_720p'))
    p.add_argument('--tag', default='native_query_jockey_gop0_energy_matched_v1')
    p.add_argument('--rounds', type=int, default=4)
    p.add_argument('--epsilon', type=int, default=4)
    p.add_argument('--step_pixel', type=float, default=1.)
    p.add_argument('--search_seed', type=int, default=20261117)
    p.add_argument('--plan_only', action='store_true')
    a = p.parse_args()
    if not 1 <= a.rounds <= 4 or not 2 <= a.epsilon <= 16 or not 0 < a.step_pixel <= 4:
        p.error('Expected 1..4 rounds, 2..16 pixel bound, and 0..4 positive step')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', a.tag):
        p.error('Tag must be a safe single directory name')
    for path in (a.data, a.initial_attack, a.wan_ckpt / 'Wan2.1_VAE.pth'):
        if not path.is_file():
            p.error(f'Missing: {path}')
    if a.plan_only:
        print(f'Native 720p only: 1 initial + {2*a.rounds} search candidates; '
              'up to 2 held-out seed runs if gain >= 0.1 dB. No GPU started.')
        return
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    root = a.results_root.resolve()
    folder = root / (a.tag + '_search')
    folder.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name('run_flf2v_experiment.py').resolve()
    config = dict(version=1, data=str(a.data.resolve()), data_sha256=sha256(a.data),
                  initial_sha256=sha256(a.initial_attack), wan_ckpt=str(a.wan_ckpt.resolve()),
                  codec_script_sha256=sha256(script), rounds=a.rounds, epsilon=a.epsilon,
                  step_pixel=a.step_pixel, search_seed=a.search_seed,
                  codec='720p33-M64-K16384-S20-tail3-g3-q4', model_seed=42,
                  heldout_seed=123, selection='mean frame PSNR against clean source',
                  energy_rule='within 1% of initial input SSE, never above it',
                  validation_gate_db=.1, min_accept_db=.001,
                  grids=[[9,16], [18,32], [9,16], [36,64]])
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    state_path = folder / 'search.json'
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding='utf-8'))
        if state['fingerprint'] != fingerprint:
            raise ValueError('Search settings/source changed; use another tag')
    else:
        state = dict(fingerprint=fingerprint, configuration=config, status='running',
                     oracle_cache={}, evaluations=[], rounds=[], codec_launches=0,
                     limitation='Local reconstruction-feedback search; selected seed/GOP is not held-out evidence.')
    raw = load_source(a.data)
    with np.load(a.initial_attack, allow_pickle=False) as saved:
        delta = saved['delta'].astype(np.float32)
    if delta.shape != raw.shape or not np.isfinite(delta).all() or np.abs(delta).max()*255 > a.epsilon+.001:
        raise ValueError('Invalid initialization')
    submitted = (np.clip(raw.astype(np.float32)/255 + delta, 0, 1)*255 + .5).astype(np.uint8)
    initial = (submitted.astype(np.int16)-raw.astype(np.int16)).astype(np.int8)
    del delta, submitted
    target_energy = sse(initial)
    assets = folder / 'assets'
    assets.mkdir(exist_ok=True)
    state['evaluations'], state['rounds'] = [], []
    state['status'] = 'running'
    state.pop('error', None)

    def query(name, actual, model_seed, phase):
        if not np.array_equal(actual[[0, -1]], initial[[0, -1]]):
            raise ValueError('Submitted first/last RGB changed')
        energy = sse(actual)
        if not 0 <= target_energy-energy <= .01*target_energy or np.abs(actual.astype(np.int16)).max() > a.epsilon:
            raise ValueError('Candidate input energy/bound invalid')
        pixel_hash = hashlib.sha256(actual.tobytes()).hexdigest()
        key = f'{model_seed}:{pixel_hash}'
        asset = assets / f'{name}.npz'
        if asset.exists():
            with np.load(asset, allow_pickle=False) as saved:
                old = np.rint(saved['delta']*255).astype(np.int8)
            if not np.array_equal(old, actual):
                raise ValueError(f'Existing asset changed: {asset}')
        else:
            np.savez_compressed(asset, delta=actual.astype(np.float32)/255)
        asset_hash = sha256(asset)
        cached = state['oracle_cache'].get(key)
        metric = root / f'{a.tag}_{name}_m{model_seed}' / 'Jockey/gop0/metrics.json'
        if cached is not None:
            prior_metric = Path(cached['metrics_path'])
            if sha256(prior_metric) != cached['metrics_sha256']:
                raise ValueError('Cached metrics changed; not silently reusing/reselecting')
            values = read_metrics(prior_metric)
            metric = prior_metric
            print(f'[CACHE] {name} seed={model_seed}', flush=True)
        else:
            cmd = [sys.executable, str(script), '--wan_ckpt', str(a.wan_ckpt),
                   '--data_dir', str(a.data), '--output_dir', str(root),
                   '--run_name', f'{a.tag}_{name}_m{model_seed}', '--num_frames_per_gop', '33',
                   '--num_gops', '1', '--height', '720', '--width', '1280', '--M', '64', '--K', '16384',
                   '--steps', '20', '--ddim_tail', '3', '--g_scale', '3.0', '--ref_codec', 'compressai',
                   '--ref_quality', '4', '--seed', str(model_seed), '--sequences', 'Jockey',
                   '--attack', 'vae-pgd', '--attack_file', str(asset), '--epsilon', str(a.epsilon),
                   '--attack_scope', 'all', '--defense', 'none', '--input_attack_diagnostics']
            if '--reuse_codebook' in cmd:
                raise RuntimeError('Codebook reuse is forbidden for native query search')
            state['codec_launches'] += 1
            state['active_query'] = dict(name=name, model_seed=model_seed, phase=phase, command=cmd)
            atomic_json(state_path, state)
            print(f'[QUERY] {name} seed={model_seed} phase={phase}', flush=True)
            with (folder / f'{name}_m{model_seed}.log').open('a', encoding='utf-8') as log:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, encoding='utf-8', errors='replace',
                                        env=dict(os.environ, PYTHONIOENCODING='utf-8'))
                for line in proc.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                    log.flush()
                code = proc.wait()
            if code != 0:
                raise RuntimeError(f'Native query failed: {name}; resume same command after fixing')
            if sha256(asset) != asset_hash:
                raise ValueError('Asset modified during codec execution')
            values = read_metrics(metric)
            state['oracle_cache'][key] = dict(metrics_path=str(metric), metrics_sha256=sha256(metric))
        expected_mse = energy / actual.size / 255**2
        if abs(values['InputMSE']-expected_mse) > 1e-9 or values['InputLinfPixel'] > a.epsilon+.001:
            raise ValueError('Native submitted input differs from planned input')
        record = dict(name=name, phase=phase, model_seed=model_seed, asset=str(asset),
                      input_sha256=pixel_hash, metrics_path=str(metric), **values)
        state['evaluations'].append(record)
        state.pop('active_query', None)
        atomic_json(state_path, state)
        print(f'[RESULT] {name}: PSNR={values["PSNR"]:.5f}, LPIPS={values["LPIPS"]:.4f}', flush=True)
        return record

    try:
        baseline = query('initial', initial, 42, 'baseline')
        best, best_delta = baseline, initial.copy()
        for r in range(a.rounds):
            direction = smooth_direction(raw.shape, a.search_seed+r, config['grids'][r])
            parent_name = best['name']
            parent_delta = best_delta
            candidates, arrays = [], {}
            for sign, label in ((1, 'plus'), (-1, 'minus')):
                name = f'r{r+1}_{label}'
                proposed = project_energy(raw, parent_delta.astype(np.float32) +
                                          sign*a.step_pixel*direction, initial, a.epsilon)
                record = query(name, proposed, 42, 'search')
                candidates.append(record)
                arrays[name] = proposed
            best = choose_best(best, candidates)
            if best['name'] in arrays:
                best_delta = arrays[best['name']]
            state['rounds'].append(dict(round=r+1, parent=parent_name, chosen=best['name'],
                                        gain_vs_initial_db=baseline['PSNR']-best['PSNR']))
            atomic_json(state_path, state)
            print(f'[ROUND {r+1}] chosen={best["name"]}; gain={baseline["PSNR"]-best["PSNR"]:.5f} dB', flush=True)
            del arrays, candidates, direction, proposed, parent_delta
        state['best'] = best
        state['search_gain_db'] = baseline['PSNR']-best['PSNR']
        state['search_delta_lpips'] = best['LPIPS']-baseline['LPIPS']
        if state['search_gain_db'] >= .1:
            hold_initial = query('holdout_initial', initial, 123, 'heldout')
            hold_best = query('holdout_best', best_delta, 123, 'heldout')
            state['heldout'] = dict(gain_db=hold_initial['PSNR']-hold_best['PSNR'],
                                    delta_lpips=hold_best['LPIPS']-hold_initial['LPIPS'],
                                    used_for_selection=False)
        else:
            state['heldout'] = dict(skipped=True, reason='Search improvement below predeclared 0.1 dB gate')
        state.update(status='completed', unique_oracle_evaluations=len(state['oracle_cache']))
        atomic_json(state_path, state)
        out = folder / 'summary.csv'
        with out.open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(state['evaluations'][0]))
            writer.writeheader()
            writer.writerows(state['evaluations'])
        print('[SUMMARY] ' + json.dumps({k:state[k] for k in ('search_gain_db','search_delta_lpips','heldout','unique_oracle_evaluations')}, indent=2))
        print(f'Saved: {out}\nSaved: {state_path}')
    except Exception as exc:
        state.update(status='failed', error=str(exc))
        atomic_json(state_path, state)
        raise


if __name__ == '__main__':
    main()
