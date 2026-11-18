"""Refine the accepted old VAE input using its crop reconstruction-deviation loss.

This is a warm start from realized RGB pixels with fresh momentum, NOT a resume
of the old optimizer. Temporal updates use three anchors; endpoints may change.
Both epsilon branches start independently from the same accepted input.
"""
import argparse
import json
import os
import shutil
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exp_flf2v.search_native_rgb_input import atomic_json, load_source, sha256
from exp_flf2v.vae_warmstart_utils import bounded_delta, crop_starts, quantized_pair, temporal_weights


def save_npz(path, **arrays):
    path = Path(path)
    tmp = path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def validate_extension(source, config):
    previous = json.loads((source/'generation.json').read_text(encoding='utf-8'))
    if previous['status'] != 'completed':
        raise ValueError('Source generation must be completed')
    keys = ('data_sha256','initial_sha256','vae_path','vae_size','geometry','crop','stride',
            'anchors','alpha_pixel','momentum','seed','schedule_offset','objective','parameterization','boundary')
    for key in keys:
        if previous['configuration'][key] != config[key]:
            raise ValueError(f'Continuation setting differs: {key}')
    source_round = max(previous['configuration']['saved_rounds'])
    with np.load(source/'eps4_optimizer.npz',allow_pickle=False) as z:
        if int(z['done']) != source_round or str(z['fingerprint'].item()) != previous['fingerprint']:
            raise ValueError('Optimizer checkpoint does not match the completed source round')
    for kind in ('candidate','random'):
        matches = [r for r in previous['assets'] if r['name'] == f'eps4_r{source_round}_{kind}']
        if len(matches) != 1 or sha256(source/matches[0]['file']) != matches[0]['sha256']:
            raise ValueError('Original final-round asset hash mismatch')
    return previous


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--vae', type=Path, required=True)
    p.add_argument('--initial', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--extend_from', type=Path,
                   help='Continue epsilon4 from a completed batch, retaining controls and momentum')
    p.add_argument('--extension_rounds', type=int, nargs='+', default=[8,12])
    a = p.parse_args()
    config = dict(version=1, data_sha256=sha256(a.data), initial_sha256=sha256(a.initial),
                  generator_sha256=sha256(__file__),
                  helpers_sha256=sha256(Path(__file__).with_name('vae_warmstart_utils.py')),
                  vae_path=str(a.vae.resolve()), vae_size=a.vae.stat().st_size,
                  geometry=[33,720,1280], crop=192, stride=144, anchors=3, alpha_pixel=.5,
                  momentum=.8, saved_rounds=[1,4], epsilon_pixels=[2,4], seed=42,
                  schedule_offset=8, objective='MSE(VAE(x+delta), VAE(x))',
                  parameterization='frozen realized initial delta plus three-anchor additive controls',
                  boundary='all frames, including first and last, can change',
                  optimizer_state='zero additional controls and zero momentum for each epsilon branch')
    source = None
    if a.extend_from:
        if a.extend_from.resolve() == a.output.resolve():
            raise ValueError('Continuation must use a new output directory')
        source = validate_extension(a.extend_from,config)
        source_round = max(source['configuration']['saved_rounds'])
        if (a.extension_rounds != sorted(set(a.extension_rounds))
                or min(a.extension_rounds) <= source_round or max(a.extension_rounds) > 64):
            p.error('Continuation rounds must be increasing, above the source round, and at most 64')
        config.update(saved_rounds=[source_round]+a.extension_rounds,epsilon_pixels=[4],
                      source_round=source_round,
                      optimizer_state=f'continue round{source_round} epsilon4 controls and momentum',
                      source_fingerprint=source['fingerprint'],
                      source_checkpoint_sha256=sha256(a.extend_from/'eps4_optimizer.npz'))
    import hashlib
    fingerprint = hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    a.output.mkdir(parents=True, exist_ok=True)
    report_path = a.output/'generation.json'
    if report_path.exists():
        previous = json.loads(report_path.read_text(encoding='utf-8'))
        if previous['fingerprint'] != fingerprint:
            raise ValueError('Warm-start configuration/source changed; use another output directory')
        if previous['status'] == 'completed':
            for row in previous['assets']:
                if sha256(a.output/row['file']) != row['sha256']:
                    raise ValueError('Completed asset hash mismatch')
            print('[SKIP] Verified warm-start candidates and controls', flush=True)
            return
    report = dict(fingerprint=fingerprint, configuration=config, status='running', assets=[])
    atomic_json(report_path,report)
    raw = load_source(a.data)
    with np.load(a.initial,allow_pickle=False) as z:
        initial = z['delta'].astype(np.float32)
    if initial.shape != raw.shape or not np.isfinite(initial).all() or np.abs(initial).max()*255 > 2.001:
        raise ValueError('Expected the accepted 33-frame epsilon=2 initialization')
    original = raw.astype(np.float32)/255
    submitted = (np.clip(original+initial,0,1)*255+.5).astype(np.uint8)
    actual = submitted.astype(np.int16)-raw.astype(np.int16)
    initial = actual.astype(np.float32)/255
    save_npz(a.output/'initial.npz',delta=initial)
    report['assets'].append(dict(name='initial',file='initial.npz',kind='initial',epsilon=2,round=0,
                                input_MSE=float(np.mean(initial**2)),
                                input_Linf_pixel=float(np.abs(initial).max()*255),
                                sha256=sha256(a.output/'initial.npz')))
    clean = torch.from_numpy(original).permute(3,0,1,2)
    base = torch.from_numpy(initial).permute(3,0,1,2)
    weights = temporal_weights(33)
    tiles = [(y,x) for y in crop_starts(720) for x in crop_starts(1280)]
    torch.manual_seed(42)
    if not torch.cuda.is_available():
        raise RuntimeError('Use the laboratory CUDA environment for VAE generation')
    from wan.modules.vae import WanVAE
    device = torch.device('cuda:0')
    vae = WanVAE(vae_pth=str(a.vae),dtype=torch.bfloat16,device=device)
    vae.model.eval().requires_grad_(False)

    def reconstruct(pixels):
        z = vae.encode([pixels*2-1])[0]
        return (vae.decode([z])[0]+1)/2

    def save_pair(epsilon, round_id, controls):
        with torch.no_grad():
            d = bounded_delta(base,controls,weights,epsilon/255).permute(1,2,3,0).numpy()
        candidate, random = quantized_pair(raw,d,42)
        for kind, delta in [('candidate',candidate),('random',random)]:
            name = f'eps{epsilon}_r{round_id}_{kind}'
            if not np.isfinite(delta).all() or np.abs(delta).max()*255 > epsilon+.001:
                raise ValueError(f'Invalid saved input: {name}')
            path = a.output/(name+'.npz')
            if source is not None and round_id == config['source_round']:
                original_row = next(r for r in source['assets'] if r['name'] == name)
                original_path = a.extend_from/original_row['file']
                with np.load(original_path,allow_pickle=False) as z:
                    if not np.array_equal(delta,z['delta']):
                        raise ValueError('Resumed optimizer does not reproduce the source checkpoint input')
                shutil.copy2(original_path,path)
            else:
                save_npz(path,delta=delta)
            row = dict(name=name,file=path.name,kind=kind,epsilon=epsilon,round=round_id,
                       input_MSE=float(np.mean(delta**2)),input_Linf_pixel=float(np.abs(delta).max()*255),
                       sha256=sha256(path))
            atomic_json(a.output/(name+'.json'),dict(fingerprint=fingerprint,asset=row))
        print(f'[CHECKPOINT] epsilon={epsilon}/255 additional rounds={round_id}',flush=True)

    try:
        with torch.enable_grad():
            for epsilon in config['epsilon_pixels']:
                checkpoint = a.output/f'eps{epsilon}_optimizer.npz'
                controls = torch.zeros(3,3,720,1280)
                velocity = torch.zeros_like(controls)
                done, history = 0, []
                loading = checkpoint if checkpoint.exists() else (a.extend_from/'eps4_optimizer.npz' if source else None)
                if loading is not None:
                    with np.load(loading,allow_pickle=False) as z:
                        expected_fp = fingerprint if loading == checkpoint else source['fingerprint']
                        if str(z['fingerprint'].item()) != expected_fp:
                            raise ValueError('Optimizer fingerprint mismatch')
                        controls = torch.from_numpy(z['controls'].copy())
                        velocity = torch.from_numpy(z['velocity'].copy())
                        done = int(z['done'])
                        history = json.loads(str(z['history'].item()))
                    if done not in range(max(config['saved_rounds'])+1):
                        raise ValueError('Invalid optimizer round')
                    if (controls.shape != (3,3,720,1280) or velocity.shape != controls.shape
                            or not torch.isfinite(controls).all() or not torch.isfinite(velocity).all()):
                        raise ValueError('Invalid saved optimizer tensors')
                if source is not None and done == config['source_round']:
                    save_pair(4,config['source_round'],controls)
                for round_id in range(done,max(config['saved_rounds'])):
                    tick = time.monotonic()
                    gradients = torch.zeros_like(controls)
                    counts = torch.zeros(3,1,720,1280)
                    losses = []
                    for tile_id,(y,x) in enumerate(tiles):
                        t = (tile_id*7+(8+round_id)*5) % 29
                        target = clean[:,t:t+5,y:y+192,x:x+192].to(device)
                        fixed = base[:,t:t+5,y:y+192,x:x+192].to(device)
                        with torch.no_grad():
                            reference = reconstruct(target)
                        c = controls[:,:,y:y+192,x:x+192].to(device).detach().requires_grad_(True)
                        w = weights[t:t+5].to(device)
                        d = bounded_delta(fixed,c,w,epsilon/255)
                        recon = reconstruct((target+d).clamp(0,1))
                        loss = (recon-reference).square().mean()
                        grad, = torch.autograd.grad(loss,c)
                        if not torch.isfinite(grad).all():
                            raise RuntimeError('Non-finite VAE input gradient')
                        gradients[:,:,y:y+192,x:x+192] += grad.detach().cpu()
                        counts[:,:,y:y+192,x:x+192] += (w.sum(0).cpu()/5)[:,None,None,None]
                        losses.append(float(loss.detach()))
                        del target,fixed,reference,c,w,d,recon,loss,grad
                        if (tile_id+1)%10 == 0 or tile_id+1 == len(tiles):
                            print(f'[VAE] eps={epsilon} round={round_id+1}/{max(config["saved_rounds"])} crops={tile_id+1}/{len(tiles)}',flush=True)
                    gradients /= counts.clamp_min(1e-8)
                    if not torch.any(gradients):
                        raise RuntimeError('Entire crop sweep produced zero gradient')
                    gradients /= gradients.abs().mean(dim=(1,2,3),keepdim=True).clamp_min(1e-12)
                    velocity = .8*velocity+gradients
                    controls = (controls+.5/255*velocity.sign()).clamp(-2*epsilon/255,2*epsilon/255)
                    history.append(dict(round=round_id+1,mean_crop_loss=float(np.mean(losses)),
                                        coverage=float((counts>0).float().mean()),seconds=time.monotonic()-tick))
                    if round_id+1 in config['saved_rounds']:
                        save_pair(epsilon,round_id+1,controls)
                    # Save only after selected assets and metadata are complete.
                    save_npz(checkpoint,controls=controls.numpy(),velocity=velocity.numpy(),
                             done=round_id+1,history=json.dumps(history),fingerprint=fingerprint)
                for round_id in config['saved_rounds']:
                    for kind in ('candidate','random'):
                        meta = a.output/f'eps{epsilon}_r{round_id}_{kind}.json'
                        item = json.loads(meta.read_text(encoding='utf-8'))
                        row = item['asset']
                        if item['fingerprint'] != fingerprint or sha256(a.output/row['file']) != row['sha256']:
                            raise ValueError('Saved checkpoint asset changed')
                        report['assets'].append(row)
                report[f'eps{epsilon}_history'] = history
                atomic_json(report_path,report)
        report.update(status='completed',note='Native codec acceptance required; endpoints can change; random controls match pixel magnitudes, not spatial spectrum.')
        atomic_json(report_path,report)
        print(f'[READY] {report_path}',flush=True)
    except Exception as exc:
        report.update(status='failed',error=str(exc))
        atomic_json(report_path,report)
        raise


if __name__ == '__main__':
    main()
