"""Standardized fresh-start VAE RGB attack across three contents; laboratory only.

This is not a reproduction of the historical staged Jockey r44 optimizer.
Native metrics, not lossy MP4 screenshots, determine acceptance.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exp_flf2v.search_native_rgb_input import atomic_json, sha256, read_metrics
from exp_flf2v.run_vae_warmstart_batch import codec_command, write_csv


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data_root', type=Path, required=True)
    p.add_argument('--wan_ckpt', type=Path, required=True)
    p.add_argument('--results_root', type=Path, default=Path('exp_flf2v/results_720p'))
    p.add_argument('--rounds', type=int, nargs='+', default=[4,20,44])
    p.add_argument('--temporal_window',type=int,choices=[5,9,17],default=5)
    p.add_argument('--plan_only', action='store_true')
    a = p.parse_args()
    if a.rounds != sorted(set(a.rounds)) or min(a.rounds)<1 or max(a.rounds)>64:
        p.error('Rounds must increase in [1,64]')
    sequences = ('Jockey','Bosphorus','HoneyBee')
    for seq in sequences:
        path = a.data_root/f'{seq}_720p.yuv'
        if not path.is_file() or path.stat().st_size < 33*1382400 or path.stat().st_size % 1382400:
            p.error(f'Missing/incomplete 720p YUV420: {path}')
    if not (a.wan_ckpt/'Wan2.1_VAE.pth').is_file():
        p.error('Missing Wan2.1_VAE.pth')
    print(f'Plan: {len(sequences)} contents, rounds={a.rounds}; '
          f'{len(sequences)*(1+2*len(a.rounds))} fresh native codec cases, sequential.', flush=True)
    print('Frozen VAE, source RGB MSE, epsilon4, alpha0.5, momentum0.8; '
          f'192x192 spatial / {a.temporal_window}-frame crops; 3 temporal control anchors. '
          'No historical r44 reuse; no Wan gradients.', flush=True)
    if a.plan_only:
        return
    import numpy as np
    root = a.results_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).resolve().parent
    all_rows, paired = [], []
    prefix = 'vae_content_rgb_v1' if a.temporal_window==5 else f'vae_content_rgb_t{a.temporal_window}_v1'
    for seq in sequences:
        data = (a.data_root/f'{seq}_720p.yuv').resolve()
        tag = f'{prefix}_{seq.lower()}_r'+ '-'.join(map(str,a.rounds))
        assets = root/(tag+'_assets')
        assets.mkdir(parents=True, exist_ok=True)
        initial = assets/'zero.npz'
        if not initial.exists():
            np.savez_compressed(initial, delta=np.zeros((33,720,1280,3), np.float32))
        subprocess.run([sys.executable, str(here/'generate_vae_warmstart.py'),
                        '--data',str(data),'--vae',str(a.wan_ckpt.resolve()/'Wan2.1_VAE.pth'),
                        '--initial',str(initial),'--output',str(assets),'--direct_rgb',
                        '--temporal_window',str(a.temporal_window),
                        '--fresh_rgb_rounds',*map(str,a.rounds)],check=True)
        report = json.loads((assets/'generation.json').read_text(encoding='utf-8'))
        if report['status'] != 'completed':
            raise RuntimeError('Generation incomplete')
        by_name = {}
        for item in report['assets']:
            asset = assets/item['file']
            if sha256(asset) != item['sha256']:
                raise ValueError('Asset changed')
            name = 'clean' if item['kind']=='initial' else item['name']
            run = tag+'_'+name
            script = here/'run_flf2v_experiment.py'
            command = codec_command(sys.executable,script,data,a.wan_ckpt.resolve(),root,run,asset,4)
            command[command.index('--sequences')+1] = seq
            metrics = root/run/seq/'gop0/metrics.json'
            receipt = assets/(name+'_native.json')
            identity = dict(command=command, asset=sha256(asset), script=sha256(script),
                            data=sha256(data), generation=report['fingerprint'])
            cached = False
            if receipt.exists():
                saved = json.loads(receipt.read_text(encoding='utf-8'))
                if saved['identity'] != identity:
                    raise ValueError('Configuration changed; use a new batch version')
                cached = metrics.exists() and saved['metrics_hash']==sha256(metrics)
            if not cached:
                subprocess.run(command,check=True)
            values = read_metrics(metrics)
            if abs(values['InputMSE']-item['input_MSE'])>1e-9 or values['InputLinfPixel']>4.001:
                raise ValueError('Native input mismatch/bound violation')
            atomic_json(receipt,dict(identity=identity,metrics_hash=sha256(metrics)))
            row = dict(Sequence=seq,Name=name,**values)
            all_rows.append(row)
            by_name[name] = row
            write_csv(root/(prefix+'_native.csv'),all_rows)
        for rounds in a.rounds:
            c,r = (by_name[f'eps4_r{rounds}_{kind}'] for kind in ('candidate','random'))
            if abs(c['InputMSE']-r['InputMSE'])>1e-9:
                raise ValueError('Control energy mismatch')
            paired.append(dict(Sequence=seq,Rounds=rounds,TemporalWindow=a.temporal_window,CleanPSNR=by_name['clean']['PSNR'],
                               CandidatePSNR=c['PSNR'],RandomPSNR=r['PSNR'],
                               ExtraDropVsRandom=r['PSNR']-c['PSNR'],
                               ExtraLPIPSVsRandom=c['LPIPS']-r['LPIPS'],
                               InputMSE=c['InputMSE'],InputLinfPixel=c['InputLinfPixel'],
                               CandidateBPP=c['BPP'],RandomBPP=r['BPP']))
        write_csv(root/(prefix+'_paired.csv'),paired)
    print(f'COMPLETE: {prefix}_paired.csv',flush=True)


if __name__ == '__main__':
    main()
