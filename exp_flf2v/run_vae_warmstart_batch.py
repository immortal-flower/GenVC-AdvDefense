"""Generate four VAE warm-start candidates and evaluate nine native codec cases."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exp_flf2v.search_native_rgb_input import atomic_json, read_metrics, sha256


def codec_command(python, script, data, checkpoint, root, run, asset, epsilon):
    return [str(python), str(script), '--wan_ckpt', str(checkpoint),
            '--data_dir', str(data), '--output_dir', str(root), '--run_name', run,
            '--num_frames_per_gop', '33', '--num_gops', '1', '--height', '720',
            '--width', '1280', '--M', '64', '--K', '16384', '--steps', '20',
            '--ddim_tail', '3', '--g_scale', '3.0', '--ref_codec', 'compressai',
            '--ref_quality', '4', '--seed', '42', '--sequences', 'Jockey',
            '--attack', 'vae-pgd', '--attack_file', str(asset), '--epsilon', str(epsilon),
            '--attack_scope', 'all', '--defense', 'none', '--input_attack_diagnostics']


def compare(rows):
    by_name = {row['name']: row for row in rows}
    initial = by_name['initial']
    result = []
    for eps in (2, 4):
        for rounds in (1, 4):
            c = by_name[f'eps{eps}_r{rounds}_candidate']
            r = by_name[f'eps{eps}_r{rounds}_random']
            if abs(c['InputMSE'] - r['InputMSE']) > 1e-9:
                raise ValueError('Native paired input MSE mismatch')
            result.append(dict(Epsilon=eps, AdditionalRounds=rounds,
                               CandidatePSNR=c['PSNR'], RandomPSNR=r['PSNR'],
                               ExtraDropVsInitial=initial['PSNR']-c['PSNR'],
                               ExtraDropVsRandom=r['PSNR']-c['PSNR'],
                               ExtraLPIPSVsInitial=c['LPIPS']-initial['LPIPS'],
                               ExtraLPIPSVsRandom=c['LPIPS']-r['LPIPS'],
                               InputMSE=c['InputMSE'], InputLinfPixel=c['InputLinfPixel'],
                               BPP=c['BPP']))
    return result


def write_csv(path, rows):
    temp = path.with_suffix('.tmp.csv')
    with temp.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--wan_ckpt', type=Path, required=True)
    p.add_argument('--initial', type=Path, required=True)
    p.add_argument('--results_root', type=Path, default=Path('exp_flf2v/results_720p'))
    p.add_argument('--tag', default='vae_warmstart_jockey_gop0_v1')
    p.add_argument('--plan_only', action='store_true')
    a = p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', a.tag):
        p.error('Invalid tag')
    for path in (a.data, a.initial, a.wan_ckpt/'Wan2.1_VAE.pth'):
        if not path.is_file():
            p.error(f'Missing file: {path}')
    print('Plan: independent epsilon 2/4 branches, checkpoints after 1/4 additional crop sweeps.', flush=True)
    print('One VAE load for generation; 9 sequential fresh native 720p codec runs (models reloaded per run).', flush=True)
    if a.plan_only:
        return
    root = a.results_root.resolve()
    assets = root/(a.tag+'_assets')
    folder = root/(a.tag+'_evaluation')
    folder.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).resolve().parent
    subprocess.run([sys.executable, str(here/'generate_vae_warmstart.py'),
                    '--data', str(a.data), '--vae', str(a.wan_ckpt/'Wan2.1_VAE.pth'),
                    '--initial', str(a.initial), '--output', str(assets)], check=True)
    generation = json.loads((assets/'generation.json').read_text(encoding='utf-8'))
    if generation['status'] != 'completed' or len(generation['assets']) != 9:
        raise ValueError('Expected completed generation with nine assets')
    script = here/'run_flf2v_experiment.py'
    rows = []
    for number, item in enumerate(generation['assets'], 1):
        asset = assets/item['file']
        if sha256(asset) != item['sha256']:
            raise ValueError('Input asset hash mismatch')
        run = a.tag+'_'+item['name']
        metrics = root/run/'Jockey/gop0/metrics.json'
        command = codec_command(sys.executable, script, a.data.resolve(),
                                a.wan_ckpt.resolve(), root, run, asset, item['epsilon'])
        identity = dict(command=command, source=generation['fingerprint'],
                        asset=item['sha256'], script=sha256(script))
        fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        receipt = folder/(item['name']+'.json')
        cached = False
        if receipt.exists():
            saved = json.loads(receipt.read_text(encoding='utf-8'))
            if saved['fingerprint'] != fingerprint:
                raise ValueError('Evaluation configuration changed; use a new tag')
            cached = metrics.exists() and sha256(metrics) == saved['metrics_sha256']
        if not cached:
            print(f'[NATIVE {number}/9] {item["name"]}: fresh encoding and decoding', flush=True)
            subprocess.run(command, check=True)
        values = read_metrics(metrics)
        if abs(values['InputMSE']-item['input_MSE']) > 1e-9:
            raise ValueError('Submitted input energy differs from generated asset')
        if values['InputLinfPixel'] > item['epsilon']+.001:
            raise ValueError('Native input exceeded epsilon')
        atomic_json(receipt, dict(fingerprint=fingerprint, metrics_sha256=sha256(metrics),
                                 configuration=identity, metrics=values))
        rows.append(dict(name=item['name'], **values))
        write_csv(folder/'summary.csv', rows)
        print(f'[{"REUSE" if cached else "DONE"}] {item["name"]}: '
              f'PSNR={values["PSNR"]:.4f}, LPIPS={values["LPIPS"]:.4f}', flush=True)
    paired = compare(rows)
    write_csv(folder/'paired.csv', paired)
    atomic_json(folder/'completed.json', dict(status='completed', generation=generation['fingerprint'],
                comparisons=paired, limitation='Single GOP/seed screening; input energies may differ across checkpoints and epsilon branches.'))
    print('\nEps Rounds  ExtraDropVsInitial  ExtraDropVsRandom  ExtraLPIPSVsRandom')
    for r in paired:
        print(f'{r["Epsilon"]:3} {r["AdditionalRounds"]:6} '
              f'{r["ExtraDropVsInitial"]:+19.4f} {r["ExtraDropVsRandom"]:+18.4f} '
              f'{r["ExtraLPIPSVsRandom"]:+19.4f}')
    print(f'Saved: {folder / "paired.csv"}', flush=True)


if __name__ == '__main__':
    main()
