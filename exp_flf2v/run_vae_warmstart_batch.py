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


def compare(rows, extension=False, source_round=4, target_rounds=(8,12)):
    by_name = {row['name']: row for row in rows}
    baseline = f'eps4_r{source_round}_candidate' if extension else 'initial'
    initial = by_name[baseline]
    result = []
    for eps in ((4,) if extension else (2,4)):
        for rounds in (target_rounds if extension else (1,4)):
            c = by_name[f'eps{eps}_r{rounds}_candidate']
            r = by_name[f'eps{eps}_r{rounds}_random']
            if abs(c['InputMSE'] - r['InputMSE']) > 1e-9:
                raise ValueError('Native paired input MSE mismatch')
            result.append(dict(Epsilon=eps, AdditionalRounds=rounds,
                               Baseline=baseline,
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
    p.add_argument('--extend_tag', help='Continue the epsilon4 optimizer from this completed source tag')
    p.add_argument('--extension_rounds',type=int,nargs='+',default=[8,12])
    p.add_argument('--plan_only', action='store_true')
    a = p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', a.tag):
        p.error('Invalid tag')
    if a.extend_tag and (not re.fullmatch(r'[A-Za-z0-9_-]+',a.extend_tag) or a.tag == a.extend_tag):
        p.error('Continuation requires valid, distinct source and destination tags')
    for path in (a.data, a.initial, a.wan_ckpt/'Wan2.1_VAE.pth'):
        if not path.is_file():
            p.error(f'Missing file: {path}')
    root = a.results_root.resolve()
    source_round = 4
    if a.extend_tag:
        source_report = json.loads((root/(a.extend_tag+'_assets')/'generation.json').read_text(encoding='utf-8'))
        if source_report['status'] != 'completed':
            p.error('Source generation is not completed')
        source_round = max(source_report['configuration']['saved_rounds'])
        if (a.extension_rounds != sorted(set(a.extension_rounds))
                or min(a.extension_rounds) <= source_round or max(a.extension_rounds)>64):
            p.error('Target rounds must increase beyond the source round, up to 64')
        print(f'Plan: retain epsilon4 controls/momentum, rounds{source_round} -> {a.extension_rounds}; bound stays 4/255.',flush=True)
        print(f'Reuse verified round{source_round} native results; {2*len(a.extension_rounds)} new 720p codec runs.',flush=True)
    else:
        print('Plan: independent epsilon 2/4 branches, checkpoints after 1/4 additional crop sweeps.', flush=True)
        print('One VAE load for generation; 9 sequential fresh native 720p codec runs (models reloaded per run).', flush=True)
    if a.plan_only:
        return
    assets = root/(a.tag+'_assets')
    folder = root/(a.tag+'_evaluation')
    folder.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).resolve().parent
    generation_command = [sys.executable, str(here/'generate_vae_warmstart.py'),
                    '--data', str(a.data), '--vae', str(a.wan_ckpt/'Wan2.1_VAE.pth'),
                    '--initial', str(a.initial), '--output', str(assets)]
    if a.extend_tag:
        generation_command += ['--extend_from',str(root/(a.extend_tag+'_assets')),
                               '--extension_rounds',*map(str,a.extension_rounds)]
    subprocess.run(generation_command,check=True)
    generation = json.loads((assets/'generation.json').read_text(encoding='utf-8'))
    expected_assets = 3+2*len(a.extension_rounds) if a.extend_tag else 9
    if generation['status'] != 'completed' or len(generation['assets']) != expected_assets:
        raise ValueError('Unexpected or incomplete generation assets')
    script = here/'run_flf2v_experiment.py'
    rows = []
    cases = [r for r in generation['assets'] if not (a.extend_tag and r['kind']=='initial')]
    for number, item in enumerate(cases, 1):
        asset = assets/item['file']
        if sha256(asset) != item['sha256']:
            raise ValueError('Input asset hash mismatch')
        if a.extend_tag and item['round'] == source_round:
            old_metrics = root/(a.extend_tag+'_'+item['name'])/'Jockey/gop0/metrics.json'
            old_receipt = root/(a.extend_tag+'_evaluation')/(item['name']+'.json')
            saved = json.loads(old_receipt.read_text(encoding='utf-8'))
            old_asset = root/(a.extend_tag+'_assets')/item['file']
            expected_command = codec_command(sys.executable,script,a.data.resolve(),a.wan_ckpt.resolve(),
                                             root,a.extend_tag+'_'+item['name'],old_asset,4)
            if (saved['configuration']['asset'] != item['sha256']
                    or saved['configuration']['script'] != sha256(script)
                    or saved['configuration']['command'] != expected_command
                    or saved['metrics_sha256'] != sha256(old_metrics)):
                raise ValueError('Source native baseline cannot be verified; no new native jobs started')
            values = read_metrics(old_metrics)
            if abs(values['InputMSE']-item['input_MSE']) > 1e-9:
                raise ValueError('Source native energy mismatch')
            rows.append(dict(name=item['name'],**values))
            write_csv(folder/'summary.csv',rows)
            print(f'[BASELINE REUSE] {item["name"]}: PSNR={values["PSNR"]:.4f} LPIPS={values["LPIPS"]:.4f}',flush=True)
            continue
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
            print(f'[CASE {number}/{len(cases)}] {item["name"]}: fresh encoding and decoding', flush=True)
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
    paired = compare(rows,extension=bool(a.extend_tag),source_round=source_round,target_rounds=a.extension_rounds)
    write_csv(folder/'paired.csv', paired)
    atomic_json(folder/'completed.json', dict(status='completed', generation=generation['fingerprint'],
                comparisons=paired, limitation='Single GOP/seed screening; input energies may differ across checkpoints and epsilon branches.'))
    print('\nEps Rounds  ExtraDropVsInitial  ExtraDropVsRandom  ExtraLPIPSVsRandom')
    if a.extend_tag:
        print(f'Initial here means the existing epsilon4 round{source_round} candidate, NOT clean video.',flush=True)
    for r in paired:
        print(f'{r["Epsilon"]:3} {r["AdditionalRounds"]:6} '
              f'{r["ExtraDropVsInitial"]:+19.4f} {r["ExtraDropVsRandom"]:+18.4f} '
              f'{r["ExtraLPIPSVsRandom"]:+19.4f}')
    print(f'Saved: {folder / "paired.csv"}', flush=True)


if __name__ == '__main__':
    main()
