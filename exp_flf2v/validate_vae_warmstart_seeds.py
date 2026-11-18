"""Fixed-candidate native model-seed validation; no optimization on held-out seeds."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exp_flf2v.run_vae_warmstart_batch import codec_command, write_csv
from exp_flf2v.search_native_rgb_input import atomic_json, read_metrics, sha256


def validation_command(script, data, checkpoint, root, run, asset, seed):
    cmd = codec_command(sys.executable, script, data, checkpoint, root, run, asset, 4)
    cmd[cmd.index('--seed')+1] = str(seed)
    if asset is None:
        cmd[cmd.index('--attack')+1] = 'none'
        i = cmd.index('--attack_file')
        del cmd[i:i+2]
    return cmd


def paired_results(rows):
    paired = []
    for seed in (123, 2026):
        group = {r['Variant']: r for r in rows if r['ModelSeed'] == seed}
        if set(group) != {'clean', 'candidate', 'random'}:
            raise ValueError(f'Missing/invalid cases for seed {seed}')
        clean, candidate, random = (group[k] for k in ('clean', 'candidate', 'random'))
        if abs(candidate['InputMSE']-random['InputMSE']) > 1e-9:
            raise ValueError('Paired input energy mismatch')
        paired.append(dict(ModelSeed=seed, CleanPSNR=clean['PSNR'],
            CandidatePSNR=candidate['PSNR'], RandomPSNR=random['PSNR'],
            CandidateDropVsClean=clean['PSNR']-candidate['PSNR'],
            RandomDropVsClean=clean['PSNR']-random['PSNR'],
            ExtraDropVsRandom=random['PSNR']-candidate['PSNR'],
            CleanLPIPS=clean['LPIPS'], CandidateLPIPS=candidate['LPIPS'], RandomLPIPS=random['LPIPS'],
            CandidateDeltaLPIPSVsClean=candidate['LPIPS']-clean['LPIPS'],
            ExtraLPIPSVsRandom=candidate['LPIPS']-random['LPIPS'],
            InputMSE=candidate['InputMSE'], InputLinfPixel=candidate['InputLinfPixel'],
            CandidateBPP=candidate['BPP'], CleanBPP=clean['BPP'], RandomBPP=random['BPP']))
    return paired


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--wan_ckpt', type=Path, required=True)
    p.add_argument('--results_root', type=Path, default=Path('exp_flf2v/results_720p'))
    p.add_argument('--source_tag', default='vae_warmstart_jockey_gop0_v1')
    p.add_argument('--tag', default='vae_warmstart_eps4_r4_heldout_v1')
    p.add_argument('--plan_only', action='store_true')
    a = p.parse_args()
    if any(not re.fullmatch(r'[A-Za-z0-9_-]+', t) for t in (a.tag,a.source_tag)):
        p.error('Invalid tag')
    root = a.results_root.resolve()
    assets = root/(a.source_tag+'_assets')
    generation_path = assets/'generation.json'
    for f in (a.data, a.wan_ckpt/'Wan2.1_VAE.pth', generation_path):
        if not f.is_file():
            p.error(f'Missing file: {f}')
    generation = json.loads(generation_path.read_text(encoding='utf-8'))
    if generation['status'] != 'completed':
        raise ValueError('Source generation is incomplete')
    if sha256(a.data) != generation['configuration']['data_sha256']:
        raise ValueError('Source video changed; fixed-candidate validation requires the original Jockey input')
    selected = {}
    for kind in ('candidate','random'):
        entries = [r for r in generation['assets'] if r['name'] == 'eps4_r4_'+kind]
        if len(entries) != 1:
            raise ValueError(f'Expected one frozen {kind} asset')
        item = entries[0]
        if item['epsilon'] != 4 or item['round'] != 4 or item['input_Linf_pixel'] > 4.001:
            raise ValueError('Unexpected selected candidate settings')
        path = assets/item['file']
        if path.resolve().parent != assets.resolve() or sha256(path) != item['sha256']:
            raise ValueError('Invalid/modified source asset')
        selected[kind] = item
    if abs(selected['candidate']['input_MSE']-selected['random']['input_MSE']) > 1e-9:
        raise ValueError('Source pair is not energy matched')
    print('Frozen eps4/r4 candidate and paired random; seeds 123,2026; clean/candidate/random = 6 native runs.', flush=True)
    print('No new candidate generation. Every case uses fresh normal 720p encoding and decoding.', flush=True)
    if a.plan_only:
        return
    folder = root/(a.tag+'_evaluation')
    folder.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name('run_flf2v_experiment.py').resolve()
    repo = script.parent.parent
    files = ['exp_flf2v/run_flf2v_experiment.py', 'exp_flf2v/attacks.py',
             'sde_rf_wan/turbo_pipeline.py', 'sde_rf_wan/turbo_codebook.py',
             'sde_rf_wan/wan_flf2v_wrapper.py', 'sde_rf_wan/ref_codec.py']
    code_hashes = {f:sha256(repo/f) for f in files}
    config = dict(source_generation_sha256=sha256(generation_path),
        source_fingerprint=generation['fingerprint'], selected=selected, seeds=[123,2026],
        source_path=str(a.data.resolve()), checkpoint=str(a.wan_ckpt.resolve()), code=code_hashes,
        protocol='fixed eps4/r4; no held-out optimization; normal 720p33 re-encoding',
        codec='M64 K16384 steps20 tail3 g3 CompressAI-q4')
    digest = hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    manifest = folder/'protocol.json'
    if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8'))['fingerprint'] != digest:
        raise ValueError('Protocol changed; use another output tag')
    atomic_json(manifest, dict(fingerprint=digest, configuration=config))
    rows = []
    for seed in (123,2026):
        for kind in ('clean','candidate','random'):
            name = f'seed{seed}_{kind}'
            run = a.tag+'_'+name
            metrics = root/run/'Jockey/gop0/metrics.json'
            asset = None if kind == 'clean' else assets/selected[kind]['file']
            cmd = validation_command(script,a.data.resolve(),a.wan_ckpt.resolve(),root,run,asset,seed)
            receipt = folder/(name+'.json')
            cached = False
            if receipt.exists():
                saved = json.loads(receipt.read_text(encoding='utf-8'))
                if saved['fingerprint'] != digest or saved['command'] != cmd:
                    raise ValueError('Receipt configuration mismatch')
                cached = metrics.is_file() and sha256(metrics) == saved['metrics_sha256']
            if not cached:
                print(f'[NATIVE {len(rows)+1}/6] {name}',flush=True)
                subprocess.run(cmd,check=True)
            values = read_metrics(metrics)
            expected = 0. if kind == 'clean' else selected[kind]['input_MSE']
            if abs(values['InputMSE']-expected) > 1e-9 or values['InputLinfPixel'] > 4.001:
                raise ValueError('Native input budget does not match frozen asset')
            if kind == 'clean' and values['InputLinfPixel'] != 0:
                raise ValueError('Clean input was modified')
            atomic_json(receipt,dict(fingerprint=digest,command=cmd,metrics_sha256=sha256(metrics),metrics=values))
            rows.append(dict(ModelSeed=seed,Variant=kind,**values))
            write_csv(folder/'summary.csv',rows)
            print(f'[{"REUSE" if cached else "DONE"}] {name}: PSNR={values["PSNR"]:.4f} LPIPS={values["LPIPS"]:.4f}',flush=True)
    paired = paired_results(rows)
    write_csv(folder/'paired.csv',paired)
    atomic_json(folder/'completed.json',dict(status='completed',fingerprint=digest,paired=paired,
        limitation='Two model-seed checks on one source GOP; not cross-video generalization or proof of imperceptibility.'))
    print('\nSeed  DropVsClean  ExtraDropVsRandom  ExtraLPIPSVsRandom')
    for r in paired:
        print(f'{r["ModelSeed"]:4} {r["CandidateDropVsClean"]:+12.4f} {r["ExtraDropVsRandom"]:+18.4f} {r["ExtraLPIPSVsRandom"]:+19.4f}')
    print(f'Saved: {folder/"paired.csv"}',flush=True)


if __name__ == '__main__':
    main()
