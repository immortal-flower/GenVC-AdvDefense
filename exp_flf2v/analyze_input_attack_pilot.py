"""Compare real re-encoded codebooks and reconstruction metrics for an input pilot.

This is post-hoc analysis only. No codec metadata is exposed to the attacker.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def compare_records(clean, attacked):
    if (clean['K'], clean['num_sde_steps'], clean['num_latent_frames']) != (
        attacked['K'], attacked['num_sde_steps'], attacked['num_latent_frames']
    ):
        raise ValueError('Incompatible clean and attacked codebooks')
    rows = []
    for step, (cs, ats) in enumerate(zip(clean['step_data'], attacked['step_data']), 1):
        for pos, ((ci, sg), (ai, ag)) in enumerate(zip(cs, ats)):
            left, right = set(ci), set(ai)
            csign, asign = dict(zip(ci, sg)), dict(zip(ai, ag))
            shared = left & right
            rows.append({
                'sde_step': step,
                'latent_position': pos,
                'clean_atoms': len(left),
                'attacked_atoms': len(right),
                'replaced_atoms': len(left - right),
                'index_jaccard': len(shared) / len(left | right) if left | right else 1.0,
                'shared_atoms_sign_flipped': sum(csign[k] != asign[k] for k in shared),
            })
    return rows


def main():
    import numpy as np
    from sde_rf_wan.turbo_codebook import TurboBitstream
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results_root', type=Path, required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--sequence', default='Jockey')
    p.add_argument('--gop', type=int, default=0)
    args = p.parse_args()
    base = args.results_root
    clean_dir = base / f'{args.tag}_clean' / args.sequence / f'gop{args.gop}'
    clean = TurboBitstream.load(str(clean_dir / 'codebook.tdcm'))
    clean_metrics = json.loads((clean_dir / 'metrics.json').read_text())
    out = []
    for name in ('random', 'old_vae', 'mean_vae', 'balanced_vae'):
        run_dir = base / f'{args.tag}_{name}' / args.sequence / f'gop{args.gop}'
        if not (run_dir / 'metrics.json').exists():
            continue
        attack = TurboBitstream.load(str(run_dir / 'codebook.tdcm'))
        metrics = json.loads((run_dir / 'metrics.json').read_text())
        records = compare_records(clean, attack)
        trace_path = run_dir / 'input_attack_trace.npz'
        clean_trace_path = clean_dir / 'input_attack_trace.npz'
        trace_rows = []
        if trace_path.exists() and clean_trace_path.exists():
            with np.load(clean_trace_path, allow_pickle=False) as ct, np.load(trace_path, allow_pickle=False) as at:
                for key in ct.files:
                    if key not in at or ct[key].shape != at[key].shape:
                        raise ValueError(f'Incompatible trace: {key}')
                    c, a = ct[key], at[key]
                    for f in range(c.shape[2]):
                        u = c[:, :, f].astype(np.float64).ravel()
                        v = a[:, :, f].astype(np.float64).ravel()
                        denom = np.linalg.norm(u)*np.linalg.norm(v)
                        trace_rows.append(dict(quantity=key, latent_position=f,
                            difference_MSE=float(np.mean((u-v)**2)),
                            cosine=float(np.dot(u,v)/denom) if denom else None,
                            clean_RMS=float(np.sqrt(np.mean(u*u))),
                            attacked_RMS=float(np.sqrt(np.mean(v*v)))))
            trace_csv = base / f'{args.tag}_{name}_trajectory_changes.csv'
            with trace_csv.open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(trace_rows[0]))
                writer.writeheader()
                writer.writerows(trace_rows)
        path = base / f'{args.tag}_{name}_codebook_changes.csv'
        with path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        early = [r for r in records if r['sde_step'] <= 2]
        out.append({
            'run': name,
            'psnr_drop_db': round(clean_metrics['PSNR_dB'] - metrics['PSNR_dB'], 3),
            'delta_lpips': round(metrics['LPIPS'] - clean_metrics['LPIPS'], 4),
            'input_psnr_db': metrics.get('input_PSNR_dB'),
            'input_linf_pixel': metrics.get('input_Linf_pixel'),
            'psnr_vs_submitted': metrics.get('reconstruction_vs_submitted_PSNR_dB'),
            'early_replaced_atoms_mean': round(sum(r['replaced_atoms'] for r in early) / len(early), 2),
            'all_replaced_atoms_mean': round(sum(r['replaced_atoms'] for r in records) / len(records), 2),
            'detail_csv': str(path),
        })
    target = base / f'{args.tag}_diagnostic.json'
    report = {
        'interpretation': 'Atom/latent/correction changes are descriptive, not proof of attack success or a causal attribution of absorption. Trace CSVs compare the actual first two SDE updates when available.',
        'primary_comparison': 'balanced_vae versus mean_vae: same resolution, anchors, initialization, optimizer and iteration budget',
        'legacy_comparison': 'old_vae is a historical reference, not a matched objective ablation',
        'results': out,
    }
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f'Saved: {target}')


if __name__ == '__main__':
    main()
