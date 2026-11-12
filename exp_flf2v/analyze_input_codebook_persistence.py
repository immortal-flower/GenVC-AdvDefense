"""Measure how input perturbations change real GVCC codebook decisions by step.

This is a post-encoding diagnostic, not an attack loss and not proof that the
selected-atom differences caused the final video-quality change.
"""
import argparse
import csv
import json
from pathlib import Path

try:
    from .analyze_combinatorial_index_robustness import read_codebook_records
except ImportError:  # Direct script execution from exp_flf2v/
    from analyze_combinatorial_index_robustness import read_codebook_records


def compare_cell(clean, attacked):
    if clean['M'] != attacked['M']:
        raise ValueError('Different M_actual; compare like-for-like streams only')
    before = dict(zip(clean['indices'], clean['signs']))
    after = dict(zip(attacked['indices'], attacked['signs']))
    common = before.keys() & after.keys()
    sign_changes = sum(before[i] != after[i] for i in common)
    m = clean['M']
    return {'replaced_atoms': m - len(common),
            'shared_atoms': len(common),
            'shared_sign_changes': sign_changes,
            'replaced_fraction': (m - len(common)) / m if m else 0.0,
            'signed_change_fraction': (m - len(common) + sign_changes) / m if m else 0.0}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results_root', type=Path, required=True)
    p.add_argument('--clean_run', default='input_pilot_v2_eps2_seed42_steps20_clean')
    p.add_argument('--tag', default='input_direction_oldvae_eps2_seed42')
    p.add_argument('--sequence', default='Jockey')
    p.add_argument('--gop', type=int, default=0)
    a = p.parse_args()
    variants = ('full_positive', 'full_positive_random',
                'full_negative', 'full_negative_random',
                'half_positive', 'half_positive_random',
                'half_negative', 'half_negative_random')
    base = a.results_root / a.clean_run / a.sequence / f'gop{a.gop}'
    clean_k, clean_m, _, clean = read_codebook_records(base / 'codebook.tdcm')
    clean_metrics = json.loads((base / 'metrics.json').read_text(encoding='utf-8'))
    rows = []
    summary = []
    for variant in variants:
        path = a.results_root / f'{a.tag}_{variant}' / a.sequence / f'gop{a.gop}'
        if not (path / 'codebook.tdcm').is_file() or not (path / 'metrics.json').is_file():
            raise FileNotFoundError(f'Incomplete variant: {variant}; finish/restart the codec batch')
        k, m, _, attacked = read_codebook_records(path / 'codebook.tdcm')
        if (k, m, len(attacked)) != (clean_k, clean_m, len(clean)):
            raise ValueError(f'Codebook geometry differs for {variant}')
        metrics = json.loads((path / 'metrics.json').read_text(encoding='utf-8'))
        for original, changed in zip(clean, attacked):
            if (original['step'], original['latent']) != (changed['step'], changed['latent']):
                raise ValueError(f'Codebook record order differs for {variant}')
            comparison = compare_cell(original, changed)
            rows.append({'variant': variant, 'sde_step': original['step'] + 1,
                         'latent_position': original['latent'], **comparison})
        own = [r for r in rows if r['variant'] == variant]
        early = [r for r in own if r['sde_step'] <= 2]
        later = [r for r in own if r['sde_step'] >= 3]
        summary.append({'variant': variant,
                        'early_mean_replaced_fraction': sum(r['replaced_fraction'] for r in early) / len(early),
                        'later_mean_replaced_fraction': sum(r['replaced_fraction'] for r in later) / len(later),
                        'early_mean_signed_change_fraction': sum(r['signed_change_fraction'] for r in early) / len(early),
                        'later_mean_signed_change_fraction': sum(r['signed_change_fraction'] for r in later) / len(later),
                        'PSNR_drop_dB': float(clean_metrics['PSNR_dB']) - float(metrics['PSNR_dB']),
                        'Delta_LPIPS': float(metrics['LPIPS']) - float(clean_metrics['LPIPS'])})
    step_rows = []
    for variant in variants:
        for step in range(1, max(r['sde_step'] for r in rows) + 1):
            selected = [r for r in rows if r['variant'] == variant and r['sde_step'] == step]
            step_rows.append({'variant': variant, 'sde_step': step,
                              'mean_replaced_fraction': sum(r['replaced_fraction'] for r in selected) / len(selected),
                              'mean_signed_change_fraction': sum(r['signed_change_fraction'] for r in selected) / len(selected)})
    prefix = a.results_root / a.tag
    for suffix, items in (('_codebook_cells.csv', rows),
                          ('_codebook_steps.csv', step_rows),
                          ('_codebook_persistence.csv', summary)):
        target = prefix.with_name(prefix.name + suffix)
        with target.open('w', newline='', encoding='utf-8-sig') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(items[0]))
            writer.writeheader()
            writer.writerows(items)
        print(f'Saved: {target}')
    print('Variant                    EarlyReplaced  LaterReplaced  PSNRDrop  DeltaLPIPS')
    for row in summary:
        print(f"{row['variant']:<26} {row['early_mean_replaced_fraction']:.3f}"
              f"          {row['later_mean_replaced_fraction']:.3f}"
              f"         {row['PSNR_drop_dB']:+.3f}      {row['Delta_LPIPS']:+.4f}")
    print('Caution: persistent codebook changes are a mechanism clue, not final attack success.')


if __name__ == '__main__':
    main()
