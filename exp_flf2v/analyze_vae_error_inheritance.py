"""Full-frame VAE audit paired with recorded native GVCC metrics, not MP4 errors.

Association is not causality. VAE and GVCC are two different reconstruction maps;
their MSE ratio is not a causal error-retention fraction.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exp_flf2v.search_native_rgb_input import atomic_json, sha256, load_source
from exp_flf2v.run_vae_warmstart_batch import write_csv


def association(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.shape != (33,) or y.shape != (33,) or not np.isfinite([x,y]).all():
        raise ValueError('Expected 33 finite frame measurements')
    return dict(Pearson=float(np.corrcoef(x,y)[0,1]) if x.std()>1e-12 and y.std()>1e-12 else None,
                BothPositiveFrames=int(((x>0)&(y>0)).sum()),
                VAEPositiveFrames=int((x>0).sum()),NativePositiveFrames=int((y>0).sum()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data_root', type=Path, required=True)
    p.add_argument('--vae', type=Path, required=True)
    p.add_argument('--root', type=Path, default=Path('exp_flf2v/results_720p'))
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    root = a.root.resolve()
    out = root/'vae_content_rgb_v1_inheritance'
    out.mkdir(parents=True, exist_ok=True)
    # Preflight every input before allocating GPU memory.
    cases = []
    for seq in ('Jockey','Bosphorus','HoneyBee'):
        tag = f'vae_content_rgb_v1_{seq.lower()}_r4-20-44'
        assets = root/(tag+'_assets')
        report = json.loads((assets/'generation.json').read_text(encoding='utf-8'))
        if report['status'] != 'completed':
            raise ValueError(f'Incomplete generation: {seq}')
        data = a.data_root/f'{seq}_720p.yuv'
        if sha256(data) != report['configuration']['data_sha256']:
            raise ValueError(f'Source video changed: {seq}')
        for item in report['assets']:
            asset = assets/item['file']
            if sha256(asset) != item['sha256']:
                raise ValueError(f'Asset changed: {asset}')
            name = 'clean' if item['kind']=='initial' else item['name']
            folder = root/(tag+'_'+name)/seq/'gop0'
            metrics = folder/'metrics.json'
            receipt = json.loads((assets/(name+'_native.json')).read_text(encoding='utf-8'))
            if receipt['metrics_hash'] != sha256(metrics) or receipt['identity']['asset'] != item['sha256']:
                raise ValueError('Native metric/asset receipt mismatch')
            m = json.loads(metrics.read_text(encoding='utf-8'))
            frames = np.asarray(m['per_frame_PSNR_dB'],dtype=float)
            association(frames,frames)
            cases.append((seq,name,item,data,asset,folder,m,frames))
    import torch
    from wan.modules.vae import WanVAE
    if not torch.cuda.is_available():
        raise RuntimeError('Run this on the laboratory GPU, not a CPU fallback')
    model = None
    rows, summary, measured = [], [], {}
    current_seq, raw = None, None
    for seq,name,item,data,asset,folder,m,native_psnr in cases:
        if seq != current_seq:
            raw = load_source(data)
            current_seq = seq
        cache = out/f'{seq}_{name}.json'
        identity = dict(asset=sha256(asset),data=sha256(data),vae=sha256(a.vae),
                        dtype='bfloat16', geometry=[33,720,1280],
                        implementation=sha256(__file__))
        if cache.exists():
            saved = json.loads(cache.read_text(encoding='utf-8'))
            if saved['identity'] != identity:
                raise ValueError('Audit configuration changed; do not reuse stale cache')
            vae_mse = np.asarray(saved['frame_MSE'])
        else:
            if model is None:
                model = WanVAE(vae_pth=str(a.vae),device=a.device,dtype=torch.bfloat16)
                model.model.eval().requires_grad_(False)
            print(f'[VAE full720 no-grad] {seq} {name}',flush=True)
            with np.load(asset,allow_pickle=False) as z:
                delta = z['delta'].astype(np.float32)
            source = raw.astype(np.float32)/255
            # Replay the same realized uint8 submission used by the normal codec.
            submitted = np.floor(np.clip(source+delta,0,1)*255+.5).astype(np.uint8)
            pixels = torch.from_numpy(submitted.astype(np.float32)/255).permute(3,0,1,2).to(a.device)
            with torch.no_grad():
                latent = model.encode([pixels*2-1])[0]
                reconstruction = ((model.decode([latent])[0]+1)/2).clamp(0,1)
                recon = reconstruction.permute(1,2,3,0).float().cpu().numpy()
            vae_mse = np.square(recon-source).mean(axis=(1,2,3),dtype=np.float64)
            # Qualitative images only. Native MP4 snapshots are not numerical evidence.
            if name in ('clean','eps4_r44_candidate','eps4_r44_random'):
                import cv2
                clean_scores = next(c[7] for c in cases if c[0]==seq and c[1]=='clean')
                attack_scores = next(c[7] for c in cases if c[0]==seq and c[1]=='eps4_r44_candidate')
                selected = np.argsort(clean_scores-attack_scores)[-3:][::-1]
                for f in selected:
                    image = np.floor(recon[f]*255+.5).astype(np.uint8)
                    cv2.imwrite(str(out/f'{seq}_{name}_frame{f:02d}_vae.png'),cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
                    cap = cv2.VideoCapture(str(folder/'reconstructed.mp4'))
                    cap.set(cv2.CAP_PROP_POS_FRAMES,int(f))
                    ok, preview = cap.read()
                    cap.release()
                    if ok:
                        cv2.imwrite(str(out/f'{seq}_{name}_frame{f:02d}_native_LOSSY_PREVIEW.png'),preview)
                    cv2.imwrite(str(out/f'{seq}_frame{f:02d}_source.png'),cv2.cvtColor(raw[f],cv2.COLOR_RGB2BGR))
            del pixels,latent,reconstruction,recon,source,submitted,delta
            torch.cuda.empty_cache()
            atomic_json(cache,dict(identity=identity,frame_MSE=vae_mse.tolist()))
        vae_psnr = -10*np.log10(np.maximum(vae_mse,1e-12))
        native_mse = 10**(-native_psnr/10)  # Derived from saved PSNR rounded to 4 decimals.
        measured[(seq,name)] = (vae_mse,vae_psnr,native_mse,native_psnr)
        for f in range(33):
            rows.append(dict(Sequence=seq,Variant=name,FrameZeroBased=f,
                             VAE_MSE=vae_mse[f],VAE_PSNR=vae_psnr[f],
                             Native_PSNR=native_psnr[f],Native_MSE_from_rounded_PSNR=native_mse[f]))
        write_csv(out/'per_frame.csv',rows)
    for seq in ('Jockey','Bosphorus','HoneyBee'):
        clean = measured[(seq,'clean')]
        for rounds in (4,20,44):
            c = measured[(seq,f'eps4_r{rounds}_candidate')]
            r = measured[(seq,f'eps4_r{rounds}_random')]
            for reference, baseline in (('clean',clean),('matched_random',r)):
                summary.append(dict(Sequence=seq,Rounds=rounds,Reference=reference,
                                    MeanVAEMSEIncrease=float((c[0]-baseline[0]).mean()),
                                    MeanNativeMSEIncrease=float((c[2]-baseline[2]).mean()),
                                    VAE_PSNRDrop=float((baseline[1]-c[1]).mean()),
                                    Native_PSNRDrop=float((baseline[3]-c[3]).mean()),
                                    **association(c[0]-baseline[0],c[2]-baseline[2])))
    write_csv(out/'summary.csv',summary)
    atomic_json(out/'completed.json',dict(status='completed',cases=len(cases),
        limitations=['Frame association is descriptive, not causal error inheritance.',
                     'Frames are temporally dependent; no significance claim is made.',
                     'Native MSE derived from original in-memory PSNR rounded to 4 decimals.',
                     'MP4 previews are lossy and are excluded from all error calculations.',
                     'No exact native spatial error map is available from these old outputs.',
                     'Full-GOP VAE differs from the 5-frame spatial crops used in optimization.']))
    print(f'COMPLETE: {out / "summary.csv"}',flush=True)


if __name__ == '__main__':
    main()
