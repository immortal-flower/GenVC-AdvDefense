"""Run from the lab checkout. Re-run the same command to resume."""
import argparse
from contextlib import redirect_stdout
from datetime import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
from time import perf_counter_ns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sde_rf_wan.turbo_codebook import TurboBitstream
from uvg_data import UVG_ORDER, find_uvg_sequences


def write_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def fingerprint(path, limit=None):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while limit is None or limit > 0:
            chunk = handle.read(min(limit, 1048576) if limit is not None else 1048576)
            if not chunk:
                if limit:
                    raise ValueError(f"Short source: {path}")
                break
            h.update(chunk)
            if limit is not None:
                limit -= len(chunk)
    return h.hexdigest()


def compare(folder, dest, m, sequence, g):
    dest.mkdir(parents=True, exist_ok=True)
    source = folder / "codebook_combinatorial.tdc2"
    data = TurboBitstream.load(str(source))
    metrics = json.loads((folder / "metrics.json").read_text())
    if (data["K"], data["M"], data["seed"], data["height"], data["width"],
        data["num_sde_steps"], data["num_latent_frames"]) != (16384, m, 42, 720, 1280, 17, 9):
        raise ValueError(f"Unexpected stream configuration in {folder}")
    if not metrics["rate_roundtrip_verified"] or (metrics["sequence"], metrics["gop"]) != (sequence, g):
        raise ValueError(f"Unexpected metrics in {folder}")
    if TurboBitstream.load(str(folder / "codebook_independent.tdcm")) != dict(data, index_coding="fixed-width"):
        raise ValueError(f"Legacy stream mismatch in {folder}")
    row = dict(M=m, sequence=sequence, gop=g, metrics=metrics, source_sha256=fingerprint(source))
    receipt = dest / "comparison.json"
    if receipt.is_file():
        saved = json.loads(receipt.read_text())
        if (all(saved[key] == value for key,value in row.items()) and
                all(fingerprint(dest/(mode+".bin")) == saved[mode]["sha256"]
                    for mode in ("packed","rice","combinatorial"))):
            return saved
        raise ValueError(f"Archived data changed: {receipt}")
    for mode in ("packed", "rice", "combinatorial"):
        path = dest / (mode + ".bin")
        timings = []
        with redirect_stdout(io.StringIO()):
            for repeat in range(6):
                start = perf_counter_ns()
                TurboBitstream.save(str(path), **dict(data, index_coding=mode))
                mid = perf_counter_ns()
                restored = TurboBitstream.load(str(path))
                end = perf_counter_ns()
                if restored != dict(data, index_coding=mode):
                    raise ValueError(f"Round trip failed: {path}")
                if repeat:
                    timings.append([(mid-start)/1e6, (end-mid)/1e6])
        row[mode] = dict(bytes=path.stat().st_size, sha256=fingerprint(path),
                         pack_unpack_ms_with_io=timings)
    write_json(dest / "comparison.json", row)
    return row


def summarize(rows, out):
    from scipy.interpolate import PchipInterpolator
    import math
    points = []
    for seq in UVG_ORDER:
        for m in (32, 64, 98, 128):
            group = sorted((r for r in rows if (r["sequence"], r["M"]) == (seq, m)), key=lambda r:r["gop"])
            if len(group) != 3:
                continue
            point = dict(sequence=seq, M=m)
            for mode in ("packed", "combinatorial", "rice"):
                bits = 8 * sum(r[mode]["bytes"] + r["metrics"]["boundary_bytes"] for r in group)
                point[mode] = bits / (97 * 720 * 1280)
            for key in ("PSNR_dB", "LPIPS"):
                point[key] = statistics.mean(r["metrics"][key] for r in group)
            points.append(point)
    bd = {}
    paired = {}
    for seq in UVG_ORDER:
        p = [v for v in points if v["sequence"] == seq]
        by_m = {v["M"]:v for v in p}
        if 64 in by_m and 98 in by_m:
            a, b = by_m[64], by_m[98]
            paired[seq] = dict(delta_BPP=b["combinatorial"]-a["packed"],
                               delta_PSNR=b["PSNR_dB"]-a["PSNR_dB"], delta_LPIPS=b["LPIPS"]-a["LPIPS"])
        if len(p) != 4:
            continue
        bd[seq] = {}
        for key, sign in (("PSNR_dB", 1), ("LPIPS", -1)):
            x = [sign*v[key] for v in p]
            if any(b <= a for a,b in zip(x,x[1:])):
                bd[seq][key] = dict(status="nonmonotone_or_tied", quality=x)
                continue
            # Identical quality knots in both representations.
            def integral(mode):
                return PchipInterpolator(x, [math.log(v[mode]) for v in p]).integrate(x[0],x[-1])
            value = 100*math.expm1((integral("combinatorial")-integral("packed"))/(x[-1]-x[0]))
            bd[seq][key] = dict(percent=value, quality_domain=[x[0],x[-1]])
    write_json(out / "paper_summary.json", dict(completed_gops=len(rows), points=points,
               matched_rate_M98_vs_M64=paired, bd_rate=bd))


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--data", type=Path, default=Path(r"D:\data\UVG"))
    ap.add_argument("--checkpoint", type=Path, default=ROOT/"exp_flf2v/Wan2.1-FLF2V-14B-720P")
    old = ROOT/"exp_flf2v/results_720p/flf2v_combinatorial_rate_uvg"
    ap.add_argument("--baseline", type=Path, default=old)
    ap.add_argument("--out", type=Path, default=old.parent/"paper_rates_v1")
    a = ap.parse_args()
    os.chdir(ROOT)
    a.out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((a.baseline/"experiment_config.json").read_text())
    params = dict(K=16384, seed=42, height=720, width=1280, num_gops=3, num_frames_per_gop=33,
                  steps=20, ddim_tail=3, g_scale=3., ref_codec="compressai", ref_quality=4,
                  attack="none", defense="none", stream_attack="none", start_gop=0)
    if any(cfg.get(k) != v for k,v in dict(params,M=64).items()):
        raise ValueError("M64 baseline configuration differs")
    files = dict(find_uvg_sequences(str(a.data)))
    if set(files) != set(UVG_ORDER) or not a.checkpoint.is_dir():
        raise ValueError("Check all seven UVG inputs and FLF2V checkpoint")
    protocol = dict(parameters=params, checkpoint=str(a.checkpoint.resolve()), python=sys.executable,
                    commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
                    inputs={s:dict(path=str(p),sha256_first_97_frames=fingerprint(Path(p),97*1920*1080*3//2))
                            for s,p in files.items()})
    manifest = a.out/"protocol.json"
    if manifest.exists() and json.loads(manifest.read_text()) != protocol:
        raise ValueError("Run configuration changed; choose a fresh --out")
    write_json(manifest, protocol)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    rows = []
    for m in (64,98,32,128):
        run = a.baseline if m == 64 else a.out/f"M{m}"
        if m != 64 and any(not (run/s/f"gop{g}/metrics.json").exists() for s in UVG_ORDER for g in range(3)):
            cmd = [sys.executable,"-u","exp_flf2v/run_flf2v_experiment.py","--archive_rate_sweep",
                   "--wan_ckpt",str(a.checkpoint),"--data_dir",str(a.data),"--output_dir",str(run),"--M",str(m)]
            for k,v in params.items():
                cmd.extend(["--"+k,str(v)])
            print(f"Starting M={m}. Log: {a.out / f'M{m}.log'}",flush=True)
            with (a.out/f"M{m}.log").open("a",encoding="utf-8") as log:
                print(datetime.now().isoformat(),cmd,file=log,flush=True)
                subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
        for s in UVG_ORDER:
            for g in range(3):
                folder = run/s/f"gop{g}"
                dest = a.out/"comparisons"/f"M{m}"/s/f"gop{g}"
                if m == 64:
                    dest.mkdir(parents=True,exist_ok=True)
                    for name in ("metrics.json","codebook_combinatorial.tdc2","codebook_independent.tdcm"):
                        if (dest/name).exists():
                            if fingerprint(folder/name) != fingerprint(dest/name):
                                raise ValueError(f"Baseline copy differs: {dest/name}")
                        else:
                            shutil.copy2(folder/name,dest/name)
                    folder = dest
                rows.append(compare(folder,dest,m,s,g))
        summarize(rows,a.out)
        print(f"Archived and verified M={m}: {len(rows)}/84 GOPs",flush=True)


if __name__ == "__main__":
    main()
