"""Analyze temporal propagation in completed GVCC codebook attacks.

This script performs no model inference.  It reads the clean and attacked
``metrics.json`` files, compares their per-frame PSNR, and separates the four
output frames represented by each attacked latent-time position from the rest
of the 33-frame GOP.
"""

import argparse
import csv
import json
import re
import statistics
from pathlib import Path


FRAME_RUN_RE = re.compile(r"sensitivity_step1_frame(\d+)_sign16(?:_seed(\d+))?$")


def mean(values):
    return sum(values) / len(values) if values else None


def represented_output_frames(latent_frame, num_output_frames):
    """Wan temporal factor 4: latent 0 is frame 0; others represent 4 frames."""
    if latent_frame == 0:
        return [0]
    start = 1 + 4 * (latent_frame - 1)
    return list(range(start, min(start + 4, num_output_frames)))


def load_metrics(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_root", type=Path, default=Path("exp_flf2v/results_720p"),
        help="Directory containing clean and sensitivity_* result folders",
    )
    parser.add_argument(
        "--clean_metrics", type=Path, default=None,
        help="Clean metrics.json; defaults to results_root/Jockey/gop0/metrics.json",
    )
    parser.add_argument("--sequence", default="Jockey")
    args = parser.parse_args()

    clean_path = args.clean_metrics
    if clean_path is None:
        candidates = [
            args.results_root / "sensitivity_clean_redecode" / args.sequence / "gop0" / "metrics.json",
            args.results_root / args.sequence / "gop0" / "metrics.json",
        ]
        clean_path = next((path for path in candidates if path.exists()), candidates[0])
    if not clean_path.exists():
        raise FileNotFoundError(f"Clean metrics not found: {clean_path}")
    clean = load_metrics(clean_path)
    clean_per_frame = clean.get("per_frame_PSNR_dB")
    if not clean_per_frame:
        raise ValueError(
            f"Clean metrics has no per_frame_PSNR_dB: {clean_path}. "
            "Run exp_flf2v/run_reused_clean_reference.ps1 first."
        )

    runs = []
    for child in args.results_root.iterdir():
        match = FRAME_RUN_RE.fullmatch(child.name)
        if not match:
            continue
        metrics_path = child / args.sequence / "gop0" / "metrics.json"
        if metrics_path.exists():
            seed = int(match.group(2)) if match.group(2) is not None else 42
            runs.append((int(match.group(1)), seed, child.name, metrics_path))
    if not runs:
        raise FileNotFoundError("No sensitivity_step1_frame*_sign16 results found")
    runs.sort()

    summary = []
    detail_rows = []
    all_frames = set(range(len(clean_per_frame)))
    for latent_frame, seed, run_name, metrics_path in runs:
        attacked = load_metrics(metrics_path)
        attacked_per_frame = attacked.get("per_frame_PSNR_dB")
        if not attacked_per_frame or len(attacked_per_frame) != len(clean_per_frame):
            raise ValueError(f"Per-frame PSNR length mismatch: {metrics_path}")
        deltas = [float(a) - float(c) for a, c in zip(attacked_per_frame, clean_per_frame)]
        represented = represented_output_frames(latent_frame, len(deltas))
        outside = sorted(all_frames - set(represented))
        worst_frame = min(range(len(deltas)), key=lambda index: deltas[index])
        row = {
            "latent_frame": latent_frame,
            "seed": seed,
            "represented_output_frames": represented,
            "PSNR_dB": attacked["PSNR_dB"],
            "LPIPS": attacked["LPIPS"],
            "gop_mean_psnr_delta_dB": round(float(attacked["PSNR_dB"]) - float(clean["PSNR_dB"]), 4),
            "represented_frames_mean_delta_dB": round(mean([deltas[i] for i in represented]), 4),
            "outside_frames_mean_delta_dB": round(mean([deltas[i] for i in outside]), 4),
            "worst_output_frame": worst_frame,
            "worst_frame_delta_dB": round(deltas[worst_frame], 4),
            "metrics_path": str(metrics_path),
        }
        summary.append(row)
        for output_frame, delta in enumerate(deltas):
            detail_rows.append({
                "latent_frame": latent_frame,
                "seed": seed,
                "output_frame": output_frame,
                "is_represented_segment": output_frame in represented,
                "clean_psnr_dB": clean_per_frame[output_frame],
                "attacked_psnr_dB": attacked_per_frame[output_frame],
                "delta_psnr_dB": round(delta, 4),
            })

    summary_path = args.results_root / "sensitivity_frame_propagation_summary.json"
    detail_path = args.results_root / "sensitivity_frame_per_output_psnr.csv"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump({"clean_metrics": str(clean_path), "runs": summary}, handle, indent=2)
    with detail_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail_rows[0]))
        writer.writeheader()
        writer.writerows(detail_rows)

    print("\nTemporal propagation of first-step sign corruption")
    print("LF  Seed  GOP-dPSNR  Local-dPSNR  Outside-dPSNR  WorstFrame  WorstDelta")
    for row in summary:
        print(
            f"{row['latent_frame']:>2}  {row['seed']:>4}  "
            f"{row['gop_mean_psnr_delta_dB']:>9.4f}  "
            f"{row['represented_frames_mean_delta_dB']:>11.4f}  "
            f"{row['outside_frames_mean_delta_dB']:>13.4f}  "
            f"{row['worst_output_frame']:>10}  {row['worst_frame_delta_dB']:>10.4f}"
        )

    grouped = {}
    for row in summary:
        grouped.setdefault(row["latent_frame"], []).append(row)
    print("\nMulti-seed aggregate (available seeds only)")
    print("LF  N  PSNR_mean  PSNR_std  LPIPS_mean  LPIPS_std  LocalDelta  OutsideDelta")
    for latent_frame, rows in sorted(grouped.items()):
        psnr = [float(row["PSNR_dB"]) for row in rows]
        lpips = [float(row["LPIPS"]) for row in rows]
        local = [row["represented_frames_mean_delta_dB"] for row in rows]
        outside = [row["outside_frames_mean_delta_dB"] for row in rows]
        print(
            f"{latent_frame:>2}  {len(rows):>1}  {statistics.mean(psnr):>9.4f}  "
            f"{statistics.pstdev(psnr):>8.4f}  {statistics.mean(lpips):>10.4f}  "
            f"{statistics.pstdev(lpips):>9.4f}  {statistics.mean(local):>10.4f}  "
            f"{statistics.mean(outside):>12.4f}"
        )
    print(f"\nSaved: {summary_path}")
    print(f"Saved: {detail_path}")


if __name__ == "__main__":
    main()
