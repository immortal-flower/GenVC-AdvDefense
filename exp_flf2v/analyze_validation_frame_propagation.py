"""Analyze frame-wise propagation for targeted codebook validation runs.

This script performs no neural-network inference.  It reads the saved
``per_frame_PSNR_dB`` arrays for the cells in a validation plan, aggregates
them across attack seeds, and compares each sensitive cell with its matched
same-step and same-position controls.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

try:
    from codebook_validation_plan import expand_jobs, load_cells
except ModuleNotFoundError:
    from exp_flf2v.codebook_validation_plan import expand_jobs, load_cells


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8-sig") as handle:
        return json.load(handle)


def represented_output_frames(latent_position: int, num_output_frames: int) -> list[int]:
    """Map Wan's temporal latent position to the output frames it represents."""
    if latent_position == 0:
        return [0] if num_output_frames else []
    start = 1 + 4 * (latent_position - 1)
    return list(range(start, min(start + 4, num_output_frames)))


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def pstdev(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect_runs(results_root: Path, sequence: str, plan: Path, count: int,
                 attack_seeds: list[int], clean_metrics: Path,
                 gop_index: int = 0):
    clean = load_json(clean_metrics)
    clean_frames = clean.get("per_frame_PSNR_dB")
    if not clean_frames:
        raise ValueError(f"Clean metrics has no per_frame_PSNR_dB: {clean_metrics}")
    clean_frames = [float(value) for value in clean_frames]

    jobs = expand_jobs(load_cells(plan), [count], attack_seeds)
    detail = []
    missing = []
    for job in jobs:
        metrics_path = (results_root / job.run_name(sequence, gop_index) /
                        sequence / f"gop{gop_index}" / "metrics.json")
        if not metrics_path.exists():
            missing.append(str(metrics_path))
            continue
        metrics = load_json(metrics_path)
        attacked_frames = metrics.get("per_frame_PSNR_dB")
        if not attacked_frames or len(attacked_frames) != len(clean_frames):
            missing.append(
                f"{metrics_path} (per-frame length mismatch: "
                f"{len(attacked_frames or [])} != {len(clean_frames)})")
            continue
        for frame, (clean_psnr, attacked_psnr) in enumerate(
                zip(clean_frames, attacked_frames)):
            detail.append({
                "role": job.role,
                "sde_step": job.step,
                "latent_position": job.frame,
                "requested_bits": job.count,
                "attack_seed": job.attack_seed,
                "output_frame": frame,
                "is_direct_segment": frame in represented_output_frames(
                    job.frame, len(clean_frames)),
                "clean_psnr_db": clean_psnr,
                "attacked_psnr_db": float(attacked_psnr),
                "psnr_loss_db": clean_psnr - float(attacked_psnr),
                "run_name": job.run_name(sequence, gop_index),
            })
    return detail, missing, len(clean_frames)


def aggregate_frames(detail: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for row in detail:
        key = (row["role"], row["sde_step"], row["latent_position"],
               row["requested_bits"], row["output_frame"])
        grouped.setdefault(key, []).append(row)
    rows = []
    for key, members in sorted(grouped.items(), key=lambda item: item[0]):
        role, step, latent, count, frame = key
        losses = [member["psnr_loss_db"] for member in members]
        rows.append({
            "role": role,
            "sde_step": step,
            "latent_position": latent,
            "requested_bits": count,
            "output_frame": frame,
            "n_attack_seeds": len(members),
            "is_direct_segment": members[0]["is_direct_segment"],
            "psnr_loss_mean_db": mean(losses),
            "psnr_loss_std_db": pstdev(losses),
        })
    return rows


def summarize_cells(detail: list[dict], num_frames: int) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for row in detail:
        key = (row["role"], row["sde_step"], row["latent_position"],
               row["requested_bits"])
        grouped.setdefault(key, []).append(row)
    rows = []
    for key, members in sorted(grouped.items(), key=lambda item: item[0]):
        role, step, latent, count = key
        by_seed: dict[int, list[float]] = {}
        by_frame: dict[int, list[float]] = {}
        for member in members:
            by_seed.setdefault(member["attack_seed"], []).append(member["psnr_loss_db"])
            by_frame.setdefault(member["output_frame"], []).append(member["psnr_loss_db"])
        seed_gop_losses = [mean(values) for values in by_seed.values()]
        frame_means = {frame: mean(values) for frame, values in by_frame.items()}
        direct = represented_output_frames(latent, num_frames)
        outside = [frame for frame in range(num_frames) if frame not in direct]
        before = [frame for frame in range(num_frames) if frame < min(direct)]
        after = [frame for frame in range(num_frames) if frame > max(direct)]
        worst_frame = max(frame_means, key=frame_means.get)
        rows.append({
            "role": role,
            "sde_step": step,
            "latent_position": latent,
            "requested_bits": count,
            "n_attack_seeds": len(by_seed),
            "direct_output_frames": (
                str(direct[0]) if len(direct) == 1 else f"{direct[0]}-{direct[-1]}"),
            "gop_psnr_loss_mean_db": mean(seed_gop_losses),
            "gop_psnr_loss_std_db": pstdev(seed_gop_losses),
            "direct_segment_loss_mean_db": mean([frame_means[f] for f in direct]),
            "outside_segment_loss_mean_db": mean([frame_means[f] for f in outside]),
            "before_segment_loss_mean_db": mean([frame_means[f] for f in before]),
            "after_segment_loss_mean_db": mean([frame_means[f] for f in after]),
            "worst_output_frame": worst_frame,
            "worst_frame_loss_mean_db": frame_means[worst_frame],
        })
    return rows


def find_cell(rows: list[dict], step: int, latent: int) -> list[dict]:
    return sorted(
        [row for row in rows
         if row["sde_step"] == step and row["latent_position"] == latent],
        key=lambda row: row["output_frame"])


def plot_pairs(frame_rows: list[dict], output_path: Path, num_frames: int) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    pairs = [
        ((1, 3), (1, 0), (9, 3)),
        ((1, 4), (1, 0), (9, 4)),
        ((2, 1), (2, 0), (9, 1)),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    colors = ("#c73e1d", "#2b6cb0", "#3a8d5d")
    labels = ("hotspot", "same-step control", "same-position control")
    for axis, cells in zip(axes, pairs):
        hotspot_step, hotspot_latent = cells[0]
        direct = represented_output_frames(hotspot_latent, num_frames)
        axis.axvspan(direct[0] - 0.5, direct[-1] + 0.5,
                     color="#f6c85f", alpha=0.20,
                     label="hotspot direct segment")
        for (step, latent), color, label in zip(cells, colors, labels):
            rows = find_cell(frame_rows, step, latent)
            if not rows:
                continue
            x = [row["output_frame"] for row in rows]
            y = [row["psnr_loss_mean_db"] for row in rows]
            std = [row["psnr_loss_std_db"] for row in rows]
            axis.plot(x, y, marker="o", markersize=2.5, linewidth=1.6,
                      color=color, label=f"{label}: SDE {step}, latent {latent}")
            axis.fill_between(x, [a - b for a, b in zip(y, std)],
                              [a + b for a, b in zip(y, std)],
                              color=color, alpha=0.12)
        axis.axhline(0.0, color="#666666", linestyle="--", linewidth=0.8)
        axis.set_ylabel("PSNR loss (dB)")
        axis.set_title(
            f"Hotspot SDE {hotspot_step}, latent {hotspot_latent} "
            f"(direct output frames {direct[0]}-{direct[-1]})")
        axis.grid(alpha=0.18)
        axis.legend(fontsize=8, ncol=2)
    axes[-1].set_xlabel("Output frame")
    axes[-1].set_xlim(0, num_frames - 1)
    fig.suptitle("GVCC codebook sensitivity: frame-wise propagation across attack seeds")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def rounded(value):
    return "N/A" if value is None or not math.isfinite(value) else f"{value:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", type=Path,
                        default=Path("exp_flf2v/results_720p"))
    parser.add_argument("--sequence", default="Jockey")
    parser.add_argument("--plan", type=Path,
                        default=Path("exp_flf2v/validation_cells_jockey.csv"))
    parser.add_argument("--clean_metrics", type=Path,
                        default=Path("exp_flf2v/results_720p/"
                                     "sensitivity_clean_redecode/Jockey/gop0/metrics.json"))
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--gop_index", type=int, default=0)
    parser.add_argument("--attack_seeds", type=int, nargs="+",
                        default=[42, 123, 2026])
    parser.add_argument("--output_prefix", default="validation_propagation")
    args = parser.parse_args()

    detail, missing, num_frames = collect_runs(
        args.results_root, args.sequence, args.plan, args.count,
        args.attack_seeds, args.clean_metrics, args.gop_index)
    if not detail:
        raise FileNotFoundError("No complete validation runs were found")
    frame_rows = aggregate_frames(detail)
    cell_rows = summarize_cells(detail, num_frames)
    sequence_tag = args.sequence.lower()
    gop_tag = "" if args.gop_index == 0 else f"_gop{args.gop_index}"
    prefix = args.results_root / f"{args.output_prefix}_{sequence_tag}{gop_tag}"
    detail_path = Path(f"{prefix}_detail.csv")
    frames_path = Path(f"{prefix}_per_frame.csv")
    cells_path = Path(f"{prefix}_cells.csv")
    plot_path = Path(f"{prefix}_paired.png")
    summary_path = Path(f"{prefix}_summary.json")
    write_csv(detail_path, detail)
    write_csv(frames_path, frame_rows)
    write_csv(cells_path, cell_rows)
    plotted = plot_pairs(frame_rows, plot_path, num_frames)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump({"sequence": args.sequence, "gop": args.gop_index,
                   "requested_bits": args.count,
                   "attack_seeds": args.attack_seeds, "num_output_frames": num_frames,
                   "missing": missing, "cells": cell_rows}, handle, indent=2)

    print("\nFrame-wise validation propagation")
    print("Role                  SDE Latent Seeds GOPLoss DirectLoss OutsideLoss WorstFrame WorstLoss")
    for row in cell_rows:
        print(f"{row['role']:<21} {row['sde_step']:>3} {row['latent_position']:>6} "
              f"{row['n_attack_seeds']:>5} "
              f"{rounded(row['gop_psnr_loss_mean_db']):>7} "
              f"{rounded(row['direct_segment_loss_mean_db']):>10} "
              f"{rounded(row['outside_segment_loss_mean_db']):>11} "
              f"{row['worst_output_frame']:>10} "
              f"{rounded(row['worst_frame_loss_mean_db']):>9}")
    if missing:
        print(f"WARNING: {len(missing)} requested runs were missing or incompatible")
    print(f"Saved: {detail_path}")
    print(f"Saved: {frames_path}")
    print(f"Saved: {cells_path}")
    print(f"Saved: {summary_path}")
    if plotted:
        print(f"Saved: {plot_path}")
    else:
        print("Plot skipped: matplotlib is not installed")


if __name__ == "__main__":
    main()
