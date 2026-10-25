"""Summarize uniform GVCC codebook sensitivity-grid experiments.

The script performs no model inference. It reads completed ``metrics.json``
files produced by ``run_codebook_sensitivity_grid.ps1`` and writes tidy CSV,
JSON, per-seed heatmap matrices, and an across-attack-seed aggregate.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


RUN_RE = re.compile(
    r"sensitivity_grid_(sign|index)(\d+)_step(\d+)_frame(\d+)_aseed(\d+)"
    r"(?:_gop(\d+))?_([a-z0-9_-]+)$"
)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_matrix(path: Path, values: dict[tuple[int, int], float | None]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sde_step", *[f"latent_{frame}" for frame in range(9)]])
        for step in range(1, 18):
            writer.writerow([step, *[values.get((step, frame), "") for frame in range(9)]])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", type=Path, default=Path("exp_flf2v/results_720p"))
    parser.add_argument("--sequence", default="Jockey")
    parser.add_argument("--gop_index", type=int, default=0)
    parser.add_argument("--clean_metrics", type=Path, required=True)
    parser.add_argument("--output_prefix", default="sensitivity_grid")
    args = parser.parse_args()

    if not args.clean_metrics.exists():
        raise FileNotFoundError(f"Clean metrics not found: {args.clean_metrics}")
    clean = load_json(args.clean_metrics)
    clean_psnr = float(clean["PSNR_dB"])
    clean_lpips = float(clean["LPIPS"])
    sequence_tag = re.sub(r"[^a-z0-9_-]", "_", args.sequence.lower())

    rows: list[dict] = []
    for run_dir in sorted(args.results_root.iterdir()):
        if not run_dir.is_dir():
            continue
        match = RUN_RE.fullmatch(run_dir.name)
        if not match:
            continue
        field, count, step, frame, attack_seed, encoded_gop, encoded_sequence = match.groups()
        run_gop = int(encoded_gop or 0)
        if encoded_sequence != sequence_tag or run_gop != args.gop_index:
            continue
        metrics_path = (run_dir / args.sequence / f"gop{run_gop}" /
                        "metrics.json")
        if not metrics_path.exists():
            continue
        metrics = load_json(metrics_path)
        attack = metrics.get("stream_attack", {})
        psnr = float(metrics["PSNR_dB"])
        lpips = float(metrics["LPIPS"])
        changed_bits = int(attack.get("changed_bits", int(count)))
        payload_ber = float(attack.get("payload_BER", 0.0))
        rows.append(
            {
                "sequence": args.sequence,
                "gop": run_gop,
                "field": field,
                "sde_step": int(step),
                "latent_position": int(frame),
                "requested_bits": int(count),
                "changed_bits": changed_bits,
                "attack_seed": int(attack_seed),
                "payload_ber": payload_ber,
                "clean_psnr_db": clean_psnr,
                "psnr_db": psnr,
                "delta_psnr_db": psnr - clean_psnr,
                "psnr_drop_db": clean_psnr - psnr,
                "clean_lpips": clean_lpips,
                "lpips": lpips,
                "delta_lpips": lpips - clean_lpips,
                "decode_status": "ok",
                "run_name": run_dir.name,
                "metrics_path": str(metrics_path),
            }
        )

    if not rows:
        raise FileNotFoundError(
            f"No completed sensitivity_grid_* runs for sequence {args.sequence!r}"
        )
    rows.sort(key=lambda row: (row["field"], row["requested_bits"], row["attack_seed"],
                               row["sde_step"], row["latent_position"]))

    gop_tag = "" if args.gop_index == 0 else f"_gop{args.gop_index}"
    output_stem = f"{args.output_prefix}_{sequence_tag}{gop_tag}"
    tidy_path = args.results_root / f"{output_stem}_tidy.csv"
    write_csv(tidy_path, rows, list(rows[0]))

    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["field"], row["requested_bits"], row["sde_step"],
                 row["latent_position"])].append(row)
    aggregate = []
    for (field, count, step, frame), cell_rows in sorted(grouped.items()):
        drops = [row["psnr_drop_db"] for row in cell_rows]
        lpips_delta = [row["delta_lpips"] for row in cell_rows]
        aggregate.append(
            {
                "field": field,
                "requested_bits": count,
                "sde_step": step,
                "latent_position": frame,
                "n_attack_seeds": len(cell_rows),
                "psnr_drop_mean_db": statistics.mean(drops),
                "psnr_drop_std_db": statistics.pstdev(drops),
                "delta_lpips_mean": statistics.mean(lpips_delta),
                "delta_lpips_std": statistics.pstdev(lpips_delta),
            }
        )
    aggregate_path = args.results_root / f"{output_stem}_aggregate.csv"
    write_csv(aggregate_path, aggregate, list(aggregate[0]))

    for (field, count, attack_seed), seed_rows in sorted(
        ((key, list(value)) for key, value in _group_rows(
            rows, lambda row: (row["field"], row["requested_bits"], row["attack_seed"])
        )), key=lambda item: item[0]
    ):
        values = {(row["sde_step"], row["latent_position"]): round(row["psnr_drop_db"], 6)
                  for row in seed_rows}
        matrix_path = args.results_root / (
            f"{output_stem}_{field}{count}_aseed{attack_seed}_psnr_drop.csv"
        )
        write_matrix(matrix_path, values)

    json_path = args.results_root / f"{output_stem}_summary.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "sequence": args.sequence,
                "clean_metrics": str(args.clean_metrics),
                "clean_psnr_db": clean_psnr,
                "clean_lpips": clean_lpips,
                "completed_runs": len(rows),
                "runs": rows,
                "aggregate": aggregate,
            },
            handle,
            indent=2,
        )

    print(f"Completed runs: {len(rows)}")
    print(f"Saved: {tidy_path}")
    print(f"Saved: {aggregate_path}")
    print(f"Saved: {json_path}")


def _group_rows(rows, key_fn):
    grouped = defaultdict(list)
    for row in rows:
        grouped[key_fn(row)].append(row)
    return grouped.items()


if __name__ == "__main__":
    main()

