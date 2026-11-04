"""Summarize sparse repetition protection under contiguous redundancy bursts."""

import argparse
import csv
import json
import statistics
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare burst robustness with and without interleaving")
    parser.add_argument("--results_root", default="exp_flf2v/results_720p")
    parser.add_argument("--sequence", default="Jockey")
    parser.add_argument("--gop_index", type=int, default=0)
    parser.add_argument("--clean_metrics", required=True)
    parser.add_argument("--output_prefix", required=True)
    parser.add_argument("--counts", type=int, nargs="+", default=[32, 64, 96])
    parser.add_argument("--burst_starts", type=int, nargs="+",
                        default=[0, 128, 256])
    parser.add_argument("--interleave_seeds", type=int, nargs="+",
                        default=[0, 42])
    return parser.parse_args()


def sequence_tag(sequence):
    return "".join(ch.lower() if ch.isalnum() or ch in "_-" else "_"
                   for ch in sequence)


def run_name(count, interleave_seed, burst_start, sequence, gop_index):
    gop_tag = "" if gop_index == 0 else f"_gop{gop_index}"
    return (f"sign_uep_rep3_burst{count}_ileave{interleave_seed}_"
            f"start{burst_start}{gop_tag}_{sequence_tag(sequence)}")


def metric_path(root, name, sequence, gop_index):
    return root / name / sequence / f"gop{gop_index}" / "metrics.json"


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def mean_std(values):
    return (statistics.mean(values),
            statistics.stdev(values) if len(values) > 1 else 0.0)


def collect(root, sequence, gop_index, counts, starts, interleave_seeds,
            clean_psnr, clean_lpips):
    detail = []
    missing = []
    for interleave_seed in interleave_seeds:
        for count in counts:
            for start in starts:
                name = run_name(
                    count, interleave_seed, start, sequence, gop_index)
                path = metric_path(root, name, sequence, gop_index)
                if not path.exists():
                    missing.append(str(path))
                    continue
                metrics = read_json(path)
                attack = metrics.get("stream_attack", {})
                detail.append({
                    "interleave_seed": interleave_seed,
                    "interleaved": interleave_seed != 0,
                    "burst_bits": count,
                    "burst_start": start,
                    "PSNR_dB": float(metrics["PSNR_dB"]),
                    "PSNR_drop_dB": clean_psnr - float(metrics["PSNR_dB"]),
                    "LPIPS": float(metrics["LPIPS"]),
                    "delta_LPIPS": float(metrics["LPIPS"]) - clean_lpips,
                    "residual_symbols": int(attack.get("changed_symbols", 0)),
                    "corrected_symbols": int(
                        attack.get("corrected_logical_symbols", 0)),
                })
    aggregate = []
    for interleave_seed in interleave_seeds:
        for count in counts:
            items = [row for row in detail
                     if row["interleave_seed"] == interleave_seed
                     and row["burst_bits"] == count]
            if len(items) != len(starts):
                continue
            drop_mean, drop_std = mean_std(
                [row["PSNR_drop_dB"] for row in items])
            lpips_mean, lpips_std = mean_std(
                [row["LPIPS"] for row in items])
            residual_mean, residual_std = mean_std(
                [row["residual_symbols"] for row in items])
            aggregate.append({
                "interleave_seed": interleave_seed,
                "interleaved": interleave_seed != 0,
                "burst_bits": count,
                "n_burst_starts": len(items),
                "PSNR_drop_mean_dB": drop_mean,
                "PSNR_drop_std_dB": drop_std,
                "LPIPS_mean": lpips_mean,
                "LPIPS_std": lpips_std,
                "residual_symbols_mean": residual_mean,
                "residual_symbols_std": residual_std,
            })
    return detail, aggregate, missing


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(path, aggregate, interleave_seeds):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.7), dpi=180)
    colors = ["#c64b43", "#24649a", "#6f52a2", "#398f75"]
    labels = {
        0: "Repetition-3, no interleaving",
        42: "Repetition-3, interleaved",
    }
    for index, interleave_seed in enumerate(interleave_seeds):
        items = sorted(
            [row for row in aggregate
             if row["interleave_seed"] == interleave_seed],
            key=lambda row: row["burst_bits"])
        x = [row["burst_bits"] for row in items]
        label = labels.get(interleave_seed, f"Interleave seed {interleave_seed}")
        color = colors[index % len(colors)]
        axes[0].errorbar(
            x, [row["PSNR_drop_mean_dB"] for row in items],
            yerr=[row["PSNR_drop_std_dB"] for row in items],
            marker="o", linewidth=2.1, capsize=4, color=color, label=label)
        axes[1].errorbar(
            x, [row["LPIPS_mean"] for row in items],
            yerr=[row["LPIPS_std"] for row in items],
            marker="o", linewidth=2.1, capsize=4, color=color, label=label)
        axes[2].errorbar(
            x, [row["residual_symbols_mean"] for row in items],
            yerr=[row["residual_symbols_std"] for row in items],
            marker="o", linewidth=2.1, capsize=4, color=color, label=label)

    titles = ["Reconstruction degradation", "Perceptual distortion",
              "Residual logical errors"]
    ylabels = ["PSNR drop (dB)", "LPIPS", "Residual sign errors"]
    for axis, title, ylabel in zip(axes, titles, ylabels):
        axis.set_title(title, fontweight="bold")
        axis.set_xlabel("Contiguous burst length (bits)")
        axis.set_ylabel(ylabel)
        axis.grid(True, color="#d9e1e8", linewidth=0.8, alpha=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Sparse repetition protection under contiguous redundancy bursts",
        fontsize=15, fontweight="bold")
    fig.text(
        0.5, 0.015,
        "Mean over three physical burst starts; error bars show sample standard deviation",
        ha="center", color="#59636e", fontsize=9)
    fig.tight_layout(rect=(0, 0.045, 1, 0.92))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    args = parse_args()
    clean = read_json(args.clean_metrics)
    detail, aggregate, missing = collect(
        Path(args.results_root), args.sequence, args.gop_index,
        args.counts, args.burst_starts, args.interleave_seeds,
        float(clean["PSNR_dB"]), float(clean["LPIPS"]))
    if missing:
        raise FileNotFoundError(
            "Missing burst metrics:\n" +
            "\n".join(f"  {value}" for value in missing))
    if not aggregate:
        raise RuntimeError("No complete burst groups found")

    prefix = Path(args.output_prefix)
    detail_path = prefix.with_name(prefix.name + "_detail.csv")
    aggregate_path = prefix.with_name(prefix.name + "_aggregate.csv")
    png_path = prefix.with_name(prefix.name + "_comparison.png")
    summary_path = prefix.with_name(prefix.name + "_summary.json")
    write_csv(detail_path, detail)
    write_csv(aggregate_path, aggregate)
    plot(png_path, aggregate, args.interleave_seeds)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump({"detail": detail, "aggregate": aggregate}, handle, indent=2)

    print("Burst-interleaving summary")
    print("Interleave  Burst  PSNRDrop  LPIPS  Residual")
    for row in aggregate:
        print(f"{row['interleave_seed']:10d}  {row['burst_bits']:5d}  "
              f"{row['PSNR_drop_mean_dB']:8.3f}  "
              f"{row['LPIPS_mean']:5.4f}  "
              f"{row['residual_symbols_mean']:8.2f}")
    print(f"Saved: {detail_path}")
    print(f"Saved: {aggregate_path}")
    print(f"Saved: {png_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
