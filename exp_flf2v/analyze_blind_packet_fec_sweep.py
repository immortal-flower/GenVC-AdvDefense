"""Multi-seed overhead/recovery sweep for blind packet-erasure FEC."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

try:
    from .analyze_blind_packet_xor_fec import exact_recovery, group_assignment
except ImportError:
    from analyze_blind_packet_xor_fec import exact_recovery, group_assignment


def parse_config(token):
    try:
        data, parity = (int(value) for value in token.split(":", 1))
    except (ValueError, AttributeError):
        raise ValueError(f"Invalid configuration {token!r}; use DATA:PARITY")
    if data < 1 or parity < 1:
        raise ValueError("FEC configuration values must be positive")
    return data, parity


def recovery_rate(n_data, data_per_group, parity_per_group, loss, seed):
    assignment, groups = group_assignment(
        n_data, data_per_group, "shared-shuffle", seed)
    total = n_data + groups * parity_per_group
    outcomes = [
        exact_recovery(
            n_data, assignment, groups, range(start, start + loss),
            parity_per_group)
        for start in range(total - loss + 1)
    ]
    return sum(outcomes) / len(outcomes)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(path, aggregate):
    import matplotlib.pyplot as plt
    fig, axis = plt.subplots(figsize=(8.8, 5.1), dpi=180)
    for label in dict.fromkeys(row["configuration"] for row in aggregate):
        rows = [row for row in aggregate if row["configuration"] == label]
        axis.errorbar(
            [row["lost_packets"] for row in rows],
            [100 * row["recovery_mean"] for row in rows],
            yerr=[100 * row["recovery_std"] for row in rows],
            marker="o", linewidth=2, capsize=4,
            label=f"{label} ({100 * rows[0]['overhead_rate']:.1f}% overhead)")
    axis.set_xlabel("Consecutively lost physical packets")
    axis.set_ylabel("Exact stream recovery (%)")
    axis.set_ylim(0, 104)
    axis.grid(True, alpha=0.3)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    axis.set_title("Blind burst recovery over shared interleaving seeds")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Packet-FEC multi-seed sweep")
    parser.add_argument("--stream", required=True)
    parser.add_argument("--packet_bytes", type=int, default=256)
    parser.add_argument("--configurations", nargs="+", default=["8:1", "16:2", "26:3"])
    parser.add_argument("--loss_lengths", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--seed_trials", type=int, default=100)
    parser.add_argument("--output_prefix", required=True)
    args = parser.parse_args()
    if args.packet_bytes < 1 or args.seed_trials < 2:
        raise ValueError("Positive packet size and at least two seeds are required")
    configs = [parse_config(token) for token in args.configurations]
    stream_bytes = Path(args.stream).stat().st_size
    n_data = math.ceil(stream_bytes / args.packet_bytes)
    detail = []
    aggregate = []
    for data_per_group, parity_per_group in configs:
        groups = math.ceil(n_data / data_per_group)
        overhead = groups * parity_per_group * args.packet_bytes / stream_bytes
        label = f"({data_per_group}+{parity_per_group})"
        for loss in args.loss_lengths:
            values = []
            for seed in range(args.seed_trials):
                value = recovery_rate(
                    n_data, data_per_group, parity_per_group, loss, seed)
                values.append(value)
                detail.append({
                    "configuration": label, "data_per_group": data_per_group,
                    "parity_per_group": parity_per_group,
                    "overhead_rate": overhead, "seed": seed,
                    "lost_packets": loss, "recovery_rate": value,
                })
            aggregate.append({
                "configuration": label, "data_per_group": data_per_group,
                "parity_per_group": parity_per_group,
                "overhead_rate": overhead, "lost_packets": loss,
                "seed_trials": args.seed_trials,
                "recovery_mean": statistics.mean(values),
                "recovery_std": statistics.stdev(values),
                "recovery_min": min(values), "recovery_max": max(values),
            })
    prefix = Path(args.output_prefix)
    detail_path = prefix.with_name(prefix.name + "_detail.csv")
    aggregate_path = prefix.with_name(prefix.name + "_aggregate.csv")
    png_path = prefix.with_name(prefix.name + "_comparison.png")
    summary_path = prefix.with_name(prefix.name + "_summary.json")
    write_csv(detail_path, detail)
    write_csv(aggregate_path, aggregate)
    plot(png_path, aggregate)
    summary_path.write_text(json.dumps({
        "stream_bytes": stream_bytes, "packet_bytes": args.packet_bytes,
        "data_packets": n_data, "aggregate": aggregate,
    }, indent=2), encoding="utf-8")
    print("Multi-seed packet-FEC sweep")
    print("Config  Overhead  Lost  RecoveryMean  Std  Min")
    for row in aggregate:
        print(f"{row['configuration']:6s}  {100*row['overhead_rate']:7.2f}%  "
              f"{row['lost_packets']:4d}  {100*row['recovery_mean']:11.2f}%  "
              f"{100*row['recovery_std']:4.2f}  {100*row['recovery_min']:5.2f}%")
    print(f"Saved: {aggregate_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
