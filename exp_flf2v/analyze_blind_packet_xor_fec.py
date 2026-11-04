"""Format-agnostic XOR packet-FEC baseline for blind burst erasures."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


LAYOUTS = ("contiguous-groups", "shared-shuffle")


def group_assignment(n_data, data_per_group, layout, seed=42):
    groups = math.ceil(n_data / data_per_group)
    order = np.arange(n_data, dtype=np.int64)
    if layout == "shared-shuffle":
        order = np.random.default_rng(seed).permutation(order)
    elif layout != "contiguous-groups":
        raise ValueError(f"Unknown FEC layout: {layout}")
    assignment = np.empty(n_data, dtype=np.int64)
    for position, packet in enumerate(order):
        assignment[packet] = position // data_per_group
    return assignment, groups


def exact_recovery(n_data, assignment, groups, erased_packets):
    erased = set(erased_packets)
    for group in range(groups):
        missing_data = sum(
            packet in erased for packet in range(n_data)
            if int(assignment[packet]) == group)
        parity_missing = n_data + group in erased
        if missing_data > 1 or (missing_data == 1 and parity_missing):
            return False
    return True


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Blind packet burst analysis with one XOR parity per group")
    parser.add_argument("--stream", required=True)
    parser.add_argument("--packet_bytes", type=int, default=1200)
    parser.add_argument("--data_per_group", type=int, default=8)
    parser.add_argument("--loss_lengths", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--shuffle_seed", type=int, default=42)
    parser.add_argument("--output_prefix", required=True)
    args = parser.parse_args()
    if args.packet_bytes < 1 or args.data_per_group < 1:
        raise ValueError("Packet size and group size must be positive")
    if any(value < 1 for value in args.loss_lengths):
        raise ValueError("Loss lengths must be positive")

    stream_bytes = Path(args.stream).stat().st_size
    n_data = math.ceil(stream_bytes / args.packet_bytes)
    detail = []
    aggregate = []
    for layout in LAYOUTS:
        assignment, groups = group_assignment(
            n_data, args.data_per_group, layout, args.shuffle_seed)
        total_packets = n_data + groups
        for loss in args.loss_lengths:
            if loss > total_packets:
                raise ValueError("Loss length exceeds transmitted packet count")
            rows = []
            for start in range(total_packets - loss + 1):
                recovered = exact_recovery(
                    n_data, assignment, groups, range(start, start + loss))
                row = {
                    "layout": layout,
                    "lost_packets": loss,
                    "start_packet": start,
                    "exact_recovery": recovered,
                }
                rows.append(row)
                detail.append(row)
            successes = sum(row["exact_recovery"] for row in rows)
            aggregate.append({
                "layout": layout,
                "lost_packets": loss,
                "evaluated_starts": len(rows),
                "exact_recovery_starts": successes,
                "exact_recovery_rate": successes / len(rows),
            })

    groups = math.ceil(n_data / args.data_per_group)
    overhead_bytes = groups * args.packet_bytes
    prefix = Path(args.output_prefix)
    detail_path = prefix.with_name(prefix.name + "_detail.csv")
    aggregate_path = prefix.with_name(prefix.name + "_aggregate.csv")
    summary_path = prefix.with_name(prefix.name + "_summary.json")
    write_csv(detail_path, detail)
    write_csv(aggregate_path, aggregate)
    summary = {
        "stream": args.stream,
        "stream_bytes": stream_bytes,
        "packet_bytes": args.packet_bytes,
        "data_packets": n_data,
        "data_per_group": args.data_per_group,
        "parity_packets": groups,
        "overhead_bytes": overhead_bytes,
        "overhead_vs_stream": overhead_bytes / stream_bytes,
        "aggregate": aggregate,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Blind XOR packet-FEC summary")
    print("Layout              Lost  Starts  ExactRecovery")
    for row in aggregate:
        print(f"{row['layout']:19s} {row['lost_packets']:4d}  "
              f"{row['evaluated_starts']:6d}  "
              f"{100 * row['exact_recovery_rate']:12.2f}%")
    print(f"Stream: {stream_bytes} bytes; data={n_data}, parity={groups}, "
          f"overhead={100 * overhead_bytes / stream_bytes:.2f}%")
    print(f"Saved: {detail_path}")
    print(f"Saved: {aggregate_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
