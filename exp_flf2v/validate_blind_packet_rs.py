"""Validate real packet-byte Cauchy RS recovery under blind burst erasures."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

try:
    from .analyze_blind_packet_xor_fec import group_assignment
    from .packet_cauchy_rs import encode_group, recover_group
except ImportError:
    from analyze_blind_packet_xor_fec import group_assignment
    from packet_cauchy_rs import encode_group, recover_group


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Real Cauchy RS packet recovery")
    parser.add_argument("--stream", required=True)
    parser.add_argument("--packet_bytes", type=int, default=256)
    parser.add_argument("--data_per_group", type=int, default=26)
    parser.add_argument("--parity_per_group", type=int, default=3)
    parser.add_argument("--loss_lengths", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--shuffle_seed", type=int, default=42)
    parser.add_argument("--output_prefix", required=True)
    args = parser.parse_args()
    raw = Path(args.stream).read_bytes()
    n_data = math.ceil(len(raw) / args.packet_bytes)
    padded = raw + bytes(n_data * args.packet_bytes - len(raw))
    packets = [np.frombuffer(
        padded[index * args.packet_bytes:(index + 1) * args.packet_bytes],
        dtype=np.uint8).copy() for index in range(n_data)]
    assignment, groups = group_assignment(
        n_data, args.data_per_group, "shared-shuffle", args.shuffle_seed)
    group_members = [np.flatnonzero(assignment == group).tolist()
                     for group in range(groups)]
    parity = [encode_group([packets[index] for index in members],
                           args.parity_per_group)
              for members in group_members]
    total_packets = n_data + groups * args.parity_per_group
    rows = []
    for loss in args.loss_lengths:
        for start in range(total_packets - loss + 1):
            erased = set(range(start, start + loss))
            recovered_packets = list(packets)
            success = True
            try:
                for group, members in enumerate(group_members):
                    group_data = [None if index in erased else packets[index]
                                  for index in members]
                    group_parity = [
                        None if (n_data + group * args.parity_per_group + copy)
                        in erased else parity[group][copy]
                        for copy in range(args.parity_per_group)]
                    recovered = recover_group(group_data, group_parity)
                    for index, packet in zip(members, recovered):
                        recovered_packets[index] = packet
                restored = b"".join(packet.tobytes()
                                    for packet in recovered_packets)[:len(raw)]
                success = restored == raw
            except ValueError:
                success = False
            rows.append({"lost_packets": loss, "start_packet": start,
                         "exact_byte_recovery": success})
    aggregate = []
    for loss in args.loss_lengths:
        items = [row for row in rows if row["lost_packets"] == loss]
        successes = sum(row["exact_byte_recovery"] for row in items)
        aggregate.append({"lost_packets": loss, "evaluated_starts": len(items),
                          "exact_recovery_starts": successes,
                          "exact_recovery_rate": successes / len(items)})
    prefix = Path(args.output_prefix)
    detail_path = prefix.with_name(prefix.name + "_detail.csv")
    aggregate_path = prefix.with_name(prefix.name + "_aggregate.csv")
    summary_path = prefix.with_name(prefix.name + "_summary.json")
    write_csv(detail_path, rows)
    write_csv(aggregate_path, aggregate)
    overhead = groups * args.parity_per_group * args.packet_bytes / len(raw)
    summary_path.write_text(json.dumps({
        "stream_bytes": len(raw), "data_packets": n_data,
        "parity_packets": groups * args.parity_per_group,
        "overhead_rate": overhead, "aggregate": aggregate,
    }, indent=2), encoding="utf-8")
    print("Real Cauchy RS byte-recovery validation")
    print("Lost  Starts  ExactRecovery")
    for row in aggregate:
        print(f"{row['lost_packets']:4d}  {row['evaluated_starts']:6d}  "
              f"{100 * row['exact_recovery_rate']:12.2f}%")
    print(f"Data={n_data}, parity={groups * args.parity_per_group}, "
          f"overhead={100 * overhead:.2f}%")
    print(f"Saved: {aggregate_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
