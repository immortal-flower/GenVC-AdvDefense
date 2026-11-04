"""Offline packet-erasure study for frozen sparse sign-field protection.

This next-stage transport model extracts the protected logical sign fields,
forms an explicit R-copy physical block, packetizes it, and exhaustively drops
contiguous packets.  Erasures are known to the receiver: a logical sign is
recovered from any surviving replica.  If every replica is erased, the pilot
uses deterministic zero concealment and can emit a valid recovered TDCM for a
later GPU decode.
"""

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

LAYOUTS = ("sequential", "replica-stripe", "shared-shuffle")


def parse_tdcm_layout(payload):
    # Delay the project codec import so pure packet-layout tests do not require
    # PyTorch or the 14B model environment.
    from bitstream_attacks import _parse_tdcm_payload_layout
    return _parse_tdcm_payload_layout(payload)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Exhaustive packet-erasure analysis for sparse sign UEP")
    parser.add_argument("--bitstream", required=True)
    parser.add_argument("--protected_cells", nargs="+", required=True,
                        help="1-based-step:0-based-frame cells")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--packet_bits", type=int, default=64)
    parser.add_argument("--loss_lengths", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--shuffle_seed", type=int, default=42)
    parser.add_argument("--output_prefix", required=True)
    parser.add_argument("--emit_recovered", action="store_true")
    return parser.parse_args()


def parse_cells(tokens, n_sde, n_lat):
    cells = []
    for token in tokens:
        try:
            step_text, frame_text = token.split(":", 1)
            step, frame = int(step_text) - 1, int(frame_text)
        except (ValueError, AttributeError):
            raise ValueError(f"Invalid protected cell {token!r}; use STEP:FRAME")
        if not 0 <= step < n_sde or not 0 <= frame < n_lat:
            raise ValueError(f"Protected cell is outside the codebook: {token}")
        cells.append((step, frame))
    if len(set(cells)) != len(cells):
        raise ValueError("Protected cells contain duplicates")
    return tuple(cells)


def physical_positions(n_symbols, repetitions, mode, seed=42):
    if repetitions < 2:
        raise ValueError("Packet-erasure analysis requires at least two replicas")
    total = n_symbols * repetitions
    if mode == "sequential":
        base = np.arange(total, dtype=np.int64).reshape(n_symbols, repetitions)
    elif mode == "replica-stripe":
        base = np.empty((n_symbols, repetitions), dtype=np.int64)
        for symbol in range(n_symbols):
            for copy in range(repetitions):
                base[symbol, copy] = copy * n_symbols + symbol
    elif mode == "shared-shuffle":
        sequential = np.arange(total, dtype=np.int64)
        permutation = np.random.default_rng(seed).permutation(total)
        base = permutation[sequential].reshape(n_symbols, repetitions)
    else:
        raise ValueError(f"Unknown packet layout: {mode}")
    return base


def simulate_erasure(positions, original_values, packet_bits,
                     loss_packets, start_packet):
    total_bits = positions.size
    packet_count = math.ceil(total_bits / packet_bits)
    if loss_packets < 1 or start_packet < 0:
        raise ValueError("Packet loss length and start must be non-negative")
    if start_packet + loss_packets > packet_count:
        raise ValueError("Packet erasure crosses the protected block boundary")
    erased_start = start_packet * packet_bits
    erased_end = min(total_bits, (start_packet + loss_packets) * packet_bits)
    erased = (positions >= erased_start) & (positions < erased_end)
    unresolved_mask = erased.all(axis=1)
    unresolved = int(unresolved_mask.sum())
    # Deterministic zero concealment changes only unresolved signs that were 1.
    changed_mask = unresolved_mask & (original_values.astype(bool))
    changed = int(changed_mask.sum())
    return {
        "packet_count": packet_count,
        "erased_start_bit": erased_start,
        "erased_end_bit": erased_end,
        "unresolved_symbols": unresolved,
        "changed_symbols_zero_fill": changed,
        "unresolved_mask": unresolved_mask,
    }


def mean_std(values):
    return (statistics.mean(values),
            statistics.stdev(values) if len(values) > 1 else 0.0)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def conceal_payload(raw, slots, unresolved_mask):
    recovered = bytearray(raw)
    for slot, unresolved in zip(slots, unresolved_mask):
        if unresolved:
            recovered[slot["sign_offset"]] &= ~slot["sign_mask"] & 0xff
    parse_tdcm_layout(recovered)
    return recovered


def plot_summary(path, aggregate):
    import matplotlib.pyplot as plt

    labels = {
        "sequential": "Sequential replicas",
        "replica-stripe": "Replica-striped",
        "shared-shuffle": "Shared-seed shuffle",
    }
    colors = {
        "sequential": "#c64b43",
        "replica-stripe": "#398f75",
        "shared-shuffle": "#24649a",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.5), dpi=180)
    for mode in LAYOUTS:
        items = sorted(
            [row for row in aggregate if row["layout"] == mode],
            key=lambda row: row["lost_packets"])
        x = [row["lost_packets"] for row in items]
        axes[0].errorbar(
            x, [row["unresolved_mean"] for row in items],
            yerr=[row["unresolved_std"] for row in items], marker="o",
            linewidth=2.1, capsize=4, color=colors[mode], label=labels[mode])
        axes[1].plot(
            x, [row["unresolved_max"] for row in items], marker="o",
            linewidth=2.1, color=colors[mode], label=labels[mode])
    axes[0].set_title("Average over all contiguous loss starts")
    axes[0].set_ylabel("Unresolved logical signs")
    axes[1].set_title("Worst contiguous loss start")
    axes[1].set_ylabel("Maximum unresolved logical signs")
    for axis in axes:
        axis.set_xlabel("Consecutively lost packets")
        axis.set_xticks(sorted({row["lost_packets"] for row in aggregate}))
        axis.grid(True, color="#d9e1e8", linewidth=0.8, alpha=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Packet-erasure robustness of frozen sparse sign protection",
        fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    args = parse_args()
    if args.repetitions < 2:
        raise ValueError("--repetitions must be at least 2")
    if args.packet_bits < 1:
        raise ValueError("--packet_bits must be positive")
    if any(value < 1 for value in args.loss_lengths):
        raise ValueError("--loss_lengths must be positive")

    bitstream = Path(args.bitstream)
    raw = bytearray(bitstream.read_bytes())
    layout = parse_tdcm_layout(raw)
    cells = parse_cells(args.protected_cells, layout["n_sde"], layout["n_lat"])
    cell_set = set(cells)
    slots = [slot for slot in layout["slots"]
             if (slot["step"], slot["frame"]) in cell_set]
    if not slots:
        raise ValueError("No sign slots matched the protected cells")
    original_values = np.asarray([
        1 if raw[slot["sign_offset"]] & slot["sign_mask"] else 0
        for slot in slots
    ], dtype=np.uint8)
    total_physical_bits = len(slots) * args.repetitions
    packet_count = math.ceil(total_physical_bits / args.packet_bits)
    if any(value > packet_count for value in args.loss_lengths):
        raise ValueError("A requested loss length exceeds the packet count")

    detail = []
    worst = {}
    for mode in LAYOUTS:
        positions = physical_positions(
            len(slots), args.repetitions, mode, args.shuffle_seed)
        for loss_packets in args.loss_lengths:
            for start in range(packet_count - loss_packets + 1):
                result = simulate_erasure(
                    positions, original_values, args.packet_bits,
                    loss_packets, start)
                row = {
                    "layout": mode,
                    "lost_packets": loss_packets,
                    "start_packet": start,
                    "packet_bits": args.packet_bits,
                    "packet_count": packet_count,
                    "unresolved_symbols": result["unresolved_symbols"],
                    "changed_symbols_zero_fill": result[
                        "changed_symbols_zero_fill"],
                }
                detail.append(row)
                key = (mode, loss_packets)
                score = (result["changed_symbols_zero_fill"],
                         result["unresolved_symbols"])
                if key not in worst or score > worst[key][0]:
                    worst[key] = (score, start, result["unresolved_mask"])

    aggregate = []
    for mode in LAYOUTS:
        for loss_packets in args.loss_lengths:
            items = [row for row in detail
                     if row["layout"] == mode
                     and row["lost_packets"] == loss_packets]
            unresolved_mean, unresolved_std = mean_std(
                [row["unresolved_symbols"] for row in items])
            changed_mean, changed_std = mean_std(
                [row["changed_symbols_zero_fill"] for row in items])
            aggregate.append({
                "layout": mode,
                "lost_packets": loss_packets,
                "evaluated_starts": len(items),
                "unresolved_mean": unresolved_mean,
                "unresolved_std": unresolved_std,
                "unresolved_max": max(row["unresolved_symbols"] for row in items),
                "changed_zero_fill_mean": changed_mean,
                "changed_zero_fill_std": changed_std,
                "changed_zero_fill_max": max(
                    row["changed_symbols_zero_fill"] for row in items),
                "worst_start_packet": worst[(mode, loss_packets)][1],
            })

    prefix = Path(args.output_prefix)
    detail_path = prefix.with_name(prefix.name + "_detail.csv")
    aggregate_path = prefix.with_name(prefix.name + "_aggregate.csv")
    png_path = prefix.with_name(prefix.name + "_comparison.png")
    summary_path = prefix.with_name(prefix.name + "_summary.json")
    write_csv(detail_path, detail)
    write_csv(aggregate_path, aggregate)
    plot_summary(png_path, aggregate)

    recovered_paths = []
    if args.emit_recovered:
        output_dir = prefix.with_name(prefix.name + "_recovered")
        output_dir.mkdir(parents=True, exist_ok=True)
        for (mode, loss_packets), (_, start, unresolved_mask) in worst.items():
            recovered = conceal_payload(raw, slots, unresolved_mask)
            output_path = output_dir / (
                f"{mode}_loss{loss_packets}_start{start}.tdcm")
            output_path.write_bytes(recovered)
            recovered_paths.append(str(output_path))

    summary = {
        "source": str(bitstream),
        "protected_cells_zero_based": [list(cell) for cell in cells],
        "logical_symbols": len(slots),
        "repetitions": args.repetitions,
        "physical_bits": total_physical_bits,
        "packet_bits": args.packet_bits,
        "packet_count": packet_count,
        "zero_concealment": True,
        "aggregate": aggregate,
        "recovered_files": recovered_paths,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Packet-erasure summary")
    print("Layout          Lost  MeanUnresolved  MaxUnresolved  WorstStart")
    for row in aggregate:
        print(f"{row['layout']:15s} {row['lost_packets']:4d}  "
              f"{row['unresolved_mean']:14.2f}  "
              f"{row['unresolved_max']:13d}  "
              f"{row['worst_start_packet']:10d}")
    print(f"Protected block: {len(slots)} logical signs x "
          f"{args.repetitions} = {total_physical_bits} bits; "
          f"{packet_count} packets")
    print(f"Saved: {detail_path}")
    print(f"Saved: {aggregate_path}")
    print(f"Saved: {png_path}")
    print(f"Saved: {summary_path}")
    if recovered_paths:
        print(f"Saved {len(recovered_paths)} worst-case recovered TDCM files")


if __name__ == "__main__":
    main()
