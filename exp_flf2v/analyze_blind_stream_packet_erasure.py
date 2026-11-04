"""Blind contiguous packet erasures over a complete protected TDS2 stream.

Unlike the protected-block study, the attacker selects only physical packet
numbers.  It has no access to SDE steps, latent positions, or sign fields.
The analysis reports whether the current sparse-repetition container can
reconstruct a syntactically valid base TDCM after each erasure.
"""

import argparse
import csv
import json
import math
import struct
from pathlib import Path

import numpy as np

try:
    from .tdcm_layout import parse_tdcm_payload_layout
except ImportError:
    from tdcm_layout import parse_tdcm_payload_layout


HEADER_FORMAT = "<4sBBBBIII"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


def parse_container(raw):
    if len(raw) < HEADER_SIZE:
        raise ValueError("Truncated protected container header")
    magic, version, repetitions, cell_count, _, payload_bytes, seed, red_bits = \
        struct.unpack_from(HEADER_FORMAT, raw, 0)
    if magic != b"TDS2" or version != 1:
        raise ValueError("Expected a version-1 sparse TDS2 container")
    cell_end = HEADER_SIZE + 2 * cell_count
    payload_end = cell_end + payload_bytes
    expected = payload_end + (red_bits + 7) // 8
    if expected != len(raw):
        raise ValueError(
            f"Protected container size mismatch: expected {expected}, got {len(raw)}")
    cells = [struct.unpack_from("<BB", raw, HEADER_SIZE + 2 * i)
             for i in range(cell_count)]
    payload = raw[cell_end:payload_end]
    layout = parse_tdcm_payload_layout(payload)
    cell_set = set(cells)
    slots = [slot for slot in layout["slots"]
             if (slot["step"], slot["frame"]) in cell_set]
    if len(slots) * (repetitions - 1) != red_bits:
        raise ValueError("TDS2 redundancy length does not match protected cells")
    return {
        "repetitions": repetitions,
        "interleave_seed": seed,
        "redundancy_bits": red_bits,
        "prefix_bytes": cell_end,
        "payload_bytes": payload_bytes,
        "payload_end": payload_end,
        "layout": layout,
        "slots": slots,
        "cells": cells,
    }


def physical_copy_bytes(info):
    repetitions = info["repetitions"]
    red_bits = info["redundancy_bits"]
    if info["interleave_seed"]:
        interleave = np.random.default_rng(
            info["interleave_seed"]).permutation(red_bits)
    else:
        interleave = np.arange(red_bits, dtype=np.int64)
    result = []
    for ordinal, slot in enumerate(info["slots"]):
        copies = [info["prefix_bytes"] + slot["sign_offset"]]
        for copy in range(repetitions - 1):
            raw_bit = ordinal * (repetitions - 1) + copy
            physical_bit = int(interleave[raw_bit])
            copies.append(info["payload_end"] + physical_bit // 8)
        result.append(copies)
    return result


def reconstructable_payload_bytes(info):
    """Return payload bytes whose meaningful bits are all protected signs."""
    all_by_byte = {}
    protected_by_byte = {}
    protected_ids = {(slot["step"], slot["frame"], slot["atom"])
                     for slot in info["slots"]}
    for slot in info["layout"]["slots"]:
        all_by_byte.setdefault(slot["sign_offset"], set()).add(slot["sign_mask"])
        key = (slot["step"], slot["frame"], slot["atom"])
        if key in protected_ids:
            protected_by_byte.setdefault(slot["sign_offset"], set()).add(
                slot["sign_mask"])
    return {offset for offset, masks in all_by_byte.items()
            if protected_by_byte.get(offset, set()) == masks}


def analyze_loss(info, copy_bytes, recoverable_offsets, packet_bytes,
                 start_packet, lost_packets, stream_bytes):
    first = start_packet * packet_bytes
    last = min(stream_bytes, (start_packet + lost_packets) * packet_bytes)
    erased = set(range(first, last))
    prefix = info["prefix_bytes"]
    payload_end = info["payload_end"]
    structural_bytes = 0
    for offset in erased:
        if offset < prefix:
            structural_bytes += 1
        elif offset < payload_end:
            payload_offset = offset - prefix
            if payload_offset not in recoverable_offsets:
                structural_bytes += 1
        # Redundancy bytes may be erased without invalidating the base syntax.
    unresolved = sum(all(offset in erased for offset in copies)
                     for copies in copy_bytes)
    return {
        "start_packet": start_packet,
        "lost_packets": lost_packets,
        "erased_bytes": last - first,
        "structural_bytes_lost": structural_bytes,
        "syntax_recoverable": structural_bytes == 0,
        "unresolved_signs": int(unresolved),
        "exact_recovery": structural_bytes == 0 and unresolved == 0,
    }


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Blind full-stream packet-erasure analysis")
    parser.add_argument("--container", required=True)
    parser.add_argument("--packet_bytes", type=int, default=64)
    parser.add_argument("--loss_lengths", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--output_prefix", required=True)
    args = parser.parse_args()
    if args.packet_bytes < 1 or any(value < 1 for value in args.loss_lengths):
        raise ValueError("Packet size and loss lengths must be positive")

    raw = Path(args.container).read_bytes()
    info = parse_container(raw)
    copy_bytes = physical_copy_bytes(info)
    recoverable = reconstructable_payload_bytes(info)
    packet_count = math.ceil(len(raw) / args.packet_bytes)
    detail = []
    for loss in args.loss_lengths:
        if loss > packet_count:
            raise ValueError("Loss length exceeds the full-stream packet count")
        for start in range(packet_count - loss + 1):
            detail.append(analyze_loss(
                info, copy_bytes, recoverable, args.packet_bytes,
                start, loss, len(raw)))

    aggregate = []
    for loss in args.loss_lengths:
        rows = [row for row in detail if row["lost_packets"] == loss]
        syntax = [row for row in rows if row["syntax_recoverable"]]
        exact = [row for row in rows if row["exact_recovery"]]
        aggregate.append({
            "lost_packets": loss,
            "evaluated_starts": len(rows),
            "syntax_recoverable_starts": len(syntax),
            "syntax_recoverable_rate": len(syntax) / len(rows),
            "exact_recovery_starts": len(exact),
            "exact_recovery_rate": len(exact) / len(rows),
            "max_unresolved_signs_when_syntax_recoverable": (
                max((row["unresolved_signs"] for row in syntax), default=0)),
        })

    prefix = Path(args.output_prefix)
    detail_path = prefix.with_name(prefix.name + "_detail.csv")
    aggregate_path = prefix.with_name(prefix.name + "_aggregate.csv")
    summary_path = prefix.with_name(prefix.name + "_summary.json")
    write_csv(detail_path, detail)
    write_csv(aggregate_path, aggregate)
    summary = {
        "container": args.container,
        "stream_bytes": len(raw),
        "packet_bytes": args.packet_bytes,
        "packet_count": packet_count,
        "protected_cells_zero_based": [list(cell) for cell in info["cells"]],
        "protected_signs": len(info["slots"]),
        "aggregate": aggregate,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Blind full-stream packet-erasure summary")
    print("Lost  Starts  SyntaxRecoverable  ExactRecovery  MaxUnresolved")
    for row in aggregate:
        print(f"{row['lost_packets']:4d}  {row['evaluated_starts']:6d}  "
              f"{100 * row['syntax_recoverable_rate']:16.2f}%  "
              f"{100 * row['exact_recovery_rate']:12.2f}%  "
              f"{row['max_unresolved_signs_when_syntax_recoverable']:13d}")
    print(f"Full stream: {len(raw)} bytes, {packet_count} packets x "
          f"{args.packet_bytes} bytes")
    print(f"Saved: {detail_path}")
    print(f"Saved: {aggregate_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
