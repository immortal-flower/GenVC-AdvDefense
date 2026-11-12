"""Create decode-only early/late codebook swaps without re-encoding a GOP.

The output is a diagnostic hybrid, not a bitstream produced by normal encoding.
It preserves serialized records byte-for-byte and requires matching headers.
"""

import argparse
import json
import math
import struct
from pathlib import Path


def split_stream(raw: bytes):
    if len(raw) < 48 or raw[:4] not in (b"TDCM", b"TDC2"):
        raise ValueError("Expected a complete TDCM or TDC2 stream")
    magic = raw[:4]
    k, m, steps, _, latent, _, ndim, _, _, _ = struct.unpack_from("<10I", raw, 4)
    prompt_bytes = struct.unpack_from("<I", raw, 44)[0]
    if not (1 <= k <= 2**24 and 1 <= m <= k and 1 <= steps <= 1000
            and 1 <= latent <= 1000 and ndim <= 16):
        raise ValueError("Implausible codebook header")
    offset = 48 + 4 * ndim + prompt_bytes
    if offset > len(raw):
        raise ValueError("Truncated header")
    header = raw[:offset]
    records = []
    index_bytes = 2 if k <= 65536 else 4
    for _ in range(steps):
        step_start = offset
        for _ in range(latent):
            if offset + 2 > len(raw):
                raise ValueError("Truncated record length")
            count = struct.unpack_from("<H", raw, offset)[0]
            if count > m:
                raise ValueError("Record exceeds declared M")
            offset += 2
            if magic == b"TDC2":
                rank_bytes = (math.comb(k, count) - 1).bit_length()
                rank_bytes = (rank_bytes + 7) // 8
                if offset + rank_bytes > len(raw):
                    raise ValueError("Truncated rank")
                rank = int.from_bytes(raw[offset:offset + rank_bytes], "little")
                if rank >= math.comb(k, count):
                    raise ValueError("Invalid combination rank")
                offset += rank_bytes
            else:
                offset += count * index_bytes
            offset += (count + 7) // 8
            if offset > len(raw):
                raise ValueError("Truncated record")
        records.append(raw[step_start:offset])
    if offset != len(raw):
        raise ValueError("Trailing data after final codebook record")
    return header, records


def make_hybrids(clean: bytes, attacked: bytes, prefix_steps: int):
    clean_header, clean_steps = split_stream(clean)
    attack_header, attack_steps = split_stream(attacked)
    if clean_header != attack_header:
        raise ValueError("Clean and attacked streams have different headers")
    if not 1 <= prefix_steps < len(clean_steps):
        raise ValueError("prefix_steps must split the SDE step range")
    early = clean_header + b"".join(
        attack_steps[:prefix_steps] + clean_steps[prefix_steps:])
    late = clean_header + b"".join(
        clean_steps[:prefix_steps] + attack_steps[prefix_steps:])
    split_stream(early)
    split_stream(late)
    return early, late


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--attacked", type=Path, required=True)
    parser.add_argument("--prefix_steps", type=int, default=2)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    clean = args.clean.read_bytes()
    attacked = args.attacked.read_bytes()
    early, late = make_hybrids(clean, attacked, args.prefix_steps)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "clean": args.output_dir / "clean.tdcm",
        "full": args.output_dir / "full.tdcm",
        "early_only": args.output_dir / "early_only.tdcm",
        "late_only": args.output_dir / "late_only.tdcm",
    }
    for label, payload in zip(outputs, (clean, attacked, early, late)):
        outputs[label].write_bytes(payload)
    manifest = {
        "kind": "decode_only_step_swap",
        "prefix_steps": args.prefix_steps,
        "warning": "Hybrids are not normal encoder outputs; do not use as evidence of attack feasibility.",
        "sources": {"clean": str(args.clean), "attacked": str(args.attacked)},
        "outputs": {label: str(path) for label, path in outputs.items()},
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
