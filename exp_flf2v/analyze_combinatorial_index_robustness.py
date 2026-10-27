"""CPU-only robustness analysis for TDC2 combinatorial index ranks.

The TDC2 stream represents all M selected atom indices with one combinatorial
rank.  A physical bit error therefore changes the rank, not one independently
stored atom.  This tool measures how many selected atoms are replaced after a
single rank-bit flip and how often the corrupted rank is outside C(K, M).

It can analyze either a real TDC2 file or deterministic synthetic M-subsets.
No Wan, CUDA, PyTorch, or video decoding is required.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import struct
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


def combinatorial_index_bits(K: int, M: int) -> int:
    if K < 0 or M < 0 or M > K:
        raise ValueError(f"Expected 0 <= M <= K, got K={K}, M={M}")
    return (math.comb(K, M) - 1).bit_length()


def rank_combination(indices: Sequence[int], K: int) -> int:
    values = list(indices)
    if values != sorted(values) or len(set(values)) != len(values):
        raise ValueError("Combination indices must be strictly increasing")
    if values and (values[0] < 0 or values[-1] >= K):
        raise ValueError(f"Combination index outside [0, {K})")
    return sum(math.comb(value, order)
               for order, value in enumerate(values, 1))


def unrank_combination(rank: int, K: int, M: int) -> List[int]:
    total = math.comb(K, M)
    if rank < 0 or rank >= total:
        raise ValueError(f"Combination rank {rank} outside [0, {total})")
    result = [0] * M
    upper = K
    remainder = rank
    for order in range(M, 0, -1):
        lo, hi = order - 1, upper - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if math.comb(mid, order) <= remainder:
                lo = mid + 1
            else:
                hi = mid - 1
        result[order - 1] = hi
        remainder -= math.comb(hi, order)
        upper = hi
    return result


def read_tdc2_records(path: Path) -> Tuple[int, int, List[Dict[str, object]]]:
    raw = path.read_bytes()
    if raw[:4] != b"TDC2":
        raise ValueError(f"Expected a TDC2 stream, got {raw[:4]!r}")
    if len(raw) < 48:
        raise ValueError("Truncated TDC2 header")
    K, M, n_sde, _n_ddim, n_lat, _seed, ndim, *_ = \
        struct.unpack_from("<10I", raw, 4)
    off = 44
    prompt_len = struct.unpack_from("<I", raw, off)[0]
    off += 4 + 4 * ndim + prompt_len
    records: List[Dict[str, object]] = []
    for step in range(n_sde):
        for latent in range(n_lat):
            if off + 2 > len(raw):
                raise ValueError("Truncated TDC2 payload before M_actual")
            m_actual = struct.unpack_from("<H", raw, off)[0]
            off += 2
            rank_bits = combinatorial_index_bits(K, m_actual)
            rank_bytes = (rank_bits + 7) // 8
            if off + rank_bytes > len(raw):
                raise ValueError("Truncated TDC2 combination rank")
            rank = int.from_bytes(raw[off:off + rank_bytes], "little")
            off += rank_bytes
            if rank >= math.comb(K, m_actual):
                raise ValueError(f"Invalid clean rank at step={step}, latent={latent}")
            indices = unrank_combination(rank, K, m_actual)
            sign_bytes = (m_actual + 7) // 8
            if off + sign_bytes > len(raw):
                raise ValueError("Truncated TDC2 sign payload")
            off += sign_bytes
            records.append(dict(step=step, latent=latent, M=m_actual,
                                rank=rank, indices=indices))
    if off != len(raw):
        raise ValueError(f"Unexpected {len(raw) - off} trailing bytes")
    return K, M, records


def choose_bits(width: int, per_region: int, rng: random.Random) -> List[Tuple[str, int]]:
    boundaries = (0, width // 3, 2 * width // 3, width)
    selected: List[Tuple[str, int]] = []
    for name, lo, hi in zip(("low", "middle", "high"),
                            boundaries[:-1], boundaries[1:]):
        candidates = list(range(lo, hi))
        count = min(per_region, len(candidates))
        for bit in sorted(rng.sample(candidates, count)):
            selected.append((name, bit))
    return selected


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1 - fraction) + ordered[upper] * fraction)


def aggregate(rows: Sequence[Dict[str, object]], group_key: str) -> List[Dict[str, object]]:
    groups: Dict[object, List[Dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(row[group_key], []).append(row)
    output: List[Dict[str, object]] = []
    for key in sorted(groups, key=lambda value: str(value)):
        group = groups[key]
        valid = [row for row in group if row["valid_rank"]]
        replacements = [float(row["replaced_atoms"]) for row in valid]
        jaccards = [float(row["jaccard"]) for row in valid]
        output.append({
            group_key: key,
            "attempts": len(group),
            "valid_flips": len(valid),
            "invalid_rank_rate": 1.0 - len(valid) / len(group),
            "replaced_atoms_mean": statistics.fmean(replacements) if replacements else None,
            "replaced_atoms_median": statistics.median(replacements) if replacements else None,
            "replaced_atoms_p95": percentile(replacements, 0.95) if replacements else None,
            "replaced_atoms_max": max(replacements) if replacements else None,
            "single_replacement_rate": (
                sum(value == 1 for value in replacements) / len(replacements)
                if replacements else None),
            "jaccard_mean": statistics.fmean(jaccards) if jaccards else None,
        })
    return output


def analyze(records: Sequence[Dict[str, object]], K: int,
            bits_per_region: int, seed: int) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    total_by_m = {int(record["M"]): math.comb(K, int(record["M"]))
                  for record in records}
    for sample_id, record in enumerate(records):
        M = int(record["M"])
        rank = int(record["rank"])
        original = list(record["indices"])
        original_set = set(original)
        width = combinatorial_index_bits(K, M)
        # Use a sample-specific RNG so adding records does not change old rows.
        rng = random.Random(seed * 1_000_003 + sample_id)
        for region, bit in choose_bits(width, bits_per_region, rng):
            corrupted_rank = rank ^ (1 << bit)
            valid = corrupted_rank < total_by_m[M]
            row: Dict[str, object] = {
                "sample": sample_id,
                "step": record.get("step", ""),
                "latent": record.get("latent", ""),
                "M": M,
                "rank_bits": width,
                "region": region,
                "bit": bit,
                "bit_fraction": bit / max(1, width - 1),
                "original_bit": (rank >> bit) & 1,
                "valid_rank": valid,
                "replaced_atoms": "",
                "shared_atoms": "",
                "jaccard": "",
            }
            if valid:
                changed = set(unrank_combination(corrupted_rank, K, M))
                shared = len(original_set & changed)
                replaced = M - shared
                row.update(replaced_atoms=replaced, shared_atoms=shared,
                           jaccard=shared / len(original_set | changed))
            rows.append(row)
    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bitstream", type=Path,
                        help="Optional real TDC2 file; otherwise use synthetic subsets")
    parser.add_argument("--K", type=int, default=16384)
    parser.add_argument("--M", type=int, default=64)
    parser.add_argument("--samples", type=int, default=128,
                        help="Number of real records or synthetic subsets")
    parser.add_argument("--bits_per_region", type=int, default=16,
                        help="Rank bits sampled from each low/middle/high third")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_prefix", type=Path, default=Path(
        "exp_flf2v/results_720p/combinatorial_index_robustness"))
    args = parser.parse_args()
    rng = random.Random(args.seed)
    if args.bitstream:
        K, _header_m, all_records = read_tdc2_records(args.bitstream)
        if args.samples < len(all_records):
            records = rng.sample(all_records, args.samples)
        else:
            records = all_records
        source = str(args.bitstream)
    else:
        K = args.K
        records = []
        for _ in range(args.samples):
            indices = sorted(rng.sample(range(K), args.M))
            records.append(dict(M=args.M,
                                rank=rank_combination(indices, K),
                                indices=indices))
        source = "synthetic"
    rows = analyze(records, K, args.bits_per_region, args.seed)
    region_summary = aggregate(rows, "region")
    bit_summary = aggregate(rows, "bit")
    prefix = args.output_prefix
    detail_path = prefix.with_name(prefix.name + "_detail.csv")
    region_path = prefix.with_name(prefix.name + "_regions.csv")
    bit_path = prefix.with_name(prefix.name + "_bits.csv")
    json_path = prefix.with_name(prefix.name + "_summary.json")
    write_csv(detail_path, rows)
    write_csv(region_path, region_summary)
    write_csv(bit_path, bit_summary)
    summary = {
        "source": source,
        "K": K,
        "samples": len(records),
        "bits_per_region": args.bits_per_region,
        "attempts": len(rows),
        "region_summary": region_summary,
        "legacy_fixed_width_reference": {
            "index_bits_per_atom": max(1, math.ceil(math.log2(K))),
            "affected_stored_index_slots_per_bit_flip": 1,
            "cross_atom_error_propagation": False,
            "note": "A legacy fixed-width index bit changes one stored atom slot; "
                    "it may still create an invalid index or duplicate.",
        },
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Combinatorial rank single-bit error propagation")
    print(f"Source: {source}; K={K}; samples={len(records)}; attempts={len(rows)}")
    print("Region  InvalidRank  MeanReplaced  P95Replaced  MaxReplaced  MeanJaccard")
    for item in region_summary:
        mean = item["replaced_atoms_mean"]
        p95 = item["replaced_atoms_p95"]
        maximum = item["replaced_atoms_max"]
        jaccard = item["jaccard_mean"]
        print(f"{item['region']:>6}  {item['invalid_rank_rate']:>11.3%}  "
              f"{mean if mean is not None else float('nan'):>12.2f}  "
              f"{p95 if p95 is not None else float('nan'):>11.2f}  "
              f"{maximum if maximum is not None else float('nan'):>11.0f}  "
              f"{jaccard if jaccard is not None else float('nan'):>11.4f}")
    print(f"Saved: {detail_path}")
    print(f"Saved: {region_path}")
    print(f"Saved: {bit_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
