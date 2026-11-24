"""Summarize completed FLF2V independent-vs-combinatorial GOP records."""

import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    "sequence", "gop", "PSNR_dB", "LPIPS", "MS_SSIM",
    "independent_codebook_bits", "combinatorial_codebook_bits",
    "codebook_saving_percent", "independent_total_BPP",
    "combinatorial_total_BPP", "total_bitrate_saving_percent",
    "independent_container_bytes", "combinatorial_container_bytes",
    "container_saving_percent", "boundary_bytes", "encode_seconds",
    "decode_seconds", "rate_roundtrip_verified",
]


def mean(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--expected_gops", type=int, default=21)
    args = parser.parse_args()

    rows = []
    for path in sorted(args.results.glob("*/gop*/metrics.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("rate_comparison_version") != 1:
            continue
        rows.append({key: record.get(key) for key in FIELDS})

    args.results.mkdir(parents=True, exist_ok=True)
    csv_path = args.results / "rate_comparison_gops.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    sequences = {}
    for row in rows:
        sequences.setdefault(row["sequence"], []).append(row)
    per_sequence = {
        name: {
            "completed_gops": len(items),
            "PSNR_dB": mean(items, "PSNR_dB"),
            "LPIPS": mean(items, "LPIPS"),
            "independent_total_BPP": mean(items, "independent_total_BPP"),
            "combinatorial_total_BPP": mean(items, "combinatorial_total_BPP"),
            "total_bitrate_saving_percent": mean(
                items, "total_bitrate_saving_percent"),
        }
        for name, items in sorted(sequences.items())
    }
    summary = {
        "status": "complete" if len(rows) == args.expected_gops else "partial",
        "expected_gops": args.expected_gops,
        "completed_gops": len(rows),
        "all_roundtrips_verified": bool(rows) and all(
            row["rate_roundtrip_verified"] is True for row in rows),
        "averages": {
            "PSNR_dB": mean(rows, "PSNR_dB"),
            "LPIPS": mean(rows, "LPIPS"),
            "independent_total_BPP": mean(rows, "independent_total_BPP"),
            "combinatorial_total_BPP": mean(rows, "combinatorial_total_BPP"),
            "codebook_saving_percent": mean(rows, "codebook_saving_percent"),
            "total_bitrate_saving_percent": mean(
                rows, "total_bitrate_saving_percent"),
        },
        "per_sequence": per_sequence,
    }
    json_path = args.results / "rate_comparison_summary.json"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
