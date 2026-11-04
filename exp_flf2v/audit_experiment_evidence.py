"""Build a paper-facing evidence ledger from GVCC ``metrics.json`` files.

The ledger does not upgrade exploratory results into formal results.  It makes
missing provenance and metric fields explicit so that only reproducible runs
are copied into paper tables.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def _json_cell(value):
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def audit_metrics(path: Path, results_root: Path) -> dict:
    try:
        metrics = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError) as error:
        return {
            "metrics_path": str(path), "run": "", "sequence": "",
            "gop": "", "formal_ready": False,
            "issues": f"unreadable_metrics:{type(error).__name__}",
        }

    relative = path.relative_to(results_root)
    run = relative.parts[0] if len(relative.parts) >= 4 else ""
    attack = metrics.get("stream_attack") or {}
    provenance = metrics.get("provenance") or {}
    model = provenance.get("model") or {}
    source = provenance.get("source") or {}
    issues = []
    if not provenance.get("configuration_id"):
        issues.append("missing_configuration_id")
    if model.get("model_seed") is None:
        issues.append("missing_model_seed")
    if not source.get("gop_segment_sha256"):
        issues.append("missing_input_gop_hash")
    per_frame = metrics.get("per_frame_PSNR_dB")
    if not isinstance(per_frame, list) or not per_frame:
        issues.append("missing_per_frame_psnr")
    if attack and attack.get("seed") is None:
        issues.append("missing_attack_seed")
    if metrics.get("codebook_bits") is None:
        issues.append("missing_codebook_bits")
    if metrics.get("boundary_bytes") is None:
        issues.append("missing_boundary_bytes")

    return {
        "metrics_path": str(relative),
        "run": run,
        "sequence": metrics.get("sequence", ""),
        "gop": metrics.get("gop", ""),
        "batch_role": metrics.get("batch_role", ""),
        "PSNR_dB": metrics.get("PSNR_dB", ""),
        "LPIPS": metrics.get("LPIPS", ""),
        "BPP": metrics.get("BPP", ""),
        "gop_total_bytes": metrics.get("gop_total_bytes", ""),
        "codebook_bytes": metrics.get("codebook_bytes", ""),
        "boundary_bytes": metrics.get("boundary_bytes", ""),
        "protection_overhead_bits": metrics.get(
            "protection_overhead_bits", ""),
        "attack": attack.get("attack", ""),
        "transport": attack.get("transport", ""),
        "attack_seed": attack.get("seed", ""),
        "channel_flipped_bits": attack.get("channel_flipped_bits", ""),
        "residual_symbols": attack.get("changed_symbols", ""),
        "protected_cells": _json_cell(attack.get("protected_cells")),
        "model_seed": model.get("model_seed", ""),
        "configuration_id": provenance.get("configuration_id", ""),
        "input_gop_sha256": source.get("gop_segment_sha256", ""),
        "per_frame_psnr_count": len(per_frame) if isinstance(per_frame, list) else 0,
        "formal_ready": not issues,
        "issues": ";".join(issues),
    }


def build_ledger(results_root: Path):
    return [audit_metrics(path, results_root) for path in sorted(
        results_root.rglob("metrics.json"))]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit GVCC metrics provenance before paper reporting")
    parser.add_argument("--results_root", required=True)
    parser.add_argument("--output_prefix", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.results_root)
    rows = build_ledger(root)
    if not rows:
        raise ValueError(f"No metrics.json files found below {root}")
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    detail_path = prefix.with_name(prefix.name + "_detail.csv")
    summary_path = prefix.with_name(prefix.name + "_summary.json")
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with detail_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    issue_counts = Counter()
    for row in rows:
        for issue in str(row.get("issues", "")).split(";"):
            if issue:
                issue_counts[issue] += 1
    summary = {
        "results_root": str(root),
        "metrics_files": len(rows),
        "formal_ready": sum(bool(row.get("formal_ready")) for row in rows),
        "exploratory_or_incomplete": sum(
            not bool(row.get("formal_ready")) for row in rows),
        "issue_counts": dict(sorted(issue_counts.items())),
        "note": (
            "BPP/gop_total_bytes in legacy runs may exclude container headers, "
            "packet headers, padding, FEC parity, and separately transmitted "
            "reference data. Verify the accounting boundary before publication."),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Experiment evidence audit")
    print(f"Metrics files: {summary['metrics_files']}")
    print(f"Formal-ready: {summary['formal_ready']}")
    print(f"Exploratory/incomplete: {summary['exploratory_or_incomplete']}")
    for issue, count in summary["issue_counts"].items():
        print(f"  {issue}: {count}")
    print(f"Saved: {detail_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
