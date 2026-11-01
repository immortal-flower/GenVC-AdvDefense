"""Plot the closed-loop sign-fault and sparse-UEP experiment.

The figure intentionally separates blind random corruption from the
profile-guided hotspot stress test.  It therefore reports a robustness
diagnostic rather than implying that an ordinary channel knows semantic
bitstream coordinates.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EXPERIMENTS = (
    ("random", "Global random\n64-bit fault", "sign64_global_random_fault"),
    ("hotspot", "Hotspot-focused\n64-bit fault", "sign64_hotspot_fault"),
    ("protected", "Hotspot fault +\nsparse repetition", "sign64_hotspot_sparse_rep3"),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot sign-field fault sensitivity and sparse UEP results")
    parser.add_argument("--results_root", default="exp_flf2v/results_720p")
    parser.add_argument("--sequence", default="Jockey")
    parser.add_argument("--gop_index", type=int, default=0)
    parser.add_argument(
        "--clean_metrics",
        default=("exp_flf2v/results_720p/sensitivity_clean_redecode/"
                 "Jockey/gop0/metrics.json"),
    )
    parser.add_argument(
        "--output_prefix",
        default="exp_flf2v/results_720p/sign_fault_uep_jockey",
    )
    return parser.parse_args()


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def main():
    args = parse_args()
    root = Path(args.results_root)
    clean = read_json(args.clean_metrics)
    clean_psnr = float(clean["PSNR_dB"])
    clean_lpips = float(clean["LPIPS"])

    rows = [{
        "key": "clean",
        "label": "Clean",
        "PSNR_dB": clean_psnr,
        "PSNR_drop_dB": 0.0,
        "LPIPS": clean_lpips,
        "delta_LPIPS": 0.0,
        "BPP": float(clean.get("BPP", np.nan)),
        "residual_symbols": 0,
        "corrected_symbols": 0,
        "overhead_bits": 0,
        "metrics": clean,
    }]
    for key, label, run_name in EXPERIMENTS:
        metrics_path = (root / run_name / args.sequence /
                        f"gop{args.gop_index}" / "metrics.json")
        metrics = read_json(metrics_path)
        attack = metrics.get("stream_attack", {})
        rows.append({
            "key": key,
            "label": label,
            "PSNR_dB": float(metrics["PSNR_dB"]),
            "PSNR_drop_dB": clean_psnr - float(metrics["PSNR_dB"]),
            "LPIPS": float(metrics["LPIPS"]),
            "delta_LPIPS": float(metrics["LPIPS"]) - clean_lpips,
            "BPP": float(metrics.get("BPP", np.nan)),
            "residual_symbols": int(attack.get("changed_symbols", 0)),
            "corrected_symbols": int(
                attack.get("corrected_logical_symbols", 0)),
            "overhead_bits": int(attack.get("protection_overhead_bits", 0)),
            "metrics": metrics,
        })

    colors = ["#6b7280", "#4c78a8", "#d1495b", "#2a9d8f"]
    labels = [row["label"] for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.4))

    psnr = [row["PSNR_dB"] for row in rows]
    bars = axes[0].bar(x, psnr, color=colors, width=0.72)
    axes[0].set_title("Reconstruction quality")
    axes[0].set_ylabel("PSNR (dB), higher is better")
    axes[0].set_ylim(0, max(psnr) + 5)
    axes[0].bar_label(bars, fmt="%.2f", padding=3, fontsize=9)

    lpips = [row["LPIPS"] for row in rows]
    bars = axes[1].bar(x, lpips, color=colors, width=0.72)
    axes[1].set_title("Perceptual distortion")
    axes[1].set_ylabel("LPIPS, lower is better")
    axes[1].set_ylim(0, max(lpips) * 1.25)
    axes[1].bar_label(bars, fmt="%.4f", padding=3, fontsize=9)

    residual = [row["residual_symbols"] for row in rows]
    bars = axes[2].bar(x, residual, color=colors, width=0.72)
    axes[2].set_title("Residual logical sign errors")
    axes[2].set_ylabel("Residual errors")
    axes[2].set_ylim(0, max(residual) * 1.22 if max(residual) else 1)
    axes[2].bar_label(bars, fmt="%d", padding=3, fontsize=9)
    axes[2].text(
        0.98, 0.95,
        (f"Sparse UEP: +{rows[-1]['overhead_bits']} bits\n"
         f"Corrected: {rows[-1]['corrected_symbols']} symbols"),
        transform=axes[2].transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", fc="#e8f5f1", ec="#2a9d8f"),
    )

    for axis in axes:
        axis.set_xticks(x)
        axis.set_xticklabels(labels, fontsize=8)
        axis.grid(axis="y", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Sign-field fault sensitivity and sparse unequal error protection",
        fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_png = prefix.with_name(prefix.name + "_summary.png")
    fig.savefig(summary_png, dpi=220, bbox_inches="tight")
    plt.close(fig)

    csv_path = prefix.with_name(prefix.name + "_summary.csv")
    fields = [
        "key", "label", "PSNR_dB", "PSNR_drop_dB", "LPIPS",
        "delta_LPIPS", "BPP", "residual_symbols", "corrected_symbols",
        "overhead_bits",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields}
                         for row in rows)

    clean_frames = clean.get("per_frame_PSNR_dB")
    propagation_png = None
    if clean_frames:
        fig, axis = plt.subplots(figsize=(9.2, 4.5))
        frame_ids = np.arange(len(clean_frames))
        for row, color in zip(rows[1:], colors[1:]):
            attacked_frames = row["metrics"].get("per_frame_PSNR_dB")
            if not attacked_frames or len(attacked_frames) != len(clean_frames):
                continue
            loss = np.asarray(clean_frames) - np.asarray(attacked_frames)
            axis.plot(frame_ids, loss, linewidth=2.1, color=color,
                      label=row["label"].replace("\n", " "))
        axis.axhline(0, color="#6b7280", linestyle="--", linewidth=1)
        axis.set_xlabel("Output frame index")
        axis.set_ylabel("Frame-wise PSNR loss (dB)")
        axis.set_title("How sign-field faults propagate across output frames")
        axis.grid(alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False)
        fig.tight_layout()
        propagation_png = prefix.with_name(prefix.name + "_propagation.png")
        fig.savefig(propagation_png, dpi=220, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved: {summary_png}")
    print(f"Saved: {csv_path}")
    if propagation_png is not None:
        print(f"Saved: {propagation_png}")
    else:
        print("Skipped propagation plot: clean per-frame PSNR is unavailable")


if __name__ == "__main__":
    main()
