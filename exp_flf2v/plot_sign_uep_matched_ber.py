"""Plot matched local-BER comparison for sparse sign-field UEP.

The protected stream repeats 192 logical sign bits three times (576 physical
bits).  For a fair channel comparison, each protected physical-error budget is
paired with the nearest unprotected error count at the same local BER.
"""

import argparse
import csv
import json
import statistics
from pathlib import Path


MATCHED_BUDGETS = (
    (32, 11),
    (64, 21),
    (96, 32),
    (128, 43),
    (192, 64),
)
DEFAULT_SEEDS = (42, 123, 2026)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot sparse sign UEP against a matched-BER baseline")
    parser.add_argument("--results_root", default="exp_flf2v/results_720p")
    parser.add_argument("--sequence", default="Jockey")
    parser.add_argument("--gop_index", type=int, default=0)
    parser.add_argument("--clean_metrics", required=True)
    parser.add_argument("--output_prefix", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    return parser.parse_args()


def read_metrics(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def sequence_tag(sequence):
    return "".join(ch.lower() if ch.isalnum() or ch in "_-" else "_"
                   for ch in sequence)


def protected_run_name(budget, seed, sequence="Jockey", gop_index=0):
    if gop_index == 0 and budget == 64 and seed == 42:
        return "sign64_hotspot_sparse_rep3"
    if gop_index:
        return (f"sign_uep_rep3_count{budget}_aseed{seed}_"
                f"gop{gop_index}_{sequence_tag(sequence)}")
    return f"sign_uep_rep3_budget{budget}_seed{seed}"


def unprotected_run_name(count, seed, sequence, gop_index=0):
    gop_tag = "" if gop_index == 0 else f"_gop{gop_index}"
    return (f"sign_uep_unprotected_count{count}_aseed{seed}"
            f"{gop_tag}_{sequence_tag(sequence)}")


def metric_path(root, run_name, sequence, gop_index):
    return root / run_name / sequence / f"gop{gop_index}" / "metrics.json"


def sample_std(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def aggregate(values):
    return statistics.mean(values), sample_std(values)


def collect_rows(root, sequence, gop_index, seeds, clean_psnr, clean_lpips):
    rows = []
    missing = []
    for protected_budget, unprotected_count in MATCHED_BUDGETS:
        protected = []
        unprotected = []
        for seed in seeds:
            protected_path = metric_path(
                root, protected_run_name(
                    protected_budget, seed, sequence, gop_index),
                sequence, gop_index)
            unprotected_path = metric_path(
                root, unprotected_run_name(
                    unprotected_count, seed, sequence, gop_index),
                sequence, gop_index)
            for label, path, destination in (
                    ("protected", protected_path, protected),
                    ("unprotected", unprotected_path, unprotected)):
                if not path.exists():
                    missing.append(str(path))
                    continue
                metrics = read_metrics(path)
                destination.append({
                    "seed": seed,
                    "psnr_drop": clean_psnr - float(metrics["PSNR_dB"]),
                    "lpips": float(metrics["LPIPS"]),
                    "delta_lpips": float(metrics["LPIPS"]) - clean_lpips,
                    "residual": int(
                        metrics.get("stream_attack", {}).get(
                            "changed_symbols", unprotected_count)),
                    "label": label,
                })
        if len(protected) != len(seeds) or len(unprotected) != len(seeds):
            continue
        p_drop, p_drop_std = aggregate([item["psnr_drop"] for item in protected])
        u_drop, u_drop_std = aggregate([item["psnr_drop"] for item in unprotected])
        p_lpips, p_lpips_std = aggregate([item["lpips"] for item in protected])
        u_lpips, u_lpips_std = aggregate([item["lpips"] for item in unprotected])
        p_residual, p_residual_std = aggregate(
            [item["residual"] for item in protected])
        rows.append({
            "local_ber_percent": protected_budget / 576 * 100,
            "protected_physical_errors": protected_budget,
            "unprotected_logical_errors": unprotected_count,
            "seeds": len(seeds),
            "unprotected_psnr_drop_mean_db": u_drop,
            "unprotected_psnr_drop_std_db": u_drop_std,
            "protected_psnr_drop_mean_db": p_drop,
            "protected_psnr_drop_std_db": p_drop_std,
            "psnr_recovery_gain_db": u_drop - p_drop,
            "unprotected_lpips_mean": u_lpips,
            "unprotected_lpips_std": u_lpips_std,
            "protected_lpips_mean": p_lpips,
            "protected_lpips_std": p_lpips_std,
            "protected_residual_mean": p_residual,
            "protected_residual_std": p_residual_std,
        })
    return rows, missing


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_rows(path, rows):
    import matplotlib.pyplot as plt

    rates = [row["local_ber_percent"] for row in rows]
    u_drop = [row["unprotected_psnr_drop_mean_db"] for row in rows]
    u_drop_std = [row["unprotected_psnr_drop_std_db"] for row in rows]
    p_drop = [row["protected_psnr_drop_mean_db"] for row in rows]
    p_drop_std = [row["protected_psnr_drop_std_db"] for row in rows]
    u_lpips = [row["unprotected_lpips_mean"] for row in rows]
    u_lpips_std = [row["unprotected_lpips_std"] for row in rows]
    p_lpips = [row["protected_lpips_mean"] for row in rows]
    p_lpips_std = [row["protected_lpips_std"] for row in rows]

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
    })
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8), dpi=180)
    colors = {"unprotected": "#d9534f", "protected": "#24649a"}

    axes[0].errorbar(
        rates, u_drop, yerr=u_drop_std, marker="o", linewidth=2.2,
        capsize=4, color=colors["unprotected"], label="Unprotected")
    axes[0].errorbar(
        rates, p_drop, yerr=p_drop_std, marker="o", linewidth=2.2,
        capsize=4, color=colors["protected"], label="Sparse repetition-3")
    axes[0].set_title("Reconstruction degradation")
    axes[0].set_xlabel("Local channel error rate (%)")
    axes[0].set_ylabel("PSNR drop (dB, lower is better)")
    axes[0].legend(frameon=False)

    axes[1].errorbar(
        rates, u_lpips, yerr=u_lpips_std, marker="o", linewidth=2.2,
        capsize=4, color=colors["unprotected"], label="Unprotected")
    axes[1].errorbar(
        rates, p_lpips, yerr=p_lpips_std, marker="o", linewidth=2.2,
        capsize=4, color=colors["protected"], label="Sparse repetition-3")
    axes[1].set_title("Perceptual distortion")
    axes[1].set_xlabel("Local channel error rate (%)")
    axes[1].set_ylabel("LPIPS (lower is better)")
    axes[1].legend(frameon=False)

    for axis in axes:
        axis.grid(True, color="#d9e1e8", linewidth=0.8, alpha=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xticks(rates)
        axis.set_xticklabels([f"{rate:.1f}" for rate in rates])
        axis.axvspan(rates[0], rates[1], color="#dceaf5", alpha=0.25,
                     zorder=0)
    axes[0].annotate(
        "Near-lossless protected region",
        xy=(rates[1], p_drop[1]), xytext=(rates[1] + 2.5, 0.55),
        arrowprops={"arrowstyle": "->", "color": colors["protected"]},
        color=colors["protected"], fontsize=9)

    fig.suptitle(
        "Matched local-BER comparison on frozen sensitive sign fields",
        fontsize=15, fontweight="bold")
    fig.text(
        0.5, 0.015,
        "Mean of three channel seeds; error bars show sample standard deviation",
        ha="center", color="#59636e", fontsize=9)
    fig.tight_layout(rect=(0, 0.045, 1, 0.93))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    args = parse_args()
    root = Path(args.results_root)
    clean = read_metrics(args.clean_metrics)
    rows, missing = collect_rows(
        root, args.sequence, args.gop_index, tuple(args.seeds),
        float(clean["PSNR_dB"]), float(clean["LPIPS"]))
    if missing:
        formatted = "\n".join(f"  {path}" for path in missing)
        raise FileNotFoundError(f"Missing matched-BER metrics:\n{formatted}")
    if not rows:
        raise RuntimeError("No complete matched-BER rows were found")

    prefix = Path(args.output_prefix)
    csv_path = prefix.with_name(prefix.name + "_comparison.csv")
    png_path = prefix.with_name(prefix.name + "_comparison.png")
    json_path = prefix.with_name(prefix.name + "_summary.json")
    write_csv(csv_path, rows)
    plot_rows(png_path, rows)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump({"rows": rows}, handle, indent=2)

    print("Matched local-BER sparse UEP comparison")
    print("BER(%)  UnprotectedDrop  ProtectedDrop  RecoveryGain")
    for row in rows:
        print(f"{row['local_ber_percent']:6.2f}  "
              f"{row['unprotected_psnr_drop_mean_db']:15.3f}  "
              f"{row['protected_psnr_drop_mean_db']:13.3f}  "
              f"{row['psnr_recovery_gain_db']:12.3f}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {png_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
