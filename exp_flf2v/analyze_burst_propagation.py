"""Analyze per-output-frame propagation of the 128-bit structured attack.

No neural-network inference is performed.  The script compares saved clean and
attacked ``per_frame_PSNR_dB`` arrays for Jockey GOPs 0-2 and Beauty GOP 0.
The attacked latent positions 3 and 4 directly represent output frames 9-16;
all other output frames are treated as propagation outside the target segment.
"""
import argparse
import csv
import json
from pathlib import Path


def load(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def mean(values):
    return sum(values) / len(values) if values else None


def represented_output_frames(latent_frames, num_output_frames):
    represented = []
    for latent_frame in latent_frames:
        if latent_frame == 0:
            represented.append(0)
        else:
            start = 1 + 4 * (latent_frame - 1)
            represented.extend(range(start, min(start + 4, num_output_frames)))
    return sorted(set(represented))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", type=Path,
                        default=Path("exp_flf2v/results_720p"))
    args = parser.parse_args()
    root = args.results_root

    cases = [
        ("Jockey", 0,
         root / "Jockey/gop0/metrics.json",
         root / "burst_step1_frames3_4_all/Jockey/gop0/metrics.json"),
        ("Jockey", 1,
         root / "Jockey/gop1/metrics.json",
         root / "cross_gop1_step1_frames3_4_all/Jockey/gop1/metrics.json"),
        ("Jockey", 2,
         root / "Jockey/gop2/metrics.json",
         root / "cross_gop2_step1_frames3_4_all/Jockey/gop2/metrics.json"),
        ("Beauty", 0,
         root / "crossseq_clean/Beauty/gop0/metrics.json",
         root / "crossseq_burst_step1_frames3_4/Beauty/gop0/metrics.json"),
    ]
    summary = []
    detail = []
    missing = []
    for sequence, gop, clean_path, attack_path in cases:
        if not clean_path.exists() or not attack_path.exists():
            missing.append({"sequence": sequence, "gop": gop,
                            "clean": str(clean_path), "attack": str(attack_path)})
            continue
        clean = load(clean_path)
        attacked = load(attack_path)
        clean_frames = clean.get("per_frame_PSNR_dB")
        attack_frames = attacked.get("per_frame_PSNR_dB")
        if not clean_frames or not attack_frames or len(clean_frames) != len(attack_frames):
            raise ValueError(f"Per-frame PSNR mismatch: {clean_path} vs {attack_path}")
        # Positive damage means a PSNR loss, which is easier to interpret than
        # the negative attacked-minus-clean convention used in older scripts.
        damage = [float(c) - float(a) for c, a in zip(clean_frames, attack_frames)]
        target = represented_output_frames([3, 4], len(damage))
        outside = [index for index in range(len(damage)) if index not in target]
        worst = max(range(len(damage)), key=lambda index: damage[index])
        row = {
            "sequence": sequence,
            "gop": gop,
            "target_output_frames": target,
            "clean_PSNR_dB": clean["PSNR_dB"],
            "attack_PSNR_dB": attacked["PSNR_dB"],
            "gop_PSNR_loss_dB": round(float(clean["PSNR_dB"]) - float(attacked["PSNR_dB"]), 4),
            "clean_LPIPS": clean["LPIPS"],
            "attack_LPIPS": attacked["LPIPS"],
            "LPIPS_increase": round(float(attacked["LPIPS"]) - float(clean["LPIPS"]), 4),
            "target_frames_mean_PSNR_loss_dB": round(mean([damage[i] for i in target]), 4),
            "outside_frames_mean_PSNR_loss_dB": round(mean([damage[i] for i in outside]), 4),
            "worst_output_frame": worst,
            "worst_frame_PSNR_loss_dB": round(damage[worst], 4),
            "clean_metrics": str(clean_path),
            "attack_metrics": str(attack_path),
        }
        summary.append(row)
        for frame, loss in enumerate(damage):
            detail.append({
                "sequence": sequence,
                "gop": gop,
                "output_frame": frame,
                "is_direct_target_segment": frame in target,
                "clean_PSNR_dB": clean_frames[frame],
                "attack_PSNR_dB": attack_frames[frame],
                "PSNR_loss_dB": round(loss, 4),
            })

    if not summary:
        raise FileNotFoundError(f"No complete clean/attack metric pairs under {root}")
    summary_path = root / "burst_propagation_summary.json"
    detail_path = root / "burst_propagation_per_frame.csv"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump({"runs": summary, "missing": missing}, handle, indent=2)
    with detail_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail[0]))
        writer.writeheader()
        writer.writerows(detail)

    print("\nStructured burst propagation: latent frames 3-4 -> output frames 9-16")
    print("Sequence  GOP  GOPLoss  TargetLoss  OutsideLoss  WorstFrame  WorstLoss  LPIPSInc")
    for row in summary:
        print(f"{row['sequence']:<9} {row['gop']:>3}  "
              f"{row['gop_PSNR_loss_dB']:>7.3f}  "
              f"{row['target_frames_mean_PSNR_loss_dB']:>10.3f}  "
              f"{row['outside_frames_mean_PSNR_loss_dB']:>11.3f}  "
              f"{row['worst_output_frame']:>10}  "
              f"{row['worst_frame_PSNR_loss_dB']:>9.3f}  "
              f"{row['LPIPS_increase']:>8.4f}")
    if missing:
        print(f"\nSkipped {len(missing)} missing clean/attack pairs.")
    print(f"Saved: {summary_path}")
    print(f"Saved: {detail_path}")


if __name__ == "__main__":
    main()
