"""Audit a bounded input candidate against its matched random control.

Only the saved outputs of normal GVCC re-encodes are read. The clean codebook
is used here for post-hoc diagnosis, never to modify the transmitted stream.
One GOP is a screening experiment, not a generalization result.
"""

import argparse
import csv
import json
from pathlib import Path

try:
    from .analyze_input_attack_pilot import compare_records, load_codebook
except ImportError:  # Direct execution from the exp_flf2v directory.
    from analyze_input_attack_pilot import compare_records, load_codebook


def read_json(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def run_summary(clean_dir, run_dir, clean_metrics):
    metrics = read_json(run_dir / "metrics.json")
    clean_book = load_codebook(clean_dir / "codebook.tdcm")
    attacked_book = load_codebook(run_dir / "codebook.tdcm")
    changes = compare_records(clean_book, attacked_book)
    target = next(row for row in changes
                  if row["sde_step"] == 1 and row["latent_position"] == 3)
    early = [row for row in changes if row["sde_step"] <= 2]
    frames = metrics.get("per_frame_PSNR_dB")
    clean_frames = clean_metrics.get("per_frame_PSNR_dB")
    if frames is not None and clean_frames is not None and len(frames) != len(clean_frames):
        raise ValueError(f"Per-frame metric length differs: {run_dir}")
    return {
        "PSNR_dB": float(metrics["PSNR_dB"]),
        "PSNR_drop_vs_clean_dB": float(clean_metrics["PSNR_dB"]) - float(metrics["PSNR_dB"]),
        "LPIPS": float(metrics["LPIPS"]),
        "delta_LPIPS_vs_clean": float(metrics["LPIPS"]) - float(clean_metrics["LPIPS"]),
        "PSNR_vs_submitted_dB": float(metrics["reconstruction_vs_submitted_PSNR_dB"]),
        "input_PSNR_dB": float(metrics["input_PSNR_dB"]),
        "input_MSE": float(metrics["input_MSE"]),
        "input_Linf_pixel": float(metrics["input_Linf_pixel"]),
        "BPP": float(metrics["BPP"]),
        "target_cell_replaced_atoms": target["replaced_atoms"],
        "target_cell_shared_sign_flips": target["shared_atoms_sign_flipped"],
        "early_mean_replaced_atoms": sum(row["replaced_atoms"] for row in early) / len(early),
        "encoding_trace_saved": (run_dir / "input_attack_trace.npz").is_file(),
        "per_frame_PSNR_dB": frames,
    }


def assess_pair(random, candidate):
    mse_r = random["input_MSE"]
    mse_c = candidate["input_MSE"]
    matched = abs(mse_c - mse_r) <= max(1e-8, 0.01 * max(mse_c, mse_r))
    within_budget = (candidate["input_Linf_pixel"] <= 2.01 and
                     random["input_Linf_pixel"] <= 2.01)
    psnr_gain = random["PSNR_dB"] - candidate["PSNR_dB"]
    lpips_gain = candidate["LPIPS"] - random["LPIPS"]
    submitted_gain = random["PSNR_vs_submitted_dB"] - candidate["PSNR_vs_submitted_dB"]
    direction_agrees = matched and within_budget and psnr_gain > 0 and lpips_gain > 0 and submitted_gain > 0
    if direction_agrees and psnr_gain >= 0.3 and lpips_gain >= 0.005:
        result = "promising_single_GOP"
    elif direction_agrees:
        result = "weak_single_GOP_signal"
    else:
        result = "not_yet_consistent"
    return {
        "input_MSE_matched_within_1pct": matched,
        "both_within_2_255_bound": within_budget,
        "candidate_extra_PSNR_drop_dB": psnr_gain,
        "candidate_extra_LPIPS": lpips_gain,
        "candidate_extra_PSNR_drop_vs_submitted_dB": submitted_gain,
        "screening_result": result,
        "screening_note": "0.3 dB PSNR and 0.005 LPIPS are pilot triage thresholds, not statistical significance.",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results_root", type=Path, required=True)
    p.add_argument("--clean_run", required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--output_prefix", type=Path, required=True)
    args = p.parse_args()
    root = args.results_root
    clean_dir = root / args.clean_run / "Jockey" / "gop0"
    clean_metrics = read_json(clean_dir / "metrics.json")
    runs = {
        label: run_summary(clean_dir, root / f"{args.prefix}_{label}" / "Jockey" / "gop0", clean_metrics)
        for label in ("random", "candidate")
    }
    comparison = assess_pair(runs["random"], runs["candidate"])
    random_frames = runs["random"].pop("per_frame_PSNR_dB")
    candidate_frames = runs["candidate"].pop("per_frame_PSNR_dB")
    if random_frames is not None and candidate_frames is not None:
        frame_rows = [
            {"frame": i, "random_PSNR_dB": float(r), "candidate_PSNR_dB": float(c),
             "candidate_extra_drop_dB": float(r) - float(c)}
            for i, (r, c) in enumerate(zip(random_frames, candidate_frames))
        ]
        comparison["frames_candidate_worse"] = sum(row["candidate_extra_drop_dB"] > 0
                                                   for row in frame_rows)
        frame_csv = args.output_prefix.with_name(args.output_prefix.name + "_per_frame.csv")
        frame_csv.parent.mkdir(parents=True, exist_ok=True)
        with frame_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(frame_rows[0]))
            writer.writeheader()
            writer.writerows(frame_rows)
        print(f"Saved: {frame_csv}")
    report = {
        "clean_run": args.clean_run,
        "candidate_prefix": args.prefix,
        "runs": runs,
        "paired_comparison": comparison,
        "limits": ["One GOP and one attack seed do not establish robustness or transfer.",
                   "Codebook changes are post-hoc observations, not the attack success metric.",
                   "The proxy fixes the clean trajectory and optimizes at reduced resolution."],
    }
    output = args.output_prefix.with_suffix(".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for label, row in runs.items():
        print(f"{label:9} PSNR={row['PSNR_dB']:.3f} LPIPS={row['LPIPS']:.4f} "
              f"vs_submitted={row['PSNR_vs_submitted_dB']:.3f} "
              f"target_replaced={row['target_cell_replaced_atoms']}")
    print(f"Candidate minus random: extra PSNR drop={comparison['candidate_extra_PSNR_drop_dB']:+.3f} dB, "
          f"extra LPIPS={comparison['candidate_extra_LPIPS']:+.4f}, "
          f"result={comparison['screening_result']}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
