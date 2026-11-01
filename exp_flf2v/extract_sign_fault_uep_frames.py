"""Extract a matched worst-frame comparison for the sign-fault UEP study."""

import argparse
import json
from pathlib import Path

try:
    import imageio.v2 as imageio
except (ImportError, ModuleNotFoundError):
    import imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract matched frames from clean/fault/UEP videos")
    parser.add_argument("--results_root", default="exp_flf2v/results_720p")
    parser.add_argument("--sequence", default="Jockey")
    parser.add_argument("--gop_index", type=int, default=0)
    parser.add_argument(
        "--clean_metrics",
        default=("exp_flf2v/results_720p/sensitivity_clean_redecode/"
                 "Jockey/gop0/metrics.json"),
    )
    parser.add_argument("--clean_video", default=None)
    parser.add_argument(
        "--output_dir",
        default="exp_flf2v/results_720p/sign_fault_uep_frames",
    )
    return parser.parse_args()


def load(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def read_frame(path, index):
    reader = imageio.get_reader(str(path))
    try:
        return np.asarray(reader.get_data(index))
    finally:
        reader.close()


def main():
    args = parse_args()
    root = Path(args.results_root)
    gop = f"gop{args.gop_index}"
    clean_metrics_path = Path(args.clean_metrics)
    clean_video = (Path(args.clean_video) if args.clean_video else
                   clean_metrics_path.parent / "reconstructed.mp4")

    specs = [
        ("clean", "Clean reconstruction", clean_metrics_path, clean_video),
        (
            "random",
            "Global random 64-bit fault",
            root / "sign64_global_random_fault" / args.sequence / gop /
            "metrics.json",
            root / "sign64_global_random_fault" / args.sequence / gop /
            "reconstructed.mp4",
        ),
        (
            "hotspot",
            "Hotspot-focused 64-bit fault",
            root / "sign64_hotspot_fault" / args.sequence / gop /
            "metrics.json",
            root / "sign64_hotspot_fault" / args.sequence / gop /
            "reconstructed.mp4",
        ),
        (
            "protected",
            "Hotspot fault + sparse repetition",
            root / "sign64_hotspot_sparse_rep3" / args.sequence / gop /
            "metrics.json",
            root / "sign64_hotspot_sparse_rep3" / args.sequence / gop /
            "reconstructed.mp4",
        ),
    ]
    missing = [str(path) for _, _, metrics, video in specs
               for path in (metrics, video) if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("Missing comparison inputs:\n" + "\n".join(missing))

    records = []
    for key, title, metrics_path, video_path in specs:
        metrics = load(metrics_path)
        per_frame = metrics.get("per_frame_PSNR_dB")
        if not per_frame:
            raise ValueError(f"No per_frame_PSNR_dB in {metrics_path}")
        records.append({
            "key": key,
            "title": title,
            "metrics": metrics,
            "per_frame": np.asarray(per_frame, dtype=float),
            "video": Path(video_path),
        })

    clean_curve = records[0]["per_frame"]
    hotspot_curve = records[2]["per_frame"]
    if len(clean_curve) != len(hotspot_curve):
        raise ValueError("Clean and hotspot per-frame arrays differ in length")
    losses = clean_curve - hotspot_curve
    worst_frame = int(np.argmax(losses))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for record in records:
        frame = read_frame(record["video"], worst_frame)
        frames.append(frame)
        imageio.imwrite(
            output_dir / f"frame{worst_frame:02d}_{record['key']}.png",
            frame,
        )

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 7.6))
    for axis, record, frame in zip(axes.flat, records, frames):
        metrics = record["metrics"]
        axis.imshow(frame)
        axis.set_title(
            f"{record['title']}\n"
            f"PSNR {float(metrics['PSNR_dB']):.2f} dB | "
            f"LPIPS {float(metrics['LPIPS']):.4f}",
            fontsize=11,
        )
        axis.axis("off")
    fig.suptitle(
        f"Matched output frame {worst_frame}: hotspot loss "
        f"{losses[worst_frame]:.2f} dB",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    contact_sheet = output_dir / f"frame{worst_frame:02d}_comparison.png"
    fig.savefig(contact_sheet, dpi=200, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "worst_frame": worst_frame,
        "hotspot_frame_psnr_loss_db": float(losses[worst_frame]),
        "clean_frame_psnr_db": float(clean_curve[worst_frame]),
        "hotspot_frame_psnr_db": float(hotspot_curve[worst_frame]),
        "outputs": [str(output_dir / f"frame{worst_frame:02d}_{r['key']}.png")
                    for r in records],
        "contact_sheet": str(contact_sheet),
    }
    summary_path = output_dir / "worst_frame_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Worst frame: {worst_frame}")
    print(f"Hotspot frame PSNR loss: {losses[worst_frame]:.2f} dB")
    print(f"Saved: {contact_sheet}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
