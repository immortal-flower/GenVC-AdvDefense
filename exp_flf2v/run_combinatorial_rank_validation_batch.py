"""Persistent-model validation of TDC2 combinatorial-rank bit errors.

The model, boundary conditions, prompt embedding and LPIPS network are loaded
once.  Each job flips one valid bit in a requested third of one combination
rank and then performs only the generative decode and metric evaluation.
"""

import argparse
import csv
import gc
import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from batch_flf2v_runtime import load_dual_gpu_flf2v_model
from bitstream_attacks import apply_serialized_codebook_attack
from codebook_validation_plan import load_cells
from run_flf2v_experiment import (
    compress_boundary_frame,
    compute_lpips,
    compute_msssim,
    compute_psnr,
    flf2v_decode,
    frames_to_tensor,
    load_yuv420_frames,
    resize_frames,
    save_video_mp4,
)
from sde_rf_wan.turbo_codebook import TurboBitstream
from sde_rf_wan.turbo_pipeline import TurboDDCMWanPipeline


@dataclass(frozen=True)
class RankJob:
    role: str
    step: int
    frame: int
    region: str
    attack_seed: int
    attack_mode: str
    attack_count: int

    def run_name(self, sequence: str, gop_index: int) -> str:
        seq = "".join(
            ch.lower() if ch.isalnum() or ch in "_-" else "_"
            for ch in sequence
        )
        gop_tag = "" if gop_index == 0 else f"_gop{gop_index}"
        if self.attack_mode == "rank-bitflip":
            prefix = f"tdc2_rank{self.attack_count}_{self.region}"
        else:
            prefix = f"tdcm_index{self.attack_count}"
        return (
            f"{prefix}_step{self.step}_frame{self.frame}_"
            f"aseed{self.attack_seed}{gop_tag}_{seq}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Persistent real-TDC2 rank-bit validation"
    )
    parser.add_argument("--plan", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--sequence", default="Jockey")
    parser.add_argument("--wan_ckpt", default="exp_flf2v/Wan2.1-FLF2V-14B-720P")
    parser.add_argument("--output_dir", default="exp_flf2v/results_720p")
    parser.add_argument("--clean_codebook", required=True)
    parser.add_argument("--clean_metrics", required=True)
    parser.add_argument(
        "--attack_mode", default="rank-bitflip",
        choices=["rank-bitflip", "index-bitflip"],
    )
    parser.add_argument("--attack_count", type=int, default=1)
    parser.add_argument(
        "--rank_regions", nargs="+", default=["low", "middle", "high"],
        choices=["low", "middle", "high"],
    )
    parser.add_argument("--attack_seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--model_seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num_frames", type=int, default=33)
    parser.add_argument("--gop_index", type=int, default=0)
    parser.add_argument("--K", type=int, default=16384)
    parser.add_argument("--M", type=int, default=64)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--ddim_tail", type=int, default=3)
    parser.add_argument("--g_scale", type=float, default=3.0)
    parser.add_argument("--flow_shift", type=float, default=5.0)
    parser.add_argument(
        "--ref_codec", default="compressai", choices=["compressai", "webp", "gt"]
    )
    parser.add_argument("--ref_quality", type=int, default=4)
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--plan_only", action="store_true")
    return parser.parse_args()


def expand_jobs(cells, regions, seeds, attack_mode, attack_count):
    if attack_mode == "index-bitflip":
        regions = ["independent"]
    return [
        RankJob(role, step, frame, region, seed, attack_mode, attack_count)
        for seed in seeds
        for role, step, frame in cells
        for region in regions
    ]


def write_manifest(path, jobs, sequence, output_dir, gop_index):
    fields = [
        "run_name", "role", "rank_region", "sde_step", "latent_position",
        "attack_seed", "status", "metrics_path",
    ]
    rows = []
    for job in jobs:
        name = job.run_name(sequence, gop_index)
        metrics = output_dir / name / sequence / f"gop{gop_index}" / "metrics.json"
        rows.append({
            "run_name": name,
            "role": job.role,
            "rank_region": job.region,
            "sde_step": job.step,
            "latent_position": job.frame,
            "attack_seed": job.attack_seed,
            "status": "complete" if metrics.exists() else "pending",
            "metrics_path": str(metrics),
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_summary(path, rows):
    fields = [
        "run_name", "role", "rank_region", "sde_step", "latent_position",
        "attack_seed", "flipped_rank_bit", "replaced_atoms", "PSNR_dB",
        "PSNR_drop_dB", "LPIPS", "delta_LPIPS", "decode_seconds",
    ]
    summary = []
    for row in rows:
        metrics_path = Path(row["metrics_path"])
        if not metrics_path.exists():
            continue
        with metrics_path.open(encoding="utf-8") as handle:
            metrics = json.load(handle)
        examples = metrics.get("stream_attack", {}).get("examples", [])
        example = examples[0] if examples else {}
        summary.append({
            "run_name": row["run_name"],
            "role": row["role"],
            "rank_region": row["rank_region"],
            "sde_step": row["sde_step"],
            "latent_position": row["latent_position"],
            "attack_seed": row["attack_seed"],
            "flipped_rank_bit": example.get("flipped_bit", ""),
            "replaced_atoms": example.get(
                "replaced_atoms",
                metrics.get("stream_attack", {}).get("changed_symbols", ""),
            ),
            "PSNR_dB": metrics["PSNR_dB"],
            "PSNR_drop_dB": round(
                float(metrics["clean_PSNR_dB"]) - float(metrics["PSNR_dB"]), 4
            ),
            "LPIPS": metrics["LPIPS"],
            "delta_LPIPS": round(
                float(metrics["LPIPS"]) - float(metrics["clean_LPIPS"]), 6
            ),
            "decode_seconds": metrics["decode_seconds"],
        })
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)


def main():
    args = parse_args()
    if args.gop_index < 0:
        raise ValueError("--gop_index must be non-negative")
    if args.attack_count < 1:
        raise ValueError("--attack_count must be positive")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    output_dir = Path(args.output_dir)
    cells = load_cells(args.plan)
    jobs = expand_jobs(
        cells, args.rank_regions, args.attack_seeds,
        args.attack_mode, args.attack_count,
    )
    gop_tag = "" if args.gop_index == 0 else f"_gop{args.gop_index}"
    sequence_tag = args.sequence.lower()
    if args.attack_mode == "rank-bitflip":
        experiment_tag = (
            "tdc2_rank" if args.attack_count == 1
            else f"tdc2_rank{args.attack_count}"
        )
    elif args.attack_count == 1:
        experiment_tag = "tdcm_index"
    else:
        experiment_tag = f"tdcm_index{args.attack_count}"
    manifest_path = output_dir / (
        f"{experiment_tag}_manifest_{sequence_tag}{gop_tag}.csv"
    )
    summary_path = output_dir / (
        f"{experiment_tag}_summary_{sequence_tag}{gop_tag}.csv"
    )
    manifest_rows = write_manifest(
        manifest_path, jobs, args.sequence, output_dir, args.gop_index
    )
    complete = sum(row["status"] == "complete" for row in manifest_rows)
    print(
        f"{args.attack_mode} jobs: {len(jobs)}; complete: {complete}; "
        f"pending: {len(jobs) - complete}"
    )
    print(f"Manifest: {manifest_path}")
    if args.plan_only:
        return

    clean_codebook = Path(args.clean_codebook)
    if not clean_codebook.exists():
        raise FileNotFoundError(f"Clean codebook not found: {clean_codebook}")
    with Path(args.clean_metrics).open(encoding="utf-8") as handle:
        clean_metrics = json.load(handle)

    source_start_frame = args.gop_index * (args.num_frames - 1)
    frames = resize_frames(
        load_yuv420_frames(
            args.data, args.num_frames, start_frame=source_start_frame
        ),
        args.width,
        args.height,
    )
    if len(frames) != args.num_frames:
        raise ValueError(f"Expected {args.num_frames} frames, found {len(frames)}")
    first_decoded, first_bytes = compress_boundary_frame(
        frames[0], args.ref_codec, args.ref_quality
    )
    last_decoded, last_bytes = compress_boundary_frame(
        frames[-1], args.ref_codec, args.ref_quality
    )
    boundary_bytes = first_bytes + last_bytes
    target_tensor = frames_to_tensor(frames)

    model = load_dual_gpu_flf2v_model(args.wan_ckpt, args.flow_shift)
    pipe = TurboDDCMWanPipeline(
        model,
        K=args.K,
        M=args.M,
        num_steps=args.steps,
        num_ddim_tail=args.ddim_tail,
        guidance_scale=1.0,
        g_scale=args.g_scale,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        seed=args.model_seed,
    )

    loaded = TurboBitstream.load(str(clean_codebook))
    expected = (
        pipe.K, pipe.M, pipe.num_sde_steps, pipe.num_latent_frames,
        pipe.seed, pipe.frame_shape,
    )
    actual = (
        loaded["K"], loaded["M"], loaded["num_sde_steps"],
        loaded["num_latent_frames"], loaded["seed"], loaded["frame_shape"],
    )
    if actual != expected:
        raise ValueError(
            f"Reused codebook metadata mismatch: expected {expected}, got {actual}"
        )
    clean_step_data = loaded["step_data"]

    print("[TDC2-BATCH] Encoding shared first/last-frame condition once...")
    condition = model.encode_first_last_frames(
        first_decoded, last_decoded, args.num_frames, args.height, args.width
    )
    print("[TDC2-BATCH] Encoding shared empty prompt once...")
    prompt_embeds = model.encode_prompt("")
    print("[TDC2-BATCH] Loading shared LPIPS metric once on CPU...")
    import lpips
    lpips_metric = lpips.LPIPS(net="alex").eval()

    cache_dir = (
        output_dir / f"_{experiment_tag}_batch_cache" /
        args.sequence / f"gop{args.gop_index}"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    canonical_clean = cache_dir / "codebook_clean.tdcm"
    if args.attack_mode == "rank-bitflip":
        # save_compressed writes current TDC2, so a legacy TDCM input can be
        # converted without repeating codebook search.
        pipe.save_compressed(clean_step_data, str(canonical_clean))
    else:
        if clean_codebook.read_bytes()[:4] != b"TDCM":
            raise ValueError(
                "index-bitflip baseline requires a legacy TDCM clean stream"
            )
        shutil.copy2(clean_codebook, canonical_clean)

    for index, (job, row) in enumerate(zip(jobs, manifest_rows), start=1):
        if row["status"] == "complete":
            print(f"[SKIP {index}/{len(jobs)}] {row['run_name']}")
            continue
        print("=" * 88)
        print(
            f"[RUN {index}/{len(jobs)}] role={job.role} step={job.step} "
            f"latent={job.frame} region={job.region} seed={job.attack_seed}"
        )
        run_dir = output_dir / row["run_name"] / args.sequence / f"gop{args.gop_index}"
        run_dir.mkdir(parents=True, exist_ok=True)
        clean_path = run_dir / "codebook_clean.tdcm"
        attacked_path = run_dir / "codebook.tdcm"
        shutil.copy2(canonical_clean, clean_path)
        attack_metadata = apply_serialized_codebook_attack(
            clean_path,
            attacked_path,
            attack=args.attack_mode,
            count=job.attack_count,
            seed=job.attack_seed,
            target_steps=[job.step - 1],
            target_frames=[job.frame],
            rank_region=(job.region if args.attack_mode == "rank-bitflip" else "all"),
        )
        attacked_step_data = TurboBitstream.load(str(attacked_path))["step_data"]

        started = time.time()
        reconstructed = flf2v_decode(
            pipe, model, attacked_step_data, condition, prompt_embeds=prompt_embeds
        )
        decode_seconds = time.time() - started
        if args.save_video:
            save_video_mp4(reconstructed, run_dir / "reconstructed.mp4")

        n = min(len(frames), len(reconstructed))
        reconstructed_tensor = frames_to_tensor(reconstructed[:n])
        mean_psnr, per_frame_psnr = compute_psnr(
            target_tensor[:n], reconstructed_tensor
        )
        mean_msssim = compute_msssim(target_tensor[:n], reconstructed_tensor)
        mean_lpips = compute_lpips(
            target_tensor[:n], reconstructed_tensor,
            device="cuda:1", loss_fn=lpips_metric,
        )
        if args.attack_mode == "rank-bitflip":
            bits_per_frame_step = pipe.codebook.bits_per_frame_step
        else:
            bits_per_frame_step = (
                pipe.M * math.ceil(math.log2(pipe.K)) + pipe.M
            )
        codebook_bits = (
            pipe.num_sde_steps * pipe.num_latent_frames
            * bits_per_frame_step
        )
        total_bits = codebook_bits + boundary_bytes * 8
        result = {
            "sequence": args.sequence,
            "gop": args.gop_index,
            "batch_role": job.role,
            "rank_region": job.region,
            "stream_attack": attack_metadata,
            "PSNR_dB": round(mean_psnr, 2),
            "per_frame_PSNR_dB": [
                round(float(value), 4) for value in per_frame_psnr.cpu().tolist()
            ],
            "LPIPS": round(mean_lpips, 4),
            "MS_SSIM": None if mean_msssim is None else round(mean_msssim, 6),
            "BPP": round(
                total_bits / (args.num_frames * args.height * args.width), 6
            ),
            "gop_total_bytes": (total_bits + 7) // 8,
            "codebook_bits": codebook_bits,
            "codebook_bytes": (codebook_bits + 7) // 8,
            "boundary_bytes": boundary_bytes,
            "protection_overhead_bits": 0,
            "encode_seconds": 0.0,
            "decode_seconds": round(decode_seconds, 3),
            "clean_PSNR_dB": clean_metrics["PSNR_dB"],
            "clean_LPIPS": clean_metrics["LPIPS"],
        }
        with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)

        row["status"] = "complete"
        manifest_rows = write_manifest(
            manifest_path, jobs, args.sequence, output_dir, args.gop_index
        )
        write_summary(summary_path, manifest_rows)
        example = attack_metadata["examples"][0]
        replaced_atoms = example.get(
            "replaced_atoms", attack_metadata.get("changed_symbols", "")
        )
        print(
            f"[DONE] bit={example['flipped_bit']} "
            f"replaced={replaced_atoms} PSNR={mean_psnr:.2f} dB "
            f"LPIPS={mean_lpips:.4f} decode={decode_seconds:.1f}s"
        )
        del reconstructed, reconstructed_tensor, attacked_step_data
        gc.collect()
        torch.cuda.empty_cache()

    write_summary(summary_path, manifest_rows)
    print("[TDC2-BATCH] All requested rank-validation jobs are complete.")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
