"""Persistent-model targeted validation for GVCC codebook sign sensitivity.

Unlike the original grid launcher, this process loads Wan2.1 once and reuses
the boundary conditions, prompt embedding, reference-frame compression and
clean codebook for every requested attack configuration.
"""

import argparse
import csv
import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bitstream_attacks import apply_serialized_codebook_attack
from batch_flf2v_runtime import load_dual_gpu_flf2v_model
from codebook_validation_plan import expand_jobs, load_cells
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Persistent-model targeted codebook validation")
    parser.add_argument("--plan", required=True,
                        help="CSV with role,sde_step,latent_position columns")
    parser.add_argument("--data", required=True, help="Single 720p YUV sequence")
    parser.add_argument("--sequence", default="Jockey")
    parser.add_argument("--wan_ckpt", default="exp_flf2v/Wan2.1-FLF2V-14B-720P")
    parser.add_argument("--output_dir", default="exp_flf2v/results_720p")
    parser.add_argument("--clean_codebook", required=True)
    parser.add_argument("--clean_metrics", required=True)
    parser.add_argument("--counts", type=int, nargs="+", default=[16])
    parser.add_argument("--attack_seeds", type=int, nargs="+", default=[42, 123, 2026])
    parser.add_argument("--model_seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num_frames", type=int, default=33)
    parser.add_argument("--K", type=int, default=16384)
    parser.add_argument("--M", type=int, default=64)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--ddim_tail", type=int, default=3)
    parser.add_argument("--g_scale", type=float, default=3.0)
    parser.add_argument("--flow_shift", type=float, default=5.0)
    parser.add_argument("--ref_codec", default="compressai",
                        choices=["compressai", "webp", "gt"])
    parser.add_argument("--ref_quality", type=int, default=4)
    parser.add_argument("--save_video", action="store_true",
                        help="Save reconstructed MP4 for every job; metrics never use it")
    parser.add_argument("--plan_only", action="store_true")
    return parser.parse_args()


def write_manifest(path, jobs, sequence, output_dir):
    fields = ["run_name", "role", "count", "sde_step", "latent_position",
              "attack_seed", "status", "metrics_path"]
    rows = []
    for job in jobs:
        name = job.run_name(sequence)
        metrics = output_dir / name / sequence / "gop0" / "metrics.json"
        rows.append({
            "run_name": name, "role": job.role, "count": job.count,
            "sde_step": job.step, "latent_position": job.frame,
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


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    output_dir = Path(args.output_dir)
    cells = load_cells(args.plan)
    jobs = expand_jobs(cells, args.counts, args.attack_seeds)
    manifest_path = output_dir / f"validation_batch_manifest_{args.sequence.lower()}.csv"
    manifest_rows = write_manifest(manifest_path, jobs, args.sequence, output_dir)
    complete = sum(row["status"] == "complete" for row in manifest_rows)
    print(f"Validation jobs: {len(jobs)}; complete: {complete}; "
          f"pending: {len(jobs) - complete}")
    print(f"Manifest: {manifest_path}")
    if args.plan_only:
        return

    clean_codebook = Path(args.clean_codebook)
    if not clean_codebook.exists():
        raise FileNotFoundError(f"Clean codebook not found: {clean_codebook}")
    clean_metrics_path = Path(args.clean_metrics)
    with clean_metrics_path.open(encoding="utf-8") as handle:
        clean_metrics = json.load(handle)

    frames = resize_frames(
        load_yuv420_frames(args.data, args.num_frames), args.width, args.height)
    if len(frames) != args.num_frames:
        raise ValueError(f"Expected {args.num_frames} frames, found {len(frames)}")
    first_decoded, first_bytes = compress_boundary_frame(
        frames[0], args.ref_codec, args.ref_quality)
    last_decoded, last_bytes = compress_boundary_frame(
        frames[-1], args.ref_codec, args.ref_quality)
    boundary_bytes = first_bytes + last_bytes
    target_tensor = frames_to_tensor(frames)

    model = load_dual_gpu_flf2v_model(args.wan_ckpt, args.flow_shift)
    pipe = TurboDDCMWanPipeline(
        model, K=args.K, M=args.M, num_steps=args.steps,
        num_ddim_tail=args.ddim_tail, guidance_scale=1.0,
        g_scale=args.g_scale, num_frames=args.num_frames,
        height=args.height, width=args.width, seed=args.model_seed)
    loaded = TurboBitstream.load(str(clean_codebook))
    expected = (pipe.K, pipe.M, pipe.num_sde_steps, pipe.num_latent_frames,
                pipe.seed, pipe.frame_shape)
    actual = (loaded["K"], loaded["M"], loaded["num_sde_steps"],
              loaded["num_latent_frames"], loaded["seed"], loaded["frame_shape"])
    if actual != expected:
        raise ValueError(f"Reused codebook metadata mismatch: expected {expected}, got {actual}")
    clean_step_data = loaded["step_data"]

    print("[GVCC-BATCH] Encoding shared first/last-frame condition once...")
    condition = model.encode_first_last_frames(
        first_decoded, last_decoded, args.num_frames, args.height, args.width)
    print("[GVCC-BATCH] Encoding shared empty prompt once...")
    prompt_embeds = model.encode_prompt("")
    print("[GVCC-BATCH] Loading shared LPIPS metric once on CPU...")
    import lpips
    lpips_metric = lpips.LPIPS(net="alex").eval()

    cache_dir = output_dir / "_validation_batch_cache" / args.sequence / "gop0"
    cache_dir.mkdir(parents=True, exist_ok=True)
    canonical_clean = cache_dir / "codebook_clean.tdcm"
    pipe.save_compressed(clean_step_data, str(canonical_clean))

    for index, (job, row) in enumerate(zip(jobs, manifest_rows), start=1):
        if row["status"] == "complete":
            print(f"[SKIP {index}/{len(jobs)}] {row['run_name']}")
            continue
        print("=" * 88)
        print(f"[RUN {index}/{len(jobs)}] role={job.role} step={job.step} "
              f"latent={job.frame} count={job.count} attack_seed={job.attack_seed}")
        run_dir = output_dir / row["run_name"] / args.sequence / "gop0"
        run_dir.mkdir(parents=True, exist_ok=True)
        clean_path = run_dir / "codebook_clean.tdcm"
        attacked_path = run_dir / "codebook.tdcm"
        shutil.copy2(canonical_clean, clean_path)
        attack_metadata = apply_serialized_codebook_attack(
            clean_path, attacked_path, attack="sign-bitflip", rate=0.0,
            count=job.count, seed=job.attack_seed, early_steps=0,
            target_steps=[job.step - 1], target_frames=[job.frame])
        attacked_step_data = TurboBitstream.load(str(attacked_path))["step_data"]

        started = time.time()
        reconstructed = flf2v_decode(
            pipe, model, attacked_step_data, condition,
            prompt_embeds=prompt_embeds)
        decode_seconds = time.time() - started
        if args.save_video:
            save_video_mp4(reconstructed, run_dir / "reconstructed.mp4")

        n = min(len(frames), len(reconstructed))
        reconstructed_tensor = frames_to_tensor(reconstructed[:n])
        mean_psnr, per_frame_psnr = compute_psnr(
            target_tensor[:n], reconstructed_tensor)
        mean_msssim = compute_msssim(target_tensor[:n], reconstructed_tensor)
        mean_lpips = compute_lpips(
            target_tensor[:n], reconstructed_tensor, device="cuda:1",
            loss_fn=lpips_metric)
        codebook_bits = (pipe.num_sde_steps * pipe.num_latent_frames *
                         pipe.codebook.bits_per_frame_step)
        total_bits = codebook_bits + boundary_bytes * 8
        result = {
            "sequence": args.sequence,
            "gop": 0,
            "batch_role": job.role,
            "stream_attack": attack_metadata,
            "PSNR_dB": round(mean_psnr, 2),
            "per_frame_PSNR_dB": [round(float(v), 4)
                                    for v in per_frame_psnr.cpu().tolist()],
            "LPIPS": round(mean_lpips, 4),
            "MS_SSIM": None if mean_msssim is None else round(mean_msssim, 6),
            "BPP": round(total_bits / (args.num_frames * args.height * args.width), 6),
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
        write_manifest(manifest_path, jobs, args.sequence, output_dir)
        print(f"[DONE] PSNR={mean_psnr:.2f} dB, LPIPS={mean_lpips:.4f}, "
              f"decode={decode_seconds:.1f}s")
        del reconstructed, reconstructed_tensor, attacked_step_data
        gc.collect()
        torch.cuda.empty_cache()

    print("[GVCC-BATCH] All requested validation jobs are complete.")


if __name__ == "__main__":
    main()
