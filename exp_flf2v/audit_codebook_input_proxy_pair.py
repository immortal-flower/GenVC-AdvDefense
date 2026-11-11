"""Audit an existing input-proxy pair without re-running the full GVCC codec.

Replays the saved candidate and matched random input through the small VAE
proxy, then checks the *saved* real-encoder first-step trace. This is a
diagnostic, not evidence that the proxy predicts final video quality.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def history_summary(history):
    values = [float(value) for value in history]
    if not values or not np.isfinite(values).all():
        raise ValueError("Missing or non-finite proxy objective history")
    return {
        "first_pre_update": values[0],
        "last_pre_update": values[-1],
        "pre_update_change": values[-1] - values[0],
        "upward_intervals": sum(b > a for a, b in zip(values, values[1:])),
        "intervals": len(values) - 1,
        "note": "Last history entry precedes the final PGD update; use replay_final for the final score.",
    }


def real_encoder_cell(trace_path, position):
    with np.load(trace_path, allow_pickle=False) as trace:
        residual = trace["residual_step1"][0, :, position].astype(np.float64)
        correction = trace["correction_step1"][0, :, position].astype(np.float64)
    residual_energy = float(np.square(residual).mean())
    return {
        "residual_MSE": residual_energy,
        "hard_uncovered_MSE": float(np.square(residual - correction).mean()),
        "hard_uncovered_ratio": float(np.square(residual - correction).mean() / max(residual_energy, 1e-20)),
    }


def quantize_asset(original, asset_path):
    with np.load(asset_path, allow_pickle=False) as saved:
        delta = saved["delta"].astype(np.float32)
    if delta.shape != original.shape or not np.isfinite(delta).all():
        raise ValueError(f"Invalid attack asset shape or values: {asset_path}")
    if float(np.abs(delta).max()) > 2 / 255 + 1e-6:
        raise ValueError(f"Attack exceeds 2/255: {asset_path}")
    attacked = np.clip(original.astype(np.float32) / 255 + delta, 0, 1)
    return np.floor(attacked * 255 + 0.5).astype(np.uint8)


def classify_pair(proxy_delta, real_delta):
    if proxy_delta <= 0:
        return "candidate_did_not_beat_random_under_its_own_proxy"
    if real_delta <= 0:
        return "proxy_improved_but_actual_first_step_hard_score_did_not"
    return "both_first_step_scores_increased_check_final_video_separately"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", required=True, help="Pilot prefix without _random/_candidate")
    p.add_argument("--results_root", type=Path, default=Path("exp_flf2v/results_720p"))
    p.add_argument("--assets_root", type=Path, default=Path("exp_flf2v/attack_assets"))
    p.add_argument("--data", type=Path, default=Path(r"D:\yzb and lmk\dataset-720P\UVG\Jockey_720p.yuv"))
    p.add_argument("--vae", type=Path, default=Path("exp_flf2v/Wan2.1-FLF2V-14B-720P/Wan2.1_VAE.pth"))
    p.add_argument("--clean_run", default="input_pilot_v2_eps2_seed42_steps20_clean")
    p.add_argument("--position", type=int, default=3)
    p.add_argument("--noise_coeff", type=float, default=0.3062)
    p.add_argument("--candidate_count", type=int, default=80)
    p.add_argument("--opt_height", type=int, default=192)
    p.add_argument("--opt_width", type=int, default=336)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    import torch
    import torch.nn.functional as F
    from generate_codebook_input_pilot import candidate_bank
    from generate_vae_fullcontext_pgd import resize_video
    from generate_vae_pgd import WanVAE, load_yuv420_frames, resize_frames
    from input_codebook_objective import input_codebook_loss_from_residual
    from sde_rf_wan.turbo_codebook import TurboBitstream

    root = a.results_root
    clean_dir = root / a.clean_run / "Jockey" / "gop0"
    clean_trace = clean_dir / "input_attack_trace.npz"
    asset = a.assets_root / f"{a.tag}.npz"
    random_asset = a.assets_root / f"{a.tag}_matched.npz"
    metadata = asset.with_suffix(".json")
    runs = {label: root / f"{a.tag}_{label}" / "Jockey" / "gop0"
            for label in ("random", "candidate")}
    required = (a.data, a.vae, clean_trace, clean_dir / "codebook.tdcm", asset, random_asset,
                metadata, *(directory / "input_attack_trace.npz" for directory in runs.values()))
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        p.error("Missing prerequisites: " + ", ".join(missing))
    if not torch.cuda.is_available():
        p.error("CUDA is required for VAE proxy replay and bank scan")
    if a.position != 3:
        p.error("This pilot's saved assets target latent position 3")
    info = json.loads(metadata.read_text(encoding="utf-8"))
    if info.get("proxy_geometry") != "native":
        p.error("This audit accepts native-geometry proxy assets only")
    for key, expected in (("candidate_count", a.candidate_count), ("noise_coeff", a.noise_coeff),
                          ("opt_height", a.opt_height), ("opt_width", a.opt_width)):
        if info.get(key) != expected:
            p.error(f"Metadata {key}={info.get(key)} differs from audit setting {expected}")
    stream = TurboBitstream.load(str(clean_dir / "codebook.tdcm"))
    with np.load(clean_trace, allow_pickle=False) as trace:
        clean_residual = torch.from_numpy(trace["residual_step1"][0].copy()).to("cuda:0")
    atoms, indices = candidate_bank(stream, clean_residual[:, a.position], 1, a.position,
                                    a.candidate_count, "cuda:0", return_indices=True)
    frames = resize_frames(load_yuv420_frames(str(a.data), 33), 1280, 720)
    if len(frames) != 33:
        p.error("The source has fewer than 33 frames")
    original = np.stack([np.asarray(frame) for frame in frames])
    del frames
    device = torch.device("cuda:0")
    vae = WanVAE(vae_pth=str(a.vae), dtype=torch.bfloat16, device=device)
    low_shape = (a.opt_height // 8, a.opt_width // 8)

    def encode_pixels(pixels):
        rgb = torch.from_numpy(pixels).permute(3, 0, 1, 2).to(device).float() / 255
        small = resize_video(rgb, (a.opt_height, a.opt_width))
        with torch.no_grad():
            return vae.encode([small * 2 - 1])[0].float()

    clean_latent = encode_pixels(original)
    def proxy_score(pixels):
        encoded = encode_pixels(pixels)
        shift = F.interpolate((encoded - clean_latent).permute(1, 0, 2, 3),
                              size=stream["frame_shape"][1:], mode="bilinear",
                              align_corners=False).permute(1, 0, 2, 3)
        residual = clean_residual[:, a.position] + shift[:, a.position]
        with torch.no_grad():
            score = input_codebook_loss_from_residual(residual, atoms, stream["M"], a.noise_coeff)
            projection = atoms @ residual.flatten()
            selected_offsets = projection.abs().topk(stream["M"]).indices.tolist()
        selected = {indices[i] for i in selected_offsets}
        clean_selected = set(stream["step_data"][0][a.position][0])
        return {"uncovered_ratio": float(score),
                "clean_topM_overlap_in_pool": len(selected & clean_selected),
                "residual_MSE": float(residual.square().mean())}

    report = {"tag": a.tag, "history": history_summary(info["objective_history"]),
              "limitation": "Proxy replays a fixed clean Wan trajectory and top-N pool; it does not predict final video quality."}
    report["proxy_clean"] = proxy_score(original)
    for label, attack_asset in (("random", random_asset), ("candidate", asset)):
        pixels = quantize_asset(original, attack_asset)
        report[f"proxy_{label}"] = proxy_score(pixels)
        report[f"real_{label}"] = real_encoder_cell(runs[label] / "input_attack_trace.npz", a.position)
        del pixels
    report["paired_deltas"] = {
        "proxy_uncovered_candidate_minus_random": (report["proxy_candidate"]["uncovered_ratio"] -
                                                   report["proxy_random"]["uncovered_ratio"]),
        "real_hard_uncovered_candidate_minus_random": (report["real_candidate"]["hard_uncovered_ratio"] -
                                                       report["real_random"]["hard_uncovered_ratio"]),
    }
    deltas = report["paired_deltas"]
    report["triage"] = classify_pair(deltas["proxy_uncovered_candidate_minus_random"],
                                     deltas["real_hard_uncovered_candidate_minus_random"])
    report["hard_score_note"] = (
        "The hard score subtracts the real first-step codebook correction from the real encoder residual. "
        "It is a cell-level diagnostic, not decoded-video distortion or a causal decomposition."
    )
    output = a.output or root / f"{a.tag}_proxy_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
