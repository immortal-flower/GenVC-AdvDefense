"""Deterministic provenance records for formal GVCC experiments.

The exploratory runners historically identified a result only by a readable
directory name.  That is convenient during development, but two different
configurations can then share the same directory and an old ``metrics.json``
may be mistaken for a completed new run.  This module builds a compact,
machine-checkable configuration fingerprint without depending on absolute
machine-specific paths.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


PROVENANCE_SCHEMA = "gvcc-validation-provenance-v1"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_file_range(path: str | Path, offset: int, length: int,
                      chunk_size: int = 1024 * 1024) -> str:
    if offset < 0 or length < 0:
        raise ValueError("File-range offset and length must be non-negative")
    digest = hashlib.sha256()
    remaining = int(length)
    with Path(path).open("rb") as handle:
        handle.seek(offset)
        while remaining:
            block = handle.read(min(chunk_size, remaining))
            if not block:
                raise ValueError(
                    f"Requested byte range exceeds input file: {path}")
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def canonical_json(value) -> str:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def configuration_id(config: dict, length: int = 12) -> str:
    if length < 8:
        raise ValueError("Configuration identifiers must use at least 8 hex digits")
    payload = dict(config)
    payload.pop("configuration_id", None)
    return _sha256_bytes(canonical_json(payload).encode("utf-8"))[:length]


def _job_attack_config(job) -> dict:
    config = {
        "role": str(job.role),
        "requested_channel_flips": int(job.count),
        "attack_seed": int(job.attack_seed),
        "sde_step": str(job.step),
        "latent_position": str(job.frame),
    }
    if hasattr(job, "cells"):
        config.update({
            "cells_zero_based": [list(cell) for cell in job.cells],
            "repetitions": int(job.repetitions),
            "disturbance_mode": str(job.disturbance_mode),
            "interleave_seed": int(job.interleave_seed),
            "burst_start": int(job.burst_start),
        })
    else:
        config["cells_zero_based"] = [[int(job.step) - 1, int(job.frame)]]
        config["repetitions"] = 1
        config["disturbance_mode"] = "random"
        config["interleave_seed"] = 0
        config["burst_start"] = 0
    if hasattr(job, "policy"):
        config.update({
            "protection_policy": str(job.policy),
            "protected_signs": int(job.protected_signs),
            "policy_seed": int(job.policy_seed),
            "channel_model": "uniform-sign-bsc",
        })
    return config


def build_shared_validation_provenance(args) -> dict:
    """Fingerprint all non-job inputs used by the persistent validation runner."""
    data_path = Path(args.data)
    codebook_path = Path(args.clean_codebook)
    metrics_path = Path(args.clean_metrics)
    for path in (data_path, codebook_path, metrics_path):
        if not path.is_file():
            raise FileNotFoundError(f"Formal-run input not found: {path}")

    # YUV420 uses 1.5 bytes per pixel.  Hash only the exact source GOP, which
    # is both cheaper than hashing a multi-GOP file and stricter than a path.
    pixels = int(args.height) * int(args.width)
    if pixels % 2:
        raise ValueError("YUV420 dimensions must contain an even pixel count")
    bytes_per_frame = pixels * 3 // 2
    source_start_frame = int(args.gop_index) * (int(args.num_frames) - 1)
    segment_offset = source_start_frame * bytes_per_frame
    segment_bytes = int(args.num_frames) * bytes_per_frame

    return {
        "schema": PROVENANCE_SCHEMA,
        "experiment_family": "persistent-codebook-validation",
        "sequence": str(args.sequence),
        "gop_index": int(args.gop_index),
        "source": {
            "file_name": data_path.name,
            "file_size_bytes": data_path.stat().st_size,
            "source_start_frame": source_start_frame,
            "num_frames": int(args.num_frames),
            "gop_segment_sha256": sha256_file_range(
                data_path, segment_offset, segment_bytes),
        },
        "clean_inputs": {
            "codebook_file_name": codebook_path.name,
            "codebook_size_bytes": codebook_path.stat().st_size,
            "codebook_sha256": sha256_file(codebook_path),
            "clean_metrics_sha256": sha256_file(metrics_path),
        },
        "model": {
            "model_seed": int(args.model_seed),
            "checkpoint_name": Path(args.wan_ckpt).name,
            "K": int(args.K),
            "M": int(args.M),
            "steps": int(args.steps),
            "ddim_tail": int(args.ddim_tail),
            "g_scale": float(args.g_scale),
            "flow_shift": float(args.flow_shift),
            "height": int(args.height),
            "width": int(args.width),
        },
        "boundary_codec": {
            "name": str(args.ref_codec),
            "quality": int(args.ref_quality),
        },
    }


def build_job_provenance(shared: dict, job) -> dict:
    record = dict(shared)
    record["attack"] = _job_attack_config(job)
    record["configuration_id"] = configuration_id(record)
    return record


def formal_run_name(base_name: str, provenance: dict,
                    experiment_tag: str = "paper") -> str:
    tag = re.sub(r"[^a-z0-9_-]+", "_", experiment_tag.lower()).strip("_")
    if not tag:
        tag = "paper"
    return f"{base_name}_{tag}_cfg{provenance['configuration_id']}"


def metrics_matches_provenance(path: str | Path, provenance: dict) -> bool:
    path = Path(path)
    if not path.is_file():
        return False
    try:
        metrics = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError):
        return False
    found = metrics.get("provenance", {}).get("configuration_id")
    return found == provenance.get("configuration_id")
