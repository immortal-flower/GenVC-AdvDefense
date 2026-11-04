"""Torch-free helpers for locating serialized Turbo-DDCM payload fields."""

import math
import struct


def combinatorial_index_bits(k, m):
    """Return the fixed width needed for the rank of an unordered subset."""
    if k < 0 or m < 0 or m > k:
        raise ValueError(f"Expected 0 <= M <= K, got K={k}, M={m}")
    return (math.comb(k, m) - 1).bit_length()


def unrank_combination(rank, k, m):
    """Decode a colex combinatorial rank without importing PyTorch."""
    total = math.comb(k, m)
    if rank < 0 or rank >= total:
        raise ValueError(f"Combination rank {rank} outside [0, {total})")
    result = [0] * m
    upper = k
    remainder = rank
    for order in range(m, 0, -1):
        lo, hi = order - 1, upper - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if math.comb(mid, order) <= remainder:
                lo = mid + 1
            else:
                hi = mid - 1
        result[order - 1] = hi
        remainder -= math.comb(hi, order)
        upper = hi
    return result


def parse_tdcm_payload_layout(raw):
    """Locate every index and packed sign bit in a TDCM or TDC2 stream.

    This module intentionally depends only on the Python standard library so
    offline transport studies do not initialize PyTorch's OpenMP runtime.
    """
    magic = bytes(raw[:4])
    if magic not in (b"TDCM", b"TDC2"):
        raise ValueError(f"Invalid TDCM magic: {bytes(raw[:4])!r}")
    if len(raw) < 48:
        raise ValueError("Truncated TDCM header")
    k, m, n_sde, n_ddim, n_lat, seed, ndim, n_fr, height, width = \
        struct.unpack_from("<10I", raw, 4)
    off = 44
    prompt_len = struct.unpack_from("<I", raw, off)[0]
    off += 4 + 4 * ndim + prompt_len
    if off > len(raw):
        raise ValueError("Truncated TDCM shape or prompt")

    index_size = 2 if k <= 65536 else 4
    index_format = "<H" if index_size == 2 else "<I"
    slots = []
    rank_entries = []
    frame_indices = {}
    payload_bits = 0
    for step in range(n_sde):
        for frame in range(n_lat):
            if off + 2 > len(raw):
                raise ValueError("Truncated TDCM payload before M_actual")
            m_actual = struct.unpack_from("<H", raw, off)[0]
            off += 2
            indices_offset = off
            if magic == b"TDC2":
                index_bits = combinatorial_index_bits(k, m_actual)
                indices_bytes = (index_bits + 7) // 8
            else:
                index_bits = m_actual * max(1, math.ceil(math.log2(k)))
                indices_bytes = m_actual * index_size
            if off + indices_bytes > len(raw):
                raise ValueError("Truncated TDCM index payload")

            if magic == b"TDC2":
                rank = int.from_bytes(raw[off:off + indices_bytes], "little")
                indices = unrank_combination(rank, k, m_actual)
                rank_entries.append({
                    "step": step, "frame": frame, "m_actual": m_actual,
                    "rank": rank, "rank_offset": indices_offset,
                    "rank_bits": index_bits, "rank_bytes": indices_bytes,
                    "indices": list(indices),
                })
            else:
                indices = [
                    struct.unpack_from(
                        index_format, raw, off + atom * index_size)[0]
                    for atom in range(m_actual)
                ]

            signs_offset = indices_offset + indices_bytes
            signs_bytes = (m_actual + 7) // 8
            if signs_offset + signs_bytes > len(raw):
                raise ValueError("Truncated TDCM index/sign payload")
            key = (step, frame)
            values = []
            for atom, index in enumerate(indices):
                if index >= k:
                    raise ValueError(f"Invalid codebook index {index} >= K={k}")
                values.append(index)
                slots.append({
                    "step": step, "frame": frame, "atom": atom,
                    "index": index,
                    "index_offset": (indices_offset + atom * index_size
                                     if magic == b"TDCM" else None),
                    "index_size": index_size,
                    "sign_offset": signs_offset + atom // 8,
                    "sign_mask": 1 << (atom % 8), "frame_key": key,
                })
            frame_indices[key] = set(values)
            payload_bits += index_bits + m_actual
            off = signs_offset + signs_bytes

    if off != len(raw):
        raise ValueError(f"Unexpected {len(raw) - off} trailing TDCM bytes")
    return {
        "K": k, "M": m, "n_sde": n_sde, "n_ddim": n_ddim,
        "n_lat": n_lat, "seed": seed, "num_frames": n_fr,
        "height": height, "width": width, "slots": slots,
        "frame_indices": frame_indices, "rank_entries": rank_entries,
        "bits_per_index": max(1, math.ceil(math.log2(k))),
        "index_coding": ("combinatorial" if magic == b"TDC2"
                         else "fixed-width"),
        "payload_bits": payload_bits,
    }
