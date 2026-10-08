"""Defense preprocessing utilities for FLF2V experiments."""

import io
from PIL import Image, ImageFilter, ImageOps

DEFENSE_CHOICES = [
    "none",
    "jpeg",
    "median",
    "jpeg-median",
    "hflip",
]


def apply_defense(frames, defense, jpeg_quality=85, median_size=3):
    if defense == "none":
        return list(frames), {"defense": "none"}

    defended = list(frames)
    metadata = {
        "defense": defense,
        "jpeg_quality": jpeg_quality,
        "median_size": median_size,
    }

    # Training-free, invertible input-transform baseline.  The same transform
    # is applied to every frame in a GOP, including FLF2V's two boundary
    # frames, so the video condition remains temporally consistent.  The
    # runner mirrors decoded frames back before it saves/evaluates them.
    # This is a single transformed route, not the paper's full two-way
    # "encode both routes then choose" method.
    if defense == "hflip":
        defended = [ImageOps.mirror(frame) for frame in defended]
        metadata["inverse_after_decode"] = "hflip"
        metadata["method_note"] = (
            "Single-route invertible horizontal-flip baseline inspired by "
            "training-free input randomization; not two-way selection."
        )
        return defended, metadata

    if defense in ("jpeg", "jpeg-median"):
        jpeg_frames = []
        for frame in defended:
            buf = io.BytesIO()
            frame.save(buf, format="JPEG", quality=jpeg_quality)
            buf.seek(0)
            jpeg_frames.append(Image.open(buf).convert("RGB"))
        defended = jpeg_frames

    if defense in ("median", "jpeg-median"):
        defended = [frame.filter(ImageFilter.MedianFilter(size=median_size)) for frame in defended]

    return defended, metadata
