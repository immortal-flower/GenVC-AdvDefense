"""Defense preprocessing utilities for FLF2V experiments."""

import io
from PIL import Image, ImageFilter

DEFENSE_CHOICES = [
    "none",
    "jpeg",
    "median",
    "jpeg-median",
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
