from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def read_rgb(path: Path, max_side: int = 0):
    with Image.open(path) as source:
        raw_size = source.size
        orientation = source.getexif().get(274, 1)
        upright = ImageOps.exif_transpose(source).convert("RGB")
    original_size = upright.size
    processed = upright.copy()
    if max_side > 0 and max(original_size) > max_side:
        ratio = max_side / max(original_size)
        size = tuple(max(1, round(x * ratio)) for x in original_size)
        processed = upright.resize(size, Image.Resampling.LANCZOS)
    return upright, np.asarray(processed), {
        "raw_file_size": list(raw_size), "exif_orientation": orientation,
        "original_size": list(original_size), "processing_size": list(processed.size),
        "coordinate_frame": "EXIF-upright RGB, x right, y down",
    }


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_json(path: Path, value):
    path.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def save_mask(path: Path, mask: np.ndarray):
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255).save(path)
