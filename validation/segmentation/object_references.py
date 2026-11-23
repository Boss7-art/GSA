import argparse
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from src.utils import save_json, save_mask
from common import sha256

LABELS = {"advertising_sign": "F", "vegetation": "G", "artificial_light": "H"}

def normalized_label(shape: dict) -> str:
    return str(shape.get("label", "")).strip().upper()


def explicit_hole(shape: dict) -> bool:
    flags = shape.get("flags") or {}
    return any(flags.get(key) is True for key in ("hole", "is_hole"))


def rasterize_shape(size_wh: tuple[int, int], shape: dict) -> np.ndarray:
    layer = Image.new("L", size_wh, 0)
    draw = ImageDraw.Draw(layer)
    points = [tuple(float(value) for value in point) for point in shape["points"]]
    shape_type = shape.get("shape_type") or "polygon"
    if shape_type == "polygon":
        draw.polygon(points, fill=255, outline=255)
    elif shape_type == "rectangle":
        (x0, y0), (x1, y1) = points
        draw.rectangle([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)], fill=255, outline=255)
    elif shape_type == "circle":
        (cx, cy), (px, py) = points
        radius = math.hypot(px - cx, py - cy)
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=255, outline=255)
    else:
        raise ValueError(f"Unsupported shape type: {shape_type}")
    return np.asarray(layer) != 0


def rasterize_category(size_wh: tuple[int, int], shapes: list[dict]) -> np.ndarray:
    width, height = size_wh
    positive = np.zeros((height, width), dtype=bool)
    holes = np.zeros_like(positive)
    for shape in shapes:
        target = holes if explicit_hole(shape) else positive
        target |= rasterize_shape(size_wh, shape)
    return positive & ~holes

def main():
    parser = argparse.ArgumentParser(description="Rasterize LabelMe object references on the upright photograph grid.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("The output directory must be empty.")
    annotation = json.loads(args.annotation.read_text(encoding="utf-8-sig"))
    with Image.open(args.image) as image:
        size = ImageOps.exif_transpose(image).size
    if size != (annotation["imageWidth"], annotation["imageHeight"]):
        raise ValueError("Annotation and upright photograph dimensions differ.")
    masks, counts = {}, {}
    for name, label in LABELS.items():
        shapes = [s for s in annotation["shapes"] if normalized_label(s) == label]
        for shape in shapes:
            points = np.asarray(shape["points"], dtype=float)
            kind = shape.get("shape_type") or "polygon"
            if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
                raise ValueError(f"Invalid {name} coordinates.")
            if (kind == "polygon" and len(points) < 3) or (kind in ("rectangle", "circle") and len(points) != 2):
                raise ValueError(f"Invalid {name} vertex count.")
        masks[name] = rasterize_category(size, shapes)
        counts[name] = {"shapes": len(shapes), "pixels": int(masks[name].sum())}
    args.output.mkdir(parents=True, exist_ok=True)
    for name, mask in masks.items():
        save_mask(args.output / f"{name}_mask.png", mask)
    save_json(args.output / "result.json", {"method": "manual_reference", "image": args.image.name,
              "image_sha256": sha256(args.image), "annotation_sha256": sha256(args.annotation), "classes": counts})


if __name__ == "__main__":
    main()
