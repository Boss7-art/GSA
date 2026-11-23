import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from src.utils import read_rgb, save_mask, save_json
from src.wall_region_refinement import rasterize_polygons, refine_wall_regions
from common import sha256

LABELS = ("wall_transparent", "wall_solid", "wall_uncertain", "foreground_occlusion")

def read_mask(path, shape):
    with Image.open(path) as source:
        array = np.asarray(source)
        if source.mode != "L" or array.shape != shape or not set(np.unique(array)) <= {0, 255}:
            raise ValueError(f"Expected native-size L-mode 0/255 mask: {path}")
    return array != 0


def canonical_polygon(points, shape):
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2 or len(array) < 3 or not np.isfinite(array).all():
        raise ValueError("Invalid review polygon vertices")
    h, w = shape
    tolerance = 1e-6
    if np.any(array < -tolerance) or np.any(array[:, 0] > w + tolerance) or np.any(array[:, 1] > h + tolerance):
        raise ValueError("Review polygon must stay inside the native image canvas")
    canonical = np.clip(array, [0, 0], [w - 1, h - 1])
    keep = [canonical[0]]
    for vertex in canonical[1:]:
        if np.linalg.norm(vertex - keep[-1]) >= 0.5:
            keep.append(vertex)
    if len(keep) > 3 and np.linalg.norm(keep[0] - keep[-1]) < 0.5:
        keep.pop()
    validated = np.asarray(keep)
    rasterize_polygons(shape, [validated])
    return validated.tolist()


def resolve_input_path(value, parent):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (parent / path).resolve()

def export_one(review_path, output):
    if output.exists() and any(output.iterdir()):
        raise ValueError("The output directory must be empty.")
    data = json.loads(review_path.read_text(encoding="utf-8"))
    photo = resolve_input_path(data["source_photo"], review_path.parent)
    photo_hash = sha256(photo)
    source_annotation = photo.with_suffix(".json")
    annotation_hash = sha256(source_annotation) if source_annotation.is_file() else None
    declared = data.get("source_input_sha256", {})
    for path, digest in ((photo, photo_hash), (source_annotation, annotation_hash)):
        if path.name in declared and declared[path.name] != digest:
            raise ValueError(f"Source input differs from reviewed file: {path}")
    original, rgb, _ = read_rgb(photo, 0)
    shape = rgb.shape[:2]
    if data["image_size"] != list(original.size):
        raise ValueError("Review and EXIF-upright native photo coordinates disagree")
    if "transparent_regions_complete" not in data:
        raise ValueError("Review must explicitly state transparent-region completeness")
    domain_path = resolve_input_path(data["wall_domain_mask"], review_path.parent)
    domain = read_mask(domain_path, shape)
    groups = {label: [] for label in LABELS}
    for region in data["polygons"]:
        label = region["label"]
        if label not in groups:
            raise ValueError(f"Unrecognized reviewed region label: {label}")
        groups[label].append(canonical_polygon(region["points"], shape))
    masks = {label: rasterize_polygons(shape, polygons) for label, polygons in groups.items()}
    transparent = masks["wall_transparent"] & ~masks["wall_solid"]
    solid = masks["wall_solid"].copy()
    if data.get("solid_seed_mask"):
        seed_path = resolve_input_path(data["solid_seed_mask"], review_path.parent)
        seed = read_mask(seed_path, shape)
        solid |= seed & ~transparent
    solid &= ~transparent
    result = refine_wall_regions(
        rgb, domain, transparent, solid_seed=solid,
        uncertain_seed=masks["wall_uncertain"],
        foreground_occlusion=masks["foreground_occlusion"],
        transparent_regions_complete=data["transparent_regions_complete"],
        config={"maximum_boundary_shift_pixels": 0}, provenance={"method": "human_review"},
    )
    output.mkdir(parents=True, exist_ok=True)
    for label in ("wall_transparent", "wall_solid", "wall_uncertain", "wall_domain", "other"):
        save_mask(output / f"{label}_mask.png", result[label])
    save_mask(output / "foreground_occlusion_reviewed_mask.png", masks["foreground_occlusion"])

    save_json(output / "result.json", {
        "method": "human_review", "image": photo.name, "image_sha256": photo_hash,
        "review_sha256": sha256(review_path), "domain_sha256": sha256(domain_path),
        "image_size": list(original.size), "transparent_regions_complete": data["transparent_regions_complete"],
        "class_pixels": {key: int(result[key].sum()) for key in ("wall_transparent", "wall_solid", "wall_uncertain", "other")},
    })


def main():
    parser = argparse.ArgumentParser(description="Rasterize reviewed facade regions.")
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export_one(args.review, args.output)


if __name__ == "__main__":
    main()
