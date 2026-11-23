"""Prepare Labelme photo masks for EMSANet inference and mIoU evaluation."""

from __future__ import annotations

from workspace_paths import record_path

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps


IMAGE_EXTS = (".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG", ".bmp", ".BMP")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path.cwd(),
        help="Root containing Labelme JSON files and images.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("prepared_photo_masks"),
        help="Output dataset bundle directory.",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional ordered class names. If omitted, labels are discovered and sorted.",
    )
    parser.add_argument(
        "--create-dummy-depth",
        action="store_true",
        help="Create all-zero uint16 depth PNGs for EMSANet RGB-D sample loaders.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files in the output bundle.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def is_labelme_annotation(path: Path) -> bool:
    try:
        data = read_json(path)
    except Exception:
        return False
    return isinstance(data.get("shapes"), list)


def discover_annotations(input_root: Path, output_root: Path) -> list[Path]:
    input_root = input_root.resolve()
    output_root = output_root.resolve()
    annotations: list[Path] = []
    for path in sorted(input_root.rglob("*.json")):
        try:
            path.resolve().relative_to(output_root)
            continue
        except ValueError:
            pass
        if is_labelme_annotation(path):
            annotations.append(path)
    return annotations


def find_image(json_path: Path, data: dict[str, Any]) -> Path:
    candidates: list[Path] = []
    image_path = data.get("imagePath")
    if image_path:
        candidates.append(json_path.parent / image_path)
    candidates.extend(json_path.with_suffix(ext) for ext in IMAGE_EXTS)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"No image found for {json_path}")


def slugify(value: str) -> str:
    value = value.replace("\\", "/")
    value = re.sub(r"[^0-9A-Za-z]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "sample"


def make_sample_id(json_path: Path, input_root: Path, used: set[str]) -> str:
    rel = json_path.resolve().relative_to(input_root.resolve())
    parts = list(rel.parts)
    stem = json_path.stem

    lower_parts = [p.lower() for p in parts]
    if "photo and json" in lower_parts:
        idx = lower_parts.index("photo and json")
        context = parts[max(0, idx - 2) : idx]
        base = slugify("_".join([*context, stem]))
    else:
        base = slugify(str(rel.with_suffix("")))

    sample_id = base
    if sample_id in used:
        digest = hashlib.sha1(str(rel).encode("utf-8")).hexdigest()[:8]
        sample_id = f"{base}_{digest}"
    used.add(sample_id)
    return sample_id


def class_mapping(annotations: list[Path], explicit_labels: list[str] | None) -> dict[str, int]:
    if explicit_labels:
        labels = explicit_labels
    else:
        found: set[str] = set()
        for path in annotations:
            data = read_json(path)
            for shape in data.get("shapes", []):
                label = str(shape.get("label", "")).strip()
                if label:
                    found.add(label)
        labels = sorted(found)

    return {name: idx + 1 for idx, name in enumerate(labels)}


def draw_shape_mask(shape: dict[str, Any], width: int, height: int) -> np.ndarray:
    points = shape.get("points") or []
    shape_type = shape.get("shape_type") or "polygon"
    image = Image.new("1", (width, height), 0)
    draw = ImageDraw.Draw(image)

    if shape_type == "rectangle" and len(points) >= 2:
        (x0, y0), (x1, y1) = points[:2]
        box = [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
        draw.rectangle(box, fill=1)
    elif len(points) >= 3:
        polygon = [(float(x), float(y)) for x, y in points]
        draw.polygon(polygon, fill=1)
    else:
        return np.zeros((height, width), dtype=bool)

    return np.asarray(image, dtype=bool)


def encode_annotation(
    data: dict[str, Any],
    width: int,
    height: int,
    label_to_id: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    semantic = np.zeros((height, width), dtype=np.uint16)
    raw_instance = np.zeros((height, width), dtype=np.uint16)
    raw_label_ids = [0]
    raw_labels = [""]

    for raw_id, shape in enumerate(data.get("shapes", []), start=1):
        label = str(shape.get("label", "")).strip()
        if not label:
            continue
        if label not in label_to_id:
            raise KeyError(f"Label {label!r} is not in label mapping")

        mask = draw_shape_mask(shape, width, height)
        if not mask.any():
            raw_label_ids.append(label_to_id[label])
            raw_labels.append(label)
            continue

        semantic[mask] = label_to_id[label]
        raw_instance[mask] = raw_id
        raw_label_ids.append(label_to_id[label])
        raw_labels.append(label)

    visible_raw_ids = [int(v) for v in np.unique(raw_instance) if v != 0]
    instance = np.zeros_like(raw_instance)
    instances: list[dict[str, Any]] = []

    for dense_id, raw_id in enumerate(visible_raw_ids, start=1):
        mask = raw_instance == raw_id
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue
        instance[mask] = dense_id
        instances.append(
            {
                "instance_id": dense_id,
                "source_shape_index": raw_id - 1,
                "label": raw_labels[raw_id],
                "semantic_id": int(raw_label_ids[raw_id]),
                "area": int(mask.sum()),
                "bbox_xyxy": [
                    int(xs.min()),
                    int(ys.min()),
                    int(xs.max()),
                    int(ys.max()),
                ],
            }
        )

    return semantic, instance, instances


def save_png_id_mask(path: Path, array: np.ndarray) -> None:
    if array.max(initial=0) <= np.iinfo(np.uint8).max:
        Image.fromarray(array.astype(np.uint8)).save(path)
    else:
        Image.fromarray(array.astype(np.uint16)).save(path)


def choose_rgb_for_annotation(image: Image.Image, data: dict[str, Any], source: Path) -> Image.Image:
    raw_rgb = image.convert("RGB")
    oriented_rgb = ImageOps.exif_transpose(image).convert("RGB")
    declared_width = data.get("imageWidth")
    declared_height = data.get("imageHeight")

    if declared_width and declared_height:
        declared_size = (int(declared_width), int(declared_height))
        if oriented_rgb.size == declared_size:
            return oriented_rgb
        if raw_rgb.size == declared_size:
            return raw_rgb
        raise ValueError(
            f"Image size mismatch for {source}: "
            f"JSON={declared_size[0]}x{declared_size[1]}, "
            f"raw={raw_rgb.size[0]}x{raw_rgb.size[1]}, "
            f"exif_transposed={oriented_rgb.size[0]}x{oriented_rgb.size[1]}"
        )

    return oriented_rgb


def ensure_empty_or_overwrite(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"{path} already exists and is not empty. Use --overwrite to refresh outputs."
        )
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> int:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()

    annotations = discover_annotations(input_root, output_root)
    if not annotations:
        raise SystemExit(f"No Labelme JSON annotations found under {input_root}")

    label_to_id = class_mapping(annotations, args.labels)
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    ensure_empty_or_overwrite(output_root, args.overwrite)
    dirs = {
        "rgb": output_root / "emsanet_samples" / "rgb",
        "depth": output_root / "emsanet_samples" / "depth",
        "semantic": output_root / "gt" / "semantic_id",
        "instance": output_root / "gt" / "instance_id",
        "npz": output_root / "gt" / "semantic_instance_npz",
        "instances": output_root / "gt" / "instances_json",
        "metadata": output_root / "metadata",
        "splits": output_root / "splits",
        "predictions": output_root / "predictions",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    classes = [{"id": 0, "name": "background", "isthing": False}]
    classes.extend(
        {"id": idx, "name": id_to_label[idx], "isthing": True}
        for idx in sorted(id_to_label)
    )
    write_json(dirs["metadata"] / "classes.json", classes)

    used_ids: set[str] = set()
    samples: list[dict[str, Any]] = []
    label_counter: Counter[str] = Counter()

    for json_path in annotations:
        data = read_json(json_path)
        image_path = find_image(json_path, data)
        sample_id = make_sample_id(json_path, input_root, used_ids)

        with Image.open(image_path) as image:
            rgb = choose_rgb_for_annotation(image, data, json_path)
            width, height = rgb.size

            rgb_path = dirs["rgb"] / f"{sample_id}.png"
            if args.overwrite or not rgb_path.exists():
                rgb.save(rgb_path)

        semantic, instance, instances = encode_annotation(data, width, height, label_to_id)
        label_counter.update(item["label"] for item in instances)

        semantic_path = dirs["semantic"] / f"{sample_id}.png"
        instance_path = dirs["instance"] / f"{sample_id}.png"
        npz_path = dirs["npz"] / f"{sample_id}.npz"
        instances_path = dirs["instances"] / f"{sample_id}.json"

        if args.overwrite or not semantic_path.exists():
            save_png_id_mask(semantic_path, semantic)
        if args.overwrite or not instance_path.exists():
            save_png_id_mask(instance_path, instance)

        instance_semantic_ids = np.zeros(len(instances) + 1, dtype=np.uint16)
        for item in instances:
            instance_semantic_ids[item["instance_id"]] = item["semantic_id"]

        if args.overwrite or not npz_path.exists():
            np.savez_compressed(
                npz_path,
                semantic=semantic,
                instance=instance,
                instance_semantic_ids=instance_semantic_ids,
                sample_id=np.asarray(sample_id),
            )

        write_json(
            instances_path,
            {
                "sample_id": sample_id,
                "source_json": record_path(json_path),
                "source_image": record_path(image_path),
                "width": width,
                "height": height,
                "instances": instances,
            },
        )

        depth_path = None
        if args.create_dummy_depth:
            depth_path = dirs["depth"] / f"{sample_id}.png"
            if args.overwrite or not depth_path.exists():
                Image.fromarray(np.zeros((height, width), dtype=np.uint16)).save(depth_path)

        samples.append(
            {
                "sample_id": sample_id,
                "width": width,
                "height": height,
                "source_json": record_path(json_path),
                "source_image": record_path(image_path),
                "rgb_path": record_path(rgb_path),
                "depth_path": record_path(depth_path) if depth_path else None,
                "semantic_path": record_path(semantic_path),
                "instance_path": record_path(instance_path),
                "semantic_instance_npz": record_path(npz_path),
                "instances_json": record_path(instances_path),
                "num_instances": len(instances),
                "labels": sorted({item["label"] for item in instances}),
            }
        )

    manifest = {
        "source_root": record_path(input_root),
        "output_root": record_path(output_root),
        "num_samples": len(samples),
        "classes": classes,
        "dummy_depth_created": bool(args.create_dummy_depth),
        "samples": samples,
    }
    write_json(dirs["metadata"] / "manifest.json", manifest)

    with (dirs["metadata"] / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "width",
                "height",
                "num_instances",
                "rgb_path",
                "depth_path",
                "semantic_path",
                "instance_path",
                "semantic_instance_npz",
            ],
        )
        writer.writeheader()
        for sample in samples:
            writer.writerow({name: sample[name] for name in writer.fieldnames})

    with (dirs["splits"] / "all.txt").open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(sample["sample_id"] + "\n")

    print(f"Prepared {len(samples)} samples in {output_root}")
    print("Class mapping:", ", ".join(f"{k}={v}" for k, v in label_to_id.items()))
    print("Visible instances by label:", dict(sorted(label_counter.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
