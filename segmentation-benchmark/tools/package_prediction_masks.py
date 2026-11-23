"""Package EMSANet prediction masks into the same npz layout as the ground truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepared-root",
        type=Path,
        default=Path("prepared_photo_masks"),
        help="Dataset bundle created by prepare_photo_masks_dataset.py.",
    )
    parser.add_argument(
        "--semantic-dir",
        type=Path,
        default=None,
        help="Directory with raw semantic id PNGs named <sample_id>.png.",
    )
    parser.add_argument(
        "--instance-dir",
        type=Path,
        default=None,
        help="Directory with raw instance id PNGs named <sample_id>.png.",
    )
    parser.add_argument(
        "--output-subdir",
        default="normalized",
        help="Subdirectory under prepared-root/predictions for normalized outputs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing normalized prediction outputs.",
    )
    return parser.parse_args()


def read_manifest(prepared_root: Path) -> dict[str, Any]:
    manifest_path = prepared_root / "metadata" / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_id_png(path: Path) -> np.ndarray:
    image = Image.open(path)
    array = np.asarray(image)
    if array.ndim == 3:
        raise ValueError(
            f"{path} looks like an RGB visualization. Please provide raw/id masks instead."
        )
    return array.astype(np.uint16, copy=False)


def save_id_png(path: Path, array: np.ndarray) -> None:
    if array.max(initial=0) <= np.iinfo(np.uint8).max:
        Image.fromarray(array.astype(np.uint8)).save(path)
    else:
        Image.fromarray(array.astype(np.uint16)).save(path)


def main() -> int:
    args = parse_args()
    prepared_root = args.prepared_root.resolve()
    manifest = read_manifest(prepared_root)

    if args.semantic_dir is None and args.instance_dir is None:
        raise SystemExit("Provide --semantic-dir, --instance-dir, or both.")

    output_root = prepared_root / "predictions" / args.output_subdir
    semantic_out = output_root / "semantic_id"
    instance_out = output_root / "instance_id"
    npz_out = output_root / "semantic_instance_npz"
    for directory in (semantic_out, instance_out, npz_out):
        directory.mkdir(parents=True, exist_ok=True)

    packaged = 0
    for sample in manifest["samples"]:
        sample_id = sample["sample_id"]
        gt_shape = (int(sample["height"]), int(sample["width"]))
        semantic = None
        instance = None

        if args.semantic_dir is not None:
            semantic_path = args.semantic_dir / f"{sample_id}.png"
            if not semantic_path.exists():
                raise FileNotFoundError(semantic_path)
            semantic = read_id_png(semantic_path)
            if semantic.shape != gt_shape:
                raise ValueError(f"{semantic_path} shape {semantic.shape} != {gt_shape}")

        if args.instance_dir is not None:
            instance_path = args.instance_dir / f"{sample_id}.png"
            if not instance_path.exists():
                raise FileNotFoundError(instance_path)
            instance = read_id_png(instance_path)
            if instance.shape != gt_shape:
                raise ValueError(f"{instance_path} shape {instance.shape} != {gt_shape}")

        if semantic is None:
            semantic = np.zeros(gt_shape, dtype=np.uint16)
        if instance is None:
            instance = np.zeros(gt_shape, dtype=np.uint16)

        semantic_path_out = semantic_out / f"{sample_id}.png"
        instance_path_out = instance_out / f"{sample_id}.png"
        npz_path = npz_out / f"{sample_id}.npz"
        if not args.overwrite and any(
            path.exists() for path in (semantic_path_out, instance_path_out, npz_path)
        ):
            raise FileExistsError(
                f"Output for {sample_id} already exists. Use --overwrite to refresh."
            )

        save_id_png(semantic_path_out, semantic)
        save_id_png(instance_path_out, instance)

        instance_ids = [int(v) for v in np.unique(instance) if v != 0]
        instance_semantic_ids = np.zeros(max(instance_ids, default=0) + 1, dtype=np.uint16)
        for instance_id in instance_ids:
            semantic_values, counts = np.unique(semantic[instance == instance_id], return_counts=True)
            if len(semantic_values):
                instance_semantic_ids[instance_id] = int(semantic_values[counts.argmax()])

        np.savez_compressed(
            npz_path,
            semantic=semantic,
            instance=instance,
            instance_semantic_ids=instance_semantic_ids,
            sample_id=np.asarray(sample_id),
        )
        packaged += 1

    print(f"Packaged {packaged} prediction samples into {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
