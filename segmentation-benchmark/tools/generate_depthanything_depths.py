"""Generate pseudo-depth PNGs for EMSANet using a Depth Anything model."""

from __future__ import annotations

from workspace_paths import record_path, resolve_reference

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


SUNRGBD_DEPTH_MEAN = 19025.14930492213
SUNRGBD_DEPTH_STD = 9880.916071806689
SUNRGBD_DEPTH_MIN = 1.0
SUNRGBD_DEPTH_MAX = 65528.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepared-root",
        type=Path,
        default=Path("prepared_photo_masks"),
        help="Dataset bundle created by prepare_photo_masks_dataset.py.",
    )
    parser.add_argument(
        "--model-id",
        default="depth-anything/Depth-Anything-V2-Small-hf",
        help="Hugging Face Depth Anything depth-estimation model id.",
    )
    parser.add_argument(
        "--output-prefix",
        default="depth_anything_v2_small_sunrgbd_stats",
        help="Prefix under prepared-root/emsanet_samples for generated depth dirs.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda", "mps"),
        help="Inference device. auto uses CUDA if the installed torch supports it.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=("direct", "inverted"),
        choices=("direct", "inverted"),
        help="Depth variants to write. Inverted is often useful for relative inverse-depth outputs.",
    )
    parser.add_argument("--target-mean", type=float, default=SUNRGBD_DEPTH_MEAN)
    parser.add_argument("--target-std", type=float, default=SUNRGBD_DEPTH_STD)
    parser.add_argument("--clip-min", type=float, default=SUNRGBD_DEPTH_MIN)
    parser.add_argument("--clip-max", type=float, default=SUNRGBD_DEPTH_MAX)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of samples to process.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def standardize_to_sunrgbd_units(
    relative_depth: np.ndarray,
    *,
    target_mean: float,
    target_std: float,
    clip_min: float,
    clip_max: float,
    inverted: bool,
) -> np.ndarray:
    depth = relative_depth.astype(np.float32, copy=False)
    if inverted:
        depth = -depth

    std = float(depth.std())
    if std < 1e-6:
        mapped = np.full_like(depth, fill_value=target_mean, dtype=np.float32)
    else:
        mapped = (depth - float(depth.mean())) / std
        mapped = mapped * target_std + target_mean

    mapped = np.clip(mapped, clip_min, clip_max)
    return np.rint(mapped).astype(np.uint16)


def save_u16_png(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array.astype(np.uint16)).save(path)


def main() -> int:
    args = parse_args()
    prepared_root = args.prepared_root.resolve()
    manifest = read_json(prepared_root / "metadata" / "manifest.json")
    samples = manifest["samples"][: args.limit]

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but this torch install has no CUDA support.")

    output_dirs = {
        variant: prepared_root / "emsanet_samples" / f"{args.output_prefix}_{variant}"
        for variant in args.variants
    }
    for output_dir in output_dirs.values():
        output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = prepared_root / "emsanet_samples" / f"{args.output_prefix}_metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    image_processor = AutoImageProcessor.from_pretrained(args.model_id)
    model = AutoModelForDepthEstimation.from_pretrained(args.model_id)
    model.to(torch.device(device))
    model.eval()

    written: list[dict[str, Any]] = []
    for sample in tqdm(samples, desc="Depth Anything"):
        sample_id = sample["sample_id"]
        out_paths = {variant: output_dirs[variant] / f"{sample_id}.png" for variant in args.variants}
        if not args.overwrite and any(path.exists() for path in out_paths.values()):
            raise FileExistsError(f"Depth output already exists for {sample_id}. Use --overwrite.")

        image = Image.open(resolve_reference(sample["rgb_path"])).convert("RGB")
        width, height = image.size
        inputs = image_processor(images=image, return_tensors="pt")
        inputs = {key: value.to(torch.device(device)) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth
            resized = F.interpolate(
                predicted_depth.unsqueeze(1),
                size=(height, width),
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        relative_depth = resized.detach().cpu().numpy().astype(np.float32, copy=False)
        record = {
            "sample_id": sample_id,
            "rgb_path": sample["rgb_path"],
            "width": width,
            "height": height,
            "relative_min": float(relative_depth.min()),
            "relative_max": float(relative_depth.max()),
            "relative_mean": float(relative_depth.mean()),
            "relative_std": float(relative_depth.std()),
            "outputs": {},
        }

        for variant, out_path in out_paths.items():
            depth_u16 = standardize_to_sunrgbd_units(
                relative_depth,
                target_mean=args.target_mean,
                target_std=args.target_std,
                clip_min=args.clip_min,
                clip_max=args.clip_max,
                inverted=(variant == "inverted"),
            )
            save_u16_png(out_path, depth_u16)
            record["outputs"][variant] = {
                "path": record_path(out_path),
                "min": int(depth_u16.min()),
                "max": int(depth_u16.max()),
                "mean": float(depth_u16.mean()),
                "std": float(depth_u16.std()),
            }

        written.append(record)

    summary = {
        "prepared_root": record_path(prepared_root),
        "model_id": args.model_id,
        "device": device,
        "torch_version": torch.__version__,
        "output_prefix": args.output_prefix,
        "variants": list(args.variants),
        "target_mean": args.target_mean,
        "target_std": args.target_std,
        "clip_min": args.clip_min,
        "clip_max": args.clip_max,
        "num_samples": len(written),
        "samples": written,
    }
    write_json(metadata_dir / "summary.json", summary)
    print(f"Saved Depth Anything depth maps under: {prepared_root / 'emsanet_samples'}")
    for variant, output_dir in output_dirs.items():
        print(f"{variant}: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
