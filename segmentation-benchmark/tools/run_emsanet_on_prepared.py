"""Run EMSANet on the prepared photo-mask dataset and save raw id masks."""

from __future__ import annotations

from workspace_paths import record_path, resolve_reference

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--emsanet-root",
        type=Path,
        default=Path("external/emsanet"),
        help="Path to the EMSANet source tree.",
    )
    parser.add_argument(
        "--prepared-root",
        type=Path,
        default=Path("prepared_photo_masks"),
        help="Dataset bundle created by prepare_photo_masks_dataset.py.",
    )
    parser.add_argument(
        "--weights-filepath",
        type=Path,
        default=Path("model/emsanet.pth"),
        help="EMSANet checkpoint to load.",
    )
    parser.add_argument(
        "--output-subdir",
        default="emsanet_sunrgbd_pre_raw",
        help="Subdirectory under prepared-root/predictions for raw EMSANet outputs.",
    )
    parser.add_argument(
        "--depth-dir",
        type=Path,
        default=None,
        help="Optional directory containing <sample_id>.png depth maps to use instead of manifest depth_path.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda", "mps"),
        help="Inference device. auto uses CUDA if the installed torch supports it.",
    )
    parser.add_argument(
        "--dataset-profile",
        default="sunrgbd",
        choices=("sunrgbd", "nyuv2"),
        help="Dataset configuration used to build the EMSANet heads.",
    )
    parser.add_argument("--input-height", type=int, default=480)
    parser.add_argument("--input-width", type=int, default=640)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of samples to process.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing raw prediction outputs.",
    )
    return parser.parse_args()


def add_emsanet_to_path(emsanet_root: Path) -> None:
    root = emsanet_root.resolve()
    paths = [
        root,
        root / "lib" / "nicr-multitask-scene-analysis" / "src",
        root / "lib" / "nicr-scene-analysis-datasets" / "src",
    ]
    for path in reversed(paths):
        sys.path.insert(0, str(path))


def read_manifest(prepared_root: Path) -> dict[str, Any]:
    with (prepared_root / "metadata" / "manifest.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def load_img(path: Path) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def save_id_png(path: Path, array: np.ndarray) -> None:
    array = np.asarray(array)
    if array.max(initial=0) <= np.iinfo(np.uint8).max:
        Image.fromarray(array.astype(np.uint8)).save(path)
    else:
        Image.fromarray(array.astype(np.uint16)).save(path)


def to_numpy_2d(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if value.ndim == 3:
        value = value[0]
    if value.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape {value.shape}")
    return value


def build_emsanet_args(user_args: argparse.Namespace, device: str) -> Any:
    from emsanet.args import ArgParserEMSANet

    parser = ArgParserEMSANet()
    arg_list = [
        "--dataset",
        user_args.dataset_profile,
        "--tasks",
        "semantic",
        "scene",
        "instance",
        "orientation",
        "--enable-panoptic",
        "--input-modalities",
        "rgb",
        "depth",
        "--rgb-encoder-backbone",
        "resnet34",
        "--rgb-encoder-backbone-block",
        "nonbottleneck1d",
        "--depth-encoder-backbone",
        "resnet34",
        "--depth-encoder-backbone-block",
        "nonbottleneck1d",
        "--no-pretrained-backbone",
        "--weights-filepath",
        str(user_args.weights_filepath.resolve()),
        "--input-height",
        str(user_args.input_height),
        "--input-width",
        str(user_args.input_width),
        "--validation-input-height",
        str(user_args.input_height),
        "--validation-input-width",
        str(user_args.input_width),
        "--validation-batch-size",
        "1",
        "--batch-size",
        "1",
        "--n-workers",
        "0",
        "--validation-only",
        "--skip-sanity-check",
        "--wandb-mode",
        "disabled",
        "--device",
        device,
    ]
    if user_args.dataset_profile == "sunrgbd":
        arg_list.extend(
            [
                "--sunrgbd-instances-version",
                "emsanet",
                "--sunrgbd-depth-do-not-force-mm",
            ]
        )
    return parser.parse_args(arg_list, verbose=False)


def sample_keys_from_args(emsanet_args: Any) -> tuple[str, ...]:
    sample_keys = list(emsanet_args.input_modalities) + list(emsanet_args.tasks)
    sample_keys.append("identifier")
    if "orientation" in sample_keys:
        sample_keys[sample_keys.index("orientation")] = "orientations"
    if "instance" in emsanet_args.tasks and "semantic" not in sample_keys:
        sample_keys.append("semantic")
    return tuple(sample_keys)


def build_config_only_dataset(emsanet_args: Any) -> Any:
    sample_keys = sample_keys_from_args(emsanet_args)
    if emsanet_args.dataset == "sunrgbd":
        from nicr_scene_analysis_datasets.datasets.sunrgbd.dataset import SUNRGBD

        return SUNRGBD(
            dataset_path=None,
            split="test",
            sample_keys=sample_keys,
            depth_mode="refined",
            depth_force_mm=False,
            semantic_use_nyuv2_colors=True,
            instances_version="emsanet",
            scene_use_indoor_domestic_labels=True,
        )
    if emsanet_args.dataset == "nyuv2":
        from nicr_scene_analysis_datasets.datasets.nyuv2.dataset import NYUv2

        return NYUv2(
            dataset_path=None,
            split="test",
            sample_keys=sample_keys,
            depth_mode="refined",
            semantic_n_classes=40,
            scene_use_indoor_domestic_labels=True,
        )
    raise ValueError(f"Unsupported dataset profile: {emsanet_args.dataset}")


def load_checkpoint(weights_filepath: Path, torch_module: Any) -> dict[str, Any]:
    try:
        return torch_module.load(
            weights_filepath,
            map_location=torch_module.device("cpu"),
            weights_only=False,
        )
    except TypeError:
        return torch_module.load(weights_filepath, map_location=torch_module.device("cpu"))


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> int:
    args = parse_args()
    mpl_cache = Path("_cache/matplotlib").resolve()
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    add_emsanet_to_path(args.emsanet_root)

    import torch
    from nicr_mt_scene_analysis.data import move_batch_to_device
    from nicr_mt_scene_analysis.data import mt_collate

    from emsanet.model import EMSANet
    from emsanet.preprocessing import get_preprocessor
    from emsanet.weights import load_weights

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but this torch install has no CUDA support.")

    prepared_root = args.prepared_root.resolve()
    output_root = prepared_root / "predictions" / args.output_subdir
    semantic_dir = output_root / "semantic_id"
    instance_dir = output_root / "instance_id"
    semantic_direct_dir = output_root / "semantic_direct_id"
    panoptic_dir = output_root / "panoptic_id"
    meta_dir = output_root / "metadata"
    for directory in (semantic_dir, instance_dir, semantic_direct_dir, panoptic_dir, meta_dir):
        directory.mkdir(parents=True, exist_ok=True)

    manifest = read_manifest(prepared_root)
    samples = manifest["samples"][: args.limit]

    emsanet_args = build_emsanet_args(args, device=device)
    dataset = build_config_only_dataset(emsanet_args)
    dataset_config = dataset.config

    model = EMSANet(emsanet_args, dataset_config=dataset_config)
    checkpoint = load_checkpoint(args.weights_filepath.resolve(), torch)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    load_weights(emsanet_args, model, state_dict, verbose=True)
    model.eval()
    model.to(torch.device(device))
    torch.set_grad_enabled(False)

    preprocessor = get_preprocessor(
        emsanet_args,
        dataset=dataset,
        phase="test",
        multiscale_downscales=None,
    )

    class_table = [
        {
            "id": idx,
            "name": label.class_name,
            "isthing": bool(label.is_thing),
            "use_orientations": bool(label.use_orientations),
            "color": [int(v) for v in label.color],
        }
        for idx, label in enumerate(dataset_config.semantic_label_list)
    ]
    write_json(meta_dir / "model_classes.json", class_table)
    write_json(meta_dir / f"{emsanet_args.dataset}_classes.json", class_table)

    summaries = []
    for index, sample in enumerate(samples, start=1):
        sample_id = sample["sample_id"]
        out_paths = [
            semantic_dir / f"{sample_id}.png",
            instance_dir / f"{sample_id}.png",
            semantic_direct_dir / f"{sample_id}.png",
            panoptic_dir / f"{sample_id}.png",
            meta_dir / f"{sample_id}.json",
        ]
        if not args.overwrite and any(path.exists() for path in out_paths):
            raise FileExistsError(f"Outputs already exist for {sample_id}. Use --overwrite.")

        rgb = load_img(resolve_reference(sample["rgb_path"]))
        depth_path = sample.get("depth_path")
        if args.depth_dir is not None:
            depth_path = args.depth_dir.resolve() / f"{sample_id}.png"
        if depth_path:
            depth_path = resolve_reference(depth_path)
            depth = load_img(depth_path).astype("float32")
        else:
            depth = np.zeros(rgb.shape[:2], dtype="float32")

        model_sample = preprocessor(
            {
                "rgb": rgb,
                "depth": depth,
                "identifier": sample_id,
            }
        )
        batch = mt_collate([model_sample])
        batch = move_batch_to_device(batch, device=torch.device(device))

        predictions = model(batch, do_postprocessing=True)

        semantic = to_numpy_2d(predictions["panoptic_segmentation_deeplab_semantic_idx_fullres"])
        instance = to_numpy_2d(predictions["panoptic_segmentation_deeplab_instance_idx_fullres"])
        semantic_direct = to_numpy_2d(predictions["semantic_segmentation_idx_fullres"]) + 1
        panoptic = to_numpy_2d(predictions["panoptic_segmentation_deeplab_fullres"])

        save_id_png(out_paths[0], semantic.astype(np.uint16, copy=False))
        save_id_png(out_paths[1], instance.astype(np.uint16, copy=False))
        save_id_png(out_paths[2], semantic_direct.astype(np.uint16, copy=False))
        save_id_png(out_paths[3], panoptic.astype(np.uint32, copy=False))

        instance_ids = [int(v) for v in np.unique(instance) if v != 0]
        instance_meta = predictions.get("panoptic_segmentation_deeplab_instance_meta", [{}])[0]
        instance_summary = []
        for instance_id in instance_ids:
            mask = instance == instance_id
            sem_values, counts = np.unique(semantic[mask], return_counts=True)
            semantic_id = int(sem_values[counts.argmax()]) if len(sem_values) else 0
            instance_summary.append(
                {
                    "instance_id": instance_id,
                    "semantic_id_majority": semantic_id,
                    "semantic_name_majority": class_table[semantic_id]["name"],
                    "area": int(mask.sum()),
                    "emsanet_meta": instance_meta.get(instance_id, {}),
                }
            )

        sample_meta = {
            "sample_id": sample_id,
            "source_image": sample["source_image"],
            "rgb_path": sample["rgb_path"],
            "depth_path": sample.get("depth_path"),
            "depth_path_used": record_path(depth_path) if depth_path else None,
            "semantic_id_path": record_path(out_paths[0]),
            "instance_id_path": record_path(out_paths[1]),
            "semantic_direct_id_path": record_path(out_paths[2]),
            "panoptic_id_path": record_path(out_paths[3]),
            "num_pred_instances": len(instance_ids),
            "unique_semantic_ids": [int(v) for v in np.unique(semantic)],
            "instances": instance_summary,
        }
        write_json(out_paths[4], sample_meta)
        summaries.append(sample_meta)

        print(
            f"[{index:03d}/{len(samples):03d}] {sample_id}: "
            f"{len(instance_ids)} instances, semantic ids {sample_meta['unique_semantic_ids']}",
            flush=True,
        )

    run_summary = {
        "prepared_root": record_path(prepared_root),
        "emsanet_root": record_path(args.emsanet_root.resolve()),
        "weights_filepath": record_path(args.weights_filepath.resolve()),
        "output_root": record_path(output_root),
        "depth_dir": record_path(args.depth_dir.resolve()) if args.depth_dir else None,
        "dataset_profile": args.dataset_profile,
        "device": device,
        "torch_version": torch.__version__,
        "input_size_hw": [args.input_height, args.input_width],
        "num_samples": len(samples),
        "samples": summaries,
    }
    write_json(meta_dir / "run_summary.json", run_summary)
    print(f"Saved EMSANet raw outputs to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
