import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from common import sha256, weights
from src.geometry import create_geometry_backend, select_device
from src.plane_fitting import fit_ceiling_plane, PlaneFittingError
from src.slab_surfaces import recover_slab_surfaces
from src.surface_openings import extract_role
from src.utils import read_rgb, save_json


def semantic_classes(rgb, model_path, device):
    import torch
    from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

    processor = AutoImageProcessor.from_pretrained(model_path, use_fast=False, local_files_only=True)
    model = SegformerForSemanticSegmentation.from_pretrained(model_path, local_files_only=True).to(device).eval()
    inputs = processor(images=rgb, input_data_format="channels_last", return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
        classes = processor.post_process_semantic_segmentation(outputs, target_sizes=[rgb.shape[:2]])[0]
    return classes.cpu().numpy(), model.config.id2label


def measure(rgb, native_size, classes, labels, geometry, plane_config, destination):
    g = geometry
    references = {}
    for role in ("floor", "ceiling"):
        ids = [int(i) for i, label in labels.items() if label.casefold() == role]
        seed = np.isin(classes, ids)
        if seed.sum() < plane_config["minimum_points"]:
            continue
        try:
            fitted = fit_ceiling_plane(g.points, seed, g.valid, normals=g.normals, confidence=g.confidence, config=plane_config)
        except PlaneFittingError:
            continue
        references[role] = {**fitted["plane"], "surface_role": role, "metadata": fitted["metadata"]}
    surfaces = recover_slab_surfaces(g.points, g.normals, g.valid, g.intrinsics, classes, labels,
                                     reference_planes=references)
    roles = {}
    for role, data in surfaces["roles"].items():
        opening, uncertain, status = extract_role(role, data, g, classes, labels, references.get(role), rgb)
        for name, mask in (("support", data["accepted_support"]), ("opening", opening), ("uncertain", uncertain)):
            binary = Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255)
            binary.resize(native_size, Image.Resampling.NEAREST).save(destination / f"{role}_{name}.png")
        roles[role] = {"status": status, "surface_status": data["status"],
                       "reference_plane": references.get(role),
                       "planes": [plane["plane"] for plane in data["planes"]]}
    return roles


def main():
    parser = argparse.ArgumentParser(description="Estimate floor and ceiling support and opening candidates.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("slabs.json"))
    parser.add_argument("--geometry", choices=("moge", "depth-anything"), default="moge")
    parser.add_argument("--depth-weights", help="Depth Anything V2 model identifier or directory")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu", "mps"))
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("The output directory must be empty.")
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    native, rgb, frame = read_rgb(args.image, cfg["max_side"])
    classes, labels = semantic_classes(rgb, weights() / "segformer_b0", device)
    options = dict(cfg["geometry"], moge_model=str(weights() / "moge" / "model.pt"))
    if args.depth_weights:
        options["depth_anything_model"] = args.depth_weights
    geom = create_geometry_backend(args.geometry, options, device, offline=args.offline).infer(rgb)
    roles = measure(rgb, native.size, classes, labels, geom, cfg["plane"], args.output)
    save_json(args.output / "result.json", {"method": "geometry_based_slab_candidates", "image": args.image.name,
             "image_sha256": sha256(args.image), "backend": geom.metadata["backend"], "frame": frame, "roles": roles})


if __name__ == "__main__":
    main()

