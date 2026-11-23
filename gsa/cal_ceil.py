"""Segment indoor photographs and measure their spatial indicators."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

sys.dont_write_bytecode = True

SLAB = "slab_transparency"
WALL = "wall_transparency"
ROLES = ("floor", "ceiling", "slab", "wall")
RATIOS = ("void_over_total", "void_over_solid")
IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
OUTPUT_FOLDERS = {"slab_transparency", "wall_transparency", "perspective", "color_masks"}


def photos_in(source, output):
    if source.is_file() and source.suffix.lower() in IMAGE_TYPES:
        return [source]
    if not source.is_dir():
        raise ValueError(f"Expected a photograph or an image directory: {source}")
    photos = []
    for directory, subdirs, names in os.walk(source):
        folder = Path(directory)
        subdirs[:] = sorted(name for name in subdirs if not name.startswith(".")
                            and name.casefold() not in OUTPUT_FOLDERS and folder / name != output)
        photos.extend(folder / name for name in sorted(names)
                      if not name.startswith(".") and Path(name).suffix.lower() in IMAGE_TYPES)
    if not photos:
        raise ValueError(f"No photographs found in {source}")
    if len({p.with_suffix("") for p in photos}) != len(photos):
        raise ValueError("Photographs in the same directory must have distinct stems.")
    return photos


def read_result(folder):
    return json.loads((folder / "result.json").read_text(encoding="utf-8"))


def geometry_fields(result):
    report = {}
    metrics = result["report_metrics"]
    groups = result["explicit_segmentation"]["groups"]
    for role in ("slab", "wall"):
        for ratio in RATIOS:
            report[f"rectified_{role}_{ratio}"] = metrics[role][ratio]
    report["rectified_enclosure_degree"] = metrics["enclosure_degree"]
    report["estimated_visible_surface_enclosure_degree"] = metrics["estimated_enclosure_degree"]
    report["fitted_reference_solid_fraction"] = result["aggregates"]["all"]["solid_over_total"]
    for role in ROLES:
        for ratio in RATIOS:
            report[f"explicit_{role}_{ratio}"] = groups[role][ratio]
    for role in ROLES:
        group = groups[role]
        assigned = group["assigned"] or {}
        for ratio in RATIOS:
            report[f"explicit_assigned_{role}_{ratio}"] = assigned.get(ratio)
        for key in ("solid", "void"):
            total = group["source_pixels"][key]
            measured = group["measured_source_pixels"][key]
            report[f"explicit_{role}_{key}_source_pixel_coverage"] = measured / total if total else None
    return report


@contextmanager
def file_lock(target):
    locks = Path(tempfile.gettempdir()) / "gsa-file-locks"
    locks.mkdir(exist_ok=True)
    key = hashlib.sha256(os.fsencode(os.path.normcase(str(target.resolve())))).hexdigest()
    with (locks / key).open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def save_report(path, fields, partial=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(path):
        values = {}
        if partial and path.is_file():
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                key, sep, value = line.partition(":")
                if sep:
                    values[key.strip()] = value.strip()
        values.update(fields)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as f:
            tmp = Path(f.name)
        try:
            tmp.write_text("".join(f"{k}: {v if v is not None else 'nan'}\n" for k, v in values.items()),
                           encoding="utf-8")
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)


def run_model(request):
    from perspective.pipeline import PerspectiveBatch

    mode = request["model"]
    device = request["device"]
    output = Path(request["output"])
    base = Path(request["base"])
    geometry = PerspectiveBatch(base=base, output=output, device=device, **request["geometry"]) if request["geometry"] else None
    if mode == "slab":
        from slab_model.predict import Predictor
        predictor = Predictor(device=device)
    elif mode == "wall":
        import cv2
        import torch
        from wall_model.predict import WallPredictor
        torch.set_num_threads(3)
        cv2.setNumThreads(1)
        predictor = WallPredictor(device=device)
    for name in request["photos"]:
        photo = Path(name)
        stem = photo.relative_to(base).with_suffix("")
        report = output / stem.parent / (stem.name + ".txt")
        if mode == "perspective":
            save_report(report, geometry_fields(geometry.process(photo)), partial=True)
            continue
        dest = output / (mode + "_transparency") / stem
        dest.mkdir(parents=True, exist_ok=True)
        camera = {} if geometry and geometry.needs_camera(photo) else None
        with tempfile.TemporaryDirectory(dir=dest) as temp:
            stage = Path(temp)
            if mode == "wall":
                result = predictor.predict_file(photo, stage, camera_out=camera)
            else:
                result = predictor.predict_file(photo, stage)
            with file_lock(dest):
                for asset in sorted(stage.iterdir()):
                    if asset.name != "result.json":
                        os.replace(asset, dest / asset.name)
                os.replace(stage / "result.json", dest / "result.json")
        if geometry:
            geometry.process(photo, camera)
        if request["partial"]:
            field = {SLAB: result["image_plane"]["void_over_total"]} if mode == "slab" else {WALL: result["void_over_total"]}
            save_report(report, field, partial=True)
        print(f"{mode}: {photo.relative_to(base)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_image", type=Path, required=True, help="Photograph or directory of photographs")
    parser.add_argument("--output_dir", "-o", type=Path, default=Path("results"))
    parser.add_argument("--slab-python", default=os.environ.get("GSA_SLAB_PYTHON", sys.executable),
                        help="Python interpreter for segmentation and perspective measurement")
    parser.add_argument("--slab-device", choices=("cpu", "cuda"), help="Segmentation device; default is automatic")
    parser.add_argument("--wall-device", choices=("cpu", "cuda"), help="Wall segmentation device")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="Scene indicator device")
    parser.add_argument("--box_threshold", type=float, default=.3)
    parser.add_argument("--text_threshold", type=float, default=.25)
    parser.add_argument("--plane-config", type=Path, help="Optional per-photograph plane controls in JSON")
    parser.add_argument("--camera-height-m", type=float, default=1.5)
    parser.add_argument("--no-camera-scale", action="store_true")
    parser.add_argument("--pixels-per-metre", type=float, default=100.)
    parser.add_argument("--floor-morphology", choices=("open", "close", "none"), default="open")
    only = parser.add_mutually_exclusive_group()
    for mode in ("slab", "wall", "perspective"):
        only.add_argument(f"--{mode}-only", dest="only", action="store_const", const=mode)
    args = parser.parse_args()
    source = args.input_image.resolve()
    output = args.output_dir.resolve()
    base = source if source.is_dir() else source.parent
    photos = photos_in(source, output)
    geometry = dict(config=str(args.plane_config.resolve()) if args.plane_config else None,
                    height=None if args.no_camera_scale else args.camera_height_m,
                    sampling=args.pixels_per_metre, floor_morphology=args.floor_morphology)
    for mode in ([args.only] if args.only else ["slab", "wall"]):
        device = args.slab_device if mode == "slab" else args.wall_device or args.slab_device
        use_geometry = mode == "perspective" or (mode == "wall" and not args.only)
        request = dict(model=mode, device=device, photos=[str(p) for p in photos], base=str(base), output=str(output),
                       partial=bool(args.only), geometry=geometry if use_geometry else None)
        subprocess.run([args.slab_python, "-B", str(Path(__file__).resolve()), "--run-model"], check=True,
                       input=json.dumps(request).encode("utf-8"))
    if args.only:
        return
    from indicators import IndicatorEngine
    engine = IndicatorEngine(device=args.device, box_threshold=args.box_threshold, text_threshold=args.text_threshold)
    for photo in photos:
        stem = photo.relative_to(base).with_suffix("")
        slab = read_result(output / "slab_transparency" / stem)
        wall = read_result(output / "wall_transparency" / stem)
        fields = {SLAB: slab["image_plane"]["void_over_total"]}
        for key, value in engine.measure(photo).items():
            fields[key] = value
            if key == "enclosure_degree":
                fields[WALL] = wall["void_over_total"]
        fields.update(geometry_fields(read_result(output / "perspective" / stem)))
        save_report(output / stem.parent / (stem.name + ".txt"), fields)
        print(f"Report: {stem}", flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["--run-model"]:
        run_model(json.load(sys.stdin.buffer))
    else:
        main()
