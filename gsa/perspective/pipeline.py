import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image, ImageOps

from .estimate import estimate_planes
from .rectify import rectify_scene

def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def report_metrics(result, diagnostics, masks, automatic):
    groups = result["aggregates"]
    coverage = {}
    min_coverage = .90
    for role in ("floor","ceiling","wall"):
        if not automatic:
            continue
        native = diagnostics["coverage"][role]
        parts = [p for p in result["planes"] if p["role"] == role]
        stats = {}
        for cls, key in (("solid","raw_solid_source_pixels"),("opening","known_void_source_pixels")):
            total = native[f"input_{cls}_pixels"]
            stats[cls] = sum(p[key] for p in parts)/total if total else None
        fractions = [f for f in stats.values() if f is not None]
        coverage[role] = dict(fractions=stats, present=bool(fractions),
                              sufficient=bool(fractions) and all(f >= min_coverage for f in fractions))
    slab_solid = (masks[0] == 1) | (masks[0] == 3)
    wall_solid = masks[1] == 1
    conflict = int(np.count_nonzero(slab_solid & wall_solid))
    union = int(np.count_nonzero(slab_solid | wall_solid))
    conflict_fraction = conflict/union if union else 0.
    output = {}
    for name, roles in (("slab",("floor","ceiling")),("wall",("wall",))):
        group = groups[name]
        represented = [coverage[r] for r in roles if automatic and coverage[r]["present"]]
        eligible = group["comparable"] and (not automatic or
                    (bool(represented) and all(r["sufficient"] for r in represented)))
        output[name] = {key:group[key] if eligible else None for key in ("void_over_total","void_over_solid")}
    all_planes = groups["all"]
    good_scene = (automatic and all(r["sufficient"] for r in coverage.values()) and
                  conflict_fraction <= .10 and all_planes["comparable"] and
                  all_planes["area_unit"] in ("m2","relative_units2"))
    output["estimated_enclosure_degree"] = (all_planes["solid_over_total"] if good_scene
                                            else result["estimated_enclosure_degree"])
    output["enclosure_degree"] = result["enclosure_degree"]
    output["coverage_checks"] = dict(minimum_class_coverage=min_coverage, roles=coverage,
                                     opaque_conflict_pixels=conflict, opaque_union_pixels=union,
                                     opaque_conflict_fraction=conflict_fraction,maximum_scene_conflict_fraction=.10)
    return output


class PerspectiveBatch:
    def __init__(self, base, output, config, height, sampling, floor_morphology, device=None):
        self.base, self.output = Path(base), Path(output)
        self.config = Path(config) if config else None
        self.controls = None
        if self.config:
            images = _read_json(self.config)["images"]
            self.controls = {key.replace(chr(92), "/").casefold(): value for key, value in images.items()}
            if len(images) != len(self.controls):
                raise ValueError("Repeated relative photo path in plane config")
        self.height, self.sampling = height, sampling
        for v in (height, sampling):
            if v is not None and (not math.isfinite(v) or v <= 0):
                raise ValueError("Camera height and sampling density must be positive and finite")
        self.morphology, self.device = floor_morphology, device
        self.geometry = None

    def entry(self, photo):
        if self.controls is None:
            return {"mode":"auto"}
        key = photo.relative_to(self.base).as_posix().casefold()
        if key not in self.controls:
            raise ValueError(f"Missing {key} in plane config {self.config}")
        entry = self.controls[key]
        mode = entry.get("mode", "manual")
        if mode not in ("auto", "manual") or (mode == "auto" and "planes" in entry):
            raise ValueError(f"Invalid plane config mode for {key}")
        return entry

    def needs_camera(self, photo):
        return self.entry(photo).get("mode", "manual") == "auto"

    def manual_planes(self, entry):
        planes = []
        for control in entry["planes"]:
            plane = dict(control)
            for field in ("footprint", "reference_mask"):
                if isinstance(plane.get(field), str):
                    plane[field] = self.config.parent / plane[field]
            planes.append(plane)
        return planes

    def process(self, photo, camera=None):
        relative = photo.relative_to(self.base)
        stem = relative.with_suffix("")
        dest = self.output / "perspective" / stem
        dest.mkdir(parents=True, exist_ok=True)
        entry = self.entry(photo)
        source_hash = _sha(photo)
        with Image.open(photo) as im:
            image = ImageOps.exif_transpose(im).convert("RGB")
        if entry.get("source_sha256", source_hash) != source_hash or list(entry.get("image_size", image.size)) != list(image.size):
            raise ValueError(f"Control points belong to a different photograph or orientation: {relative}")
        masks, hashes, identities = [], {}, {}
        for name in ("slab_transparency","wall_transparency"):
            folder = self.output / name / stem
            mask_path = folder / "classes.png"
            if not mask_path.is_file():
                raise FileNotFoundError(f"Missing {mask_path}. Run full GSA or generate both segmentation masks first.")
            marker = folder / "result.json"
            mask_source_hash = _read_json(marker).get("source_sha256") if marker.is_file() else None
            if mask_source_hash is not None and source_hash != mask_source_hash:
                raise ValueError(f"The {name} mask belongs to a different photo: {relative}")
            with Image.open(mask_path) as im:
                if im.size != image.size:
                    raise ValueError(f"Mask grid differs from EXIF-upright photo: {mask_path}")
                masks.append(np.array(im))
            hashes[name] = _sha(mask_path)
            identities[name] = "matched_source_hash" if mask_source_hash else "mask_path_and_size_only"
        auto = self.needs_camera(photo)
        if auto:
            if camera is None:
                if self.geometry is None:
                    from model_assets import assets_root
                    from slab_model.geometry_features import Geometry
                    self.geometry = Geometry(assets_root() / "moge" / "model.pt", self.device)
                _, camera = self.geometry.camera_input(image)
            height = entry.get("camera_height_m", self.height) if self.height is not None else None
            planes, diagnostics = estimate_planes(*masks, **camera, camera_height_m=height)
            trusted_roles = entry.get("all_labels_coplanar_roles", [])
            if not isinstance(trusted_roles, list) or any(r not in ("floor","ceiling","wall") for r in trusted_roles):
                raise ValueError("all_labels_coplanar_roles must list floor, ceiling or wall")
            for role in trusted_roles:
                matches = [plane for plane in planes if plane["role"] == role]
                if len(matches) > 1:
                    raise ValueError(f"Cannot assign whole-role labels to multiple fitted {role} planes")
                if matches:
                    matches[0]["all_role_labels_on_this_plane"] = True
            diagnostics["intrinsics_normalized"] = camera["intrinsics"].tolist()
            diagnostics["depth_size"] = list(camera["depth"].shape[::-1])
            complete, absent = False, []
        else:
            planes = self.manual_planes(entry)
            complete, absent = entry.get("declared_complete", False), entry.get("absent_roles", [])
            diagnostics = dict(scale_source="per_plane_control_dimensions",camera_height_applied=False)
        with tempfile.TemporaryDirectory(dir=dest, prefix=".rectify-") as tmp:
            stage = Path(tmp)
            bundle = stage / "masks"
            result = rectify_scene(*masks, planes, bundle, pixels_per_metre=self.sampling,
                                   declared_complete=complete, absent_roles=absent, floor_morphology=self.morphology)
            result.update(schema_version=1, source=relative.as_posix(), source_sha256=source_hash,
                          class_map_sha256=hashes, mask_source_verification=identities,
                          mode="auto" if auto else "manual", geometry_diagnostics=diagnostics,
                          sampling_density=self.sampling)
            result["report_metrics"] = report_metrics(result, diagnostics, masks, auto)
            result["estimated_enclosure_degree"] = result["report_metrics"]["estimated_enclosure_degree"]
            signature = hashlib.sha256(json.dumps(result, sort_keys=True,allow_nan=False).encode())
            for path in sorted(bundle.glob("*/*.png")):
                signature.update(_sha(path).encode())
            mask_directory = "planes-" + signature.hexdigest()[:24]
            (bundle / "result.json").unlink()
            for plane in result["planes"]:
                for key, value in plane.items():
                    if isinstance(value,str) and value.endswith(".png"):
                        plane[key] = mask_directory + "/" + value
            manifest = stage / "result.json"
            manifest.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
            target = dest / mask_directory
            if not target.is_dir():
                try:
                    bundle.rename(target)
                except OSError:
                    if not target.is_dir():
                        raise
            os.replace(manifest, dest / "result.json")
        print(f"Perspective: {relative} -> {len(result['planes'])} planes; {dest}",flush=True)
        return result
