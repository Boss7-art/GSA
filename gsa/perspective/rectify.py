import json
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from . import explicit

ROLES = ("floor", "ceiling", "wall")
CLASSES = {"floor": (1, 2), "ceiling": (3, 4), "wall": (1, 2)}
MAX_TARGET_PIXELS = 24_000_000
MAX_SIDE = 32766


def _positive(value, name):
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _size(value):
    wh = np.asarray(value, dtype=float)
    if (wh.shape != (2, ) or not np.isfinite(wh).all() or (wh != np.floor(wh)).any() or (wh < 2).any()
            or (wh > MAX_SIDE).any() or wh.prod() > MAX_TARGET_PIXELS):
        raise ValueError(f"Unsafe or degenerate rectified canvas: {value}")
    return tuple(int(v) for v in wh)


def _quad(value):
    pts = np.asarray(value, dtype=np.float64)
    if (pts.shape != (4, 2) or not np.isfinite(pts).all() or np.abs(pts).max() > 1.e7
            or not cv2.isContourConvex(pts.astype(np.float32))
            or cv2.contourArea(pts.astype(np.float32)) < 1.):
        raise ValueError("Expected four finite, cyclic, non-crossing corners with nonzero area")
    return pts


def _geometry(plane, sampling):
    if "homography" in plane:
        matrix = np.asarray(plane["homography"], np.float64)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) != 3:
            raise ValueError("Singular or invalid plane homography")
        size = _size(plane["target_size"])
        unit = plane.get("scale_unit", "unknown")
        if unit not in ("metre", "relative", "unknown"):
            raise ValueError(f"Invalid scale unit: {unit}")
        density = _positive(plane["pixels_per_unit"], "pixels_per_unit") if unit != "unknown" else 1.
        return matrix, size, unit, density
    quad = _quad(plane["quad_tl_tr_br_bl"])
    keys = [k for k in ("width_height_metres", "width_height_relative", "aspect_ratio") if k in plane]
    if len(keys) != 1:
        raise ValueError("Specify one of width_height_metres, width_height_relative or aspect_ratio")
    if keys[0] == "aspect_ratio":
        wh = np.array([_positive(plane["aspect_ratio"], "aspect_ratio"), 1.])
        density, unit = _positive(plane.get("pixels_per_unit", 512), "sampling resolution"), "unknown"
    else:
        wh = np.asarray(plane[keys[0]], np.float64)
        if wh.shape != (2, ) or not np.isfinite(wh).all() or (wh <= 0).any():
            raise ValueError("The target rectangle needs a positive finite width and height")
        unit = "metre" if keys[0] == "width_height_metres" else "relative"
        density = _positive(plane.get("pixels_per_unit", sampling), "pixels_per_unit")
    size = _size(np.rint(wh * density))
    w, h = size
    target = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    matrix = cv2.getPerspectiveTransform(quad.astype(np.float32), target)
    if np.linalg.matrix_rank(matrix) != 3:
        raise ValueError("Degenerate plane homography")
    return matrix, size, unit, density if unit != "unknown" else 1.


def _mask(value, shape, *, polygon=False):
    if isinstance(value, (str, Path)):
        with Image.open(value) as im:
            if im.size != (shape[1], shape[0]):
                raise ValueError(f"Mask does not match the upright photograph: {value}")
            arr = np.array(im)
    else:
        arr = np.asarray(value)
    if arr.shape == shape:
        if arr.dtype != np.bool_ and not np.isin(arr, [0, 1, 255]).all():
            raise ValueError("Expected a binary source reference or footprint")
        return np.ascontiguousarray(arr != 0, dtype=np.uint8)
    if (polygon and arr.ndim == 2 and arr.shape[1] == 2 and len(arr) >= 3 and np.isfinite(arr).all()
            and np.abs(arr).max() < 1.e7):
        points = np.rint(arr).astype(np.int32)
        return cv2.fillPoly(np.zeros(shape, np.uint8), [points], 1)
    raise ValueError("Expected an upright binary mask or an XY footprint polygon")


def _check_projection(region, matrix, size):
    contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("The plane reference is empty in this photograph")
    boundary = np.concatenate(contours)[:, 0, :].astype(np.float64)
    denom = boundary @ matrix[2, :2] + matrix[2, 2]
    if np.any(denom * np.sign(denom[0]) <= 1.e-8 * np.max(np.abs(denom))):
        raise ValueError("The reference crosses or touches the homography horizon")
    mapped = (boundary @ matrix[:2, :2].T + matrix[:2, 2]) / denom[:, None]
    jac = (matrix[:2, :2][None, :, :] - mapped[:, :, None] * matrix[2, :2]) / denom[:, None, None]
    tolerance = 1. + np.abs(jac).sum(2)
    if np.any(mapped < -tolerance) or np.any(mapped > (np.asarray(size) - 1) + tolerance):
        raise ValueError("The destination clips the plane reference; correct H or the extent")


def _write_mask(path, data):
    Image.fromarray(data.astype(np.uint8) * 255).save(path)


def _fraction(n, d):
    return float(n / d) if d else None


def _areas(solid, reference):
    void = reference - solid
    return dict(solid_area=solid,
                reference_area=reference,
                void_area=void,
                void_over_solid=_fraction(void, solid),
                void_over_total=_fraction(void, reference),
                solid_over_total=_fraction(solid, reference))


def _kernel(k):
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or not (1 <= k <= 127 and k % 2 == 1):
        raise ValueError("Morphology kernels must be odd integer diameters from 1 to 127 pixels")
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _measure(group, role, plane_id, plane, reference, classes, matrix, size, unit, density, options):
    raw = np.asarray(classes == CLASSES[role][0], np.uint8) & reference
    known_open = np.asarray(classes == CLASSES[role][1], np.uint8) & reference
    measured = raw
    mode, kernel, dilation = options
    if role == "floor" and mode != "none":
        op = cv2.MORPH_OPEN if mode == "open" else cv2.MORPH_CLOSE
        measured = cv2.morphologyEx(raw, op, kernel)
        measured = cv2.dilate(measured, dilation) & reference & (1 - known_open)
    codes = reference + measured
    corrected = cv2.warpPerspective(codes,
                                    matrix,
                                    size,
                                    flags=cv2.INTER_NEAREST,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=0)
    ref = corrected != 0
    solid = corrected == 2
    s, r = int(solid.sum()), int(ref.sum())
    if not r:
        raise ValueError(f"The destination grid contains no reference samples for {plane_id}")
    group.mkdir(parents=True, exist_ok=True)
    for name, mask in (("source_reference", reference), ("source_solid_raw", raw), ("source_solid", measured),
                       ("reference", ref), ("solid", solid), ("void", corrected == 1)):
        _write_mask(group / (name + ".png"), mask)
    record = dict(id=plane_id,
                  role=role,
                  source=plane.get("source", "manual"),
                  source_reference=f"{plane_id}/source_reference.png",
                  source_solid_raw=f"{plane_id}/source_solid_raw.png",
                  source_solid=f"{plane_id}/source_solid.png",
                  homography=matrix.tolist(),
                  target_size=list(size),
                  scale_unit=unit,
                  scale_group=plane.get("scale_group"),
                  pixels_per_unit=density,
                  area_unit={
                      "metre": "m2",
                      "relative": "relative_units2",
                      "unknown": "rectified_pixels"
                  }[unit],
                  raw_solid_source_pixels=int(raw.sum()),
                  processed_solid_source_pixels=int(measured.sum()),
                  known_void_source_pixels=int(known_open.sum()),
                  reference_source_pixels=int(reference.sum()),
                  unclassified_reference_source_pixels=int(reference.sum()) - int(raw.sum()) -
                  int(known_open.sum()),
                  solid_rectified_pixels=s,
                  void_rectified_pixels=r - s,
                  reference_rectified_pixels=r,
                  rectified_solid=f"{plane_id}/solid.png",
                  rectified_void=f"{plane_id}/void.png",
                  rectified_reference=f"{plane_id}/reference.png",
                  **_areas(s / density**2, r / density**2))
    for key in ("quad_tl_tr_br_bl", "width_height_metres", "width_height_relative", "aspect_ratio",
                "normal_camera", "offset_model_units", "fit_support_pixels", "median_residual_model_units",
                "all_role_labels_on_this_plane"):
        if key in plane:
            record[key] = np.asarray(plane[key]).tolist()
    return record


def _combine(planes, roles, complete, absent):
    parts = [p for p in planes if p["role"] in roles]
    represented = {p["role"] for p in parts}
    scope = "complete" if complete and roles <= (represented | absent) else "partial"
    support = dict(scope=scope,
                   plane_ids=[p["id"] for p in parts],
                   missing_roles=sorted(roles - represented - absent))
    if not parts:
        return dict(**support, comparable=False, area_unit=None, **{k: None for k in _areas(0., 0.)})
    comparable = len(parts) == 1 or all(p["scale_unit"] == "metre" for p in parts)
    if all(p["scale_unit"] == "relative" for p in parts):
        groups = {p["scale_group"] for p in parts}
        comparable = len(parts) == 1 or (len(groups) == 1 and bool(next(iter(groups))))
    if not comparable:
        return dict(**support, comparable=False, area_unit=None, **{k: None for k in _areas(0., 0.)})
    return dict(**support,
                comparable=True,
                area_unit=parts[0]["area_unit"],
                **_areas(sum(p["solid_area"] for p in parts), sum(p["reference_area"] for p in parts)))


def _measure_plane(plane, occupied, classes, root, sampling, morphology, full_role_candidate):
    plane_id = plane["id"]
    matrix, size, unit, density = _geometry(plane, sampling)
    region = plane.get("footprint", plane.get("quad_tl_tr_br_bl"))
    if region is None:
        region = plane.get("reference_mask")
    if region is None:
        raise ValueError(f"Missing source footprint for {plane_id}")
    source = _mask(region, classes.shape, polygon=True)
    if plane.get("reference_mask") is not None:
        source &= _mask(plane["reference_mask"], classes.shape)
    _check_projection(source, matrix, size)
    overlap = occupied & source
    if overlap.any():
        kernel = np.ones((3, 3), np.uint8)
        inside = cv2.erode(source, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        occupied_interior = cv2.erode(occupied, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        if np.any(overlap & (inside | occupied_interior)) or np.array_equal(overlap, source):
            raise ValueError(f"References of the same role overlap: {plane_id}")
        source &= 1 - occupied
    record = _measure(root / plane_id, plane["role"], plane_id, plane, source, classes, matrix, size, unit,
                      density, morphology)
    record["explicit_segmentation"] = explicit.measure(plane, source, classes, CLASSES[plane["role"]],
                                                       matrix, size, unit, density, full_role_candidate)
    record["single_plane_label_extension"] = bool(full_role_candidate)
    occupied |= source
    return record


def rectify_scene(slab_classes,
                  wall_classes,
                  planes,
                  output,
                  *,
                  pixels_per_metre=100.,
                  declared_complete=False,
                  absent_roles=(),
                  floor_morphology="open",
                  kernel_size=3,
                  dilation_size=3):
    slab, wall = np.asarray(slab_classes), np.asarray(wall_classes)
    if slab.ndim != 2 or slab.shape != wall.shape or any(not 1 <= n <= MAX_SIDE for n in slab.shape):
        raise ValueError("Need matching EXIF-upright 2D slab and wall class maps")
    if (slab.dtype.kind not in "uib" or wall.dtype.kind not in "uib"
            or not np.isin(slab, [0, 1, 2, 3, 4]).all() or not np.isin(wall, [0, 1, 2]).all()):
        raise ValueError("Unexpected input mask class")
    if not isinstance(declared_complete,
                      (bool, np.bool_)) or floor_morphology not in ("none", "open", "close"):
        raise ValueError("Invalid declared_complete or floor morphology")
    absent = set(absent_roles)
    if not absent <= set(ROLES):
        raise ValueError("Unknown absent plane role")
    kernel, dilation = _kernel(kernel_size), _kernel(dilation_size)
    sampling = _positive(pixels_per_metre, "pixels_per_metre")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    records, rejected, seen, occupied = [], [], set(), {}
    planes = list(planes)
    role_counts = {role: sum(p["role"] == role for p in planes) for role in ROLES}
    for plane in planes:
        role, key = plane["role"], plane["id"]
        if role not in ROLES or role in absent:
            raise ValueError(f"Unknown or contradictory plane role: {role}")
        if (not isinstance(key, str) or not re.fullmatch(r"[\w-][\w.-]{0,119}", key) or key.endswith(".")
                or key.casefold() in seen or Path(key).is_reserved()):
            raise ValueError(f"Unsafe or duplicate plane id: {key!r}")
        seen.add(key.casefold())
        if role not in occupied:
            occupied[role] = np.zeros(slab.shape, np.uint8)
        classes = wall if role == "wall" else slab
        trusted = plane.get("all_role_labels_on_this_plane", False)
        if not isinstance(trusted, (bool,np.bool_)) or (trusted and role_counts[role] != 1):
            raise ValueError("All labels of a role may only be assigned explicitly to one plane")
        auto = plane.get("source") in ("automatic_solid_depth_plane", "frozen_automatic_solid_depth_plane")
        full_role_candidate = role_counts[role] == 1 and (trusted or auto) and plane.get("single_plane_label_extension", True)
        try:
            record = _measure_plane(plane, occupied[role], classes, root, sampling,
                                    (floor_morphology, kernel, dilation), full_role_candidate)
        except ValueError as error:
            if plane.get("source") != "automatic_solid_depth_plane":
                raise
            rejected.append(dict(id=key, role=role, reason=str(error)))
            continue
        records.append(record)
    groups = {name: {name} for name in ROLES}
    groups.update(slab={"floor", "ceiling"}, all=set(ROLES))
    rejected_roles = {p["role"] for p in rejected}
    aggregates = {
        k: _combine(records, r, declared_complete and r.isdisjoint(rejected_roles), absent)
        for k, r in groups.items()
    }
    inputs = {role: [int(np.count_nonzero((wall if role == "wall" else slab) == code)) for code in CLASSES[role]]
              for role in ROLES}
    explicit_groups = {name: groups[name] for name in ("floor", "ceiling", "slab", "wall")}
    all_planes = aggregates["all"]
    ready = all_planes["scope"] == "complete" and all_planes["comparable"]
    metric = ready and all_planes["area_unit"] == "m2"
    relative = ready and all_planes["area_unit"] == "relative_units2" and all(p["scale_group"]
                                                                              for p in records)
    result = dict(image_size=[slab.shape[1], slab.shape[0]],
                  declared_complete=bool(declared_complete),
                  absent_roles=sorted(absent),
                  floor_morphology=dict(mode=floor_morphology,
                                        kernel_size=kernel_size,
                                        dilation_size=dilation_size,
                                        grid="upright_source_pixels",
                                        known_openings_protected=True),
                  planes=records,
                  rejected_planes=rejected,
                  aggregates=aggregates,
                  explicit_segmentation=explicit.summarize(records, explicit_groups, aggregates, inputs),
                  enclosure_degree=all_planes["solid_over_total"] if metric else None,
                  estimated_enclosure_degree=all_planes["solid_over_total"] if relative else None)
    (root / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
                                      encoding="utf-8")
    return result


