from __future__ import annotations

from collections.abc import Mapping

import cv2
import numpy as np


DEFAULT_CONFIG = {
    "normal_angle_degrees": 25.0,
    "reference_offset_quantiles": [0.01, 0.99],
    "offset_margin_median_depth_fraction": 0.15,
    "closing_radius_long_edge_fraction": 0.006,
    "minimum_reference_pixels": 100,
    "minimum_component_pixels": 100,
    "minimum_component_seed_pixels": 32,
    "minimum_component_seed_fraction": 0.001,
    "tta_rotations_degrees": [0, 90, 180, 270],
}


def validated_config(config=None):
    if config is not None and not isinstance(config, Mapping):
        raise ValueError("Ceiling recovery config must be a mapping.")
    settings = {**DEFAULT_CONFIG, **(config or {})}
    unknown = set(settings) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"Unknown ceiling recovery parameters: {sorted(unknown)}")
    for key in ("minimum_reference_pixels", "minimum_component_pixels", "minimum_component_seed_pixels"):
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{key} must be a positive integer.")
    for key in ("normal_angle_degrees", "offset_margin_median_depth_fraction", "closing_radius_long_edge_fraction", "minimum_component_seed_fraction"):
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)) or not np.isfinite(value):
            raise ValueError(f"{key} must be finite numeric.")
    if not 0 < settings["normal_angle_degrees"] < 90:
        raise ValueError("normal_angle_degrees must lie between 0 and 90.")
    for key in ("offset_margin_median_depth_fraction", "closing_radius_long_edge_fraction", "minimum_component_seed_fraction"):
        if not 0 <= settings[key] <= 1:
            raise ValueError(f"{key} must lie between 0 and 1.")
    q = np.asarray(settings["reference_offset_quantiles"], dtype=float)
    if q.shape != (2,) or not np.isfinite(q).all() or not 0 <= q[0] < q[1] <= 1:
        raise ValueError("reference_offset_quantiles must contain ordered fractions.")
    rotations = settings["tta_rotations_degrees"]
    if not isinstance(rotations, (tuple, list)) or not rotations or len(set(rotations)) != len(rotations):
        raise ValueError("tta_rotations_degrees must be a nonempty unique list.")
    if any(isinstance(x, bool) or x not in (0, 90, 180, 270) for x in rotations):
        raise ValueError("TTA rotations must be 0, 90, 180, or 270 degrees.")
    return settings


def recover_ceiling_support(semantic, points, normals, valid, plane, supplemental_seeds=None, config=None):
    settings = validated_config(config)
    semantic = np.asarray(semantic)
    if semantic.ndim != 2 or semantic.dtype != bool or min(semantic.shape) < 1:
        raise ValueError("semantic must be a nonempty boolean HxW ceiling grid.")
    shape = semantic.shape
    points = np.asarray(points, dtype=float)
    normals = np.asarray(normals, dtype=float)
    valid = np.asarray(valid)
    if points.shape != (*shape, 3) or normals.shape != (*shape, 3) or valid.shape != shape or valid.dtype != bool:
        raise ValueError("All geometry grids must share the semantic frame; validity must be bool.")
    n = np.asarray(plane["normal"], dtype=float)
    d = float(plane["d"])
    if n.shape != (3,) or not np.isfinite(n).all() or not np.isfinite(d) or np.linalg.norm(n) <= 1e-12:
        raise ValueError("Plane must contain a finite nonzero normal and finite d.")
    scale = np.linalg.norm(n)
    n, d = n / scale, d / scale
    norm_lengths = np.linalg.norm(normals, axis=-1)
    usable = valid & np.isfinite(points).all(axis=-1) & (points[..., 2] > 0)
    usable &= np.isfinite(normals).all(axis=-1) & (norm_lengths > 1e-12)
    dot = np.einsum("hwk,k->hw", normals, n) / np.maximum(norm_lengths, 1e-12)
    angle = np.degrees(np.arccos(np.clip(np.abs(dot), 0, 1)))
    parallel = usable & (angle <= settings["normal_angle_degrees"])
    offset = np.einsum("hwk,k->hw", points, n) + d
    reference = semantic & parallel
    empty = np.zeros(shape, bool)
    base = {
        "effective_config": settings,
        "original_semantic_pixels": int(semantic.sum()),
        "reference_pixels": int(reference.sum()),
        "opening_annotation_used": False,
        "supplemental_seed_policy": "offset-gated only; cannot enlarge reference band",
        "coordinate_frame": "same as raw cached RGB and geometry",
    }
    if int(reference.sum()) < settings["minimum_reference_pixels"]:
        return {"recovered_ceiling": semantic.copy(), "added_support": empty.copy(), "accepted_support": empty.copy(), "offset_gated_support": empty.copy(), "domain_envelope": _hull(semantic), "normal_angle": angle, "signed_offset": offset, "components": [], "metadata": {**base, "status": "insufficient_reference", "added_pixels": 0}}
    median_depth = float(np.median(points[..., 2][reference]))
    bounds = np.quantile(offset[reference], settings["reference_offset_quantiles"])
    margin = settings["offset_margin_median_depth_fraction"] * median_depth
    lower, upper = float(bounds[0] - margin), float(bounds[1] + margin)
    gated = parallel & (offset >= lower) & (offset <= upper)
    seeds = semantic.copy()
    supplemental_records = []
    if supplemental_seeds is not None and not isinstance(supplemental_seeds, (list, tuple)):
        raise ValueError("Supplemental seeds must be a list or tuple of automatic boolean ceiling grids.")
    for value in ([] if supplemental_seeds is None else supplemental_seeds):
        seed = np.asarray(value)
        if seed.shape != shape or seed.dtype != bool:
            raise ValueError("Supplemental seeds must be bool and share the geometry pixel frame.")
        seeds |= seed & gated
        supplemental_records.append({"raw_seed_pixels": int(seed.sum()), "offset_gated_seed_pixels": int((seed & gated).sum())})
    radius = round(max(shape) * settings["closing_radius_long_edge_fraction"])
    connected = gated.astype(np.uint8)
    if radius > 0:
        connected = cv2.morphologyEx(connected, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(connected, connectivity=8)
    accepted = np.zeros(shape, bool)
    records = []
    for label in range(1, count):
        component = (labels == label) & gated
        area = int(component.sum())
        if area < settings["minimum_component_pixels"]:
            continue
        seed_pixels = int((component & seeds).sum())
        fraction = seed_pixels / area
        keep = seed_pixels >= settings["minimum_component_seed_pixels"] and fraction >= settings["minimum_component_seed_fraction"]
        if keep:
            accepted |= component
        records.append({"label": label, "bbox_xywh": stats[label, :4].tolist(), "support_pixels": area, "seed_pixels": seed_pixels, "seed_fraction": fraction, "accepted": bool(keep)})
    recovered = semantic | accepted
    added = accepted & ~semantic
    return {
        "recovered_ceiling": recovered, "added_support": added,
        "accepted_support": accepted, "offset_gated_support": gated,
        "domain_envelope": _hull(recovered), "normal_angle": angle,
        "signed_offset": offset, "components": records,
        "metadata": {**base, "status": "support_recovered" if added.any() else "unchanged", "added_pixels": int(added.sum()), "recovered_ceiling_pixels": int(recovered.sum()), "median_reference_depth": median_depth, "offset_band": [lower, upper], "closing_radius_pixels": radius, "supplemental_seeds": supplemental_records, "opening_mask_generated": False, "physical_void_confirmed": False, "limitation": "Original semantics must define a usable ceiling reference. If it already includes another parallel plane, the offset gate may inherit that contamination. The convex envelope is only a search domain."},
    }


def _hull(mask):
    result = np.zeros(mask.shape, np.uint8)
    ys, xs = np.nonzero(mask)
    if len(xs) >= 3:
        cv2.fillConvexPoly(result, cv2.convexHull(np.column_stack([xs, ys]).astype(np.int32)), 1)
    else:
        result[ys, xs] = 1
    return result.astype(bool)
