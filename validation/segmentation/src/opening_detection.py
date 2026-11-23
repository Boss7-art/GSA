from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import cv2
import numpy as np
from scipy.ndimage import median_filter


DEFAULT_CONFIG: dict[str, Any] = {
    "support_mode": "orientation",
    "normal_angle_threshold": 25.0,
    "envelope_method": "convex_hull",
    "close_radius_fraction": 0.035,
    "candidate_smoothing_radius_fraction": 0.002,
    "minimum_area_image_fraction": 0.0002,
    "minimum_probable_area_image_fraction": 0.002,
    "maximum_area_envelope_fraction": 0.35,
    "maximum_aspect_ratio": 12.0,
    "minimum_compactness": 0.035,
    "minimum_bbox_fill": 0.20,
    "ring_radius_fraction": 0.015,
    "minimum_ring_support": 0.25,
    "minimum_side_support": 0.25,
    "minimum_nonplanar_fraction": 0.50,
    "minimum_recess_fraction": 0.30,
    "minimum_local_aperture_normal_fraction": 0.30,
    "local_recess_distance_multiplier": 1.0,
    "row_reference_minimum_width_fraction": 0.04,
    "row_reference_smoothing_fraction": 0.015,
    "use_recess_cores": True,
    "extract_coherent_bodies": True,
    "body_radius_peak_fraction": 0.70,
    "boundary_body_radius_peak_fraction": 0.30,
    "maximum_probable_fringe_fraction": 0.30,
    "minimum_geometry_coverage": 0.35,
    "foreground_distance_multiplier": 2.0,
    "maximum_protrusion_fraction": 0.70,
    "remove_protrusion_pixels": True,
    "brightness_filter_enabled": False,
    "minimum_relative_brightness": 0.0,
}


def _config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    if config is not None:
        if not isinstance(config, Mapping):
            raise TypeError("config must be a mapping or None")
        unknown = set(config) - set(cfg)
        if unknown:
            raise ValueError(f"Unknown opening configuration: {sorted(unknown)}")
        cfg.update(config)
    if cfg["support_mode"] not in ("orientation", "strict", "orientation_local_depth"):
        raise ValueError("support_mode must be orientation, strict, or orientation_local_depth")
    if cfg["envelope_method"] not in ("convex_hull", "morphological_close"):
        raise ValueError("envelope_method must be convex_hull or morphological_close")
    for key, value in cfg.items():
        if key in ("support_mode", "envelope_method"):
            continue
        if key in ("remove_protrusion_pixels", "brightness_filter_enabled", "use_recess_cores", "extract_coherent_bodies"):
            if not isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{key} must be a boolean")
        elif isinstance(value, (bool, np.bool_)) or not np.isfinite(float(value)):
            raise ValueError(f"{key} must be finite numeric")
        elif key != "minimum_relative_brightness" and float(value) < 0:
            raise ValueError(f"{key} must be nonnegative")
        else:
            cfg[key] = float(value)
    for key in ("minimum_area_image_fraction", "minimum_probable_area_image_fraction", "maximum_area_envelope_fraction",
                "minimum_bbox_fill", "minimum_ring_support", "minimum_side_support",
                "minimum_nonplanar_fraction", "minimum_geometry_coverage",
                "maximum_protrusion_fraction", "minimum_recess_fraction",
                "minimum_local_aperture_normal_fraction",
                "row_reference_minimum_width_fraction", "row_reference_smoothing_fraction",
                "body_radius_peak_fraction", "boundary_body_radius_peak_fraction",
                "maximum_probable_fringe_fraction"):
        if cfg[key] > 1:
            raise ValueError(f"{key} must be in [0, 1]")
    if cfg["normal_angle_threshold"] > 90:
        raise ValueError("normal_angle_threshold must be in [0, 90]")
    if cfg["maximum_aspect_ratio"] < 1:
        raise ValueError("maximum_aspect_ratio must be >= 1")
    return cfg


def _mask(array: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    result = np.asarray(array)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    return result.astype(bool, copy=False)


def _kernel(radius: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    return cv2.dilate(mask.astype(np.uint8), _kernel(max(1, radius))) > 0


def _hull(support: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(support)
    envelope = np.zeros(support.shape, dtype=np.uint8)
    if len(xs) >= 3:
        hull = cv2.convexHull(np.column_stack([xs, ys]).astype(np.int32))
        cv2.fillConvexPoly(envelope, hull, 1)
    else:
        envelope[support] = 1
    return envelope.astype(bool)


def _geometry(shape: tuple[int, int], points: np.ndarray | None,
              valid: np.ndarray | None, normal_angle: np.ndarray | None,
              plane: Mapping[str, Any] | None, cfg: dict[str, Any]) -> dict[str, Any]:
    geometry_valid = np.zeros(shape, dtype=bool)
    angle_valid = np.zeros(shape, dtype=bool)
    angles = np.full(shape, np.nan, dtype=np.float64)
    signed_away = np.full(shape, np.nan, dtype=np.float64)
    residual = np.full(shape, np.nan, dtype=np.float64)
    warnings: list[str] = []
    threshold = None
    quality = None
    normal = None
    offset = None
    if points is not None:
        points = np.asarray(points, dtype=np.float64)
        if points.shape != (*shape, 3):
            raise ValueError("points must be H x W x 3 camera-space coordinates")
        geometry_valid = np.isfinite(points).all(axis=2) & (points[..., 2] > 0)
        if valid is not None:
            geometry_valid &= _mask(valid, shape, "valid")
    elif valid is not None:
        _mask(valid, shape, "valid")
    if normal_angle is not None:
        angles = np.asarray(normal_angle, dtype=np.float64)
        if angles.shape != shape:
            raise ValueError("normal_angle must be H x W degrees")
        angle_valid = np.isfinite(angles) & (angles >= 0) & (angles <= 90)
        if valid is not None:
            angle_valid &= _mask(valid, shape, "valid")
        if points is not None:
            angle_valid &= geometry_valid
    if plane is not None:
        if not isinstance(plane, Mapping):
            raise TypeError("plane must be a mapping or None")
        coeff = plane.get("plane", plane)
        metadata = plane.get("metadata", {})
        normal = np.asarray(coeff.get("normal"), dtype=np.float64)
        offset = float(coeff.get("d", np.nan))
        if normal.shape != (3,) or not np.isfinite(normal).all() or not np.isfinite(offset):
            raise ValueError("plane needs finite normal[3] and d")
        if not np.isclose(np.linalg.norm(normal), 1.0, rtol=1e-5, atol=1e-7):
            raise ValueError("plane.normal must have unit length")
        quality = metadata.get("quality_passed")
        if quality is not None:
            quality = bool(quality)
        raw_threshold = metadata.get("effective_plane_distance_threshold")
        if raw_threshold is not None:
            threshold = float(raw_threshold)
            if not np.isfinite(threshold) or threshold <= 0:
                raise ValueError("effective_plane_distance_threshold must be positive")
        if points is not None:
            signed = np.einsum("ijk,k->ij", points, normal) + offset
            residual[geometry_valid] = np.abs(signed[geometry_valid])
            if abs(offset) > 1e-10:
                signed_away[geometry_valid] = -np.sign(offset) * signed[geometry_valid]
            else:
                warnings.append("Plane passes through the camera origin; foreground/recess sign is unavailable.")
        if quality is False:
            warnings.append("Single-plane quality gate failed. Apertures remain exploratory image-space hypotheses; plane residual is not an accuracy score.")
    if not angle_valid.any():
        warnings.append("Normal evidence unavailable; residual-only ceiling support is used when a saved distance tolerance exists.")
    if not np.isfinite(signed_away).any():
        warnings.append("Signed geometry unavailable; foreground beam rejection cannot be verified.")
    if threshold is None:
        warnings.append("Saved plane-distance tolerance unavailable; no distance threshold is fabricated.")
    protrusion = np.zeros(shape, dtype=bool)
    recess = np.zeros(shape, dtype=bool)
    if threshold is not None:
        sign_threshold = threshold * cfg["foreground_distance_multiplier"]
        protrusion = geometry_valid & (signed_away < -sign_threshold)
        recess = geometry_valid & (signed_away > sign_threshold)
    return {"valid": geometry_valid, "angle_valid": angle_valid, "angles": angles,
            "residual": residual, "signed_away": signed_away,
            "distance_threshold": threshold, "plane_quality_passed": quality,
            "protrusion": protrusion, "recess": recess, "warnings": warnings}


def _side_support(component: np.ndarray, support: np.ndarray, radius: int) -> dict[str, float | None]:
    h, w = component.shape
    sides: dict[str, float | None] = {}
    for name, axis, sign in (("left", 1, -1), ("right", 1, 1),
                             ("top", 0, -1), ("bottom", 0, 1)):
        strip = np.zeros((h, w), dtype=bool)
        for delta in range(1, radius + 1):
            if axis == 1 and delta < w:
                if sign < 0:
                    strip[:, :-delta] |= component[:, delta:]
                else:
                    strip[:, delta:] |= component[:, :-delta]
            elif axis == 0 and delta < h:
                if sign < 0:
                    strip[:-delta, :] |= component[delta:, :]
                else:
                    strip[delta:, :] |= component[:-delta, :]
        strip &= ~component
        denominator = int(strip.sum())
        sides[name] = float(np.count_nonzero(strip & support) / denominator) if denominator else None
    return sides


def _row_recession(geo: dict[str, Any], reference: np.ndarray,
                   cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = reference.shape
    baseline = np.full(shape, np.nan, dtype=np.float64)
    local = np.full(shape, np.nan, dtype=np.float64)
    strong_recess = np.zeros(shape, dtype=bool)
    usable = reference & geo["angle_valid"] & np.isfinite(geo["signed_away"])
    usable &= geo["angles"] <= cfg["normal_angle_threshold"]
    minimum = max(3, int(np.ceil(shape[1] * cfg["row_reference_minimum_width_fraction"])))
    medians = np.full(shape[0], np.nan, dtype=float)
    for row in range(shape[0]):
        if np.count_nonzero(usable[row]) >= minimum:
            medians[row] = np.median(geo["signed_away"][row, usable[row]])
    good = np.flatnonzero(np.isfinite(medians))
    if len(good) and geo["distance_threshold"] is not None:
        values = np.interp(np.arange(shape[0]), good, medians[good])
        radius = int(round(shape[0] * cfg["row_reference_smoothing_fraction"]))
        if radius:
            values = median_filter(values, size=2 * radius + 1, mode="nearest")
        rows = np.any(reference, axis=1)
        baseline[rows] = values[rows, None]
        local = geo["signed_away"] - baseline
        strong_recess = local > cfg["local_recess_distance_multiplier"] * geo["distance_threshold"]
    return baseline, local, strong_recess


def filter_opening_candidates(candidate: np.ndarray, envelope: np.ndarray,
                              support: np.ndarray, rgb: np.ndarray,
                              points: np.ndarray | None, valid: np.ndarray | None,
                              normal_angle: np.ndarray | None,
                              plane: Mapping[str, Any] | None,
                              config: Mapping[str, Any] | None = None,
                              *, reference_ceiling: np.ndarray | None = None) -> dict[str, Any]:
    cfg = _config(config)
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must be H x W x 3")
    shape = rgb.shape[:2]
    envelope = _mask(envelope, shape, "envelope")
    support = _mask(support, shape, "support") & envelope
    raw_candidate = _mask(candidate, shape, "candidate") & envelope & ~support
    geo = _geometry(shape, points, valid, normal_angle, plane, cfg)
    reference = support if reference_ceiling is None else _mask(reference_ceiling, shape, "reference_ceiling")
    row_reference, local_recession, local_recess = _row_recession(geo, reference, cfg)
    candidate = raw_candidate.copy()
    rejected_protrusion = np.zeros(shape, dtype=bool)
    if cfg["remove_protrusion_pixels"]:
        rejected_protrusion = candidate & geo["protrusion"]
        candidate &= ~rejected_protrusion
    unconfirmed_candidates = np.zeros(shape, dtype=bool)
    if cfg["support_mode"] == "orientation_local_depth" and cfg["use_recess_cores"]:
        local_known = np.isfinite(local_recession)
        positive_global_away = geo["signed_away"] > 0
        supported_core = local_recess & positive_global_away
        unconfirmed_candidates = candidate & local_known & ~supported_core
        candidate &= ~local_known | supported_core
    radius = int(round(min(shape) * cfg["candidate_smoothing_radius_fraction"]))
    if radius:
        candidate = cv2.morphologyEx(candidate.astype(np.uint8), cv2.MORPH_OPEN,
                                     _kernel(radius), borderType=cv2.BORDER_CONSTANT,
                                     borderValue=1).astype(bool)
        candidate &= envelope & ~support & ~rejected_protrusion
    candidate_before_body = candidate.copy()
    body_fringe = np.zeros(shape, dtype=bool)
    body_parent_labels = np.zeros(shape, dtype=np.int32)
    body_info: dict[int, dict[str, Any]] = {}
    if cfg["support_mode"] == "orientation_local_depth" and cfg["extract_coherent_bodies"]:
        parent_n, body_parent_labels, parent_stats, _ = cv2.connectedComponentsWithStats(candidate.astype(np.uint8), connectivity=8)
        bodies = np.zeros(shape, dtype=bool)
        for parent in range(1, parent_n):
            x, y, width, height, parent_area = map(int, parent_stats[parent])
            original = body_parent_labels == parent
            distance = cv2.distanceTransform(original.astype(np.uint8), cv2.DIST_L2, 5)
            peak = float(distance.max())
            boundary = x == 0 or y == 0 or x + width == shape[1] or y + height == shape[0]
            fraction = cfg["boundary_body_radius_peak_fraction"] if boundary else cfg["body_radius_peak_fraction"]
            body_radius = int(round(peak * fraction))
            if body_radius:
                body = cv2.morphologyEx(original.astype(np.uint8), cv2.MORPH_OPEN,
                                        _kernel(body_radius), borderType=cv2.BORDER_CONSTANT,
                                        borderValue=1).astype(bool) & original
            else:
                body = original.copy()
            bodies |= body
            body_fringe |= original & ~body
            body_info[parent] = {"parent_area_pixels": parent_area,
                                 "body_radius_pixels": body_radius,
                                 "distance_transform_peak_pixels": peak,
                                 "fringe_fraction": 1 - int(body.sum()) / parent_area}
        candidate = bodies
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate.astype(np.uint8), connectivity=8)
    ring_radius = max(1, int(round(min(shape) * cfg["ring_radius_fraction"])))
    min_area = max(1, int(np.ceil(np.prod(shape) * cfg["minimum_area_image_fraction"])))
    envelope_area = int(envelope.sum())
    gray = cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(float) / 255
    final = np.zeros(shape, dtype=bool)
    ambiguous = np.zeros(shape, dtype=bool)
    rejected = rejected_protrusion.copy()
    components: list[dict[str, Any]] = []
    for index in range(1, n):
        component = labels == index
        x, y, width, height, area = map(int, stats[index])
        contours, _ = cv2.findContours(component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = sum(cv2.arcLength(contour, True) for contour in contours)
        compactness = float(4 * np.pi * area / (perimeter * perimeter)) if perimeter else 0.0
        aspect = max(width / height, height / width)
        fill = area / (width * height)
        ring = _dilate(component, ring_radius) & ~component & envelope
        ring_pixels = int(ring.sum())
        ring_support = float(np.count_nonzero(ring & support) / ring_pixels) if ring_pixels else 0.0
        sides = _side_support(component, support, ring_radius)
        bracketed = ((sides["left"] or 0) >= cfg["minimum_side_support"] and
                     (sides["right"] or 0) >= cfg["minimum_side_support"]) or (
                     (sides["top"] or 0) >= cfg["minimum_side_support"] and
                     (sides["bottom"] or 0) >= cfg["minimum_side_support"])
        boundary_sides = [name for name, flag in (("left", x == 0), ("top", y == 0),
                          ("right", x + width == shape[1]), ("bottom", y + height == shape[0])) if flag]
        v = component & geo["valid"]
        av = component & geo["angle_valid"]
        count_valid, count_angle = int(v.sum()), int(av.sum())
        valid_fraction, angle_fraction = count_valid / area, count_angle / area
        nonplanar = float(np.mean(geo["angles"][av] > cfg["normal_angle_threshold"])) if count_angle else None
        signed_valid = component & np.isfinite(geo["signed_away"])
        signed_count = int(signed_valid.sum())
        protrusion_fraction = float(np.count_nonzero(component & geo["protrusion"]) / signed_count) if signed_count else None
        recess_fraction = float(np.count_nonzero(component & geo["recess"]) / signed_count) if signed_count else None
        local_valid = component & np.isfinite(local_recession)
        local_count = int(local_valid.sum())
        local_recess_fraction = float(np.count_nonzero(component & local_recess) / local_count) if local_count else None
        local_median = float(np.median(local_recession[local_valid])) if local_count else None
        offset_median = float(np.median(geo["signed_away"][signed_valid])) if signed_count else None
        relative_brightness = float(np.mean(gray[component]) - np.median(gray[ring & support])) if np.any(ring & support) else None
        reasons: list[str] = []
        if area < min_area:
            reasons.append("area_below_minimum")
        if envelope_area and area / envelope_area > cfg["maximum_area_envelope_fraction"]:
            reasons.append("area_too_large_for_bounded_aperture")
        if aspect > cfg["maximum_aspect_ratio"]:
            reasons.append("extreme_aspect_ratio")
        if compactness < cfg["minimum_compactness"] or fill < cfg["minimum_bbox_fill"]:
            reasons.append("fragmented_or_incoherent_shape")
        if ring_support < cfg["minimum_ring_support"] and not bracketed:
            reasons.append("insufficient_surrounding_ceiling")
        if boundary_sides and not bracketed:
            reasons.append("unbracketed_image_boundary_candidate")
        if protrusion_fraction is not None and signed_count / area >= cfg["minimum_geometry_coverage"]:
            if protrusion_fraction > cfg["maximum_protrusion_fraction"]:
                reasons.append("foreground_protrusion_consistent_with_beam")
        if nonplanar is not None and angle_fraction >= cfg["minimum_geometry_coverage"]:
            if nonplanar < cfg["minimum_nonplanar_fraction"] and not (
                cfg["support_mode"] == "orientation_local_depth" and
                local_recess_fraction is not None and local_recess_fraction >= cfg["minimum_recess_fraction"]
            ):
                reasons.append("parallel_surface_consistent_with_coffer_or_mask_gap")
        if cfg["brightness_filter_enabled"] and relative_brightness is not None:
            if relative_brightness < cfg["minimum_relative_brightness"]:
                reasons.append("optional_brightness_filter_failed")
        ambiguities: list[str] = []
        parent_ids = body_parent_labels[component]
        parent_ids = parent_ids[parent_ids > 0]
        parent = int(parent_ids[0]) if parent_ids.size else None
        body_evidence = body_info.get(parent, {})
        if body_evidence.get("fringe_fraction", 0) > cfg["maximum_probable_fringe_fraction"]:
            ambiguities.append("large_connected_fringe_coffer_or_bulkhead_confusion")
        if area / np.prod(shape) < cfg["minimum_probable_area_image_fraction"]:
            ambiguities.append("aperture_below_conservative_resolution_threshold")
        if angle_fraction < cfg["minimum_geometry_coverage"]:
            ambiguities.append("normal_evidence_insufficient")
        if signed_count / area < cfg["minimum_geometry_coverage"] or geo["distance_threshold"] is None:
            ambiguities.append("foreground_recess_evidence_insufficient")
        if recess_fraction is not None:
            if recess_fraction < cfg["minimum_recess_fraction"]:
                ambiguities.append("normal_interruption_without_recess_confirmation")
        if cfg["support_mode"] == "orientation_local_depth" and nonplanar is not None:
            if nonplanar < cfg["minimum_local_aperture_normal_fraction"]:
                ambiguities.append("parallel_recess_coffer_or_glass_ambiguous")
        if reasons:
            status = "rejected"
            rejected |= component
        elif ambiguities:
            status = "ambiguous"
            ambiguous |= component
        else:
            status = "probable_opening"
            final |= component
        components.append({
            "id": index, "status": status, "bbox_xywh": [x, y, width, height],
            "centroid_xy": list(map(float, centroids[index])), "area_pixels": area,
            "area_image_fraction": float(area / np.prod(shape)),
            "area_envelope_fraction": float(area / envelope_area) if envelope_area else 0.0,
            "aspect_ratio": float(aspect), "compactness": compactness,
            "bbox_fill_ratio": float(fill), "ring_support_fraction": ring_support,
            "side_support": sides, "bracketed": bool(bracketed),
            "image_boundary_sides": boundary_sides,
            "valid_geometry_fraction": valid_fraction, "usable_normal_fraction": angle_fraction,
            "nonplanar_normal_fraction": nonplanar,
            "foreground_protrusion_fraction": protrusion_fraction, "recess_fraction": recess_fraction,
            "local_recess_fraction": local_recess_fraction, "local_recession_median": local_median,
            "signed_away_median": offset_median, "relative_brightness": relative_brightness,
            "rejection_reasons": reasons, "ambiguity_reasons": ambiguities,
            "body_extraction": body_evidence,
            "interpretation": "Exploratory aperture-footprint hypothesis. Recession cannot establish a through-slab void; deep coffers and glass are confounders. No probability calibration or opening ground truth was used.",
        })
    return {
        "expected_ceiling_envelope_image": envelope.copy(),
        "observed_ceiling_support": support.copy(),
        "opening_candidate_image": raw_candidate,
        "candidate_after_foreground_filter": candidate,
        "candidate_before_body_extraction": candidate_before_body,
        "body_extraction_fringe_mask": body_fringe,
        "foreground_protrusion_rejection_mask": rejected_protrusion,
        "unconfirmed_normal_interruption_mask": unconfirmed_candidates,
        "row_ceiling_recession_reference": row_reference,
        "local_recession_raw": local_recession,
        "local_recess_evidence_mask": local_recess & envelope,
        "final_opening_mask": final, "ambiguous_opening_mask": ambiguous,
        "rejected_opening_mask": rejected, "components": components,
        "metadata": {
            "method": "bounded_ceiling_interruption_with_normal_and_signed_geometry_evidence",
            "config": cfg, "plane_quality_passed": geo["plane_quality_passed"],
            "plane_quality_low": geo["plane_quality_passed"] is False,
            "warnings": geo["warnings"],
            "signed_distance_convention": "away = -sign(d) * (normal dot X + d); positive=recess, negative=foreground",
            "minimum_area_pixels": min_area, "ring_radius_pixels": ring_radius,
            "minimum_probable_area_pixels": max(1, int(np.ceil(np.prod(shape) * cfg["minimum_probable_area_image_fraction"]))),
            "ceiling_envelope_pixels": envelope_area, "observed_support_pixels": int(support.sum()),
            "raw_candidate_pixels": int(raw_candidate.sum()),
            "foreground_rejected_pixels": int(rejected_protrusion.sum()),
            "unconfirmed_normal_interruption_pixels": int(unconfirmed_candidates.sum()),
            "body_extraction_fringe_pixels": int(body_fringe.sum()),
            "local_recession_reference": "per-row median signed-away of observed parallel semantic ceiling; sparse rows interpolated, row median smoothing",
            "invalid_geometry_candidate_pixels": int(np.count_nonzero(raw_candidate & ~geo["valid"])),
            "unknown_normal_candidate_pixels": int(np.count_nonzero(raw_candidate & ~geo["angle_valid"])),
            "probable_opening_count": sum(c["status"] == "probable_opening" for c in components),
            "ambiguous_count": sum(c["status"] == "ambiguous" for c in components),
            "rejected_count": sum(c["status"] == "rejected" for c in components),
            "accuracy": None,
            "limitation": "A single low-quality plane can confuse recesses, beams, and skylight glass. Probable labels are exploratory; an independent annotation is required for accuracy.",
        },
    }


def detect_openings(rgb: np.ndarray, ceiling: np.ndarray,
                    points: np.ndarray | None, valid: np.ndarray | None,
                    normal_angle: np.ndarray | None, plane: Mapping[str, Any] | None,
                    config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    cfg = _config(config)
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must be H x W x 3")
    shape = rgb.shape[:2]
    ceiling = _mask(ceiling, shape, "ceiling")
    geo = _geometry(shape, points, valid, normal_angle, plane, cfg)
    normals_available = geo["angle_valid"].any()
    distance_ok = np.zeros(shape, dtype=bool)
    if geo["distance_threshold"] is not None:
        distance_ok = geo["valid"] & (geo["residual"] <= geo["distance_threshold"])
    if normals_available:
        support = ceiling & geo["angle_valid"] & (geo["angles"] <= cfg["normal_angle_threshold"])
        support |= ceiling & ~geo["angle_valid"] & distance_ok
        if cfg["support_mode"] == "strict":
            support &= distance_ok
        elif cfg["support_mode"] == "orientation_local_depth":
            _, _, local_recess = _row_recession(geo, ceiling, cfg)
            support &= ~local_recess
        support_source = cfg["support_mode"]
    elif geo["distance_threshold"] is not None and geo["valid"].any():
        support = ceiling & distance_ok
        support_source = "residual_fallback"
    else:
        support = ceiling.copy()
        support_source = "semantic_only_unknown_geometry"
    convex_hull = _hull(support)
    close_radius = max(1, int(round(min(shape) * cfg["close_radius_fraction"])))
    closed = cv2.morphologyEx(support.astype(np.uint8), cv2.MORPH_CLOSE,
                             _kernel(close_radius), borderType=cv2.BORDER_CONSTANT,
                             borderValue=1).astype(bool)
    closed = (closed | support) & convex_hull
    envelope = convex_hull if cfg["envelope_method"] == "convex_hull" else closed
    result = filter_opening_candidates(envelope & ~support, envelope, support,
                                      rgb, points, valid, normal_angle, plane, cfg,
                                      reference_ceiling=ceiling)
    result["convex_hull_envelope"] = convex_hull
    result["morphological_close_envelope"] = closed
    result["convex_hull_candidate_baseline"] = convex_hull & ~support
    result["morphological_close_candidate_baseline"] = closed & ~support
    result["metadata"].update({"support_source": support_source,
                               "envelope_method": cfg["envelope_method"],
                               "close_radius_pixels": close_radius,
                               "semantic_ceiling_pixels": int(ceiling.sum())})
    return result
