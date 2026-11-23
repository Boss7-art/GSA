from __future__ import annotations

from collections.abc import Mapping
from typing import Any
import cv2
import numpy as np


DEFAULT_CONFIG = {
    "plane_distance_multiplier": 2.0,
    "normal_angle_degrees": 25.0,
    "minimum_seed_pixels": 32,
    "minimum_seed_fraction": 0.0001,
    "support_closing_fraction": 0.002,
    "min_ray_cosine": 0.08,
    "recess_distance_multiplier": 2.0,
    "minimum_candidate_image_fraction": 0.005,
    "minimum_boundary_contact_fraction": 0.10,
    "minimum_slab_adjacency_fraction": 0.10,
    "adjacency_radius_fraction": 0.004,
    "minimum_recess_fraction_for_probable": 0.10,
    "minimum_geometry_coverage": 0.50,
    "dominant_gap_only": True,
    "semantic_opening_context_weight": 2.0,
    "semantic_wall_context_weight": 2.0,
}


def validated_config(settings: Mapping[str, Any] | None = None) -> dict[str, Any]:
    result = dict(DEFAULT_CONFIG)
    if settings is not None:
        if not isinstance(settings, Mapping):
            raise ValueError("slab_edge settings must be a mapping")
        unknown = set(settings) - set(result)
        if unknown:
            raise ValueError(f"Unknown slab_edge settings: {sorted(unknown)}")
        result.update(settings)
    for key, value in result.items():
        if key == "dominant_gap_only":
            if not isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{key} must be boolean")
            continue
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{key} must be finite and nonnegative")
        try:
            value = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{key} must be numeric") from error
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
        result[key] = value
        if "fraction" in key or key == "min_ray_cosine":
            if value > 1:
                raise ValueError(f"{key} must be in [0, 1]")
        if key == "minimum_seed_pixels" and (int(value) != value or value < 1):
            raise ValueError("minimum_seed_pixels must be a positive integer")
        if key == "minimum_seed_pixels":
            result[key] = int(value)
        if key == "normal_angle_degrees" and value > 90:
            raise ValueError("normal_angle_degrees must be <= 90")
    if result["min_ray_cosine"] <= 0 or result["plane_distance_multiplier"] <= 0:
        raise ValueError("Ray cosine and plane distance multiplier must be positive")
    if result["minimum_recess_fraction_for_probable"] <= 0:
        raise ValueError("minimum_recess_fraction_for_probable must be positive")
    return result


def _mask(value, shape, name):
    value = np.asarray(value)
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if value.dtype != bool:
        raise ValueError(f"{name} must be boolean")
    return value


def _kernel(radius):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1,) * 2)


def extract_slab_edge_openings(semantic, points, normals, valid, intrinsics,
                              plane, config=None, *, semantic_classes=None,
                              semantic_labels=None):
    cfg = validated_config(config)
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError("points must be H x W x 3")
    shape = points.shape[:2]
    semantic = _mask(semantic, shape, "semantic")
    valid = _mask(valid, shape, "valid")
    normals = np.asarray(normals, dtype=np.float64)
    if normals.shape != points.shape:
        raise ValueError("normals must match points")
    opening_context = np.zeros(shape, bool)
    wall_context = np.zeros(shape, bool)
    if (semantic_classes is None) != (semantic_labels is None):
        raise ValueError("Full semantic classes and label mapping must be supplied together")
    semantic_context_used = semantic_classes is not None
    context_names = {"railing", "fence", "bannister", "banister", "escalator", "stairs", "stairway"}
    wall_names = {"wall", "door", "building"}
    resolved_context_ids, resolved_wall_ids = [], []
    if semantic_context_used:
        classes = np.asarray(semantic_classes)
        if classes.shape != shape or not np.issubdtype(classes.dtype, np.integer):
            raise ValueError("semantic_classes must be integer H x W on the geometry grid")
        if not isinstance(semantic_labels, Mapping):
            raise ValueError("semantic_labels must be id-to-name mapping")
        for class_id, name in semantic_labels.items():
            if not isinstance(name, str):
                raise ValueError("Semantic label names must be strings")
            canonical = name.strip().casefold()
            if canonical in context_names:
                resolved_context_ids.append(int(class_id))
            if canonical in wall_names:
                resolved_wall_ids.append(int(class_id))
        opening_context = np.isin(classes, resolved_context_ids)
        wall_context = np.isin(classes, resolved_wall_ids)
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all() or abs(np.linalg.det(intrinsics)) < 1e-12:
        raise ValueError("intrinsics must be finite and invertible normalized 3 x 3")
    coeff = plane.get("plane", plane)
    metadata = plane.get("metadata", {})
    normal = np.asarray(coeff.get("normal"), dtype=np.float64)
    d = float(coeff.get("d", np.nan))
    if normal.shape != (3,) or not np.isfinite(normal).all() or not np.isfinite(d):
        raise ValueError("plane must have finite normal and d")
    if not np.isclose(np.linalg.norm(normal), 1) or abs(d) < 1e-10:
        raise ValueError("plane normal must be unit and plane must not pass through camera")
    threshold = float(metadata.get("effective_plane_distance_threshold", np.nan))
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("Saved positive plane-distance tolerance is required")
    geometry_valid = valid & np.isfinite(points).all(axis=-1) & (points[..., 2] > 0)
    signed_away = -np.sign(d) * (points @ normal + d)
    signed_away[~geometry_valid] = np.nan
    lengths = np.linalg.norm(normals, axis=-1)
    normal_valid = geometry_valid & np.isfinite(normals).all(axis=-1) & (lengths > 1e-8)
    angle = np.full(shape, np.nan)
    angle[normal_valid] = np.degrees(np.arccos(np.clip(np.abs(normals[normal_valid] @ normal / lengths[normal_valid]), 0, 1)))
    strict = geometry_valid & (np.abs(signed_away) <= threshold * cfg["plane_distance_multiplier"])
    strict &= normal_valid & (angle <= cfg["normal_angle_degrees"])
    nc, labels, stats, _ = cv2.connectedComponentsWithStats(strict.astype(np.uint8), connectivity=8)
    slab = np.zeros(shape, bool)
    support_components = []
    for index in range(1, nc):
        component = labels == index
        area = int(stats[index, cv2.CC_STAT_AREA])
        hits = int(np.count_nonzero(component & semantic))
        accepted = hits >= cfg["minimum_seed_pixels"] and hits / area >= cfg["minimum_seed_fraction"]
        if accepted:
            slab |= component
        if area >= cfg["minimum_seed_pixels"]:
            support_components.append({"id": index, "area_pixels": area, "semantic_seed_pixels": hits,
                                       "accepted": accepted, "bbox_xywh": stats[index, :4].tolist()})
    if not slab.any():
        raise ValueError("No coplanar component is anchored by the supplied semantic slab")
    radius = int(round(min(shape) * cfg["support_closing_fraction"]))
    recovered = slab.copy()
    if radius:
        recovered = cv2.morphologyEx(slab.astype(np.uint8), cv2.MORPH_CLOSE, _kernel(radius),
                                     borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool) | slab
    y, x = np.indices(shape, dtype=np.float64)
    pixels = np.stack([(x + .5) / shape[1], (y + .5) / shape[0], np.ones(shape)], axis=-1)
    rays = pixels @ np.linalg.inv(intrinsics).T
    denominator = rays @ normal
    cosine = np.abs(denominator) / np.linalg.norm(rays, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = -d / denominator
    positive_ray = np.isfinite(scale) & (scale > 0)
    domain = positive_ray & (cosine >= cfg["min_ray_cosine"])
    gaps = domain & ~recovered
    recess = geometry_valid & (signed_away > threshold * cfg["recess_distance_multiplier"])
    foreground = geometry_valid & (signed_away < -threshold * cfg["recess_distance_multiplier"])
    nc, labels, stats, _ = cv2.connectedComponentsWithStats(gaps.astype(np.uint8), connectivity=8)
    radius = max(1, int(round(min(shape) * cfg["adjacency_radius_fraction"])))
    candidates = []
    eligible = []
    for index in range(1, nc):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < max(1, shape[0] * shape[1] * cfg["minimum_candidate_image_fraction"]):
            continue
        component = labels == index
        border_counts = {"left": int(component[:, 0].sum()), "right": int(component[:, -1].sum()),
                         "top": int(component[0].sum()), "bottom": int(component[-1].sum())}
        border_fraction = max(border_counts["left"], border_counts["right"]) / shape[0]
        border_fraction = max(border_fraction, max(border_counts["top"], border_counts["bottom"]) / shape[1])
        ring = cv2.dilate(component.astype(np.uint8), _kernel(radius)).astype(bool) & ~component
        contacts = int(np.count_nonzero(ring & recovered))
        adjacency_fraction = contacts / max(1, int(ring.sum()))
        geo_count = int(np.count_nonzero(component & geometry_valid))
        recess_pixels = int(np.count_nonzero(component & recess))
        fraction = recess_pixels / max(1, geo_count)
        context_fraction = float(np.count_nonzero(component & opening_context)) / area
        wall_fraction = float(np.count_nonzero(component & wall_context)) / area
        rank_score = area * (1 + cfg["semantic_opening_context_weight"] * context_fraction)
        rank_score /= 1 + cfg["semantic_wall_context_weight"] * wall_fraction
        rejected = []
        if border_fraction < cfg["minimum_boundary_contact_fraction"]:
            rejected.append("insufficient_image_boundary_contact")
        if adjacency_fraction < cfg["minimum_slab_adjacency_fraction"]:
            rejected.append("insufficient_target_slab_adjacency")
        status = "eligible_exterior_gap" if not rejected else "rejected_exterior_gap"
        if not rejected:
            eligible.append((rank_score, index))
        candidates.append({"id": index, "status": status, "area_pixels": area,
                           "bbox_xywh": stats[index, :4].tolist(), "boundary_counts": border_counts,
                           "boundary_contact_fraction": border_fraction, "slab_adjacency_pixels": contacts,
                           "slab_adjacency_fraction": adjacency_fraction,
                           "geometry_coverage": geo_count / area, "positive_recess_pixels": recess_pixels,
                           "positive_recess_fraction": fraction,
                           "semantic_opening_context_fraction": context_fraction,
                           "semantic_wall_context_fraction": wall_fraction,
                           "gap_rank_score": rank_score,
                           "foreground_or_glass_pixels": int(np.count_nonzero(component & foreground)),
                           "rejection_reasons": rejected})
    chosen_ids = [index for _, index in sorted(eligible, reverse=True)]
    if cfg["dominant_gap_only"]:
        chosen_ids = chosen_ids[:1]
    hypothesis = np.isin(labels, chosen_ids) if chosen_ids else np.zeros(shape, bool)
    probable = np.zeros(shape, bool)
    for candidate in candidates:
        if candidate["id"] in chosen_ids:
            if (metadata.get("quality_passed") is not False and
                candidate["positive_recess_fraction"] >= cfg["minimum_recess_fraction_for_probable"] and
                candidate["geometry_coverage"] >= cfg["minimum_geometry_coverage"]):
                candidate["status"] = "recession_supported_opening_hypothesis"
                probable |= labels == candidate["id"]
            else:
                candidate["status"] = "ambiguous_edge_opening_hypothesis"
                candidate["ambiguity_reasons"] = ["missing_positive_recession_support_wall_glass_or_void_unresolved"]
        elif candidate["status"] == "eligible_exterior_gap":
            candidate["status"] = "unselected_external_gap"
    observed = hypothesis & recess
    return {
        "strict_coplanar_support": strict, "semantic_anchored_support": slab,
        "recovered_slab_support": recovered, "support_inferred_bridge_mask": recovered & ~slab,
        "positive_ray_domain": positive_ray, "ray_domain": domain,
        "grazing_ray_unknown_mask": positive_ray & ~domain,
        "raw_exterior_gap_mask": gaps, "opening_hypothesis_mask": hypothesis,
        "probable_opening_mask": probable, "ambiguous_opening_mask": hypothesis & ~probable,
        "other_external_gap_mask": gaps & ~hypothesis,
        "observed_recess_mask": observed, "inferred_void_mask": hypothesis & ~observed,
        "foreground_or_glass_mask": hypothesis & foreground,
        "unknown_geometry_mask": hypothesis & ~geometry_valid,
        "signed_away": signed_away, "normal_angle": angle, "ray_cosine": cosine,
        "components": candidates, "support_components": support_components,
        "metadata": {
            "method": "semantic_anchored_single_slab_coplanarity_then_rank_border_connected_external_gaps",
            "config": cfg, "fitted_plane_quality_passed": metadata.get("quality_passed"),
            "semantic_pixels": int(semantic.sum()), "strict_support_pixels": int(strict.sum()),
            "recovered_slab_pixels": int(recovered.sum()), "slab_ray_domain_pixels": int(domain.sum()),
            "hypothesis_pixels": int(hypothesis.sum()), "probable_pixels": int(probable.sum()),
            "observed_positive_recess_pixels": int(observed.sum()),
            "inferred_pixels": int((hypothesis & ~observed).sum()),
            "signed_distance_convention": "away=-sign(d)*(n dot X+d), positive=opposite camera side",
            "opening_annotation_used": False,
            "auxiliary_semantic_context_used": semantic_context_used,
            "auxiliary_opening_context_ids": resolved_context_ids,
            "auxiliary_wall_context_ids": resolved_wall_ids,
            "interpretation": "A border-connected exterior gap is ranked by area and optional inferred railing/stair versus wall semantic context. It is an image-space slab-edge opening hypothesis; positive recession pixels alone are observed geometry evidence. Other gap pixels are inferred, including glass, occluders and unresolved surfaces. Geometry does not establish a through-slab void.",
            "limitation": "The grazing-ray limit clips distant extent. Geometry through glass may land on the parapet. Gap area and auxiliary semantic ranking cannot independently distinguish an atrium from a foreground wall. Masks may include visible railing and lower-storey objects within the inferred footprint.",
        },
    }
