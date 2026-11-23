from __future__ import annotations

from collections.abc import Mapping
import cv2
import numpy as np

from src.ceiling_recovery import recover_ceiling_support


DEFAULT_CONFIG = {
    "normal_angle_degrees": 25.0,
    "offset_margin_median_depth_fraction": 0.50,
    "reference_offset_quantiles": [0.01, 0.99],
    "connectivity_closing_long_edge_fraction": 0.006,
    "support_closing_long_edge_fraction": 0.0026,
    "minimum_semantic_anchor_pixels": 32,
    "minimum_top_contact_pixels": 32,
    "recession_distance_multiplier": 5.5,
    "minimum_border_recession_fraction": 0.50,
    "minimum_candidate_image_fraction": 0.0006,
    "minimum_candidate_recession_fraction": 0.60,
    "maximum_candidate_aspect_ratio": 4.0,
    "minimum_candidate_extent_long_edge_fraction": 0.015,
    "grouping_radius_long_edge_fraction": 0.015,
    "maximum_inferred_to_seed_fraction": 0.75,
}


def validated_config(config=None):
    if config is not None and not isinstance(config, Mapping):
        raise ValueError("ceiling_edge_gaps config must be a mapping")
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    if set(cfg) - set(DEFAULT_CONFIG):
        raise ValueError("Unknown ceiling edge-gap parameters")
    for key, value in cfg.items():
        if key == "reference_offset_quantiles":
            q = np.asarray(value, dtype=float)
            if q.shape != (2,) or not np.isfinite(q).all() or not 0 <= q[0] < q[1] <= 1:
                raise ValueError("Reference quantiles must be two ordered fractions")
            continue
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)) or not np.isfinite(value):
            raise ValueError(f"{key} must be a finite number")
        if key in ("minimum_semantic_anchor_pixels", "minimum_top_contact_pixels"):
            if int(value) != value or value < 1:
                raise ValueError(f"{key} must be a positive integer")
            cfg[key] = int(value)
        elif "fraction" in key:
            if not 0 <= value <= 1:
                raise ValueError(f"{key} must be a fraction")
        elif value <= 0:
            raise ValueError(f"{key} must be positive")
    if not 0 < cfg["normal_angle_degrees"] < 90:
        raise ValueError("Normal angle must lie in (0, 90)")
    if cfg["maximum_candidate_aspect_ratio"] < 1:
        raise ValueError("Aspect ratio limit must be at least one")
    return cfg


def _kernel(radius):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)


def _hull(mask, extra_points=()):
    y, x = np.nonzero(mask)
    result = np.zeros(mask.shape, np.uint8)
    points = np.column_stack([x, y]).astype(np.int32)
    if len(extra_points):
        points = np.concatenate([points, np.asarray(extra_points, dtype=np.int32)], axis=0)
    if len(points) >= 3:
        cv2.fillConvexPoly(result, cv2.convexHull(points), 1)
    elif len(points):
        result[points[:, 1], points[:, 0]] = 1
    return result.astype(bool)


def complete_curved_mouth(seed, component_count, maximum_addition_fraction):
    if component_count < 2:
        return seed.copy(), {"method": "unchanged_single_gap", "inferred_pixels": 0}
    rows = np.flatnonzero(seed.any(axis=1))
    if len(rows) < 3:
        return seed.copy(), {"method": "unchanged_insufficient_rows", "inferred_pixels": 0}
    width = max(1, len(rows) // 5)
    top_x = np.nonzero(seed[rows[:width]])[1].mean()
    bottom_x = np.nonzero(seed[rows[-width:]])[1].mean()
    bends_left = bottom_x < top_x
    hull = _hull(seed)
    result = seed.copy()
    for y in rows:
        xs = np.flatnonzero(seed[y])
        bound = np.flatnonzero(hull[y])
        left = xs.min() if bends_left else bound.min()
        right = bound.max() if bends_left else xs.max()
        result[y, left:right + 1] = True
    added = result & ~seed
    if added.sum() > seed.sum() * maximum_addition_fraction:
        return seed.copy(), {"method": "rejected_excessive_row_completion", "proposed_inferred_pixels": int(added.sum()), "inferred_pixels": 0}
    return result, {"method": "row_concavity_preserved_outer_convex_bound", "bend_direction": "left" if bends_left else "right", "inferred_pixels": int(added.sum()), "boundary_confirmation": "inferred_between_geometry_gap_bodies; RGB edge review remains required"}


def extract_ceiling_edge_gaps(semantic, points, normals, valid, intrinsics, plane,
                              config=None, *, supplemental_seeds=None):
    cfg = validated_config(config)
    semantic = np.asarray(semantic)
    if semantic.ndim != 2 or semantic.dtype != bool or not semantic.any():
        raise ValueError("A nonempty automatic boolean ceiling seed is required")
    shape = semantic.shape
    k = np.asarray(intrinsics, dtype=float)
    if k.shape != (3, 3) or not np.isfinite(k).all() or abs(np.linalg.det(k)) < 1e-12:
        raise ValueError("Normalized intrinsics must be finite and invertible")
    n = np.asarray(plane.get("normal"), dtype=float)
    d = float(plane.get("d", np.nan))
    if n.shape != (3,) or not np.isfinite(n).all() or not np.isclose(np.linalg.norm(n), 1) or not np.isfinite(d) or abs(d) < 1e-10:
        raise ValueError("A saved unit-normal plane separate from the camera is required")
    threshold = float(plane.get("metadata", {}).get("effective_plane_distance_threshold", np.nan))
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("A positive saved plane-distance tolerance is required")
    result = recover_ceiling_support(semantic, points, normals, valid, plane,
        supplemental_seeds=supplemental_seeds, config={
            "normal_angle_degrees": cfg["normal_angle_degrees"],
            "reference_offset_quantiles": cfg["reference_offset_quantiles"],
            "offset_margin_median_depth_fraction": cfg["offset_margin_median_depth_fraction"],
            "closing_radius_long_edge_fraction": cfg["connectivity_closing_long_edge_fraction"],
        })
    recovered = result["recovered_ceiling"]
    nc, labels, stats, _ = cv2.connectedComponentsWithStats(recovered.astype(np.uint8), connectivity=8)
    primary = np.zeros(shape, bool)
    support_records = []
    for i in range(1, nc):
        mask = labels == i
        anchors = int((mask & semantic).sum())
        top_contact = int(mask[0].sum())
        accepted = anchors >= cfg["minimum_semantic_anchor_pixels"] or top_contact >= cfg["minimum_top_contact_pixels"]
        if accepted:
            primary |= mask
        if stats[i, cv2.CC_STAT_AREA] >= cfg["minimum_semantic_anchor_pixels"]:
            support_records.append({"id": i, "pixels": int(mask.sum()), "original_semantic_anchor_pixels": anchors, "top_boundary_pixels": top_contact, "accepted": accepted})
    if not primary.any():
        raise ValueError("No primary upper slab support is anchored by original semantics or image top")
    radius = round(max(shape) * cfg["support_closing_long_edge_fraction"])
    support = primary.copy()
    if radius:
        support = cv2.morphologyEx(primary.astype(np.uint8), cv2.MORPH_CLOSE, _kernel(radius)).astype(bool) | primary
    points = np.asarray(points)
    usable = valid & np.isfinite(points).all(-1) & (points[..., 2] > 0)
    away = -np.sign(d) * (points @ n + d)
    recession = usable & (away > threshold * cfg["recession_distance_multiplier"])
    y, x = np.indices(shape, dtype=float)
    rays = np.stack([(x + .5) / shape[1], (y + .5) / shape[0], np.ones(shape)], -1) @ np.linalg.inv(k).T
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = -d / (rays @ n)
    positive_ray = np.isfinite(depth) & (depth > 0)
    last_row = int(np.nonzero(primary)[0].max())
    additions, closures = [], []
    for side, column in (("left", 0), ("right", shape[1] - 1)):
        border_gap = (~support[:last_row + 1, column]) & positive_ray[:last_row + 1, column]
        coverage = int((border_gap & usable[:last_row + 1, column]).sum())
        fraction = int((border_gap & recession[:last_row + 1, column]).sum()) / max(1, coverage)
        accepted = coverage > 0 and fraction >= cfg["minimum_border_recession_fraction"]
        if accepted:
            first_row = int(np.flatnonzero(positive_ray[:last_row + 1, column])[0])
            additions.append([column, first_row])
            additions.append([column, last_row])
        closures.append({"side": side, "positive_depth_border_gap_pixels": int(border_gap.sum()), "valid_pixels": coverage, "recession_fraction": fraction, "border_closed": accepted, "inferred_end_row": last_row if accepted else None})
    envelope = _hull(support, additions)
    domain = envelope & positive_ray
    gaps = domain & ~support
    nc, labels, stats, _ = cv2.connectedComponentsWithStats(gaps.astype(np.uint8), connectivity=8)
    seeds, uncertain = np.zeros(shape, bool), np.zeros(shape, bool)
    components = []
    for i in range(1, nc):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < semantic.size * cfg["minimum_candidate_image_fraction"]:
            continue
        mask = labels == i
        x, y, w, h = stats[i, :4]
        aspect = max(w / h, h / w)
        geo = int((mask & usable).sum())
        fraction = int((mask & recession).sum()) / max(1, geo)
        reasons = []
        if fraction < cfg["minimum_candidate_recession_fraction"]:
            reasons.append("insufficient_positive_recession")
        if aspect > cfg["maximum_candidate_aspect_ratio"]:
            reasons.append("elongated_exterior_band_may_include_other_storey_or_solid_slab")
        if min(w, h) < max(shape) * cfg["minimum_candidate_extent_long_edge_fraction"]:
            reasons.append("thin_groove_or_light_feature")
        accepted = not reasons
        if accepted:
            seeds |= mask
        else:
            uncertain |= mask
        components.append({"id": i, "area_pixels": area, "bbox_xywh": [int(x), int(y), int(w), int(h)], "positive_recession_fraction": fraction, "accepted_as_candidate": accepted, "status": "ambiguous_single_plane_slab_gap_candidate" if accepted else "excluded_uncertain_exterior_gap", "reasons": reasons})
    group_radius = round(max(shape) * cfg["grouping_radius_long_edge_fraction"])
    grouped = cv2.dilate(seeds.astype(np.uint8), _kernel(group_radius)) if group_radius else seeds.astype(np.uint8)
    ng, groups, _, _ = cv2.connectedComponentsWithStats(grouped, connectivity=8)
    candidate = seeds.copy()
    completions = []
    for i in range(1, ng):
        seed = seeds & (groups == i)
        count = len(np.unique(labels[seed])) - int(0 in labels[seed])
        filled, record = complete_curved_mouth(seed, count, cfg["maximum_inferred_to_seed_fraction"])
        filled &= domain
        candidate |= filled
        completions.append({"group": i, "geometry_gap_components": count, "seed_pixels": int(seed.sum()), "final_pixels": int(filled.sum()), **record})
    observed = candidate & recession
    completion = candidate & ~seeds
    return {
        "original_semantic": semantic, "recovered_support": support,
        "primary_geometry_support": primary, "search_envelope": envelope,
        "search_domain": domain, "positive_ray_domain": positive_ray,
        "raw_gap_mask": gaps, "geometry_gap_seed_mask": seeds,
        "opening_candidate_mask": candidate, "observed_positive_recession_mask": observed,
        "inferred_between_gap_completion_mask": completion,
        "unconfirmed_candidate_mask": candidate.copy(),
        "excluded_uncertain_gap_mask": uncertain,
        "signed_away": away, "components": components, "support_components": support_records,
        "completion_records": completions, "border_closure_records": closures,
        "metadata": {"method": "upper_semantic_anchored_support_then_recession_gated_border_envelope_gaps_with_concavity_preserved_well_completion", "surface_role": "ceiling", "effective_config": cfg, "opening_annotation_read": False, "manual_slab_mask_read": False, "same_single_plane_reused": True, "plane_refitted": False, "original_semantic_pixels": int(semantic.sum()), "original_semantic_fraction": float(semantic.mean()), "recovered_slab_support_pixels": int(support.sum()), "candidate_pixels": int(candidate.sum()), "observed_positive_recession_pixels": int(observed.sum()), "inferred_between_gap_pixels": int(completion.sum()), "excluded_uncertain_pixels": int(uncertain.sum()), "physical_through_slab_void_confirmed": False, "recovery": result["metadata"], "interpretation": "All delivered openings are visible-footprint candidates. Positive distance is geometry recession evidence, not proof of a physical void. Border closure and well/frame additions are inferred. The initial semantic ceiling may cover only a small roof patch; the offset band is deliberately broad and other-storey surfaces can remain ambiguous. Long exterior bands are excluded separately."},
    }
