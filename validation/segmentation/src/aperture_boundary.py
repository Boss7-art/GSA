from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import cv2
import numpy as np


DEFAULT_CONFIG = {
    "roi_padding_x_fraction": 0.55,
    "roi_padding_y_fraction": 1.0,
    "minimum_seed_pixels": 16,
    "minimum_segment_length_fraction": 0.18,
    "maximum_side_candidates": 14,
    "maximum_cap_candidates": 10,
    "maximum_alternatives": 8,
    "maximum_vertical_deviation_degrees": 40.0,
    "maximum_horizontal_deviation_degrees": 35.0,
    "canny_low": 35,
    "canny_high": 95,
    "edge_tolerance_fraction": 0.018,
    "minimum_edge_support": 0.60,
    "minimum_side_edge_support": 0.55,
    "minimum_segment_coverage": 0.45,
    "minimum_seed_recall": 0.32,
    "minimum_deep_seed_recall": 0.38,
    "minimum_area_seed_ratio": 0.30,
    "maximum_area_seed_ratio": 4.0,
    "minimum_outside_support": 0.25,
    "minimum_support_transition": 0.08,
    "support_band_fractions": [0.035, 0.055, 0.085],
    "minimum_supported_boundary_sides": 2,
    "strong_boundary_mean_edge_support": 0.90,
    "strong_lateral_boundary_contrast": 0.08,
    "contrast_band_fraction": 0.055,
    "minimum_lateral_boundary_contrast": 0.022,
    "minimum_score_margin": 0.035,
    "alternative_equivalence_iou": 0.90,
    "line_merge_tolerance_fraction": 0.018,
    "allow_image_boundary_cap": True,
    "seed_sample_limit": 2048,
}


def _config(config):
    cfg = dict(DEFAULT_CONFIG)
    if config is not None:
        if not isinstance(config, Mapping):
            raise TypeError("config must be a mapping")
        unknown = set(config) - set(cfg)
        if unknown:
            raise ValueError(f"Unknown aperture boundary configuration: {sorted(unknown)}")
        cfg.update(config)
    for key, value in cfg.items():
        if key == "support_band_fractions":
            array = np.asarray(value, dtype=float)
            if array.ndim != 1 or len(array) < 2 or not np.isfinite(array).all() or np.any(array <= 0) or np.any(array > 0.5):
                raise ValueError("support_band_fractions must contain at least two fractions in (0, 0.5]")
            cfg[key] = array.tolist()
        elif key == "allow_image_boundary_cap":
            if not isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{key} must be boolean")
        elif isinstance(value, (bool, np.bool_)) or not np.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
        elif key.endswith("candidates") or key in ("maximum_alternatives", "seed_sample_limit", "minimum_seed_pixels", "minimum_supported_boundary_sides"):
            if int(value) != float(value) or int(value) < 1:
                raise ValueError(f"{key} must be a positive integer")
            cfg[key] = int(value)
        else:
            cfg[key] = float(value)
    for key in ("minimum_edge_support", "minimum_side_edge_support", "minimum_segment_coverage",
                "minimum_seed_recall", "minimum_deep_seed_recall", "minimum_outside_support",
                "minimum_support_transition", "minimum_lateral_boundary_contrast",
                "strong_boundary_mean_edge_support", "strong_lateral_boundary_contrast",
                "minimum_score_margin", "alternative_equivalence_iou"):
        if cfg[key] > 1:
            raise ValueError(f"{key} must be in [0, 1]")
    if cfg["minimum_area_seed_ratio"] > cfg["maximum_area_seed_ratio"]:
        raise ValueError("minimum_area_seed_ratio exceeds maximum_area_seed_ratio")
    if cfg["canny_low"] >= cfg["canny_high"]:
        raise ValueError("canny_low must be below canny_high")
    if cfg["minimum_supported_boundary_sides"] > 4:
        raise ValueError("minimum_supported_boundary_sides must be in [1, 4]")
    return cfg


def _line(points, virtual=False):
    p = np.asarray(points, dtype=np.float64).reshape(2, 2)
    delta = p[1] - p[0]
    length = float(np.linalg.norm(delta))
    n = np.array([-delta[1], delta[0]]) / max(length, 1e-12)
    return {"points": p, "normal": n, "offset": -float(n @ p[0]),
            "length": length, "virtual": bool(virtual)}


def _intersect(a, b):
    matrix = np.stack((a["normal"], b["normal"]))
    if abs(np.linalg.det(matrix)) < 0.05:
        return None
    return np.linalg.solve(matrix, -np.array((a["offset"], b["offset"])))


def _x_at(line, y):
    n = line["normal"]
    return -(n[1] * y + line["offset"]) / n[0] if abs(n[0]) > 1e-8 else np.inf


def _y_at(line, x):
    n = line["normal"]
    return -(n[0] * x + line["offset"]) / n[1] if abs(n[1]) > 1e-8 else np.inf


def _contains(points, corners):
    delta = np.roll(corners, -1, axis=0) - corners
    rel = points[:, None, :] - corners[None, :, :]
    cross = delta[None, :, 0] * rel[..., 1] - delta[None, :, 1] * rel[..., 0]
    return (cross >= -1e-5).all(axis=1)


def _poly_mask(shape, corners):
    mask = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(mask, np.round(corners).astype(np.int32), 1)
    return mask.astype(bool)


def _samples(a, b, number=96):
    t = np.linspace(0.02, 0.98, number)
    return a[None] + t[:, None] * (b - a)[None]


def _lookup(array, points):
    h, w = array.shape
    x = np.clip(np.round(points[:, 0]).astype(int), 0, w - 1)
    y = np.clip(np.round(points[:, 1]).astype(int), 0, h - 1)
    return array[y, x]


def _colour_lookup(array, points):
    h, w = array.shape[:2]
    x = np.clip(np.round(points[:, 0]).astype(int), 0, w - 1)
    y = np.clip(np.round(points[:, 1]).astype(int), 0, h - 1)
    return array[y, x].astype(np.float64) / 255.0


def _merge_lines(lines, tolerance):
    groups = []
    for line in sorted(lines, key=lambda value: value["length"], reverse=True):
        merged = False
        for i, other in enumerate(groups):
            angle = abs(float(line["normal"] @ other["normal"]))
            separation = np.max(abs(line["points"] @ other["normal"] + other["offset"]))
            if angle >= np.cos(np.deg2rad(3.0)) and separation <= tolerance:
                all_points = np.concatenate((other["points"], line["points"]))
                tangent = np.array([other["normal"][1], -other["normal"][0]])
                projection = all_points @ tangent
                endpoints = all_points[[np.argmin(projection), np.argmax(projection)]]
                groups[i] = _line(endpoints)
                merged = True
                break
        if not merged:
            groups.append(line)
    return groups


def propose_aperture_boundary(rgb, seed, ceiling=None, envelope=None, support=None, config=None):
    cfg = _config(config)
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("rgb must have shape H x W x 3")
    if image.dtype != np.uint8:
        raise ValueError("rgb must be uint8 RGB")
    shape = image.shape[:2]
    masks = {}
    for name, value in (("seed", seed), ("ceiling", ceiling), ("envelope", envelope), ("support", support)):
        if value is None:
            masks[name] = None
        else:
            array = np.asarray(value)
            if array.shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
            masks[name] = array.astype(bool)
    if masks["seed"] is None:
        raise ValueError("seed is required")
    blank = np.zeros(shape, dtype=bool)
    metadata = {"method": "geometry_seeded_LSD_perspective_mouth_boundary", "config": cfg,
                "geometry_seed_required": True, "through_slab_void_confirmed": False,
                "assumption": "Near-rectangular aperture viewed in perspective; RGB alone cannot establish through-slab topology.",
                "rejection_reasons": [], "ambiguity_reasons": []}
    result = {"proposal_mask": blank.copy(), "accepted_mask": blank.copy(), "corners_xy": [],
              "status": "rejected", "segments": [], "alternatives": [], "edge_map": blank.copy(),
              "metadata": metadata}
    ys, xs = np.nonzero(masks["seed"])
    if len(xs) < cfg["minimum_seed_pixels"]:
        metadata["rejection_reasons"].append("insufficient_geometry_seed")
        return result
    h, w = shape
    sx0, sx1, sy0, sy1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    sw, sh = sx1 - sx0 + 1, sy1 - sy0 + 1
    px = int(np.ceil(sw * cfg["roi_padding_x_fraction"]))
    py = int(np.ceil(sh * cfg["roi_padding_y_fraction"]))
    x0, x1 = max(0, sx0 - px), min(w, sx1 + 1 + px)
    y0, y1 = max(0, sy0 - py), min(h, sy1 + 1 + py)
    metadata["roi_xyxy"] = [x0, y0, x1, y1]
    metadata["seed_bbox_xywh"] = [sx0, sy0, sw, sh]
    metadata["seed_area_pixels"] = len(xs)
    local = image[y0:y1, x0:x1]
    gray = cv2.cvtColor(local, cv2.COLOR_RGB2GRAY)
    smooth = cv2.GaussianBlur(gray, (3, 3), 0.7)
    edge = cv2.Canny(smooth, cfg["canny_low"], cfg["canny_high"])
    result["edge_map"][y0:y1, x0:x1] = edge > 0
    edge_distance = cv2.distanceTransform((edge == 0).astype(np.uint8), cv2.DIST_L2, 3)
    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    detected = detector.detect(smooth)[0]
    lines = []
    minimum_length = max(3.0, min(sw, sh) * cfg["minimum_segment_length_fraction"])
    if detected is not None:
        for segment in detected.reshape(-1, 4):
            line = _line(segment.reshape(2, 2))
            if line["length"] >= minimum_length:
                lines.append(line)
    merge_tolerance = max(1.2, min(sw, sh) * cfg["line_merge_tolerance_fraction"])
    lines = _merge_lines(lines, merge_tolerance)
    result["segments"] = [{"endpoints_xy": (line["points"] + [x0, y0]).tolist(),
                           "length_pixels": line["length"], "virtual": False} for line in lines]
    metadata["detected_segment_count"] = len(lines)
    centre = np.array([np.median(xs) - x0, np.median(ys) - y0])
    local_seed = masks["seed"][y0:y1, x0:x1]
    local_support = (masks["support"] if masks["support"] is not None else masks["ceiling"])
    local_support = None if local_support is None else local_support[y0:y1, x0:x1]
    seed_distance = cv2.distanceTransform(local_seed.astype(np.uint8), cv2.DIST_L2, 3)
    dy, dx = np.nonzero(seed_distance >= max(1.0, float(seed_distance.max()) * 0.45))
    seed_xy = np.column_stack((xs - x0, ys - y0)).astype(float)
    deep_xy = np.column_stack((dx, dy)).astype(float)
    if len(seed_xy) > cfg["seed_sample_limit"]:
        seed_xy = seed_xy[np.linspace(0, len(seed_xy) - 1, cfg["seed_sample_limit"]).astype(int)]
    if len(deep_xy) > cfg["seed_sample_limit"]:
        deep_xy = deep_xy[np.linspace(0, len(deep_xy) - 1, cfg["seed_sample_limit"]).astype(int)]
    sides = []
    caps = []
    for line in lines:
        delta = line["points"][1] - line["points"][0]
        angle_horizontal = np.rad2deg(np.arctan2(abs(delta[1]), abs(delta[0])))
        if 90 - angle_horizontal <= cfg["maximum_vertical_deviation_degrees"]:
            line_y = np.sort(line["points"][:, 1])
            overlap = max(0.0, min(line_y[1], sy1 - y0) - max(line_y[0], sy0 - y0))
            if overlap > min(3, sh * 0.05):
                sides.append(line)
        if angle_horizontal <= cfg["maximum_horizontal_deviation_degrees"]:
            caps.append(line)
    left = sorted([line for line in sides if _x_at(line, centre[1]) < centre[0]],
                  key=lambda line: line["length"], reverse=True)[:cfg["maximum_side_candidates"]]
    right = sorted([line for line in sides if _x_at(line, centre[1]) > centre[0]],
                   key=lambda line: line["length"], reverse=True)[:cfg["maximum_side_candidates"]]
    top = sorted([line for line in caps if _y_at(line, centre[0]) < centre[1]],
                 key=lambda line: abs(_y_at(line, centre[0]) - (sy0 - y0)))[:cfg["maximum_cap_candidates"]]
    bottom = sorted([line for line in caps if _y_at(line, centre[0]) > centre[1]],
                    key=lambda line: abs(_y_at(line, centre[0]) - (sy1 - y0)))[:cfg["maximum_cap_candidates"]]
    if cfg["allow_image_boundary_cap"] and sy0 == 0:
        top = [_line([[0, 0], [x1 - x0 - 1, 0]], virtual=True)] + top
    if cfg["allow_image_boundary_cap"] and sy1 == h - 1:
        bottom = [_line([[0, y1 - y0 - 1], [x1 - x0 - 1, y1 - y0 - 1]], virtual=True)] + bottom
    metadata["line_family_counts"] = {"left": len(left), "right": len(right), "top": len(top), "bottom": len(bottom)}
    tolerance = max(1.5, min(sw, sh) * cfg["edge_tolerance_fraction"])
    proposals = []
    evaluated = 0
    for ll in left:
        for rr in right:
            width = _x_at(rr, centre[1]) - _x_at(ll, centre[1])
            if width < max(3, sw * 0.20) or width > sw * 2.0:
                continue
            for tt in top:
                for bb in bottom:
                    evaluated += 1
                    vertices = [_intersect(ll, tt), _intersect(rr, tt), _intersect(rr, bb), _intersect(ll, bb)]
                    if any(vertex is None for vertex in vertices):
                        continue
                    corners = np.stack(vertices)
                    if not np.isfinite(corners).all() or corners.min() < -2:
                        continue
                    if np.max(corners[:, 0]) >= x1 - x0 + 2 or np.max(corners[:, 1]) >= y1 - y0 + 2:
                        continue
                    if not cv2.isContourConvex(corners.astype(np.float32)):
                        continue
                    area = float(cv2.contourArea(corners.astype(np.float32)))
                    area_ratio = area / len(xs)
                    if not cfg["minimum_area_seed_ratio"] <= area_ratio <= cfg["maximum_area_seed_ratio"]:
                        continue
                    recall = float(np.mean(_contains(seed_xy, corners)))
                    if recall < cfg["minimum_seed_recall"]:
                        continue
                    deep_recall = float(np.mean(_contains(deep_xy, corners))) if len(deep_xy) else recall
                    if deep_recall < cfg["minimum_deep_seed_recall"]:
                        continue
                    edge_fractions, coverages, transitions, outside_supports, contrasts = [], [], [], [], []
                    multi_band_support = []
                    boundary_lines = (tt, rr, bb, ll)
                    virtual_sides = []
                    for i, line in enumerate(boundary_lines):
                        a, b = corners[i], corners[(i + 1) % 4]
                        sample = _samples(a, b)
                        length = float(np.linalg.norm(b - a))
                        tangent = (b - a) / max(length, 1e-12)
                        outside = np.array([tangent[1], -tangent[0]])
                        if line["virtual"]:
                            virtual_sides.append(i)
                            edge_fractions.append(None)
                            coverages.append(None)
                            contrasts.append(None)
                            continue
                        edge_fractions.append(float(np.mean(_lookup(edge_distance, sample) <= tolerance)))
                        segment_proj = (line["points"] - a) @ tangent
                        coverages.append(float(max(0, min(length, segment_proj.max()) - max(0, segment_proj.min())) / max(length, 1e-12)))
                        band = max(2.0, min(sw, sh) * cfg["contrast_band_fraction"])
                        outer_colour = _colour_lookup(local, sample + outside * band)
                        inner_colour = _colour_lookup(local, sample - outside * band)
                        contrasts.append(float(np.median(np.linalg.norm(outer_colour - inner_colour, axis=1) / np.sqrt(3))))
                        if local_support is not None:
                            bands = []
                            for fraction in cfg["support_band_fractions"]:
                                ring_distance = max(2.0, min(sw, sh) * fraction)
                                outside_fraction = float(np.mean(_lookup(local_support, sample + outside * ring_distance)))
                                inside_fraction = float(np.mean(_lookup(local_support, sample - outside * ring_distance)))
                                bands.append({"band_distance_pixels": ring_distance,
                                              "outside_support_fraction": outside_fraction,
                                              "inside_support_fraction": inside_fraction,
                                              "transition": outside_fraction - inside_fraction})
                            outside_mean = float(np.mean([record["outside_support_fraction"] for record in bands]))
                            inside_mean = float(np.mean([record["inside_support_fraction"] for record in bands]))
                            outside_supports.append(outside_mean)
                            transitions.append(outside_mean - inside_mean)
                            multi_band_support.append({"side_index": i, "bands": bands})
                    visible = [value for value in edge_fractions if value is not None]
                    coverage_values = [value for value in coverages if value is not None]
                    if len(visible) < 3:
                        continue
                    average_edge = float(np.mean(visible))
                    min_edge = float(min(visible))
                    transition = float(np.mean(transitions)) if transitions else None
                    outside_support_mean = float(np.mean(outside_supports)) if outside_supports else None
                    score = (0.30 * average_edge + 0.23 * min_edge + 0.12 * np.mean(coverage_values)
                             + 0.18 * deep_recall + 0.07 * recall)
                    lateral_contrast = float(min(contrasts[1], contrasts[3]))
                    score += 0.15 * min(1.0, lateral_contrast / 0.10)
                    if transition is not None:
                        score += 0.07 * np.clip(transition, 0, 1) + 0.03 * outside_support_mean
                    else:
                        score += 0.08 * np.mean(coverage_values)
                    reasons = []
                    if average_edge < cfg["minimum_edge_support"]:
                        reasons.append("insufficient_average_RGB_edge_evidence")
                    if min_edge < cfg["minimum_side_edge_support"]:
                        reasons.append("one_or_more_visible_mouth_edges_incomplete")
                    if min(coverage_values) < cfg["minimum_segment_coverage"]:
                        reasons.append("excessive_line_extrapolation")
                    if lateral_contrast < cfg["minimum_lateral_boundary_contrast"]:
                        reasons.append("lateral_edges_are_thin_lines_without_mouth_region_transition")
                    if outside_support_mean is not None and outside_support_mean < cfg["minimum_outside_support"]:
                        reasons.append("outside_not_supported_by_observed_slab")
                    support_gate = "unavailable" if transition is None else "mean_visible_slab_transition"
                    partial_support = (transition is not None
                                       and np.count_nonzero(np.asarray(transitions) >= cfg["minimum_support_transition"]) >= cfg["minimum_supported_boundary_sides"]
                                       and average_edge >= cfg["strong_boundary_mean_edge_support"]
                                       and lateral_contrast >= cfg["strong_lateral_boundary_contrast"])
                    if transition is not None and transition < cfg["minimum_support_transition"]:
                        if partial_support:
                            support_gate = "partial_slab_support_with_strong_RGB_boundary"
                        else:
                            reasons.append("no_observed_slab_to_aperture_transition_coffer_or_internal_frame")
                    if masks["envelope"] is not None:
                        global_corners = corners + [x0, y0]
                        inside_fraction = float(np.mean(_lookup(masks["envelope"], _samples(global_corners[0], global_corners[2], 32))))
                        if inside_fraction < 0.9:
                            reasons.append("boundary_outside_expected_ceiling_domain")
                    proposals.append({"score": float(score), "corners": corners,
                                      "seed_recall": recall, "deep_seed_recall": deep_recall,
                                      "area_seed_ratio": area_ratio, "edge_support_per_side": edge_fractions,
                                      "segment_coverage_per_side": coverages, "mean_edge_support": average_edge,
                                      "outside_support_fraction": outside_support_mean,
                                      "support_transition": transition, "virtual_sides": virtual_sides,
                                      "outside_support_per_visible_side": outside_supports,
                                      "support_transition_per_visible_side": transitions,
                                      "multi_band_support_evidence": multi_band_support,
                                      "support_gate": support_gate,
                                      "RGB_boundary_contrast_per_side": contrasts,
                                      "minimum_lateral_boundary_contrast": lateral_contrast,
                                      "ambiguity_reasons": reasons})
    metadata["evaluated_combinations"] = evaluated
    metadata["viable_proposals"] = len(proposals)
    if not proposals:
        metadata["ambiguity_reasons"].append("no_four_sided_boundary_supported_in_search_region")
        result["status"] = "ambiguous"
        return result
    proposals.sort(key=lambda value: value["score"], reverse=True)
    distinct = []
    for proposal in proposals:
        poly = _poly_mask(local_seed.shape, proposal["corners"])
        same = False
        for previous, prev_poly in distinct:
            intersection = np.count_nonzero(poly & prev_poly)
            union = np.count_nonzero(poly | prev_poly)
            if union and intersection / union >= cfg["alternative_equivalence_iou"]:
                same = True
                break
        if not same:
            distinct.append((proposal, poly))
        if len(distinct) >= cfg["maximum_alternatives"]:
            break
    best, best_mask = distinct[0]
    alternative_margin = best["score"] - distinct[1][0]["score"] if len(distinct) > 1 else None
    reasons = list(best["ambiguity_reasons"])
    if alternative_margin is not None and alternative_margin < cfg["minimum_score_margin"]:
        reasons.append("competing_outer_and_inner_boundaries_have_similar_evidence")
    for rank, (proposal, _) in enumerate(distinct):
        record = {key: value for key, value in proposal.items() if key != "corners"}
        record["corners_xy"] = (proposal["corners"] + [x0, y0]).tolist()
        record["rank"] = rank + 1
        result["alternatives"].append(record)
    result["proposal_mask"][y0:y1, x0:x1] = best_mask
    result["corners_xy"] = (best["corners"] + [x0, y0]).tolist()
    result["status"] = "ambiguous" if reasons else "supported_boundary"
    if result["status"] == "supported_boundary":
        result["accepted_mask"] = result["proposal_mask"].copy()
    metadata["ambiguity_reasons"] = reasons
    metadata["best_score"] = best["score"]
    metadata["score_margin_to_distinct_alternative"] = alternative_margin
    metadata["visible_edge_count"] = 4 - len(best["virtual_sides"])
    metadata["image_boundary_virtual_sides"] = best["virtual_sides"]
    metadata["scope"] = "Boundary support only; the geometric seed remains an exploratory aperture hypothesis."
    return result
