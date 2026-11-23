from __future__ import annotations

from collections.abc import Mapping, Sequence

import cv2
import numpy as np


DEFAULT_CONFIG = {
    "maximum_boundary_shift_pixels": 4,
    "minimum_gradient": 12.0,
    "minimum_edge_support_fraction": 0.60,
    "minimum_score_improvement": 1.20,
    "minimum_distinct_peak_ratio": 1.15,
    "maximum_edge_samples": 512,
}


def _config(config):
    if config is not None and not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    values = {**DEFAULT_CONFIG, **(config or {})}
    unknown = set(values) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"Unknown boundary configuration: {sorted(unknown)}")
    for key, value in values.items():
        if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or not np.isfinite(value):
            raise ValueError(f"{key} must be finite numeric")
        if key in ("maximum_boundary_shift_pixels", "maximum_edge_samples"):
            minimum = 0 if key == "maximum_boundary_shift_pixels" else 8
            if int(value) != value or value < minimum:
                raise ValueError(f"{key} must be an integer >= {minimum}")
            values[key] = int(value)
        elif key == "minimum_edge_support_fraction":
            if not 0 < value <= 1:
                raise ValueError(f"{key} must be in (0, 1]")
        elif key.endswith("ratio") or key == "minimum_score_improvement":
            if value <= 1:
                raise ValueError(f"{key} must exceed 1")
        elif value <= 0:
            raise ValueError(f"{key} must be positive")
    if values["maximum_boundary_shift_pixels"] > 64:
        raise ValueError("Boundary shifts must be <= 64 native pixels")
    return values


def _shape(shape):
    if len(shape) != 2 or any(isinstance(v, (bool, np.bool_)) or int(v) != v or v <= 0 for v in shape):
        raise ValueError("shape must contain positive integer height and width")
    return tuple(int(v) for v in shape)


def _mask(value, shape, name):
    if value is None:
        return np.zeros(shape, bool)
    array = np.asarray(value)
    if array.shape != shape or array.dtype != bool:
        raise ValueError(f"{name} must be boolean on the native RGB H x W grid")
    return array.copy()


def _cross(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def _polygon(polygon, shape):
    points = np.asarray(polygon, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
        raise ValueError("Each polygon must contain at least three finite [x, y] vertices")
    if len(points) > 3 and np.allclose(points[0], points[-1]):
        points = points[:-1]
    h, w = shape
    if np.any(points < 0) or np.any(points[:, 0] > w - 1) or np.any(points[:, 1] > h - 1):
        raise ValueError("Polygon vertices must use native pixel coordinates inside the RGB image")
    if np.any(np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1) < 0.5):
        raise ValueError("Polygon edges must have distinct vertices")
    if abs(cv2.contourArea(points.astype(np.float32))) < 0.5:
        raise ValueError("Polygon must have positive area")
    for i in range(len(points)):
        a, b = points[i], points[(i + 1) % len(points)]
        for j in range(i + 1, len(points)):
            if j == i + 1 or (i == 0 and j == len(points) - 1):
                continue
            c, d = points[j], points[(j + 1) % len(points)]
            u, v = b - a, d - c
            denominator = _cross(u, v)
            if abs(denominator) < 1e-8:
                continue
            t = _cross(c - a, v) / denominator
            s = _cross(c - a, u) / denominator
            if -1e-8 <= t <= 1 + 1e-8 and -1e-8 <= s <= 1 + 1e-8:
                raise ValueError("Polygon outline must not self-intersect")
    return points


def rasterize_polygons(shape, polygons):
    shape = _shape(shape)
    if not isinstance(polygons, Sequence) and not isinstance(polygons, np.ndarray):
        raise TypeError("polygons must be a sequence of polygon vertex sequences")
    output = np.zeros(shape, np.uint8)
    for polygon in polygons:
        points = _polygon(polygon, shape)
        cv2.fillPoly(output, [np.rint(points).astype(np.int32)], 1)
    return output.astype(bool)


def _rgb_proposals(image, polygons, cfg):
    shape = image.shape[:2]
    proposals = np.zeros(shape, bool)
    records = []
    if not polygons or not cfg["maximum_boundary_shift_pixels"]:
        return proposals, records
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray = cv2.GaussianBlur(gray, (3, 3), 0.6)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3, scale=0.125)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3, scale=0.125)
    radius = cfg["maximum_boundary_shift_pixels"]
    offsets = np.arange(-radius, radius + 1)
    for index, raw in enumerate(polygons):
        points = _polygon(raw, shape)
        lines = []
        edges = []
        shifts = 0
        for a, b in zip(points, np.roll(points, -1, axis=0)):
            tangent = b - a
            length = float(np.linalg.norm(tangent))
            normal = np.array([-tangent[1], tangent[0]]) / length
            constant = float(normal @ a)
            sample_count = min(cfg["maximum_edge_samples"], max(8, int(length)))
            locations = a[None, :] + np.linspace(0.1, 0.9, sample_count)[:, None] * tangent[None, :]
            score = []
            support = []
            for offset in offsets:
                samples = (locations + normal[None, :] * offset).astype(np.float32)
                x, y = samples[:, 0, None], samples[:, 1, None]
                projected = np.abs(cv2.remap(gx, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT) * normal[0]
                                   + cv2.remap(gy, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT) * normal[1])
                score.append(float(np.mean(projected)))
                support.append(float(np.mean(projected >= cfg["minimum_gradient"])))
            best = int(np.argmax(score))
            offset = int(offsets[best])
            alternatives = [score[k] for k in range(len(offsets)) if abs(int(offsets[k]) - offset) > 1]
            distinct = not alternatives or score[best] > max(alternatives) * cfg["minimum_distinct_peak_ratio"]
            supported = support[best] >= cfg["minimum_edge_support_fraction"]
            improved = score[best] > max(score[radius], 1e-6) * cfg["minimum_score_improvement"]
            move = offset != 0 and supported and improved and distinct
            if move:
                constant += offset
                shifts += 1
            lines.append((normal, constant))
            edges.append({"offset_pixels": offset if move else 0,
                          "status": "RGB_boundary_proposal" if move else "original_reviewed_boundary_retained",
                          "peak_gradient": score[best], "peak_support_fraction": support[best],
                          "distinct_peak": bool(distinct)})
        record = {"polygon_index": index, "shifted_edges": shifts, "edges": edges,
                  "original_vertices_xy": points.tolist(), "accepted_without_review": False}
        if not shifts:
            record["status"] = "no_distinct_supported_local_RGB_adjustment"
            records.append(record)
            continue
        vertices = []
        stable = True
        for k in range(len(lines)):
            matrix = np.stack((lines[k - 1][0], lines[k][0]))
            if abs(np.linalg.det(matrix)) < 1e-6:
                stable = False
                break
            point = np.linalg.solve(matrix, [lines[k - 1][1], lines[k][1]])
            if np.linalg.norm(point - points[k]) > 2.5 * radius:
                stable = False
                break
            vertices.append(point)
        try:
            adjusted = _polygon(vertices, shape) if stable else None
        except ValueError:
            adjusted = None
        if adjusted is None:
            record["status"] = "rejected_unstable_line_intersections"
        else:
            proposals |= rasterize_polygons(shape, [adjusted])
            record.update(status="local_RGB_outline_requires_review", proposed_vertices_xy=adjusted.tolist())
        records.append(record)
    return proposals, records


def refine_wall_regions(rgb, wall_domain, transparent_seed, *, solid_seed=None,
                        uncertain_seed=None, foreground_occlusion=None,
                        transparent_regions_complete=False, region_polygons=None,
                        config=None, provenance=None):
    image = np.asarray(rgb)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or 0 in image.shape:
        raise ValueError("rgb must be a nonempty native-grid H x W x 3 uint8 RGB image")
    shape = image.shape[:2]
    domain = _mask(wall_domain, shape, "wall_domain")
    transparent = _mask(transparent_seed, shape, "transparent_seed")
    solid = _mask(solid_seed, shape, "solid_seed")
    uncertain = _mask(uncertain_seed, shape, "uncertain_seed")
    occlusion = _mask(foreground_occlusion, shape, "foreground_occlusion")
    if not isinstance(transparent_regions_complete, (bool, np.bool_)):
        raise ValueError("transparent_regions_complete must be an explicit boolean")
    if provenance is not None and not isinstance(provenance, Mapping):
        raise TypeError("provenance must be a mapping")
    cfg = _config(config)
    polygons = list(region_polygons) if region_polygons is not None else []
    if np.any(transparent & solid):
        raise ValueError("Reviewed transparent and solid seeds overlap")
    source_counts = {"transparent_seed": int(transparent.sum()), "solid_seed": int(solid.sum()),
                     "uncertain_seed": int(uncertain.sum()), "wall_domain": int(domain.sum()),
                     "foreground_occlusion": int(occlusion.sum())}
    visible = domain & ~occlusion
    uncertain &= visible
    transparent &= visible & ~uncertain
    solid &= visible & ~uncertain & ~transparent
    proposals, boundary_records = _rgb_proposals(image, polygons, cfg)
    proposals &= visible
    proposal_additions = proposals & ~transparent
    uncertain |= proposal_additions & ~solid
    if transparent_regions_complete:
        solid |= visible & ~transparent & ~uncertain
    else:
        uncertain |= visible & ~transparent & ~solid
    other = ~visible
    masks = {"wall_domain": domain, "wall_visible_domain": visible,
             "wall_transparent": transparent, "wall_solid": solid,
             "wall_uncertain": uncertain, "other": other,
             "rgb_boundary_proposal": proposals,
             "rgb_boundary_additions_for_review": proposal_additions}
    report = {
        "method": "reviewed_wall_domain_and_opening_glass_regions_with_local_RGB_boundary_proposals",
        "status": "reviewed_segmentation_with_uncertain_regions" if uncertain.any() else "reviewed_segmentation",
        "annotation_assisted": True, "fully_automatic": False,
        "native_grid_height_width": list(shape),
        "transparent_definition": "reviewed wall openings plus glass interfaces",
        "transparent_regions_complete_reviewed": bool(transparent_regions_complete),
        "solid_policy": "reviewed complete visible wall domain minus transparent and unknown" if transparent_regions_complete
                        else "independently reviewed solid seed only; residual wall uncertain",
        "foreground_policy": "provided foreground occluders and non-wall pixels remain other",
        "RGB_policy": "local supported RGB shifts are separate proposals; never classify new glass automatically",
        "physical_optical_transmission_measured": False,
        "config": cfg, "provenance": dict(provenance or {}),
        "input_mask_pixels": source_counts,
        "boundary_proposals": boundary_records,
        "mask_pixels": {key: int(value.sum()) for key, value in masks.items()},
        "clipped_transparent_seed_pixels": source_counts["transparent_seed"] - int(transparent.sum()),
    }
    classified = transparent.astype(np.uint8) + solid + uncertain + other
    if not np.all(classified == 1):
        raise AssertionError("Transparent/solid/uncertain/other must partition the native RGB grid")
    if np.any((transparent | solid | uncertain) & ~visible):
        raise AssertionError("Every wall class must stay inside visible wall domain")
    return {**masks, "report": report}
