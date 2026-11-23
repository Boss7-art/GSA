from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


class PlaneFittingError(ValueError):
    pass


DEFAULT_CONFIG: dict[str, Any] = {
    "ransac_iterations": 250,
    "ransac_max_points": 20000,
    "ransac_threshold": 0.015,
    "plane_distance_threshold": 0.025,
    "plane_normal_angle_threshold": 25.0,
    "minimum_plane_inlier_ratio": 0.35,
    "mask_boundary_erosion": 2,
    "depth_outlier_quantiles": [0.01, 0.99],
    "trim_depth_for_support": False,
    "minimum_points": 30,
    "minimum_confidence": 0.0,
    "distance_units": "relative_to_median_depth",
    "seed": 42,
}


def _integer(value: Any, key: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{key} must be an integer >= {minimum}.")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{key} must be an integer >= {minimum}.") from exc
    if result != value or result < minimum:
        raise ValueError(f"{key} must be an integer >= {minimum}.")
    return result


def _number(value: Any, key: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{key} must be finite.") from exc
    if not np.isfinite(result):
        raise ValueError(f"{key} must be finite.")
    return result


def _validated_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    if config is not None:
        if not isinstance(config, Mapping):
            raise TypeError("config must be a mapping or None.")
        cfg.update(config)
    for key, minimum in (
        ("ransac_iterations", 1),
        ("ransac_max_points", 3),
        ("minimum_points", 3),
        ("mask_boundary_erosion", 0),
        ("seed", 0),
    ):
        cfg[key] = _integer(cfg[key], key, minimum)
    for key in (
        "ransac_threshold",
        "plane_distance_threshold",
        "plane_normal_angle_threshold",
        "minimum_plane_inlier_ratio",
        "minimum_confidence",
    ):
        cfg[key] = _number(cfg[key], key)
    for key in ("ransac_threshold", "plane_distance_threshold"):
        if cfg[key] <= 0:
            raise ValueError(f"{key} must be positive.")
    if not 0 <= cfg["plane_normal_angle_threshold"] <= 90:
        raise ValueError("plane_normal_angle_threshold must be between 0 and 90 degrees.")
    if not 0 <= cfg["minimum_plane_inlier_ratio"] <= 1:
        raise ValueError("minimum_plane_inlier_ratio must be between 0 and 1.")
    if not isinstance(cfg["trim_depth_for_support"], (bool, np.bool_)):
        raise ValueError("trim_depth_for_support must be a boolean.")
    cfg["trim_depth_for_support"] = bool(cfg["trim_depth_for_support"])
    if cfg["distance_units"] not in ("relative_to_median_depth", "absolute"):
        raise ValueError("distance_units must be 'relative_to_median_depth' or 'absolute'.")
    quantiles = cfg["depth_outlier_quantiles"]
    if quantiles is None:
        quantiles = (0.0, 1.0)
    if isinstance(quantiles, (str, bytes)):
        raise ValueError("depth_outlier_quantiles must contain two quantiles between 0 and 1.")
    try:
        if len(quantiles) != 2:
            raise ValueError("depth_outlier_quantiles must contain exactly two values.")
        low = _number(quantiles[0], "depth_outlier_quantiles[0]")
        high = _number(quantiles[1], "depth_outlier_quantiles[1]")
    except TypeError as exc:
        raise ValueError("depth_outlier_quantiles must contain two values.") from exc
    if not 0 <= low < high <= 1:
        raise ValueError("depth_outlier_quantiles must satisfy 0 <= low < high <= 1.")
    cfg["depth_outlier_quantiles"] = (low, high)
    return cfg


def _erode_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    result = mask.copy()
    for _ in range(radius):
        if not result.any():
            break
        padded = np.pad(result, 1, constant_values=False)
        height, width = result.shape
        eroded = np.ones_like(result)
        for row in range(3):
            for col in range(3):
                eroded &= padded[row : row + height, col : col + width]
        result = eroded
    return result


def _canonical_plane(normal: np.ndarray, d: float) -> tuple[np.ndarray, float]:
    if normal[np.argmax(np.abs(normal))] < 0:
        return -normal, -float(d)
    return normal, float(d)


def _svd_plane(points: np.ndarray) -> tuple[np.ndarray, float]:
    if len(points) < 3:
        raise PlaneFittingError("At least three ceiling points are required to define a plane.")
    centroid = points.mean(axis=0)
    centered = points - centroid
    scatter = centered.T @ centered
    if not np.isfinite(scatter).all():
        raise PlaneFittingError("Ceiling coordinates overflowed during plane fitting.")
    _, singular_values, vectors = np.linalg.svd(scatter, full_matrices=False)
    if singular_values[0] <= 0 or singular_values[1] <= singular_values[0] * 1e-12:
        raise PlaneFittingError("Ceiling points are coincident or collinear; no unique plane exists.")
    normal = vectors[-1]
    normal = normal / np.linalg.norm(normal)
    return _canonical_plane(normal, -float(centroid @ normal))


def _three_point_plane(points: np.ndarray) -> tuple[np.ndarray, float] | None:
    edge_a = points[1] - points[0]
    edge_b = points[2] - points[0]
    lengths = np.linalg.norm([edge_a, edge_b], axis=1)
    if not np.isfinite(lengths).all() or np.any(lengths == 0):
        return None
    cross = np.cross(edge_a / lengths[0], edge_b / lengths[1])
    magnitude = np.linalg.norm(cross)
    if not np.isfinite(magnitude) or magnitude <= 1e-7:
        return None
    normal = cross / magnitude
    return _canonical_plane(normal, -float(points[0] @ normal))


def _refine_consensus(
    points: np.ndarray, normal: np.ndarray, d: float, threshold: float
) -> tuple[np.ndarray, float, np.ndarray]:
    inliers = np.abs(points @ normal + d) <= threshold
    for _ in range(5):
        try:
            refined_normal, refined_d = _svd_plane(points[inliers])
        except PlaneFittingError:
            break
        next_inliers = np.abs(points @ refined_normal + refined_d) <= threshold
        if np.count_nonzero(next_inliers) < 3:
            break
        normal, d = refined_normal, refined_d
        unchanged = np.array_equal(inliers, next_inliers)
        inliers = next_inliers
        if unchanged:
            break
    return normal, d, inliers


def fit_ceiling_plane(
    points: np.ndarray,
    ceiling: np.ndarray,
    valid: np.ndarray,
    normals: np.ndarray | None = None,
    confidence: np.ndarray | None = None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = _validated_config(config)
    xyz = np.asarray(points, dtype=np.float64)
    if xyz.ndim != 3 or xyz.shape[-1] != 3:
        raise ValueError("points must have shape (height, width, 3).")
    shape = xyz.shape[:2]
    semantic = np.asarray(ceiling, dtype=bool)
    supplied_valid = np.asarray(valid, dtype=bool)
    if semantic.shape != shape or supplied_valid.shape != shape:
        raise ValueError("ceiling and valid masks must match the points' image dimensions.")
    normal_values = None
    if normals is not None:
        normal_values = np.asarray(normals, dtype=np.float64)
        if normal_values.shape != xyz.shape:
            raise ValueError("normals must have the same shape as points.")
    confidence_values = None
    if confidence is not None:
        confidence_values = np.asarray(confidence, dtype=np.float64)
        if confidence_values.shape != shape:
            raise ValueError("confidence must match the points' image dimensions.")

    warnings: list[str] = []
    geometry_valid = supplied_valid & np.isfinite(xyz).all(axis=2) & (xyz[..., 2] > 0)
    confidence_eligible = np.ones(shape, dtype=bool)
    if confidence_values is not None:
        confidence_eligible = np.isfinite(confidence_values) & (
            confidence_values >= cfg["minimum_confidence"]
        )
    elif cfg["minimum_confidence"] > 0:
        warnings.append("minimum_confidence was configured, but no confidence array was provided.")

    eroded = _erode_mask(semantic, cfg["mask_boundary_erosion"])
    before_depth_filter = eroded & geometry_valid & confidence_eligible
    preliminary_count = int(np.count_nonzero(before_depth_filter))
    if preliminary_count < cfg["minimum_points"]:
        raise PlaneFittingError(
            f"Only {preliminary_count} eligible ceiling points remain after validity, "
            f"confidence and boundary filtering; need {cfg['minimum_points']}."
        )
    low_q, high_q = cfg["depth_outlier_quantiles"]
    depth_low, depth_high = np.quantile(xyz[..., 2][before_depth_filter], [low_q, high_q])
    depth_eligible = (xyz[..., 2] >= depth_low) & (xyz[..., 2] <= depth_high)
    candidate_mask = before_depth_filter & depth_eligible
    candidate_count = int(np.count_nonzero(candidate_mask))
    if candidate_count < cfg["minimum_points"]:
        raise PlaneFittingError(
            f"Only {candidate_count} ceiling points remain after depth-outlier filtering; "
            f"need {cfg['minimum_points']}."
        )

    candidate_points = xyz[candidate_mask]
    median_depth = float(np.median(candidate_points[:, 2]))
    distance_scale = median_depth if cfg["distance_units"] == "relative_to_median_depth" else 1.0
    ransac_threshold = cfg["ransac_threshold"] * distance_scale
    support_threshold = cfg["plane_distance_threshold"] * distance_scale
    rng = np.random.default_rng(cfg["seed"])
    if candidate_count > cfg["ransac_max_points"]:
        sample_indices = rng.choice(candidate_count, cfg["ransac_max_points"], replace=False)
        sampled_points = candidate_points[sample_indices]
    else:
        sampled_points = candidate_points
    sample_count = len(sampled_points)
    _svd_plane(sampled_points)

    best_normal = None
    best_d = 0.0
    best_count = 0
    best_median = float("inf")
    nondegenerate_trials = 0
    for _ in range(cfg["ransac_iterations"]):
        triplet = sampled_points[rng.choice(sample_count, 3, replace=False)]
        hypothesis = _three_point_plane(triplet)
        if hypothesis is None:
            continue
        nondegenerate_trials += 1
        normal, d = hypothesis
        distances = np.abs(sampled_points @ normal + d)
        sample_inliers = distances <= ransac_threshold
        count = int(np.count_nonzero(sample_inliers))
        if count < best_count or count < 3:
            continue
        median_residual = float(np.median(distances[sample_inliers]))
        if count > best_count or median_residual < best_median:
            best_normal, best_d = normal, d
            best_count, best_median = count, median_residual
    if best_normal is None:
        raise PlaneFittingError("RANSAC could not find a nondegenerate ceiling-plane consensus.")
    normal, d, _ = _refine_consensus(sampled_points, best_normal, best_d, ransac_threshold)
    normal, d, full_inliers = _refine_consensus(candidate_points, normal, d, ransac_threshold)
    inlier_count = int(np.count_nonzero(full_inliers))
    if inlier_count < cfg["minimum_points"]:
        raise PlaneFittingError(
            f"The dominant plane has only {inlier_count} full-population inliers; "
            f"need {cfg['minimum_points']}."
        )
    _svd_plane(candidate_points[full_inliers])
    inlier_ratio = inlier_count / candidate_count
    sampled_final_inliers = int(
        np.count_nonzero(np.abs(sampled_points @ normal + d) <= ransac_threshold)
    )

    residual = np.full(shape, np.nan, dtype=np.float32)
    residual[geometry_valid] = np.abs(xyz[geometry_valid] @ normal + d).astype(np.float32)
    inliers = np.zeros(shape, dtype=bool)
    inliers[candidate_mask] = full_inliers
    planar_support = (
        semantic
        & geometry_valid
        & confidence_eligible
        & (residual <= support_threshold)
    )
    if cfg["trim_depth_for_support"]:
        planar_support &= depth_eligible

    normal_angle = None
    usable_normal_count = None
    normal_checked_support_count = 0
    normal_fallback_support_count = int(np.count_nonzero(planar_support))
    if normal_values is not None:
        with np.errstate(over="ignore", invalid="ignore"):
            magnitudes = np.linalg.norm(normal_values, axis=2)
        usable_normals = (
            geometry_valid
            & np.isfinite(normal_values).all(axis=2)
            & np.isfinite(magnitudes)
            & (magnitudes > 0)
        )
        normal_angle = np.full(shape, np.nan, dtype=np.float32)
        unit_normals = normal_values[usable_normals] / magnitudes[usable_normals, None]
        alignment = np.clip(np.abs(unit_normals @ normal), 0.0, 1.0)
        normal_angle[usable_normals] = np.degrees(np.arccos(alignment)).astype(np.float32)
        usable_normal_count = int(np.count_nonzero(usable_normals & semantic))
        normal_checked_support_count = int(np.count_nonzero(planar_support & usable_normals))
        normal_fallback_support_count = int(np.count_nonzero(planar_support & ~usable_normals))
        planar_support &= ~usable_normals | (normal_angle <= cfg["plane_normal_angle_threshold"])
        if normal_fallback_support_count:
            warnings.append(
                f"{normal_fallback_support_count} distance-supported ceiling pixels have "
                "unavailable normals; those pixels use residual-only planar support."
            )

    quality_passed = True
    if inlier_ratio < cfg["minimum_plane_inlier_ratio"]:
        quality_passed = False
        warnings.append(
            f"Dominant ceiling-plane inlier ratio {inlier_ratio:.3f} is below "
            f"minimum_plane_inlier_ratio={cfg['minimum_plane_inlier_ratio']:.3f}."
        )
    support_count = int(np.count_nonzero(planar_support))
    if support_count < cfg["minimum_points"]:
        quality_passed = False
        warnings.append(
            f"Only {support_count} semantic ceiling pixels pass the plane-distance and "
            f"normal checks; need at least {cfg['minimum_points']} for usable planar support."
        )

    return {
        "plane": {
            "normal": normal.tolist(),
            "d": float(d),
            "num_points": candidate_count,
            "num_inliers": inlier_count,
            "inlier_ratio": float(inlier_ratio),
            "ransac_threshold": float(ransac_threshold),
        },
        "inliers": inliers,
        "residual": residual,
        "normal_angle": normal_angle,
        "candidate_mask": candidate_mask,
        "planar_support": planar_support,
        "warnings": warnings,
        "metadata": {
            "quality_passed": quality_passed,
            "distance_units": cfg["distance_units"],
            "coordinate_units": "same as input points; metric scale is not inferred",
            "median_candidate_depth": median_depth,
            "distance_scale": distance_scale,
            "configured_ransac_threshold": cfg["ransac_threshold"],
            "configured_plane_distance_threshold": cfg["plane_distance_threshold"],
            "effective_ransac_threshold": float(ransac_threshold),
            "effective_plane_distance_threshold": float(support_threshold),
            "plane_normal_angle_threshold_degrees": cfg["plane_normal_angle_threshold"],
            "normal_check_applied": normal_checked_support_count > 0,
            "normal_support_policy": "gate usable normals; residual-only fallback for unavailable normals",
            "num_distance_supported_pixels_checked_by_normals": normal_checked_support_count,
            "num_distance_supported_pixels_without_usable_normals": normal_fallback_support_count,
            "normal_comparison": "degrees(arccos(abs(dot(unit_normals, plane_normal))))",
            "plane_normal_sign": "largest-magnitude component is positive",
            "semantic_ceiling_pixels": int(np.count_nonzero(semantic)),
            "valid_positive_depth_pixels": int(np.count_nonzero(geometry_valid)),
            "eroded_ceiling_pixels": int(np.count_nonzero(eroded)),
            "candidates_before_depth_filter": preliminary_count,
            "depth_outlier_quantiles": [low_q, high_q],
            "depth_bounds": [float(depth_low), float(depth_high)],
            "support_depth_trim_applied": cfg["trim_depth_for_support"],
            "depth_rejected_candidate_pixels": preliminary_count - candidate_count,
            "num_candidates": candidate_count,
            "num_sampled_candidates": sample_count,
            "num_sampled_inliers": sampled_final_inliers,
            "sampled_inlier_ratio": sampled_final_inliers / sample_count,
            "inlier_ratio_denominator": "all filtered candidate_mask pixels, before subsampling",
            "num_planar_support_pixels": support_count,
            "num_semantic_pixels_with_usable_normals": usable_normal_count,
            "confidence_filter_applied": confidence_values is not None,
            "minimum_confidence": cfg["minimum_confidence"],
            "mask_boundary_erosion": cfg["mask_boundary_erosion"],
            "ransac_iterations": cfg["ransac_iterations"],
            "nondegenerate_ransac_trials": nondegenerate_trials,
            "seed": cfg["seed"],
        },
    }
