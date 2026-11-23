from __future__ import annotations

from collections.abc import Mapping
import cv2
import numpy as np


DEFAULT_CONFIG = {
    "normal_family_angle_degrees": 22.0,
    "maximum_families": 6,
    "maximum_normal_samples": 24000,
    "role_parallel_angle_degrees": 35.0,
    "role_dominance_ratio": 1.5,
    "minimum_seed_pixels": 100,
    "minimum_plane_seed_fraction": .025,
    "maximum_planes_per_role": 6,
    "ransac_iterations": 160,
    "maximum_fitting_samples": 4000,
    "ransac_depth_fraction": .015,
    "support_depth_fraction": .035,
    "support_pixel_depth_fraction": .035,
    "cap_fitting_scale_at_initial_role_median": True,
    "normal_support_angle_degrees": 25.0,
    "maximum_plane_axis_angle_degrees": 22.0,
    "minimum_component_seed_pixels": 20,
    "minimum_component_pixels": 100,
    "floor_layer_seed_fraction_of_largest": .15,
    "reference_minimum_inlier_ratio": .75,
    "reference_support_distance_multiplier": 2.0,
    "weak_context_enabled": True,
    "seed": 42,
}


def validated_config(settings=None):
    if settings is not None and not isinstance(settings, Mapping):
        raise ValueError("Surface settings must be a mapping.")
    cfg = {**DEFAULT_CONFIG, **(settings or {})}
    unknown = set(cfg) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"Unknown surface settings: {sorted(unknown)}")
    for key, value in cfg.items():
        if key in {"weak_context_enabled", "cap_fitting_scale_at_initial_role_median"}:
            if not isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{key} must be bool")
            continue
        if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or not np.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite nonnegative numeric")
        if isinstance(DEFAULT_CONFIG[key], int):
            if int(value) != value or (key != "seed" and value < 1):
                raise ValueError(f"{key} must be a positive integer")
            cfg[key] = int(value)
        if "angle_degrees" in key and not 0 < value < 90:
            raise ValueError(f"{key} must lie between 0 and 90")
        if "fraction" in key and not 0 < value <= 1:
            raise ValueError(f"{key} must lie in (0, 1]")
    if cfg["role_dominance_ratio"] <= 1:
        raise ValueError("role_dominance_ratio must be > 1")
    return cfg


def _canonical(n):
    n = np.asarray(n, dtype=np.float64)
    n = n / np.linalg.norm(n)
    return n if n[np.argmax(np.abs(n))] >= 0 else -n


def _principal(normals):
    _, vectors = np.linalg.eigh(normals.T @ normals)
    return _canonical(vectors[:, -1])


def _families(normals, masks, cfg):
    pool = np.flatnonzero(np.logical_or.reduce(list(masks.values())).ravel())
    if not len(pool):
        return []
    if len(pool) > cfg["maximum_normal_samples"]:
        pool = pool[np.linspace(0, len(pool) - 1, cfg["maximum_normal_samples"], dtype=int)]
    sample = normals.reshape(-1, 3)[pool].copy()
    flip = sample[np.arange(len(sample)), np.argmax(np.abs(sample), axis=1)] < 0
    sample[flip] *= -1
    cosine = np.cos(np.deg2rad(cfg["normal_family_angle_degrees"]))
    families = []
    for _ in range(cfg["maximum_families"]):
        if len(sample) < 10:
            break
        bins = np.round(sample * 5).astype(np.int8)
        _, inverse, counts = np.unique(bins, axis=0, return_inverse=True, return_counts=True)
        selected = sample[inverse == int(np.argmax(counts))]
        axis = _principal(selected)
        for _ in range(2):
            close = np.abs(sample @ axis) >= cosine
            if close.any():
                axis = _principal(sample[close])
        close = np.abs(sample @ axis) >= cosine
        if not close.any():
            break
        hits = {}
        for role, mask in masks.items():
            values = normals[mask]
            hits[role] = int(np.count_nonzero(np.abs(values @ axis) >= cosine))
        families.append({"id": len(families), "normal": axis.tolist(), "original_semantic_pixels": hits,
                         "sample_pixels": int(close.sum())})
        sample = sample[~close]
    return families


def _fit_population(xyz, surface_normals, axis, cfg, rng, *, maximum_depth_scale=None):
    if len(xyz) < cfg["minimum_seed_pixels"]:
        return None
    count = min(len(xyz), cfg["maximum_fitting_samples"])
    ids = rng.choice(len(xyz), size=count, replace=False)
    sample, ns = xyz[ids], surface_normals[ids]
    raw_scale = float(np.median(xyz[:, 2]))
    scale = min(raw_scale, maximum_depth_scale) if maximum_depth_scale is not None else raw_scale
    threshold = cfg["ransac_depth_fraction"] * scale
    normal_candidates = [axis]
    anchors = [sample[len(sample) // 2]]
    cosine = np.cos(np.deg2rad(cfg["maximum_plane_axis_angle_degrees"]))
    for _ in range(cfg["ransac_iterations"]):
        tri = sample[rng.choice(len(sample), 3, replace=False)]
        cross = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        if np.linalg.norm(cross) <= 1e-10:
            continue
        normal = _canonical(cross)
        if abs(normal @ axis) < cosine:
            continue
        normal_candidates.append(normal)
        anchors.append(tri[0])
    normals = np.array(normal_candidates)
    offsets = -np.einsum("ij,ij->i", normals, np.array(anchors))
    residuals = np.abs(sample @ normals.T + offsets)
    votes = np.count_nonzero(residuals <= threshold, axis=0)
    best = int(np.argmax(votes))
    normal, d = normals[best], float(offsets[best])
    for _ in range(4):
        inliers = np.abs(xyz @ normal + d) <= threshold
        if int(inliers.sum()) < cfg["minimum_seed_pixels"]:
            return None
        population = xyz[inliers]
        center = population.mean(axis=0)
        centered = population - center
        eig, vectors = np.linalg.eigh(centered.T @ centered)
        if eig[1] <= max(eig[2], 1e-10) * 1e-8:
            return None
        n2 = _canonical(vectors[:, 0])
        if abs(n2 @ axis) < cosine:
            break
        normal, d = n2, -float(n2 @ center)
    inliers = np.abs(xyz @ normal + d) <= threshold
    return normal, d, inliers, threshold, scale, raw_scale


def _anchored_components(strict, anchors, cfg):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(strict.astype(np.uint8), connectivity=8)
    keep = np.zeros(strict.shape, bool)
    records = []
    for component_id in range(1, count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < cfg["minimum_component_pixels"]:
            continue
        component = labels == component_id
        hits = int(np.count_nonzero(component & anchors))
        accepted = hits >= cfg["minimum_component_seed_pixels"]
        records.append({"component_id": component_id, "area_pixels": area, "anchor_pixels": hits,
                        "accepted": accepted, "bbox_xywh": stats[component_id, :4].tolist()})
        if accepted:
            keep |= component
    return keep, records


def _validated_reference(reference, role, axis, xyz, unit, usable, seeds, anchors, cfg):
    if not isinstance(reference, Mapping):
        return None
    coeff = reference.get("plane", reference)
    metadata = reference.get("metadata", {})
    normal = np.asarray(coeff.get("normal"), dtype=float)
    d = float(coeff.get("d", np.nan))
    threshold = float(metadata.get("effective_plane_distance_threshold", np.nan))
    inlier_ratio = float(coeff.get("inlier_ratio", reference.get("inlier_ratio", 0)))
    if normal.shape != (3,) or not np.isfinite(normal).all() or not np.isfinite(d) or not np.isfinite(threshold) or threshold <= 0:
        return None
    magnitude = np.linalg.norm(normal)
    if magnitude < 1e-8 or not metadata.get("quality_passed", False) or inlier_ratio < cfg["reference_minimum_inlier_ratio"]:
        return None
    normal, d = normal / magnitude, d / magnitude
    if abs(normal @ axis) < np.cos(np.deg2rad(cfg["maximum_plane_axis_angle_degrees"])):
        return None
    parallel = usable & (np.abs(unit @ normal) >= np.cos(np.deg2rad(cfg["normal_support_angle_degrees"])))
    support_threshold = threshold * cfg["reference_support_distance_multiplier"]
    strict = parallel & (np.abs(xyz @ normal + d) <= support_threshold)
    hits = {r: int(np.count_nonzero(strict & mask)) for r, mask in seeds.items()}
    opposite = "ceiling" if role == "floor" else "floor"
    if hits[role] < cfg["minimum_seed_pixels"] or hits[role] < cfg["role_dominance_ratio"] * hits[opposite]:
        return None
    support, components = _anchored_components(strict, anchors, cfg)
    if not support.any():
        return None
    return {"plane": {"normal": normal.tolist(), "d": d}, "support_mask": support,
            "strict_mask": strict, "seed_inlier_mask": seeds[role] & strict, "components": components,
            "metadata": {"quality_passed": True, "surface_role": role, "role_original_semantic_pixels": hits,
                         "role_identity_conflict": False, "effective_plane_distance_threshold": threshold,
                         "effective_support_distance_threshold": support_threshold,
                         "reference_support_distance_multiplier": cfg["reference_support_distance_multiplier"],
                         "seed_inlier_pixels": int(np.count_nonzero(seeds[role] & strict)),
                         "baseline_inlier_ratio": inlier_ratio, "baseline_plane_distance_threshold": threshold,
                         "median_candidate_depth": float(metadata.get("median_candidate_depth", np.median(xyz[..., 2][strict]))),
                         "effective_ransac_threshold": float(metadata.get("effective_ransac_threshold", threshold)),
                         "fit_source": "validated_original_baseline", "layer_selected": False,
                         "opening_annotation_used": False}}


def recover_slab_surfaces(points, normals, valid, intrinsics, semantic_classes, semantic_labels,
                          *, supplemental_seeds=None, reference_planes=None, config=None):
    cfg = validated_config(config)
    xyz = np.asarray(points, dtype=np.float64)
    ns = np.asarray(normals, dtype=np.float64)
    validity = np.asarray(valid)
    classes = np.asarray(semantic_classes)
    k = np.asarray(intrinsics, dtype=np.float64)
    if xyz.ndim != 3 or xyz.shape[-1] != 3 or min(xyz.shape[:2]) < 1:
        raise ValueError("points must be nonempty H x W x 3")
    shape = xyz.shape[:2]
    if ns.shape != xyz.shape or validity.shape != shape or validity.dtype != bool:
        raise ValueError("Normals/validity must share the full geometry frame; validity is bool")
    if classes.shape != shape or not np.issubdtype(classes.dtype, np.integer):
        raise ValueError("semantic_classes must be integer H x W on the geometry frame")
    if k.shape != (3, 3) or not np.isfinite(k).all() or abs(np.linalg.det(k)) < 1e-10 or not np.allclose(k[2], [0, 0, 1]):
        raise ValueError("intrinsics must be finite invertible normalized 3 x 3")
    if not isinstance(semantic_labels, Mapping):
        raise ValueError("semantic_labels must be an id-to-name mapping")
    lengths = np.linalg.norm(ns, axis=-1)
    usable = validity & np.isfinite(xyz).all(axis=-1) & (xyz[..., 2] > 0)
    usable &= np.isfinite(ns).all(axis=-1) & (lengths > 1e-8)
    unit = np.zeros_like(ns)
    unit[usable] = ns[usable] / lengths[usable, None]
    roles = ("floor", "ceiling")
    resolved = {role: [int(i) for i, name in semantic_labels.items() if str(name).strip().casefold() == role] for role in roles}
    seed_masks = {role: np.isin(classes, ids) & usable for role, ids in resolved.items()}
    supplemental = {role: [] for role in roles}
    if supplemental_seeds is not None:
        if not isinstance(supplemental_seeds, Mapping) or set(supplemental_seeds) - set(roles):
            raise ValueError("supplemental_seeds must map floor/ceiling to lists of automatic bool masks")
        for role, values in supplemental_seeds.items():
            if not isinstance(values, (list, tuple)):
                raise ValueError("Each supplemental role must contain a mask list")
            for value in values:
                mask = np.asarray(value)
                if mask.shape != shape or mask.dtype != bool:
                    raise ValueError("Supplemental seeds must be bool on the geometry frame")
                supplemental[role].append(mask & usable)
    families = _families(unit, seed_masks, cfg)
    role_family = {}
    for role in roles:
        ranked = sorted(families, key=lambda item: item["original_semantic_pixels"][role], reverse=True)
        role_family[role] = ranked[0] if ranked and ranked[0]["original_semantic_pixels"][role] >= cfg["minimum_seed_pixels"] else None
    decisions = {role: {"status": "anchored" if role_family[role] is not None else "unknown_no_role_anchor", "reasons": []} for role in roles}
    conflict = None
    if all(role_family.values()):
        fa, ca = role_family["floor"], role_family["ceiling"]
        angle = float(np.degrees(np.arccos(np.clip(abs(np.dot(fa["normal"], ca["normal"])), 0, 1))))
        if angle > cfg["role_parallel_angle_degrees"]:
            scores = {role: role_family[role]["original_semantic_pixels"][role] for role in roles}
            strongest = max(scores, key=scores.get)
            weakest = "ceiling" if strongest == "floor" else "floor"
            dominance = scores[strongest] / max(scores[weakest], 1)
            rejected = [weakest] if dominance >= cfg["role_dominance_ratio"] else list(roles)
            for role in rejected:
                decisions[role] = {"status": "unknown_role_orientation_conflict", "reasons": ["The two strongest semantic slab directions are incompatible; no gravity/image-half role guess was made."]}
            conflict = {"angle_degrees": angle, "role_counts_on_dominant_families": scores, "dominance_ratio": dominance,
                        "rejected_roles": rejected, "policy": "shared orientation compatibility + original semantic dominance"}
    weak_ids = [int(i) for i, name in semantic_labels.items() if str(name).strip().casefold() in {"building", "sidewalk"}]
    weak_context = np.isin(classes, weak_ids) & usable
    result_roles = {}
    rng = np.random.default_rng(cfg["seed"])
    for role in roles:
        original = seed_masks[role]
        empty = np.zeros(shape, bool)
        decision = decisions[role]
        family = role_family[role]
        if decision["status"] != "anchored":
            result_roles[role] = {**decision, "accepted_support": empty, "seed_mask": original,
                                  "planes": [], "primary_plane": None, "weak_context_proposal": empty.copy()}
            continue
        axis = np.asarray(family["normal"])
        orientation = usable & (np.abs(unit @ axis) >= np.cos(np.deg2rad(cfg["normal_support_angle_degrees"])))
        population = original & orientation
        initial_pixels = int(population.sum())
        initial_depth_scale = float(np.median(xyz[..., 2][population])) if initial_pixels else None
        planes = []
        anchors = original.copy()
        for mask in supplemental[role]:
            anchors |= mask & orientation
        for _ in range(cfg["maximum_planes_per_role"]):
            population_ids = np.flatnonzero(population.ravel())
            minimum = max(cfg["minimum_seed_pixels"], int(initial_pixels * cfg["minimum_plane_seed_fraction"]))
            if len(population_ids) < minimum:
                break
            values = xyz.reshape(-1, 3)[population_ids]
            fits = _fit_population(values, unit.reshape(-1, 3)[population_ids], axis, cfg, rng,
                                   maximum_depth_scale=initial_depth_scale if cfg["cap_fitting_scale_at_initial_role_median"] else None)
            if fits is None:
                break
            normal, d, inliers, threshold, scale, raw_scale = fits
            if int(inliers.sum()) < minimum:
                break
            distance = np.abs(xyz @ normal + d)
            parallel = usable & (np.abs(unit @ normal) >= np.cos(np.deg2rad(cfg["normal_support_angle_degrees"])))
            support_threshold = cfg["support_depth_fraction"] * scale
            tolerance_grid = np.minimum(support_threshold, cfg["support_pixel_depth_fraction"] * xyz[..., 2])
            strict = parallel & (distance <= tolerance_grid)
            support, components = _anchored_components(strict, anchors, cfg)
            seed_inliers = original & parallel & (distance <= threshold)
            semantic_hits = {other: int(np.count_nonzero(strict & seed_masks[other])) for other in roles}
            opposed = "ceiling" if role == "floor" else "floor"
            identity_ratio = semantic_hits[role] / max(semantic_hits[opposed], 1)
            identity_conflict = semantic_hits[opposed] >= minimum and identity_ratio < 1 / cfg["role_dominance_ratio"]
            planes.append({"id": len(planes), "plane": {"normal": normal.tolist(), "d": d},
                           "metadata": {"effective_plane_distance_threshold": support_threshold,
                                        "effective_ransac_threshold": threshold, "median_candidate_depth": scale,
                                        "raw_population_median_depth": raw_scale,
                                        "initial_role_orientation_median_depth": initial_depth_scale,
                                        "effective_support_distance_threshold": support_threshold,
                                        "support_tolerance_policy": "min(support_depth_fraction * capped_population_median_depth, support_pixel_depth_fraction * pixel_Z)",
                                        "support_pixel_depth_fraction": cfg["support_pixel_depth_fraction"],
                                        "fitting_scale_capped_at_initial_role_median": cfg["cap_fitting_scale_at_initial_role_median"],
                                        "quality_passed": bool(support.any() and not identity_conflict),
                                        "surface_role": role, "role_original_semantic_pixels": semantic_hits,
                                        "role_identity_conflict": identity_conflict, "family_id": family["id"],
                                        "seed_inlier_pixels": int(seed_inliers.sum()), "sequential_fit_pixels": int(inliers.sum()),
                                        "inlier_ratio_local_population": float(inliers.mean()),
                                        "inlier_ratio_original_role": int(seed_inliers.sum()) / max(int(original.sum()), 1),
                                        "layer_selected": False, "opening_annotation_used": False},
                           "support_mask": support, "seed_inlier_mask": seed_inliers, "strict_mask": strict,
                           "components": components})
            remove = np.abs(xyz @ normal + d) <= tolerance_grid
            population &= ~remove
        reference_used = False
        reference = _validated_reference((reference_planes or {}).get(role), role, axis, xyz, unit, usable,
                                         seed_masks, anchors, cfg)
        if reference is not None:
            reference["id"] = len(planes)
            reference["metadata"]["family_id"] = family["id"]
            planes.append(reference)
            reference_used = True
        usable_planes = [p for p in planes if p["metadata"]["quality_passed"]]
        if reference_used and role == "floor":
            selected = [reference]
        elif role == "floor" and usable_planes:
            largest = max(p["metadata"]["seed_inlier_pixels"] for p in usable_planes)
            robust = [p for p in usable_planes if p["metadata"]["seed_inlier_pixels"] >= largest * cfg["floor_layer_seed_fraction_of_largest"]]
            selected = [min(robust, key=lambda p: abs(p["plane"]["d"]))]
        else:
            selected = usable_planes
        support = empty.copy()
        for record in selected:
            record["metadata"]["layer_selected"] = True
            support |= record["support_mask"]
        if selected:
            primary = reference if reference_used else max(selected, key=lambda p: p["metadata"]["seed_inlier_pixels"])
            status = "supported_multi_plane" if len(selected) > 1 else "supported_single_plane"
        else:
            primary, status = None, "unknown_insufficient_planar_support"
        result_roles[role] = {"status": status, "reasons": decision["reasons"], "accepted_support": support,
                              "seed_mask": original, "planes": planes, "primary_plane": primary,
                              "weak_context_proposal": weak_context & orientation & ~support if cfg["weak_context_enabled"] else empty.copy(),
                              "selected_family_id": family["id"],
                              "original_seed_pixels": int(original.sum()), "orientation_seed_pixels": initial_pixels,
                              "accepted_support_pixels": int(support.sum()),
                              "selected_seed_coverage": int((support & original).sum()) / max(int(original.sum()), 1),
                              "validated_reference_used": reference_used,
                              "supplemental_seed_pixels": [int(mask.sum()) for mask in supplemental[role]]}
    return {"roles": result_roles, "orientation_families": families, "role_orientation_conflict": conflict,
            "metadata": {"effective_config": cfg, "processing_size": [shape[1], shape[0]],
                         "intrinsics_normalized": k.tolist(), "coordinate_frame": "unchanged input full-frame geometry",
                         "geometry_reinferred": False, "opening_annotation_used": False, "opening_mask_generated": False,
                         "role_identity_policy": "original semantic anchors + role orientation compatibility; never normal.y or image halves",
                         "weak_context_policy": "building/sidewalk only a separately saved proposal; cannot supply a role anchor",
                         "floor_layer_policy": "validated strong original reference, otherwise closest sufficiently supported plane; other offsets diagnostic",
                         "ceiling_layer_policy": "validated strong original reference is primary, otherwise largest original seed support; multiple-plane union is diagnostic only, not one slab for opening subtraction",
                         "reference_planes_supplied": list((reference_planes or {}).keys()),
                         "limitation": "Monocular semantic roles and normals can jointly be wrong. Multi-plane support improves fitting coverage but does not prove a through-slab void."}}
