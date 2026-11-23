import cv2
import numpy as np


ROLE_CHANNELS = ("near_anchor", "coplanar", "extent", "behind", "in_front", "rim", "bridged_gap", "confidence")


def _unit_interval(x, low, high):
    return np.clip((x - low) / (high - low), 0, 1)


def _fit_reference(design, depth, seed_score, near_weight, rng):
    ids = np.flatnonzero(seed_score > .22)
    if len(ids) < 48:
        return None
    if len(ids) > 1400:
        ids = rng.choice(ids, 1400, replace=False)
    xy = design.reshape(-1, 3)[ids]
    inv = 1. / depth.ravel()[ids]
    weight = seed_score.ravel()[ids]
    near = weight * near_weight.ravel()[ids]
    tri = rng.choice(len(ids), (180, 3), p=near.astype(float) / near.sum(dtype=float))
    tri = tri[np.abs(np.linalg.det(xy[tri])) > .009]
    if not len(tri):
        return None
    params = np.linalg.solve(xy[tri], inv[tri][..., None])[..., 0]
    errors = np.abs((xy @ params.T) / inv[:, None] - 1.)
    agreement = np.clip(1. - errors / .15, 0, 1)
    global_fit = weight @ agreement / weight.sum()
    local_fit = near @ agreement / near.sum()
    best = np.argmax(.33 * global_fit + .67 * local_fit)
    inliers = errors[:, best] < .12
    if inliers.sum() < 32 or local_fit[best] < .20:
        return None
    confidence = float(np.sqrt(global_fit[best] * local_fit[best]))
    w = np.sqrt(near[inliers]) / inv[inliers]
    params = np.linalg.lstsq(xy[inliers] * w[:, None], inv[inliers] * w, rcond=None)[0]
    return design @ params, confidence


def _role_cues(depth, valid, normals, design, edge, semantic, role):
    h, w = depth.shape
    maps = np.zeros((len(ROLE_CHANNELS), h, w), np.float32)
    orientation = np.array([0, (-1, 1)[role], 0], np.float32)
    tagged = valid & (semantic > .50)
    if tagged.sum() >= 48:
        normal = np.median(normals[:, tagged], axis=1)
        length = np.linalg.norm(normal)
        if length > .60:
            orientation = normal / length
    alignment = np.einsum("c,cyx->yx", orientation, normals)
    vertical = (design[:, :, 1] + 1) * .5
    toward_frame = vertical if role == 0 else 1. - vertical
    fallback = .65 * _unit_interval(alignment, .70, .96) * (.15 + .85 * toward_frame)
    seed_score = np.maximum(semantic * (.60 + .40 * _unit_interval(alignment, .30, .85)), fallback) * valid
    anchors = seed_score > .22
    if np.count_nonzero(anchors) < 48:
        return maps
    near_depth = float(np.quantile(depth[anchors], .28))
    proximity = np.clip(near_depth / np.maximum(depth, .04), .1, 1.) ** 2
    near_weight = (.14 + .86 * proximity) * (.45 + .55 * toward_frame)
    maps[0] = seed_score * near_weight
    fitted = _fit_reference(design, depth, seed_score, near_weight, np.random.default_rng(9281 + role))
    if fitted is None:
        return maps
    inv_plane, confidence = fitted
    inv_scale = np.median(1. / depth[anchors])
    front = _unit_interval(inv_plane, .025 * inv_scale, .18 * inv_scale)
    log_ratio = np.log(np.clip(depth * inv_plane, .04, 25.))
    coplanar = np.exp(-.5 * (log_ratio / .12) ** 2) * valid * front
    support = cv2.medianBlur(((coplanar > .64) & (seed_score > .22)).astype(np.uint8), 3)
    pts = cv2.findNonZero(support)
    if pts is None or len(pts) < 30:
        return maps
    hull = cv2.fillConvexPoly(np.zeros_like(support), cv2.convexHull(pts), 1)
    distance = cv2.distanceTransform(1 - support, cv2.DIST_L2, 5)
    extent = np.maximum(.94 * cv2.GaussianBlur(hull.astype(np.float32), (0, 0), 1.8),
                        np.exp(-distance / (.22 * min(h, w)))) * front
    behind = _unit_interval(log_ratio, .06, .44) * valid * extent
    in_front = _unit_interval(-log_ratio, .06, .44) * valid * extent
    reliable_support = coplanar * np.maximum(seed_score, .32 * _unit_interval(alignment, .45, .92))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    nearby_support = cv2.dilate(reliable_support, kernel)
    nearby_far = cv2.dilate(behind, kernel)
    rim = np.minimum(nearby_support, nearby_far) * (.40 + .60 * edge)
    gap = cv2.morphologyEx((behind > .25).astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(gap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bridge = np.zeros_like(depth)
    for contour in contours:
        if cv2.contourArea(contour) < 12:
            continue
        part = cv2.drawContours(np.zeros_like(support), [contour], -1, 1, cv2.FILLED)
        boundary = cv2.morphologyEx(part, cv2.MORPH_GRADIENT, kernel).astype(bool)
        if boundary.any():
            r = np.mean(rim[boundary] > .15)
            bridge[part > 0] = min(1., r / .28)
    maps[1:] = [reliable_support, extent, behind, in_front, rim, bridge * extent,
                np.full_like(depth, confidence)]
    return maps


def storey_geometry(feat, geom, image):
    semantic = np.asarray(feat["surfaces"], np.float32)
    depth = np.expm1(np.asarray(geom[81], np.float32) / 3.)
    valid = (np.asarray(geom[80]) > 0) & np.isfinite(depth) & (depth > .02)
    depth = np.where(valid, depth, 0)
    normals = np.nan_to_num(np.asarray(geom[82:85], np.float32) / 3., posinf=0., neginf=0.)
    h, w = depth.shape
    y, x = np.mgrid[:h, :w].astype(np.float32)
    design = np.stack([2 * (x + .5) / w - 1, 2 * (y + .5) / h - 1, np.ones_like(x)], axis=-1)
    rgb = cv2.resize(np.asarray(image), (w, h), interpolation=cv2.INTER_AREA)
    light = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.
    dx = cv2.Sobel(light, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(light, cv2.CV_32F, 0, 1, ksize=3)
    edge = _unit_interval(cv2.magnitude(dx, dy), .06, .75)
    roles = [_role_cues(depth, valid, normals, design, edge, p, role) for role, p in enumerate(semantic)]
    return np.concatenate(roles).clip(0, 1).astype(np.float32, copy=False)


def add_supported_openings(roles, cues):
    floors, ceilings = cues.reshape(2, len(ROLE_CHANNELS), *cues.shape[-2:])
    size = (roles.shape[1], roles.shape[0])
    def evidence(maps):
        score = np.minimum(maps[3], maps[6]) * np.sqrt(maps[7])
        return cv2.resize(score, size, interpolation=cv2.INTER_LINEAR)
    floor, ceiling = evidence(floors), evidence(ceilings)
    add = (np.maximum(floor, ceiling) > .5) & (roles != 2) & (roles != 4)
    return np.where(add, np.where(ceiling > floor, 4, 2), roles).astype(np.uint8)
