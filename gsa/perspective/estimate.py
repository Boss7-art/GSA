import cv2
import numpy as np


GRID_LONG_EDGE = 384
MIN_SUPPORT = 32
MAX_WALL_PLANES = 8


def _grid(width, height, camera):
    y, x = np.mgrid[:height, :width].astype(np.float64)
    uv = np.stack(((x + .5) / width, (y + .5) / height, np.ones_like(x)), axis=-1)
    return uv @ np.linalg.inv(camera).T


def _orientation(normals, offsets, gravity, role, upward_view):
    toward = normals * np.sign(offsets)[:, None]
    down = toward @ gravity
    if role == "floor":
        return down > .64
    if role == "wall":
        return np.abs(down) < .48
    matches = down < -.64
    if upward_view:
        matches |= (down < .18) & (toward[:, 2] > .45)
    return matches


def _fit(points, normals, solid, role, gravity, rng, upward_view=False):
    available = solid & np.isfinite(points).all(axis=2)
    total = int(np.count_nonzero(available))
    if total < MIN_SUPPORT:
        return []
    minimum = max(MIN_SUPPORT, round(.025 * total)) if role == "wall" else max(MIN_SUPPORT, round(.16 * total))
    planes = []
    for _ in range(MAX_WALL_PLANES if role == "wall" else 1):
        coords = np.argwhere(available)
        if len(coords) < minimum:
            break
        idx = rng.choice(len(coords), min(1600, len(coords)), replace=False)
        y, x = coords[idx].T
        samples = points[y, x]
        tri = samples[rng.integers(len(samples), size=(420, 3))]
        normals_try = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        spans = np.linalg.norm(normals_try, axis=1)
        valid = spans > 1.e-5
        normals_try = normals_try[valid] / spans[valid, None]
        offsets = np.einsum("ij,ij->i", normals_try, tri[valid, 0])
        valid = _orientation(normals_try, offsets, gravity, role, upward_view)
        valid &= np.abs(offsets) > .012
        normals_try, offsets = normals_try[valid], offsets[valid]
        if not len(offsets):
            break
        tol = .012 + .030 * samples[:, 2]
        residuals = np.abs(samples @ normals_try.T - offsets)
        inliers = residuals < tol[:, None]
        if normals is not None:
            ns = normals[y, x]
            inliers &= (np.abs(ns @ normals_try.T) > .70) | (np.linalg.norm(ns, axis=1) < .25)[:, None]
        score = (inliers * (1 - .35 * np.minimum(1., residuals / tol[:, None]))).sum(axis=0)
        best = int(np.argmax(score))
        n, d = normals_try[best], offsets[best]
        if inliers[:, best].sum() < max(9, .6 * minimum * len(y) / len(coords)):
            break
        for _ in range(2):
            distances = np.abs(points @ n - d)
            support = available & (distances < .012 + .030 * points[..., 2])
            if normals is not None:
                support &= (np.abs(normals @ n) > .68) | (np.linalg.norm(normals, axis=2) < .25)
            yy, xx = np.nonzero(support)
            if len(xx) < minimum:
                break
            subset = rng.choice(len(xx), min(3500, len(xx)), replace=False)
            fit_points = points[yy[subset], xx[subset]]
            center = np.median(fit_points, axis=0)
            _, _, axes = np.linalg.svd(fit_points - center, full_matrices=False)
            n = axes[-1]
            d = float(np.median(fit_points @ n))
        if len(xx) < minimum:
            break
        spread = np.linalg.eigvalsh(np.cov(np.column_stack((xx, yy)).T))
        oriented = _orientation(n[None, :], np.asarray([d]), gravity, role, upward_view)[0]
        if spread[0] < 1.5 or abs(d) < .012 or not oriented:
            available[support] = False
            continue
        if d < 0:
            n, d = -n, -d
        z = points[..., 2][support]
        planes.append(dict(normal=n, offset=float(d), support=support, z_range=np.quantile(z, [.02,.98]),
                           residual=float(np.median(np.abs(points[support] @ n - d))),
                           support_pixels=int(support.sum())))
        available[support] = False
    return planes


def _floor_anchor(ground, solid):
    support = ground["support"]
    yy, xx = np.nonzero(support)
    h, w = support.shape
    x10, x90 = np.quantile((xx + .5) / w, [.10, .90])
    y10, y50, y90 = np.quantile((yy + .5) / h, [.10, .50, .90])
    measures = dict(
        support_image_fraction=len(xx) / (w*h),
        support_floor_solid_fraction=len(xx) / np.count_nonzero(solid),
        central_width_fraction=x90 - x10,
        central_height_fraction=y90 - y10,
        median_y_fraction=y50,
        lower_y_fraction=y90,
        median_residual_over_distance=ground["residual"] / ground["offset"],
    )
    failures = []
    if measures["support_image_fraction"] < .008:
        failures.append("small_image_support")
    if measures["support_floor_solid_fraction"] < .40:
        failures.append("floor_solid_poorly_explained")
    if x90 - x10 < .15 or y90 - y10 < .055:
        failures.append("narrow_spatial_support")
    if y50 < .50 or y90 < .68:
        failures.append("no_near_lower_view_floor_evidence")
    if measures["median_residual_over_distance"] > .10:
        failures.append("large_relative_plane_residual")
    return dict(eligible=not failures, reasons=failures, **measures)


def _assign(planes, points, rays, solid, opening):
    h, w = solid.shape
    labels = np.zeros((h, w), np.uint8)
    best = np.full((h, w), np.inf, np.float32)
    domain = solid | opening
    kernel = np.ones((3,3), np.uint8)
    for i, plane in enumerate(planes, 1):
        n, d = plane["normal"], plane["offset"]
        cosine_factor = rays @ n
        z = np.divide(d, cosine_factor, out=np.zeros_like(cosine_factor), where=cosine_factor > 1.e-8)
        zlo, zhi = plane["z_range"]
        stable = ((cosine_factor > .075 * np.linalg.norm(rays, axis=2)) &
                  (z > max(.008,.35*zlo)) & (z < 2.5*zhi))
        solid_supported = ~np.isfinite(points[..., 2]) | (np.abs(points @ n - d) < .04 + .085*z)
        traversable = domain & stable & (opening | solid_supported)
        frontier = plane["support"] & traversable
        if not frontier.any():
            continue
        distance = np.full((h,w), np.inf, np.float32)
        distance[frontier] = 0
        for step in range(1, int(.55*max(w,h))+1):
            frontier = (cv2.dilate(frontier.astype(np.uint8), kernel).astype(bool) &
                        traversable & ~np.isfinite(distance))
            if not frontier.any():
                break
            distance[frontier] = step
        residual_penalty = np.nan_to_num(np.abs(points @ n - d), nan=0., posinf=0., neginf=0.)
        cost = distance + np.where(solid, np.minimum(residual_penalty / (.012+.030*z), 6.), 0.)
        wins = cost < best
        labels[wins] = i
        best[wins] = cost[wins]
    return labels


def _wall_reference_gaps(wall, slab, ids, planes, points, small_wall):
    h, w = wall.shape
    gh, gw = small_wall.shape
    extra = np.zeros((h,w),np.uint8)
    details = dict(method="enclosed_wall_gap_with_solid_ring_and_depth_continuity",
                   accepted_components=0,rejected_components={})
    if not planes or not np.any(wall == 0):
        return extra, details
    count, components, stats, _ = cv2.connectedComponentsWithStats((wall == 0).astype(np.uint8), connectivity=8)
    small = cv2.resize(components,(gw,gh),interpolation=cv2.INTER_NEAREST_EXACT)
    samples = np.bincount(small.ravel(),minlength=count)
    bins = len(planes) + 1
    sizes = cv2.calcHist([ids], [0], None, [bins], [0,bins]).ravel()
    kernel = np.ones((3,3),np.uint8)
    def reject(reason):
        rejected = details["rejected_components"]
        rejected[reason] = rejected.get(reason,0) + 1
    for j in np.flatnonzero(samples >= 4):
        if j == 0:
            continue
        x,y,bw,bh,area = stats[j]
        if x <= 0 or y <= 0 or x+bw >= w or y+bh >= h:
            continue
        if area > .004*h*w or bw > .12*w or bh > .12*h:
            reject("large_gap")
            continue
        roi = np.s_[y-1:y+bh+1, x-1:x+bw+1]
        inside = components[roi] == j
        neighborhood = cv2.dilate(inside.astype(np.uint8),kernel).astype(bool)
        ring = neighborhood & ~inside
        if np.any(slab[roi][neighborhood]):
            reject("adjacent_to_slab")
            continue
        neighbors = ids[roi][ring]
        owners = np.unique(neighbors[neighbors != 0])
        if len(owners) != 1 or np.mean(neighbors != 0) < .80:
            reject("not_enclosed_by_one_assigned_wall")
            continue
        owner = int(owners[0])
        if area > .05*sizes[owner]:
            reject("large_relative_to_wall")
            continue
        left, top = max(0,int(x*gw/w)-2), max(0,int(y*gh/h)-2)
        right, bottom = min(gw,int(np.ceil((x+bw)*gw/w))+2), min(gh,int(np.ceil((y+bh)*gh/h))+2)
        grid_roi = np.s_[top:bottom,left:right]
        candidate = small[grid_roi] == j
        border = cv2.dilate(candidate.astype(np.uint8),kernel).astype(bool) & ~candidate
        plane = planes[owner-1]
        support = plane["support"][grid_roi] & border & (small_wall[grid_roi] == 1)
        if np.count_nonzero(support) < max(4,.40*border.sum()):
            reject("insufficient_coplanar_boundary")
            continue
        ratio = (points[grid_roi] @ plane["normal"]) / plane["offset"]
        known = np.isfinite(ratio)
        local = (candidate | support) & known
        if np.count_nonzero(candidate & known) < .90*candidate.sum():
            reject("insufficient_gap_depth")
            continue
        jumps_x = local[:,1:] & local[:,:-1] & (np.abs(np.diff(ratio,axis=1)) > .12)
        jumps_y = local[1:] & local[:-1] & (np.abs(np.diff(ratio,axis=0)) > .12)
        if np.count_nonzero(jumps_x) + np.count_nonzero(jumps_y) >= 2:
            reject("depth_discontinuity")
            continue
        if np.quantile(np.abs(ratio[candidate & known]-1.),.90) > .07:
            reject("depth_incompatible_with_wall")
            continue
        extra[roi][inside] = owner
        details["accepted_components"] += 1
    return extra, details


def _bounds(mask, plane, camera, gravity, role):
    h, w = mask.shape
    contours, _ = cv2.findContours(mask.view(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    xy = np.concatenate(contours).reshape(-1,2).astype(np.float64)
    xy = cv2.convexHull(xy.astype(np.float32)).reshape(-1,2).astype(np.float64)
    xy = (xy[:,None,:] + np.array([[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]])).reshape(-1,2)
    homogeneous = np.column_stack(((xy[:,0]+.5)/w, (xy[:,1]+.5)/h, np.ones(len(xy))))
    rays = homogeneous @ np.linalg.inv(camera).T
    n, d = plane["normal"], plane["offset"]
    denom = rays @ n
    if np.any(denom < .06 * np.linalg.norm(rays,axis=1)):
        return None
    xyz = rays * (d / denom)[:,None]
    if role == "wall":
        vertical = gravity - n*(gravity@n)
        vertical /= np.linalg.norm(vertical)
        horizontal = np.cross(vertical,n)
    else:
        horizontal = np.array([1.,0.,0.]) - n*n[0]
        if np.linalg.norm(horizontal) < .1:
            return None
        horizontal /= np.linalg.norm(horizontal)
        vertical = np.cross(n,horizontal)
    c = np.mean(xyz,axis=0)
    if horizontal[0]*c[2] - horizontal[2]*c[0] < 0:
        horizontal = -horizontal
    if vertical[1]*c[2] - vertical[2]*c[1] < 0:
        vertical = -vertical
    axes = np.column_stack((horizontal, vertical))
    coords = xyz @ axes
    low, high = np.min(coords,axis=0), np.max(coords,axis=0)
    dimensions = high - low
    if np.min(dimensions) <= 1.e-6:
        return None
    rect = np.array([low,[high[0],low[1]],high,[low[0],high[1]]])
    corners_3d = n*d + rect @ axes.T
    if np.any(corners_3d[:,2] <= .02*plane["z_range"][0]):
        return None
    proj = corners_3d @ camera.T
    quad = proj[:,:2]/proj[:,2:] * (w,h) - .5
    if not np.isfinite(quad).all() or np.any(np.abs(quad) > 12*np.array([w,h])):
        return None
    if cv2.contourArea(quad.astype(np.float32),oriented=True) < 0:
        quad = quad[[3,2,1,0]]
    return quad.tolist(), dimensions


def estimate_planes(slab, wall, depth, intrinsics, normals=None, camera_height_m=None,
                    complete_wall_reference=True):
    slab, wall = np.asarray(slab), np.asarray(wall)
    depth = np.asarray(depth, dtype=np.float64)
    k = np.asarray(intrinsics, dtype=np.float64)
    if slab.ndim != 2 or slab.shape != wall.shape or depth.ndim != 2 or not depth.size:
        raise ValueError("Expected aligned 2-D slab/wall masks and a 2-D depth map")
    if k.shape != (3,3) or not np.isfinite(k).all() or abs(np.linalg.det(k)) < 1.e-12 or min(k[0,0],k[1,1]) <= 0:
        raise ValueError("Expected finite normalized pinhole intrinsics")
    if camera_height_m is not None and (not np.isfinite(camera_height_m) or camera_height_m <= 0):
        raise ValueError("Expected a positive camera height in metres")
    native_h, native_w = slab.shape
    factor = min(1., GRID_LONG_EDGE/max(native_w,native_h), depth.shape[1]/native_w, depth.shape[0]/native_h)
    size = (max(1,round(native_w*factor)),max(1,round(native_h*factor)))
    small = lambda im: cv2.resize(im,size,interpolation=cv2.INTER_NEAREST_EXACT)
    z = small(depth)
    good = np.isfinite(z) & (z > 0)
    small_slab, small_wall = small(slab), small(wall)
    anchor_depth = good & ((small_slab == 1) | (small_slab == 3) | (small_wall == 1))
    z_unit = float(np.median(z[anchor_depth])) if np.any(anchor_depth) else 1.
    z = np.where(good,z/z_unit,np.nan)
    rays = _grid(*size,k)
    points = rays * z[...,None]
    if normals is not None:
        normals = np.asarray(normals,dtype=np.float32)
        if normals.ndim != 3 or normals.shape[2] != 3:
            raise ValueError("Expected a normal map with shape (height,width,3)")
        normals = np.nan_to_num(small(normals),nan=0.,posinf=0.,neginf=0.)
        normals /= np.maximum(np.linalg.norm(normals,axis=2,keepdims=True),1.e-8)
    roles = (("floor",slab,small_slab,1,2),("ceiling",slab,small_slab,3,4),
             ("wall",wall,small_wall,1,2))
    gravity = np.array([0.,1.,0.])
    height_scale = None
    floor_distance = None
    anchor_check = dict(eligible=False,reasons=["no_supported_floor_plane"])
    wall_completion = dict(method="disabled",added_unclassified_source_pixels=0,by_plane={})
    result, coverage, failures = [], {}, []
    for index, (role, native, classes, solid_id, opening_id) in enumerate(roles):
        solid, opening = classes==solid_id, classes==opening_id
        upward_view = (role == "ceiling" and floor_distance is None and solid.mean() >= .15
                       and (solid | opening).mean() >= .55)
        planes = _fit(points,normals,solid,role,gravity,np.random.default_rng(6247+index),upward_view)
        assigned = _assign(planes,points,rays,solid,opening)
        native_ids = cv2.resize(assigned,(native_w,native_h),interpolation=cv2.INTER_NEAREST_EXACT)
        native_solid, native_opening = native==solid_id, native==opening_id
        native_domain = native_solid | native_opening
        if role == "wall" and complete_wall_reference:
            additions, wall_completion = _wall_reference_gaps(wall,slab,native_ids,planes,points,small_wall)
            wall_completion.update(added_unclassified_source_pixels=0,by_plane={})
        total_s, total_v = int(np.count_nonzero(native_solid)), int(np.count_nonzero(native_opening))
        covered_s = covered_v = 0
        for i, plane in enumerate(planes,1):
            footprint = (native_ids==i) & native_domain
            added_count = 0
            if role == "wall" and complete_wall_reference:
                extra = additions==i
                added_count = int(np.count_nonzero(extra))
                footprint |= extra
            geometry = _bounds(footprint,plane,k,gravity,role)
            if geometry is None:
                failures.append({"id":f"{role}_{i}","reason":"unstable_or_empty_plane_rectangle"})
                if role == "floor":
                    anchor_check = dict(eligible=False,reasons=["unstable_floor_rectangle"])
                continue
            quad, dimensions = geometry
            if role == "floor":
                anchor_check = _floor_anchor(plane,solid)
                anchor_check["candidate_perpendicular_distance_model_units"] = plane["offset"] * z_unit
                if anchor_check["eligible"]:
                    gravity = plane["normal"].copy()
                    floor_distance = plane["offset"] * z_unit
                    if camera_height_m is not None:
                        height_scale = float(camera_height_m / floor_distance)
            elif role == "ceiling" and floor_distance is None:
                gravity = -plane["normal"].copy()
            if role == "wall" and complete_wall_reference:
                wall_completion["by_plane"][f"wall_{i}"] = added_count
                wall_completion["added_unclassified_source_pixels"] += added_count
            description = dict(id=f"{role}_{i}",role=role,source="automatic_solid_depth_plane",footprint=footprint,
                               quad_tl_tr_br_bl=quad,normal_camera=plane["normal"].tolist(),
                               offset_model_units=plane["offset"]*z_unit,
                               fit_support_pixels=plane["support_pixels"],
                               median_residual_model_units=plane["residual"]*z_unit)
            if height_scale is None:
                description.update(width_height_relative=(dimensions*z_unit).tolist(),scale_group="moge_same_camera")
            else:
                description["width_height_metres"] = (dimensions*z_unit*height_scale).tolist()
            result.append(description)
            covered_s += int(np.count_nonzero(footprint & native_solid))
            covered_v += int(np.count_nonzero(footprint & native_opening))
        if not planes and (total_s+total_v):
            failures.append({"id":role,"reason":"insufficient_coplanar_solid_support"})
        coverage[role] = dict(input_solid_pixels=total_s,input_opening_pixels=total_v,
                              assigned_solid_pixels=covered_s,assigned_opening_pixels=covered_v,
                              unassigned_solid_pixels=total_s-covered_s,unassigned_opening_pixels=total_v-covered_v,
                              fraction=(covered_s+covered_v)/(total_s+total_v) if total_s+total_v else None)
    diagnostics = dict(geometry_grid_width_height=list(size),coverage=coverage,failures=failures,
                       reference_policy="Assigned visible solid plus opening classes. The wall reference can include independently checked, enclosed depth-compatible unknown gaps.",
                       wall_reference_completion=wall_completion,
                       floor_reference_id="floor_1" if floor_distance is not None else None,
                       floor_anchor_check=anchor_check,
                       camera_height_m=camera_height_m,camera_height_applied=height_scale is not None,
                       floor_distance_model_units=floor_distance,scale_m_per_model_unit=height_scale,
                       scale_source="camera_height_over_perpendicular_floor_distance" if height_scale is not None else "shared_monocular_relative_depth")
    return result,diagnostics
