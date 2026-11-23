import cv2
import numpy as np

from perspective.estimate import _fit, _floor_anchor, _grid


def refine_floor_apertures(roles, depth, intrinsics, normals):
    roles = np.asarray(roles)
    result = roles.copy()
    if not np.any(roles == 2) or not np.any(roles == 1):
        return result
    h,w = roles.shape
    scale = min(1., 384/max(w,h))
    fit_size = max(1,round(w*scale)), max(1,round(h*scale))
    fitted_classes = cv2.resize(roles,fit_size,interpolation=cv2.INTER_NEAREST_EXACT)
    z = cv2.resize(np.asarray(depth,np.float32),fit_size,interpolation=cv2.INTER_NEAREST_EXACT)
    valid = np.isfinite(z) & (z > 0)
    ground = (fitted_classes == 1) & valid
    if ground.sum() < 32:
        return result
    typical_depth = float(np.median(z[ground]))
    xyz = _grid(*fit_size,intrinsics)*np.where(valid,z/typical_depth,np.nan)[...,None]
    n = cv2.resize(np.asarray(normals,np.float32),fit_size,interpolation=cv2.INTER_NEAREST_EXACT)
    n = np.nan_to_num(n,nan=0.,posinf=0.,neginf=0.)
    n /= np.maximum(np.linalg.norm(n,axis=2,keepdims=True),1e-8)
    planes = _fit(xyz,n,ground,"floor",np.array([0.,1.,0.]),np.random.default_rng(6247))
    if not planes or not _floor_anchor(planes[0],ground)["eligible"]:
        return result
    plane=planes[0]
    z = cv2.resize(np.asarray(depth,np.float32),(w,h),interpolation=cv2.INTER_NEAREST_EXACT)
    valid = np.isfinite(z) & (z > 0)
    residual = z * (_grid(w,h,intrinsics) @ plane["normal"]) / (plane["offset"]*typical_depth) - 1.
    n = cv2.resize(np.asarray(normals,np.float32),(w,h),interpolation=cv2.INTER_NEAREST_EXACT)
    aligned = np.abs(n @ plane["normal"]) >= .75
    solid_evidence = valid & ((residual < -.12) | ((np.abs(residual) <= .08) & aligned))
    below = valid & (residual > .08)
    count, component, sizes, _ = cv2.connectedComponentsWithStats((roles == 2).astype(np.uint8),connectivity=8)
    kernel = np.ones((3,3),np.uint8)
    for i in range(1,count):
        area = sizes[i,cv2.CC_STAT_AREA]
        if area < max(40,.002*h*w):
            continue
        region = component==i
        exterior = cv2.dilate(region.astype(np.uint8),kernel).astype(bool) & ~region
        if np.count_nonzero(exterior & (roles==1)) < .15*np.count_nonzero(exterior):
            continue
        seeds = cv2.medianBlur((region & below).astype(np.uint8),3)
        _, parts, part_sizes, _ = cv2.connectedComponentsWithStats(seeds,connectivity=8)
        strong = part_sizes[:,cv2.CC_STAT_AREA] >= max(9,area*.006)
        strong[0] = False
        supported = strong[parts] & region
        if np.count_nonzero(supported) < .07*area:
            continue
        hull = cv2.convexHull(cv2.findNonZero(supported.astype(np.uint8)))
        core = cv2.fillConvexPoly(np.zeros_like(roles,dtype=np.uint8),hull,1)
        distance = cv2.distanceTransform(1-core,cv2.DIST_L2,5)
        margin = max(2., .035*min(sizes[i,cv2.CC_STAT_WIDTH],sizes[i,cv2.CC_STAT_HEIGHT]))
        possible = (region & solid_evidence & (distance > margin)).astype(np.uint8)
        _, patch = cv2.connectedComponents(possible,connectivity=8)
        boundary = region & ~cv2.erode(region.astype(np.uint8),kernel,borderType=cv2.BORDER_CONSTANT,
                                        borderValue=0).astype(bool)
        border_parts = np.unique(patch[boundary & possible.astype(bool)])
        if len(border_parts):
            result[np.isin(patch,border_parts)] = 1
    return result
