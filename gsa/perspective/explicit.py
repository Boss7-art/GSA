import cv2
import numpy as np


def _near_limit(mask, h):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    boundary = np.concatenate(contours)[:, 0, :]
    values = boundary @ h[2, :2] + h[2, 2]
    return float(values.min() - .5 * np.abs(h[2, :2]).sum())


def _projected_area(mask, h):
    area = 0.
    for start in range(0, mask.shape[0], 96):
        padded = np.pad(mask[start:start + 96], ((0, 0), (1, 1)))
        rows, edges = np.nonzero(padded[:, 1:] != padded[:, :-1])
        if not len(edges):
            continue
        left, right = edges[::2] - .5, edges[1::2] - .5
        top = rows[::2] + start - .5
        x = np.column_stack((left, right, right, left))
        y = np.column_stack((top, top, top + 1, top + 1))
        w = h[2, 0]*x + h[2, 1]*y + h[2, 2]
        u = (h[0, 0]*x + h[0, 1]*y + h[0, 2]) / w
        v = (h[1, 0]*x + h[1, 1]*y + h[1, 2]) / w
        du, dv = u[:, 1:] - u[:, :1], v[:, 1:] - v[:, :1]
        area += .5 * np.abs((du[:, :-1]*dv[:, 1:] - dv[:, :-1]*du[:, 1:]).sum(axis=1)).sum()
    return float(area)


def _cell_scale(plane, matrix, size, unit, density):
    key = {"metre": "width_height_metres", "relative": "width_height_relative"}.get(unit)
    if key in plane and "quad_tl_tr_br_bl" in plane:
        corners = np.asarray(plane["quad_tl_tr_br_bl"], np.float64)
        wh = np.asarray(plane[key], np.float64)
        if corners.shape == (4, 2) and wh.shape == (2,) and np.all(np.isfinite(wh) & (wh > 0)):
            projected = cv2.perspectiveTransform(corners[None, :, :], matrix)[0]
            w, h = np.asarray(size) - 1
            expected = [[0, 0], [w, 0], [w, h], [0, h]]
            if np.allclose(projected, expected, atol=.02, rtol=1e-6):
                return float(wh.prod() / (w*h)), "control_rectangle"
    return float(1./density**2), "declared_sampling"


def _ratios(solid, opening):
    labelled = solid + opening
    return dict(solid_area=solid, void_area=opening, labelled_area=labelled,
                void_over_total=opening/labelled if labelled else None,
                void_over_solid=opening/solid if solid else None)


def _restrict_wall(solid, opening, reference):
    if np.any(solid & (1 - reference)):
        solid &= reference
        opening &= reference
        return False
    beyond = opening & (1 - reference)
    if not beyond.any():
        return True
    contours, _ = cv2.findContours(solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    enclosed = np.zeros_like(solid)
    for contour in contours:
        cv2.drawContours(enclosed, [contour], -1, 1, cv2.FILLED)
    complete = not np.any(beyond & (1 - enclosed))
    opening &= reference | enclosed
    return complete


def measure(plane, reference, classes, codes, matrix, size, unit, density, full_role_candidate):
    row = int(np.flatnonzero(reference.any(axis=1))[0])
    col = int(np.flatnonzero(reference[row])[0])
    value = matrix[2] @ (col, row, 1.)
    h = matrix/value
    support_min = _near_limit(reference, h)
    cell_scale, scale_source = _cell_scale(plane, matrix, size, unit, density)
    area = dict(scope="assigned_footprint", source_solid_pixels=0, source_void_pixels=0,
                cell_area=cell_scale, cell_area_source=scale_source, reason=None)
    height, width = reference.shape
    error = 8*np.finfo(float).eps * (abs(h[2, 0])*width + abs(h[2, 1])*height + abs(h[2, 2]))
    if support_min is None or support_min <= error:
        area.update(scope="unavailable", reason="reference_pixel_cells_cross_horizon",
                    solid_area=None, void_area=None, labelled_area=None, void_over_total=None, void_over_solid=None)
        return area
    solid = np.asarray(classes == codes[0], np.uint8)
    opening = np.asarray(classes == codes[1], np.uint8)
    if full_role_candidate:
        trusted = plane.get("all_role_labels_on_this_plane", False)
        complete = True
        if plane["role"] == "wall" and not trusted:
            complete = _restrict_wall(solid, opening, reference)
            if not complete:
                area["reason"] = "wall_labels_outside_confirmed_facade"
        full_min = _near_limit(solid | opening, h)
        limit = error if trusted else max(error, support_min - error)
        if full_min is not None:
            area["min_label_denominator_over_reference_min"] = full_min / support_min
        if full_min is not None and full_min < limit:
            complete = False
            area["reason"] = ("label_cells_touch_or_cross_horizon" if trusted
                              else "full_role_extends_toward_or_across_horizon")
            half_cell = .5 * np.abs(h[2, :2]).sum()
            x_part = h[2, 0]*np.arange(width) + h[2, 2] - half_cell
            for begin in range(0, height, 96):
                stop = min(height, begin + 96)
                safe = (x_part + h[2, 1]*np.arange(begin, stop)[:, None]) >= limit
                solid[begin:stop] &= safe
                opening[begin:stop] &= safe
        area["scope"] = "all_role_labels" if complete else "geometrically_supported_role_labels"
    else:
        solid &= reference
        opening &= reference
    area["source_solid_pixels"] = int(solid.sum())
    area["source_void_pixels"] = int(opening.sum())
    area.update(_ratios(_projected_area(solid, h)*cell_scale, _projected_area(opening, h)*cell_scale))
    return area


def summarize(records, groups, aggregates, inputs):
    out = {}
    for name, roles in groups.items():
        parts = [p for p in records if p["role"] in roles]
        values = [p["explicit_segmentation"] for p in parts]
        covered = [sum(v[k] for v in values) for k in ("source_solid_pixels", "source_void_pixels")]
        totals = [sum(inputs[r][i] for r in roles) for i in (0, 1)]
        compatible = aggregates[name]["comparable"]
        valid = [v for v in values if v["solid_area"] is not None]
        measured = (_ratios(sum(v["solid_area"] for v in valid), sum(v["void_area"] for v in valid))
                    if valid and compatible else None)
        if not sum(totals):
            status = "no_role_labels"
        elif not valid:
            status = "no_valid_plane"
        elif not compatible:
            status = "incomparable_scales"
        elif covered != totals:
            status = "partial_label_coverage"
        else:
            status = "ok"
        row = dict(status=status, area_unit=aggregates[name]["area_unit"], plane_ids=[p["id"] for p in parts],
                   source_pixels=dict(solid=totals[0], void=totals[1]),
                   measured_source_pixels=dict(solid=covered[0], void=covered[1]), assigned=measured)
        for key in ("void_over_total", "void_over_solid"):
            row[key] = measured[key] if status == "ok" else None
        out[name] = row
    return dict(method="projected_labelled_pixel_cells", groups=out)
