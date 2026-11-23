import numpy as np

from .slab_ceiling import detect_ceiling_slab
from .slab_edge_opening import extract_slab_edge_openings
from .slab_holes import detect_slab_holes


def opening_status(accepted, uncertain):
    if accepted.any():
        return "supported_openings_with_uncertainty" if uncertain.any() else "supported_openings"
    return "uncertain_extent" if uncertain.any() else "no_supported_opening"


def extract_role(role, data, geometry, classes, labels, reference, rgb):
    if data["status"].startswith("unknown") or data["primary_plane"] is None:
        empty = np.zeros(classes.shape, dtype=bool)
        return empty, empty.copy(), data["status"]
    seed = data["seed_mask"]
    args = (geometry.points, geometry.normals, geometry.valid, geometry.intrinsics)
    if role == "ceiling":
        result = detect_ceiling_slab(seed, *args, reference, classes=classes, labels=labels, rgb=rgb)
        accepted = result["opening_hypothesis_mask"]
        uncertain = result["bounded_uncertain_mask"]
        status = result["metadata"]["status"]
        return accepted, uncertain, status if status.startswith("unknown") else opening_status(accepted, uncertain)
    primary = data["primary_plane"]
    result = detect_slab_holes(seed, *args, primary, semantic_classes=classes,
                               semantic_labels=labels, surface_role=role, verified_support=primary["support_mask"])
    accepted = result["opening_hypothesis_mask"]
    uncertain = result.get("bounded_uncertain_mask", result["ambiguous_opening_mask"] & result["enclosed_gap_mask"]).copy()
    try:
        edge = extract_slab_edge_openings(seed, *args, primary, {"dominant_gap_only": True},
                                          semantic_classes=classes, semantic_labels=labels)
    except ValueError:
        edge = None
    if edge is not None:
        uncertain |= edge["opening_hypothesis_mask"] & ~result["rejected_opening_mask"]
    uncertain &= ~accepted
    return accepted, uncertain, opening_status(accepted, uncertain)

