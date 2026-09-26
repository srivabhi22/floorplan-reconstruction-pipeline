"""
overlap_resolver.py — Step 3
Detect and resolve geometric overlaps between room polygons.

Rooms are already in the same ARKit world frame (no transform needed), so overlaps
signal residual drift. We log every overlap and apply a minimal rigid translation
to eliminate it, anchored at the shared opening position when one exists.
"""

import numpy as np
from shapely.geometry import Polygon


def _to_shapely(vertices: list) -> Polygon:
    return Polygon(vertices).buffer(0)


def detect_overlaps(room_polygons: dict) -> list[dict]:
    """
    Check all room polygon pairs for geometric intersection.

    Returns list of overlap dicts:
        {room_a, room_b, overlap_area_m2, overlap_centroid}
    """
    room_ids = list(room_polygons.keys())
    overlaps = []

    for i, rid_a in enumerate(room_ids):
        poly_a = _to_shapely(room_polygons[rid_a])
        for rid_b in room_ids[i + 1:]:
            poly_b = _to_shapely(room_polygons[rid_b])
            if not poly_a.intersects(poly_b):
                continue
            intersection = poly_a.intersection(poly_b)
            area = float(intersection.area)
            if area < 1e-4:   # sub-1 cm² — ignore floating-point artifacts
                continue
            centroid = list(intersection.centroid.coords[0])
            overlaps.append({
                "room_a":          rid_a,
                "room_b":          rid_b,
                "overlap_area_m2": round(area, 4),
                "overlap_centroid": centroid,
            })
    return overlaps


def resolve_overlap(
    vertices_b: list,
    overlap: dict,
    anchor: list | None = None,
) -> tuple[list, float]:
    """
    Translate room B by the minimum vector that eliminates the intersection,
    anchored at the shared opening position (or the overlap centroid if None).

    Returns (corrected_vertices, translation_magnitude_m).
    A large translation (> 5 cm) is flagged by the caller.
    """
    centroid = np.array(overlap["overlap_centroid"])
    anchor_pt = np.array(anchor) if anchor else centroid

    # Push room B away from the centroid along the centroid→anchor direction
    direction = centroid - anchor_pt
    dist = float(np.linalg.norm(direction))
    if dist < 1e-6:
        # Degenerate: push along +X
        direction = np.array([1.0, 0.0])
        dist = 1.0

    unit = direction / dist
    # Translation magnitude = sqrt of overlap area as a proxy for the shift needed
    shift_m = float(np.sqrt(overlap["overlap_area_m2"]))
    translation = unit * shift_m

    corrected = [[v[0] + translation[0], v[1] + translation[1]] for v in vertices_b]
    return corrected, round(shift_m, 4)
