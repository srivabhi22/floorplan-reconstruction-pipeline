"""
footprint.py — Step 5
Compute the global property footprint by unioning all room polygons via Shapely.
"""

from shapely.geometry import Polygon
from shapely.ops import unary_union


def compute_footprint(room_polygons: dict) -> tuple[float, list]:
    """
    Union all room polygons and return (total_area_m2, outer_boundary_vertices).

    Args:
        room_polygons: dict  room_id -> list of [x, y] vertices (closed polygon)

    Returns:
        (footprint_area_m2, boundary_coords)
        boundary_coords is a list of [x, y] pairs forming the outer hull.
    """
    polys = [Polygon(verts).buffer(0) for verts in room_polygons.values() if len(verts) >= 4]
    polys = [p for p in polys if p.is_valid and not p.is_empty]
    if not polys:
        return 0.0, []

    union = unary_union(polys)
    area  = float(union.area)

    # Extract outer boundary coordinates
    if union.geom_type == "Polygon":
        coords = [list(c) for c in union.exterior.coords]
    elif union.geom_type == "MultiPolygon":
        # Multiple disconnected buildings — take the largest
        largest = max(union.geoms, key=lambda g: g.area)
        coords = [list(c) for c in largest.exterior.coords]
    else:
        coords = []

    return round(area, 3), coords
