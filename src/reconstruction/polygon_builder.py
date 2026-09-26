"""polygon_builder.py — Step 5: assemble rectilinear polygon from merged segments."""

import numpy as np
from src.reconstruction.segment_snapping import rotate_points


def build_polygon(merged_segments: list[dict], theta: float) -> np.ndarray:
    """Build closed (N,2) world-coord polygon from axis-aligned merged segments."""
    if not merged_segments:
        return np.empty((0, 2), dtype=np.float64)

    vertices = []
    n = len(merged_segments)

    for i in range(n):
        curr = merged_segments[i]
        nxt  = merged_segments[(i + 1) % n]

        # Intersection of curr and nxt gives the corner vertex
        if curr['type'] == 'H' and nxt['type'] == 'V':
            vertex = np.array([nxt['p1'][0], curr['p1'][1]])
        elif curr['type'] == 'V' and nxt['type'] == 'H':
            vertex = np.array([curr['p1'][0], nxt['p1'][1]])
        else:
            # Same-type consecutive segments — use the endpoint of curr
            vertex = curr['p2'].copy()

        vertices.append(vertex)

    vertices = np.array(vertices, dtype=np.float64)

    # Rotate back to world coordinates
    vertices = rotate_points(vertices, theta)

    # Close the polygon
    return np.vstack([vertices, vertices[0]])


def is_closed(polygon: np.ndarray, tol: float = 1e-6) -> bool:
    """True if first and last vertex are within tol of each other."""
    if len(polygon) < 2:
        return False
    return bool(np.linalg.norm(polygon[0] - polygon[-1]) <= tol)
