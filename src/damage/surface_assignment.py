"""
surface_assignment.py — Step 4
Assign a 3D damage cluster to the nearest named wall/floor/ceiling surface.
"""

import numpy as np


def _wall_segment_name(idx: int) -> str:
    """Map wall index to a cardinal direction name (best-effort for labelling)."""
    names = ["north", "east", "south", "west"]
    return names[idx % len(names)]


def build_surface_planes(wall_data: dict, z_floor: float = 0.0, z_ceiling: float = 2.5
                         ) -> list[dict]:
    """
    Build a list of named surface descriptors from wall_polygons.json.

    Each surface dict:
        {room_id, name, normal (3,), point_on_plane (3,), type ('wall'|'floor'|'ceiling')}
    """
    surfaces = []
    for room_id, room in wall_data["rooms"].items():
        verts = np.array(room["vertices"])
        n = len(verts) - 1

        # Wall surfaces: one per polygon edge
        for i in range(n):
            p1, p2 = verts[i, :2], verts[i + 1, :2]
            edge = p2 - p1
            length = np.linalg.norm(edge)
            if length < 1e-6:
                continue
            # Normal pointing inward (toward room centroid)
            outward = np.array([-edge[1], edge[0]]) / length
            mid2d   = (p1 + p2) / 2
            normal3 = np.array([outward[0], outward[1], 0.0])
            point3  = np.array([mid2d[0], mid2d[1], (z_floor + z_ceiling) / 2])
            surfaces.append({
                "room_id": room_id,
                "name":    _wall_segment_name(i) + "_wall",
                "normal":  normal3,
                "point":   point3,
                "type":    "wall",
            })

        # Floor
        centroid = verts[:-1].mean(axis=0)
        surfaces.append({
            "room_id": room_id,
            "name":    "floor",
            "normal":  np.array([0.0, 0.0, 1.0]),
            "point":   np.array([centroid[0], centroid[1], z_floor]),
            "type":    "floor",
        })
        # Ceiling
        surfaces.append({
            "room_id": room_id,
            "name":    "ceiling",
            "normal":  np.array([0.0, 0.0, -1.0]),
            "point":   np.array([centroid[0], centroid[1], z_ceiling]),
            "type":    "ceiling",
        })

    return surfaces


def assign_to_surface(
    cluster_pts: np.ndarray,
    surfaces: list[dict],
) -> dict:
    """
    Find the closest named surface to the centroid of a damage cluster.

    Returns the matched surface dict.
    """
    centroid = cluster_pts.mean(axis=0)
    best, best_dist = None, np.inf

    for surf in surfaces:
        # Signed distance from centroid to the plane (normal · (point - plane_point))
        dist = abs(float(np.dot(surf["normal"], centroid - surf["point"])))
        if dist < best_dist:
            best_dist = dist
            best = surf

    return best
