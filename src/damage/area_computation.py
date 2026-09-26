"""
area_computation.py — Step 5
Compute the metric surface area of a 3D damage cluster.

Method:
  1. Fit a plane to the cluster via SVD (robust to small clusters)
  2. Project all points onto that plane
  3. Compute the 2D convex hull area of the projected points
"""

import numpy as np
from scipy.spatial import ConvexHull


def fit_plane_svd(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit a plane to pts via SVD.
    Returns (normal, centroid) where normal is unit-length.
    """
    centroid = pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts - centroid)
    normal = Vt[-1]   # last right singular vector = normal of best-fit plane
    return normal / np.linalg.norm(normal), centroid


def project_to_plane(pts: np.ndarray, normal: np.ndarray, centroid: np.ndarray
                     ) -> np.ndarray:
    """
    Project 3D pts onto the plane defined by (normal, centroid).
    Returns (N, 2) 2D coordinates in the plane's local frame.
    """
    # Build two orthogonal in-plane basis vectors
    arbitrary = np.array([1, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1, 0])
    u = np.cross(normal, arbitrary)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)

    centred = pts - centroid
    coords_2d = np.stack([centred @ u, centred @ v], axis=1)
    return coords_2d


def convex_hull_area(pts_2d: np.ndarray) -> float:
    """Return the convex hull area of a (N, 2) point set in m²."""
    if len(pts_2d) < 3:
        return 0.0
    try:
        hull = ConvexHull(pts_2d)
        return float(hull.volume)   # ConvexHull.volume = area in 2D
    except Exception:
        return 0.0


def damage_area_m2(cluster_pts: np.ndarray) -> float:
    """Full pipeline: 3D cluster → metric surface area in m²."""
    if len(cluster_pts) < 3:
        return 0.0
    normal, centroid = fit_plane_svd(cluster_pts)
    pts_2d = project_to_plane(cluster_pts, normal, centroid)
    return round(convex_hull_area(pts_2d), 4)
