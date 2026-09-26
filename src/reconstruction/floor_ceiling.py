"""
floor_ceiling.py — Steps 1, 2, 6
RANSAC-based floor and ceiling detection, plus per-room ceiling refinement.
"""

import numpy as np
import open3d as o3d


def _ransac_plane(pcd: o3d.geometry.PointCloud, dist_thresh: float = 0.02
                  ) -> tuple[float, np.ndarray] | None:
    """
    Fit a plane to pcd via RANSAC.
    Returns (z_intercept, normal) where normal is unit-length, or None if too few points.

    z_intercept is the signed distance from the origin along the dominant axis
    (i.e. the d in  normal·x + d = 0  →  z = -d / normal[v_axis]).
    """
    if len(pcd.points) < 10:
        return None
    model, _ = pcd.segment_plane(
        distance_threshold=dist_thresh,
        ransac_n=3,
        num_iterations=1000,
    )
    a, b, c, d = model
    normal = np.array([a, b, c], dtype=np.float64)
    norm = np.linalg.norm(normal)
    if norm < 1e-9:
        return None
    normal /= norm
    d /= norm
    # intercept along the axis with the largest normal component
    v = int(np.argmax(np.abs(normal)))
    z_intercept = -d / normal[v]
    return z_intercept, normal


def detect_floor(pts: np.ndarray, dist_thresh: float = 0.02) -> float:
    """
    Step 1 — Find floor height by RANSAC on the lowest 20 % of points.

    Returns z_floor in the original coordinate system.
    """
    v = int(np.argmax(pts.max(axis=0) - pts.min(axis=0)))
    threshold = np.percentile(pts[:, v], 20)
    mask = pts[:, v] <= threshold
    subset = pts[mask]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(subset)
    result = _ransac_plane(pcd, dist_thresh)
    if result is None:
        # Fallback: just take the percentile value
        return float(np.percentile(pts[:, v], 5))
    z_floor, _ = result
    return float(z_floor)


def detect_ceiling(pts_norm: np.ndarray, dist_thresh: float = 0.02) -> float:
    """
    Step 2 — Find ceiling height by RANSAC on the top 20 % of z_norm points.

    Args:
        pts_norm: (N, 3) points already normalized so floor = 0.
    Returns:
        z_ceiling in normalized coordinates.
    """
    threshold = np.percentile(pts_norm[:, 2], 80)
    mask = pts_norm[:, 2] >= threshold
    subset = pts_norm[mask]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(subset)
    result = _ransac_plane(pcd, dist_thresh)
    if result is None:
        return float(np.percentile(pts_norm[:, 2], 95))
    z_ceiling, _ = result
    return float(z_ceiling)


def detect_ceiling_per_room(
    pts_norm: np.ndarray,
    room_labels: np.ndarray,
    room_2d_bounds: dict,        # room_id → (x_min, x_max, y_min, y_max)
    dist_thresh: float = 0.02,
    min_height: float = 1.8,
) -> dict:
    """
    Step 6 — Re-run ceiling RANSAC per room on points above min_height.

    Args:
        pts_norm:       (N, 3) floor-normalized points.
        room_labels:    (M,) room id per wall-slice point (from DBSCAN).
        room_2d_bounds: dict mapping room_id → (x_min, x_max, y_min, y_max).
        min_height:     Only use points above this z_norm for ceiling fitting.

    Returns:
        dict  room_id → ceiling_height_m  (float)
    """
    result = {}
    high_mask = pts_norm[:, 2] >= min_height
    high_pts = pts_norm[high_mask]

    for room_id, (x_min, x_max, y_min, y_max) in room_2d_bounds.items():
        in_room = (
            (high_pts[:, 0] >= x_min) & (high_pts[:, 0] <= x_max) &
            (high_pts[:, 1] >= y_min) & (high_pts[:, 1] <= y_max)
        )
        room_pts = high_pts[in_room]
        if len(room_pts) < 10:
            # Not enough ceiling points — use global estimate
            result[room_id] = float(np.percentile(room_pts[:, 2], 95)) if len(room_pts) else None
            continue

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(room_pts)
        fit = _ransac_plane(pcd, dist_thresh)
        result[room_id] = float(fit[0]) if fit else float(np.percentile(room_pts[:, 2], 95))

    return result
