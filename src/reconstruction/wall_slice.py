"""
wall_slice.py — Step 3
Extract the mid-height band (1.0–1.5 m above floor) from the floor-normalized
point cloud. This band cuts through all walls cleanly while missing furniture
tops and ceiling fixtures.
"""

import numpy as np
import open3d as o3d


def extract_wall_slice(
    pts_norm: np.ndarray,
    z_min: float = 1.0,
    z_max: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Filter floor-normalized points to the wall-slice height band.

    Args:
        pts_norm: (N, 3) points with z = 0 at floor level.
        z_min:    Lower bound of the slice in metres.
        z_max:    Upper bound of the slice in metres.

    Returns:
        (slice_pts, slice_indices) — the filtered points and their indices
        into pts_norm, so they can be mapped back to room labels later.
    """
    mask = (pts_norm[:, 2] >= z_min) & (pts_norm[:, 2] <= z_max)
    indices = np.where(mask)[0]
    return pts_norm[mask], indices


def save_wall_slice(pts: np.ndarray, path: str) -> None:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    o3d.io.write_point_cloud(path, pcd)
