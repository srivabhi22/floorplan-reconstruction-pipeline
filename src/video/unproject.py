"""unproject.py — Step 4: depth maps → world-space point cloud (video version).

Reuses the same pinhole math as src/lidar/unproject.py but adds:
  - depth range filtering (0.3–8.0 m)
  - depth gradient filter (reject pixels at depth discontinuities)
"""

import numpy as np
from src.lidar.unproject import depth_to_camera_points, camera_to_world

_DEPTH_MIN = 0.3   # metres
_DEPTH_MAX = 8.0
_GRAD_THRESH = 0.1  # relative depth gradient threshold


def filter_depth(depth: np.ndarray) -> np.ndarray:
    """Zero out invalid depth pixels (range + discontinuity)."""
    d = depth.copy()

    # Range filter
    d[(d < _DEPTH_MIN) | (d > _DEPTH_MAX)] = 0

    # Gradient filter — reject pixels where neighbours have very different depth
    if d.shape[0] > 2 and d.shape[1] > 2:
        grad_x = np.abs(np.diff(d, axis=1, prepend=d[:, :1]))
        grad_y = np.abs(np.diff(d, axis=0, prepend=d[:1, :]))
        grad = np.maximum(grad_x, grad_y)
        # Relative gradient > threshold relative to local depth → discontinuity
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_grad = np.where(d > 0, grad / d, 0)
        d[rel_grad > _GRAD_THRESH] = 0

    return d


def unproject_frame(
    depth: np.ndarray,
    T: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """Depth map → world-space point cloud for one video frame."""
    d_filtered = filter_depth(depth)
    pts_cam = depth_to_camera_points(d_filtered, fx, fy, cx, cy)
    if pts_cam.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64)
    return camera_to_world(pts_cam, T)
