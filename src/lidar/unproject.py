"""
unproject.py — Steps 3 & 4
Converts a masked depth map to a world-space point cloud using the frame's
camera intrinsics and 4×4 camera-to-world transform.
"""

import numpy as np


def depth_to_camera_points(
    depth: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """
    Unproject every valid depth pixel into camera-space 3D coordinates.

    Uses the standard pinhole formula:
        X_cam = (u - cx) * d / fx
        Y_cam = (v - cy) * d / fy
        Z_cam = d

    Args:
        depth: (H, W) float32 depth in metres; 0 = invalid.
        fx, fy, cx, cy: Camera intrinsics for this frame.

    Returns:
        (N, 3) float64 array of camera-space points (only valid pixels).
    """
    H, W = depth.shape
    u_coords, v_coords = np.meshgrid(np.arange(W), np.arange(H))  # (H,W) each

    valid = depth > 0
    d = depth[valid].astype(np.float64)
    u = u_coords[valid].astype(np.float64)
    v = v_coords[valid].astype(np.float64)

    X = (u - cx) * d / fx
    Y = (v - cy) * d / fy
    Z = d

    return np.stack([X, Y, Z], axis=1)  # (N, 3)


def camera_to_world(points_cam: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Apply a 4×4 camera-to-world transform to a batch of camera-space points.

    Args:
        points_cam: (N, 3) camera-space points.
        T:          (4, 4) camera-to-world homogeneous transform.

    Returns:
        (N, 3) world-space points.
    """
    N = points_cam.shape[0]
    # Homogeneous coordinates: (N, 4)
    ones = np.ones((N, 1), dtype=np.float64)
    pts_h = np.hstack([points_cam, ones])

    # T @ pts_h.T → (4, N), then transpose back and drop the w row
    pts_world = (T @ pts_h.T).T  # (N, 4)
    return pts_world[:, :3]


def unproject_frame(
    depth: np.ndarray,
    T: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """
    Full pipeline for a single frame: depth map → world-space point cloud.

    Returns:
        (N, 3) float64 world-space xyz, or empty (0, 3) if no valid pixels.
    """
    pts_cam = depth_to_camera_points(depth, fx, fy, cx, cy)
    if pts_cam.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64)
    return camera_to_world(pts_cam, T)
