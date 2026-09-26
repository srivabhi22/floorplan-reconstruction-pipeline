"""
backproject.py — Step 3
Back-project a pixel damage mask into 3D world-space points using depth + pose.
"""

import numpy as np
import cv2


def mask_to_world_points(
    mask: np.ndarray,
    depth_path: str,
    T: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    depth_scale: float = 0.001,
    min_confidence: int = 1,
    confidence_path: str | None = None,
) -> np.ndarray:
    """
    Convert a boolean pixel mask to 3D world-space points.

    Args:
        mask:             (H, W) bool — True pixels are damage candidates.
        depth_path:       Path to 16-bit depth PNG for this frame.
        T:                (4, 4) camera-to-world transform from odometry.
        fx,fy,cx,cy:      Camera intrinsics for this frame.
        depth_scale:      Multiplier converting raw 16-bit value to metres.
        min_confidence:   Drop pixels below this confidence level (0–2).
        confidence_path:  Path to confidence PNG (optional).

    Returns:
        (N, 3) float64 world-space points, or empty (0, 3) if none valid.
    """
    depth_raw = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
    if depth_raw is None:
        return np.empty((0, 3), dtype=np.float64)

    depth = depth_raw.astype(np.float64) * depth_scale

    if confidence_path:
        conf = cv2.imread(confidence_path, cv2.IMREAD_GRAYSCALE)
        if conf is not None:
            depth[conf < min_confidence] = 0.0

    # Resize mask to depth resolution if they differ (RGB vs LiDAR resolution mismatch)
    dH, dW = depth.shape
    if mask.shape != depth.shape:
        mask_resized = cv2.resize(
            mask.astype(np.uint8), (dW, dH), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
    else:
        mask_resized = mask

    valid = mask_resized & (depth > 0)
    if not valid.any():
        return np.empty((0, 3), dtype=np.float64)

    v_idx, u_idx = np.where(valid)
    d = depth[v_idx, u_idx]

    X_cam = (u_idx - cx) * d / fx
    Y_cam = (v_idx - cy) * d / fy
    Z_cam = d

    pts_cam = np.stack([X_cam, Y_cam, Z_cam], axis=1)   # (N, 3)
    ones    = np.ones((len(pts_cam), 1), dtype=np.float64)
    pts_h   = np.hstack([pts_cam, ones])                 # (N, 4)
    pts_world = (T @ pts_h.T).T[:, :3]                   # (N, 3)
    return pts_world
