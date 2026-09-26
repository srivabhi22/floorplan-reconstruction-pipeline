"""
segment.py — Step 1
Divides the fused point cloud into overlapping frame-range segments.

Because Module 1 fuses all frames into a single PLY without per-point frame IDs,
we re-derive which world-space region each segment covers by computing the camera
positions for that segment's frame range and cropping the cloud to a convex hull
bounding box around those positions (with a padding margin).
"""

import numpy as np
import open3d as o3d
import pandas as pd


def make_segments(n_frames: int, seg_size: int, overlap: int) -> list[tuple[int, int]]:
    """
    Return a list of (start, end) frame-index ranges covering [0, n_frames).
    Adjacent segments overlap by `overlap` frames.

    Example with n_frames=10, seg_size=4, overlap=1:
        [(0,4), (3,7), (6,10)]
    """
    step = seg_size - overlap
    segments = []
    start = 0
    while start < n_frames:
        end = min(start + seg_size, n_frames)
        segments.append((start, end))
        if end == n_frames:
            break
        start += step
    return segments


def segment_cloud_by_camera_bbox(
    pcd: o3d.geometry.PointCloud,
    odom_df: pd.DataFrame,
    seg_start: int,
    seg_end: int,
    padding: float = 0.5,
) -> o3d.geometry.PointCloud:
    """
    Crop the full point cloud to points near the camera path of a frame segment.

    Strategy: compute the axis-aligned bounding box of the camera positions for
    frames [seg_start, seg_end), expand it by `padding` metres, and return the
    cloud subset inside that box. This is an approximation — it may include a few
    points from adjacent areas but is fast and good enough for plane fitting.

    Args:
        pcd:       Full fused point cloud.
        odom_df:   Odometry DataFrame (must have x, y, z columns and a row per frame).
        seg_start: First frame index (inclusive).
        seg_end:   Last frame index (exclusive).
        padding:   Metres to expand the bounding box on each side.

    Returns:
        Cropped Open3D PointCloud for this segment.
    """
    rows = odom_df.iloc[seg_start:seg_end]
    cam_pos = rows[["x", "y", "z"]].values  # (N_frames, 3)

    lo = cam_pos.min(axis=0) - padding
    hi = cam_pos.max(axis=0) + padding

    bbox = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=lo.tolist(),
        max_bound=hi.tolist(),
    )
    return pcd.crop(bbox)
