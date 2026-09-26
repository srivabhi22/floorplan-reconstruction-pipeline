"""
reproject.py — Step 5
Re-runs the Module 1 unprojection pipeline using drift-corrected poses.

For each frame we look up which segment it belongs to, apply that segment's
4×4 correction transform on top of the original ARKit pose, then unproject
the depth frame into corrected world space.
"""

import numpy as np
import open3d as o3d
import pandas as pd
from pathlib import Path

# Reuse Module 1 helpers
from src.lidar.load_frames import load_masked_depth
from src.lidar.unproject import unproject_frame


def reproject_corrected(
    odom_df: pd.DataFrame,
    segments: list[tuple[int, int]],
    correction_transforms: list[np.ndarray],   # one (4,4) per segment
    depth_dir: Path,
    conf_dir: Path,
    fallback_intrinsics: dict,
    depth_scale: float,
    min_confidence: int,
    batch_size: int = 50,
) -> np.ndarray:
    """
    Reproject all depth frames using corrected per-segment poses.

    Each frame's corrected world transform is:
        T_corrected = C_seg @ T_arkit
    where C_seg is the 4×4 correction for the segment that frame belongs to.

    Returns:
        (N, 3) float64 array of corrected world-space points.
    """
    # Map every frame index to a segment index
    frame_to_seg = {}
    for seg_idx, (start, end) in enumerate(segments):
        for fi in range(start, end):
            # Later segments overwrite earlier ones in overlap regions — fine,
            # because the last segment touching a frame has the most recent correction.
            frame_to_seg[fi] = seg_idx

    all_points = []
    rows = list(odom_df.iterrows())
    n_frames = len(rows)

    for batch_start in range(0, n_frames, batch_size):
        batch = rows[batch_start : batch_start + batch_size]
        batch_pts = []

        for fi, (_, row) in enumerate(batch, start=batch_start):
            frame_id = row["frame"]
            depth_path = depth_dir / f"{frame_id}.png"
            conf_path  = conf_dir  / f"{frame_id}.png"

            if not depth_path.exists() or not conf_path.exists():
                continue

            depth = load_masked_depth(
                str(depth_path), str(conf_path),
                depth_scale=depth_scale,
                min_confidence=min_confidence,
            )

            fx = row.get("fx", np.nan)
            fy = row.get("fy", np.nan)
            cx = row.get("cx", np.nan)
            cy = row.get("cy", np.nan)
            if any(np.isnan(v) for v in [fx, fy, cx, cy]):
                fx, fy = fallback_intrinsics["fx"], fallback_intrinsics["fy"]
                cx, cy = fallback_intrinsics["cx"], fallback_intrinsics["cy"]

            T_arkit = row["T"]   # original 4×4 from odometry
            seg_idx = frame_to_seg.get(fi, 0)
            C = correction_transforms[seg_idx]
            T_corrected = C @ T_arkit

            pts = unproject_frame(depth, T_corrected, fx, fy, cx, cy)
            if pts.shape[0] > 0:
                batch_pts.append(pts)

        if batch_pts:
            all_points.append(np.vstack(batch_pts))

        done = min(batch_start + batch_size, n_frames)
        print(f"  Reprojecting {done}/{n_frames} frames …", end="\r", flush=True)

    print()
    return np.vstack(all_points) if all_points else np.empty((0, 3), dtype=np.float64)


def downsample_and_clean(
    xyz: np.ndarray,
    voxel_size: float = 0.02,
    sor_nb: int = 20,
    sor_std: float = 2.0,
) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd = pcd.voxel_down_sample(voxel_size)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=sor_nb, std_ratio=sor_std)
    return pcd
