"""
fuse.py — Steps 5, 6 & 7  (main entry point for Module 1)

Fuses all depth frames into a single world-space point cloud, then:
  - voxel-downsamples to reduce size
  - removes statistical outliers
  - saves the result as point_cloud.ply

Usage:
    python src/lidar/fuse.py --data_dir data/ --output_dir outputs/
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from dotenv import load_dotenv

# Allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.lidar.load_odometry import load_global_intrinsics, load_odometry
from src.lidar.load_frames import load_masked_depth
from src.lidar.unproject import unproject_frame

# ── Load .env defaults ────────────────────────────────────────────────────────
load_dotenv()

DEPTH_SCALE    = float(os.getenv("DEPTH_SCALE",    0.001))
MIN_CONFIDENCE = int(  os.getenv("MIN_CONFIDENCE", 1))
VOXEL_SIZE     = float(os.getenv("VOXEL_SIZE",     0.02))
SOR_NB         = int(  os.getenv("SOR_NB_NEIGHBORS", 20))
SOR_STD        = float(os.getenv("SOR_STD_RATIO",  2.0))
BATCH_SIZE     = int(  os.getenv("BATCH_SIZE",     50))


def build_fused_cloud(
    odom_df,
    depth_dir: Path,
    conf_dir: Path,
    fallback_intrinsics: dict,
) -> np.ndarray:
    """
    Iterate over all frames in batches, unproject each depth map to world space,
    and accumulate all valid points into one large (N, 3) array.
    """
    all_points = []
    total_raw = 0
    total_masked = 0

    rows = list(odom_df.iterrows())
    n_frames = len(rows)

    for batch_start in range(0, n_frames, BATCH_SIZE):
        batch = rows[batch_start : batch_start + BATCH_SIZE]
        batch_points = []

        for _, row in batch:
            frame_id = row["frame"]
            depth_path = depth_dir / f"{frame_id}.png"
            conf_path  = conf_dir  / f"{frame_id}.png"

            if not depth_path.exists() or not conf_path.exists():
                # Frame may have been dropped during capture — skip silently
                continue

            depth = load_masked_depth(
                str(depth_path),
                str(conf_path),
                depth_scale=DEPTH_SCALE,
                min_confidence=MIN_CONFIDENCE,
            )

            total_raw    += int((depth >= 0).sum())
            total_masked += int((depth > 0).sum())

            # Use per-frame intrinsics; fall back to global if any are NaN/missing
            fx = row.get("fx", np.nan)
            fy = row.get("fy", np.nan)
            cx = row.get("cx", np.nan)
            cy = row.get("cy", np.nan)
            if any(np.isnan(v) for v in [fx, fy, cx, cy]):
                fx = fallback_intrinsics["fx"]
                fy = fallback_intrinsics["fy"]
                cx = fallback_intrinsics["cx"]
                cy = fallback_intrinsics["cy"]

            pts = unproject_frame(depth, row["T"], fx, fy, cx, cy)
            if pts.shape[0] > 0:
                batch_points.append(pts)

        if batch_points:
            all_points.append(np.vstack(batch_points))

        done = min(batch_start + BATCH_SIZE, n_frames)
        print(f"  Processed {done}/{n_frames} frames …", end="\r", flush=True)

    print()
    print(f"  Raw pixel-depth readings : {total_raw:,}")
    print(f"  After confidence masking : {total_masked:,}")

    if not all_points:
        return np.empty((0, 3), dtype=np.float64)
    return np.vstack(all_points)


def downsample_and_clean(xyz: np.ndarray) -> o3d.geometry.PointCloud:
    """
    Convert raw xyz array to Open3D point cloud, voxel-downsample, then
    remove statistical outliers.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    # Step 5 — voxel downsampling
    pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
    print(f"  After voxel downsampling ({VOXEL_SIZE*100:.0f} cm): {len(pcd.points):,} points")

    # Step 6 — statistical outlier removal
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=SOR_NB, std_ratio=SOR_STD)
    print(f"  After statistical outlier removal         : {len(pcd.points):,} points")

    return pcd


def validate(pcd: o3d.geometry.PointCloud) -> None:
    """Print a bounding-box sanity check so obvious failures are visible."""
    if len(pcd.points) == 0:
        print("WARNING: point cloud is empty!")
        return
    pts = np.asarray(pcd.points)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = hi - lo
    print(f"  Bounding box  : [{lo[0]:.2f}, {lo[1]:.2f}, {lo[2]:.2f}]"
          f"  →  [{hi[0]:.2f}, {hi[1]:.2f}, {hi[2]:.2f}]")
    print(f"  Span (x,y,z)  : {span[0]:.2f} m, {span[1]:.2f} m, {span[2]:.2f} m")
    if any(s < 0.5 for s in span):
        print("WARNING: at least one axis span < 0.5 m — check transform or intrinsics.")


def main(data_dir: str, output_dir: str) -> None:
    data    = Path(data_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    odom_path = data / "odometry.csv"
    cam_path  = data / "camera_matrix.csv"
    depth_dir = data / "depth"
    conf_dir  = data / "confidence"

    print("── Module 1: LiDAR Preprocessing ───────────────────────────────")
    print("Step 1 — Loading odometry …")
    odom_df  = load_odometry(str(odom_path))
    fallback = load_global_intrinsics(str(cam_path))
    print(f"  {len(odom_df)} frames loaded from odometry.csv")

    print("Steps 2-4 — Unprojecting depth frames to world space …")
    xyz = build_fused_cloud(odom_df, depth_dir, conf_dir, fallback)
    print(f"  Total points before downsampling: {len(xyz):,}")

    if len(xyz) == 0:
        print("ERROR: no points accumulated — check data paths and frame files.")
        sys.exit(1)

    print("Steps 5-6 — Downsampling and cleaning …")
    pcd = downsample_and_clean(xyz)

    print("Validation …")
    validate(pcd)

    print("Step 7 — Saving …")
    out_path = out_dir / "point_cloud.ply"
    o3d.io.write_point_cloud(str(out_path), pcd)
    print(f"  Saved → {out_path}")
    print("────────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module 1 — LiDAR fuse pipeline")
    parser.add_argument("--data_dir",   default=os.getenv("DATA_DIR",   "data/"))
    parser.add_argument("--output_dir", default=os.getenv("OUTPUT_DIR", "outputs/"))
    args = parser.parse_args()
    main(args.data_dir, args.output_dir)
