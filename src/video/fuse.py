"""
fuse.py — Module 9: Video Preprocessing

Entry point for the video input path. Takes a walkthrough MP4 and produces
  outputs/point_cloud.ply       (same format as Module 1)
  outputs/odometry_video.csv    (per-frame poses, same schema as odometry.csv)

Usage:
    python src/video/fuse.py --input_dir input/ --output_dir outputs/

Or from pipeline.py (tier="video"):
    module9(input_dir=..., output_dir=...)
"""

import argparse
import csv
import sys
import tempfile
from pathlib import Path

# Ensure project root is on sys.path when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import open3d as o3d
from dotenv import load_dotenv
import os

load_dotenv()

_FRAME_STRIDE   = int(os.getenv("VIDEO_FRAME_STRIDE", "5"))
_MAX_FRAMES     = int(os.getenv("VIDEO_MAX_FRAMES", "200"))
_VOXEL_SIZE     = float(os.getenv("VOXEL_SIZE", "0.02"))
_SOR_K          = int(os.getenv("SOR_NB_NEIGHBORS", "20"))
_SOR_STD        = float(os.getenv("SOR_STD_RATIO", "2.0"))


def main(input_dir: str = "input/", output_dir: str = "outputs/") -> None:
    from src.video.frame_extractor import extract_frames
    from src.video.intrinsics import load_intrinsics
    from src.video.mast3r_runner import run_mast3r
    from src.video.unproject import unproject_frame

    print("── Module 9: Video Preprocessing ───────────────────────────────")

    inp = Path(input_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Locate video
    video_path = _find_video(inp)
    print(f"  Video: {video_path}")

    cam_csv = inp / "camera_matrix.csv"

    # Step 1 — Extract frames
    print("\nStep 1 — Extracting frames …")
    with tempfile.TemporaryDirectory(prefix="vframes_") as tmpdir:
        frames = extract_frames(str(video_path), tmpdir,
                                frame_stride=_FRAME_STRIDE,
                                max_frames=_MAX_FRAMES)

        # Step 2 — Intrinsics
        print("\nStep 2 — Loading camera intrinsics …")
        K = load_intrinsics(str(video_path), str(cam_csv) if cam_csv.exists() else None)
        print(f"  fx={K['fx']:.1f}  fy={K['fy']:.1f}  cx={K['cx']:.1f}  cy={K['cy']:.1f}")

        # Step 3 — Depth + poses (lightweight: Depth Anything V2 + ORB)
        print(f"\nStep 3 — Estimating depth + poses for {len(frames)} frames …")
        from src.video.depth_pose_estimator import run as run_depth_pose
        frame_paths = [f["path"] for f in frames]
        depth_maps, poses = run_depth_pose(frame_paths, K, encoder=os.getenv("DEPTH_ENCODER", "vits"))

        # Step 4 — Unproject to world-space points
        print("\nStep 4 — Unprojecting depth maps to world-space …")
        all_pts = []
        fx, fy, cx, cy = K["fx"], K["fy"], K["cx"], K["cy"]
        for i, (depth, T) in enumerate(zip(depth_maps, poses)):
            pts = unproject_frame(depth, T, fx, fy, cx, cy)
            if pts.shape[0] > 0:
                all_pts.append(pts)
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(frames)} frames processed")

    if not all_pts:
        raise RuntimeError("No valid points after unprojection — check video quality")

    points = np.vstack(all_pts)
    print(f"  Total raw points: {len(points):,}")

    # Step 5 — Voxel downsample + SOR
    print("\nStep 5 — Voxel downsampling + outlier removal …")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd = pcd.voxel_down_sample(_VOXEL_SIZE)
    print(f"  After voxel ({_VOXEL_SIZE} m): {len(pcd.points):,}")
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=_SOR_K, std_ratio=_SOR_STD)
    print(f"  After SOR: {len(pcd.points):,}")

    # Step 6 — Save outputs
    print("\nStep 6 — Saving outputs …")
    ply_path = out / "point_cloud.ply"
    o3d.io.write_point_cloud(str(ply_path), pcd)
    print(f"  Saved → {ply_path}")

    odo_path = out / "odometry_video.csv"
    _save_odometry(frames, poses, odo_path)
    print(f"  Saved → {odo_path}")

    print("\n── Summary ──────────────────────────────────────────────────────")
    print(f"  Frames processed : {len(frames)}")
    print(f"  Final point cloud: {len(pcd.points):,} points")
    print(f"  Output PLY       : {ply_path}")
    print("────────────────────────────────────────────────────────────────")


def _find_video(inp: Path) -> Path:
    for ext in ("*.mp4", "*.MP4", "*.mov", "*.MOV", "*.avi"):
        matches = list(inp.glob(ext))
        if matches:
            return matches[0]
    # Also check common name used in project data dir
    for name in ("walkthrough.mp4", "rgb.mp4", "video.mp4"):
        p = inp / name
        if p.exists():
            return p
    raise FileNotFoundError(f"No video file found in {inp}")


def _save_odometry(frames: list[dict], poses: list[np.ndarray], path: Path) -> None:
    """Save poses as CSV matching the odometry.csv schema used by Module 2."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "frame_index",
            "tx", "ty", "tz",
            "qw", "qx", "qy", "qz",
        ])
        for frame, T in zip(frames, poses):
            t = T[:3, 3]
            # Convert rotation matrix to quaternion
            R = T[:3, :3]
            qw, qx, qy, qz = _rot_to_quat(R)
            writer.writerow([
                frame["timestamp_s"], frame["index"],
                round(t[0], 6), round(t[1], 6), round(t[2], 6),
                round(qw, 8), round(qx, 8), round(qy, 8), round(qz, 8),
            ])


def _rot_to_quat(R: np.ndarray):
    """3×3 rotation matrix → (qw, qx, qy, qz)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return qw, qx, qy, qz


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module 9 — Video Preprocessing")
    parser.add_argument("--input_dir",  default="input/")
    parser.add_argument("--output_dir", default="outputs/")
    args = parser.parse_args()
    main(args.input_dir, args.output_dir)
