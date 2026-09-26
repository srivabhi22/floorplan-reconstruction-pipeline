"""
drift_correction.py — Module 2 entry point

Corrects accumulated ARKit pose drift in the fused point cloud using a
lightweight pose-graph optimization (scipy least-squares, no GTSAM required).

Pipeline:
  Step 1  segment cloud into overlapping chunks
  Step 2  RANSAC plane detection per segment
  Step 3  build pose graph (sequential + plane-match edges)
  Step 4  optimize pose graph → per-segment correction transforms
  Step 5  reproject all frames with corrected poses
  Step 6  save ablation outputs and corrected_poses.csv

Usage:
    python src/reconstruction/drift_correction.py \
        --point_cloud outputs/point_cloud.ply \
        --data_dir data/ \
        --output_dir outputs/
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.lidar.load_odometry import load_global_intrinsics, load_odometry
from src.reconstruction.segment import make_segments, segment_cloud_by_camera_bbox
from src.reconstruction.plane_detection import detect_dominant_planes
from src.reconstruction.pose_graph import build_pose_graph
from src.reconstruction.optimize import optimize_pose_graph, delta_to_transform
from src.reconstruction.reproject import reproject_corrected, downsample_and_clean
from src.reconstruction.ablation import (
    save_ablation,
    save_corrected_poses,
    print_ablation_report,
)

load_dotenv()

# ── Config (overridable via .env) ─────────────────────────────────────────────
SEG_SIZE       = int(  os.getenv("DRIFT_SEG_SIZE",   150))
OVERLAP        = int(  os.getenv("DRIFT_OVERLAP",     30))
DIST_THRESH    = float(os.getenv("DRIFT_DIST_THRESH", 0.02))
DEPTH_SCALE    = float(os.getenv("DEPTH_SCALE",       0.001))
MIN_CONFIDENCE = int(  os.getenv("MIN_CONFIDENCE",    1))
VOXEL_SIZE     = float(os.getenv("VOXEL_SIZE",        0.02))
SOR_NB         = int(  os.getenv("SOR_NB_NEIGHBORS",  20))
SOR_STD        = float(os.getenv("SOR_STD_RATIO",     2.0))
BATCH_SIZE     = int(  os.getenv("BATCH_SIZE",        50))


def main(point_cloud_path: str, data_dir: str, output_dir: str) -> None:
    data    = Path(data_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    depth_dir = data / "depth"
    conf_dir  = data / "confidence"

    print("── Module 2: Drift Correction ───────────────────────────────────")

    # ── Load inputs ───────────────────────────────────────────────────────────
    print("Loading point cloud and odometry …")
    raw_pcd  = o3d.io.read_point_cloud(point_cloud_path)
    odom_df  = load_odometry(str(data / "odometry.csv"))
    fallback = load_global_intrinsics(str(data / "camera_matrix.csv"))
    n_frames = len(odom_df)
    print(f"  {len(raw_pcd.points):,} points  |  {n_frames} frames")

    # ── Step 1: Segment ───────────────────────────────────────────────────────
    print(f"\nStep 1 — Segmenting into overlapping chunks (size={SEG_SIZE}, overlap={OVERLAP}) …")
    segments = make_segments(n_frames, SEG_SIZE, OVERLAP)
    print(f"  {len(segments)} segments")

    # ── Step 2: Plane detection per segment ───────────────────────────────────
    print("\nStep 2 — RANSAC plane detection per segment …")
    plane_results = []
    for i, (start, end) in enumerate(segments):
        sub = segment_cloud_by_camera_bbox(raw_pcd, odom_df, start, end, padding=0.5)
        planes = detect_dominant_planes(sub, dist_thresh=DIST_THRESH)
        plane_results.append(planes)
        detected = [k for k, v in planes.items() if v is not None]
        print(f"  Seg {i:02d} [{start:4d}-{end:4d}]: detected {detected}", flush=True)

    # ── Step 3: Pose graph ────────────────────────────────────────────────────
    print("\nStep 3 — Building pose graph …")
    nodes, edges = build_pose_graph(segments, odom_df, plane_results, overlap=OVERLAP)
    n_seq   = sum(1 for e in edges if e.kind == "sequential")
    n_plane = sum(1 for e in edges if e.kind == "plane_match")
    print(f"  {len(nodes)} nodes  |  {n_seq} sequential edges  |  {n_plane} plane-match edges")

    # ── Step 4: Optimize ──────────────────────────────────────────────────────
    print("\nStep 4 — Optimizing pose graph …")
    deltas = optimize_pose_graph(nodes, edges)
    correction_transforms = [delta_to_transform(d) for d in deltas]
    max_t = max(np.linalg.norm(d[:3]) for d in deltas)
    print(f"  Max translation correction: {max_t*100:.2f} cm")

    # ── Step 5: Reproject with corrected poses ────────────────────────────────
    print("\nStep 5 — Reprojecting all frames with corrected poses …")
    xyz_corrected = reproject_corrected(
        odom_df, segments, correction_transforms,
        depth_dir, conf_dir, fallback,
        depth_scale=DEPTH_SCALE,
        min_confidence=MIN_CONFIDENCE,
        batch_size=BATCH_SIZE,
    )
    print(f"  Raw corrected points: {len(xyz_corrected):,}")

    corrected_pcd = downsample_and_clean(xyz_corrected, VOXEL_SIZE, SOR_NB, SOR_STD)
    print(f"  After downsample + clean: {len(corrected_pcd.points):,} points")

    # ── Step 6: Save outputs ──────────────────────────────────────────────────
    print("\nStep 6 — Saving outputs …")

    # Main corrected cloud
    main_out = out_dir / "point_cloud_corrected.ply"
    o3d.io.write_point_cloud(str(main_out), corrected_pcd)
    print(f"  Corrected cloud → {main_out}")

    # Ablation: raw vs corrected side-by-side
    save_ablation(raw_pcd, corrected_pcd, out_dir / "drift_ablation")

    # Per-frame corrected poses CSV
    save_corrected_poses(odom_df, segments, correction_transforms,
                         out_dir / "corrected_poses.csv")

    # Ablation report (floor plane alignment before vs after)
    floor_before = [plane_results[i].get("floor") for i in range(len(segments))]
    # Re-detect planes on corrected cloud for "after" metric
    corrected_plane_results = []
    for i, (start, end) in enumerate(segments):
        sub = segment_cloud_by_camera_bbox(corrected_pcd, odom_df, start, end, padding=0.5)
        corrected_plane_results.append(detect_dominant_planes(sub, dist_thresh=DIST_THRESH))
    floor_after = [corrected_plane_results[i].get("floor") for i in range(len(segments))]

    print_ablation_report(floor_before, floor_after, odom_df, n_seq, n_plane)
    print("────────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module 2 — Drift Correction")
    parser.add_argument("--point_cloud", default=os.getenv("OUTPUT_DIR", "outputs/") + "point_cloud.ply")
    parser.add_argument("--data_dir",    default=os.getenv("DATA_DIR",   "data/"))
    parser.add_argument("--output_dir",  default=os.getenv("OUTPUT_DIR", "outputs/"))
    args = parser.parse_args()
    main(args.point_cloud, args.data_dir, args.output_dir)
