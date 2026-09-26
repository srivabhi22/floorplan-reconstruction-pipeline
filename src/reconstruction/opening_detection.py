"""
opening_detection.py — Module 5 entry point

Steps:
  1  Find gaps along each wall polygon edge via occupancy grid sampling
  2  Refine gap width against wall-slice 3D points
  3  Classify each gap (door / window / archway) from vertical point profile
  4  Flag concealed (glass) openings with low LiDAR density
  5  Save openings.json

Usage:
    python src/reconstruction/opening_detection.py \
      --wall_polygons outputs/wall_polygons.json \
      --occupancy_grid outputs/occupancy_grid.npy \
      --point_cloud outputs/point_cloud_corrected.ply \
      --output_dir outputs/
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.reconstruction.gap_detection import find_gaps, refine_gap_width
from src.reconstruction.opening_classifier import classify_opening, confidence_interval
from src.reconstruction.floor_ceiling import detect_floor

load_dotenv()

MIN_GAP_M      = float(os.getenv("OPENING_MIN_GAP_M",  0.5))
MAX_GAP_M      = float(os.getenv("OPENING_MAX_GAP_M",  4.0))
SEARCH_RADIUS  = float(os.getenv("OPENING_SEARCH_R",   0.25))
WALL_Z_MIN     = float(os.getenv("WALL_SLICE_Z_MIN",   1.0))
WALL_Z_MAX     = float(os.getenv("WALL_SLICE_Z_MAX",   1.5))


def main(wall_polygons_path: str, occupancy_grid_path: str,
         point_cloud_path: str, output_dir: str) -> None:

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("── Module 5: Opening Detection ──────────────────────────────────")

    # ── Load inputs ───────────────────────────────────────────────────────────
    with open(wall_polygons_path) as f:
        wall_data = json.load(f)

    grid_data = np.load(occupancy_grid_path, allow_pickle=True).item()
    grid      = grid_data["grid"]
    origin    = grid_data["origin"]
    cell_size = float(grid_data["cell_size"])

    print("Loading point cloud …")
    pcd = o3d.io.read_point_cloud(point_cloud_path)
    pts = np.asarray(pcd.points, dtype=np.float64)

    # Floor-normalize the full cloud (needed for vertical classification)
    z_floor  = detect_floor(pts)
    pts_norm = pts.copy()
    pts_norm[:, 2] -= z_floor
    print(f"  Floor z = {z_floor:.4f} m  |  {len(pts):,} points")

    # Wall-slice sub-cloud for width refinement
    slice_mask = (pts_norm[:, 2] >= WALL_Z_MIN) & (pts_norm[:, 2] <= WALL_Z_MAX)
    slice_pcd  = o3d.geometry.PointCloud()
    slice_pcd.points = o3d.utility.Vector3dVector(pts_norm[slice_mask])

    # ── Process each room ─────────────────────────────────────────────────────
    openings_out = {}
    total_openings = 0

    for room_id_str, room in wall_data["rooms"].items():
        polygon  = room["vertices"]
        room_openings = []

        print(f"\n  Room {room_id_str}: {len(polygon)-1} wall edges")

        # Step 1 — find gaps
        gaps = find_gaps(polygon, grid, origin, cell_size,
                         min_gap_m=MIN_GAP_M, max_gap_m=MAX_GAP_M)
        print(f"    {len(gaps)} gap(s) found (pre-filter)")

        for gap in gaps:
            # Step 2 — refine width
            refined_width = refine_gap_width(gap, slice_pcd)

            # Step 3+4 — classify
            clf = classify_opening(gap, pts_norm, search_radius=SEARCH_RADIUS)

            ci = confidence_interval(refined_width, clf["type"])

            opening = {
                "room_id":           room_id_str,
                "type":              clf["type"],
                "width_m":           round(refined_width, 4),
                "sill_height_m":     clf["sill_height_m"],
                "header_height_m":   clf["header_height_m"],
                "position":          gap["mid_pt"],
                "wall_segment":      gap["wall_idx"],
                "confidence_interval_m": ci,
                "concealed":         clf["concealed"],
            }
            room_openings.append(opening)
            print(f"    {clf['type']:8s}  width={refined_width:.3f} m  "
                  f"sill={clf['sill_height_m']}  header={clf['header_height_m']}  "
                  f"concealed={clf['concealed']}")

        openings_out[room_id_str] = room_openings
        total_openings += len(room_openings)

    # ── Step 5 — save ─────────────────────────────────────────────────────────
    out_path = out / "openings.json"
    with open(out_path, "w") as f:
        json.dump({"openings": openings_out}, f, indent=2)

    print(f"\n── Summary ──────────────────────────────────────────────────────")
    print(f"  Total openings detected : {total_openings}")
    for room_id_str, ops in openings_out.items():
        doors   = sum(1 for o in ops if o["type"] == "door")
        windows = sum(1 for o in ops if o["type"] == "window")
        print(f"  Room {room_id_str}: {doors} door(s), {windows} window(s)")
    print(f"  Saved → {out_path}")
    print("────────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module 5 — Opening Detection")
    base = os.getenv("OUTPUT_DIR", "outputs/")
    parser.add_argument("--wall_polygons",  default=base + "wall_polygons.json")
    parser.add_argument("--occupancy_grid", default=base + "occupancy_grid.npy")
    parser.add_argument("--point_cloud",    default=base + "point_cloud_corrected.ply")
    parser.add_argument("--output_dir",     default=base)
    args = parser.parse_args()
    main(args.wall_polygons, args.occupancy_grid, args.point_cloud, args.output_dir)
