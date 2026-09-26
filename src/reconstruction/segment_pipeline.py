"""
segment_pipeline.py — Module 3 entry point

Steps:
  1  Detect floor height via RANSAC on lowest 20 % of points
  2  Detect ceiling height via RANSAC on top 20 % of normalized points
  3  Extract wall slice (1.0–1.5 m above floor)
  4  Build 2D occupancy grid from wall-slice XY projection
  5  DBSCAN room segmentation on occupied grid cells
  6  Per-room ceiling height refinement
  7  Save all outputs

Usage:
    python src/reconstruction/segment_pipeline.py \
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

from src.reconstruction.floor_ceiling import (
    detect_floor,
    detect_ceiling,
    detect_ceiling_per_room,
)
from src.reconstruction.wall_slice import extract_wall_slice, save_wall_slice
from src.reconstruction.occupancy_grid import (
    build_occupancy_grid,
    save_occupancy_grid,
    visualise_occupancy_grid,
)
from src.reconstruction.room_segmentation import segment_rooms, label_slice_points

load_dotenv()

# ── Config (overridable via .env) ─────────────────────────────────────────────
CELL_SIZE         = float(os.getenv("GRID_CELL_SIZE",      0.02))
WALL_Z_MIN        = float(os.getenv("WALL_SLICE_Z_MIN",    1.0))
WALL_Z_MAX        = float(os.getenv("WALL_SLICE_Z_MAX",    1.5))
DBSCAN_EPS        = float(os.getenv("DBSCAN_EPS",          0.05))
DBSCAN_MIN        = int(  os.getenv("DBSCAN_MIN_SAMPLES",  10))
MIN_CLUSTER_CELLS = int(  os.getenv("MIN_CLUSTER_CELLS",   50))
RANSAC_DIST       = float(os.getenv("DRIFT_DIST_THRESH",   0.02))


def main(point_cloud_path: str, output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("── Module 3: Room Segmentation ──────────────────────────────────")

    # ── Load cloud ────────────────────────────────────────────────────────────
    print("Loading point cloud …")
    pcd = o3d.io.read_point_cloud(point_cloud_path)
    pts = np.asarray(pcd.points, dtype=np.float64)
    print(f"  {len(pts):,} points")

    # ── Step 1: Floor detection ───────────────────────────────────────────────
    print("\nStep 1 — Floor detection …")
    z_floor = detect_floor(pts, dist_thresh=RANSAC_DIST)
    print(f"  Floor z = {z_floor:.4f} m")

    # Normalize all points so floor = 0
    pts_norm = pts.copy()
    pts_norm[:, 2] -= z_floor

    # ── Step 2: Ceiling detection ─────────────────────────────────────────────
    print("Step 2 — Ceiling detection …")
    z_ceiling = detect_ceiling(pts_norm, dist_thresh=RANSAC_DIST)
    print(f"  Ceiling height = {z_ceiling:.4f} m")
    if not (2.0 <= z_ceiling <= 3.5):
        print(f"  WARNING: ceiling height {z_ceiling:.2f} m outside typical 2.0–3.5 m range")

    # ── Step 3: Wall slice ────────────────────────────────────────────────────
    print("Step 3 — Extracting wall slice …")
    slice_pts, slice_idx = extract_wall_slice(pts_norm, z_min=WALL_Z_MIN, z_max=WALL_Z_MAX)
    print(f"  Wall slice: {len(slice_pts):,} points "
          f"(z ∈ [{WALL_Z_MIN}, {WALL_Z_MAX}] m)")
    save_wall_slice(slice_pts, str(out / "wall_slice.ply"))
    print(f"  Saved → {out / 'wall_slice.ply'}")

    # ── Step 4: Occupancy grid ────────────────────────────────────────────────
    print("Step 4 — Building 2D occupancy grid …")
    slice_xy = slice_pts[:, :2]
    grid, origin, cell_size = build_occupancy_grid(slice_xy, cell_size=CELL_SIZE)
    n_occupied = int(grid.sum())
    print(f"  Grid size: {grid.shape[0]} × {grid.shape[1]} cells "
          f"({grid.shape[0]*CELL_SIZE:.1f} m × {grid.shape[1]*CELL_SIZE:.1f} m)")
    print(f"  Occupied cells: {n_occupied:,}")
    save_occupancy_grid(grid, origin, cell_size, str(out / "occupancy_grid.npy"))
    visualise_occupancy_grid(grid, origin, cell_size,
                             save_path=str(out / "occupancy_grid.png"))
    print(f"  Saved → {out / 'occupancy_grid.npy'}  +  occupancy_grid.png")

    # ── Step 5: Room segmentation ─────────────────────────────────────────────
    print("Step 5 — DBSCAN room segmentation …")
    occupied_ij, cell_labels, room_bounds = segment_rooms(
        grid, origin, cell_size,
        eps=DBSCAN_EPS,
        min_samples=DBSCAN_MIN,
        min_cluster_cells=MIN_CLUSTER_CELLS,
    )
    n_rooms = int(cell_labels.max()) + 1 if len(cell_labels) and cell_labels.max() >= 0 else 0
    n_noise = int((cell_labels == -1).sum())
    print(f"  Rooms detected: {n_rooms}  (noise cells: {n_noise})")

    # Map labels back to wall-slice points
    point_labels = label_slice_points(slice_xy, occupied_ij, cell_labels, origin, cell_size)

    # ── Step 6: Per-room ceiling refinement ───────────────────────────────────
    print("Step 6 — Per-room ceiling height refinement …")
    per_room_ceiling = detect_ceiling_per_room(
        pts_norm, point_labels, room_bounds,
        dist_thresh=RANSAC_DIST,
    )
    for rid, h in per_room_ceiling.items():
        print(f"  Room {rid}: ceiling = {h:.3f} m" if h else f"  Room {rid}: ceiling = (not detected)")

    # ── Step 7: Save outputs ──────────────────────────────────────────────────
    print("Step 7 — Saving outputs …")

    # floor_ceiling.json
    floor_ceiling_data = {
        "z_floor_world": round(z_floor, 6),
        "z_ceiling_global": round(z_ceiling, 4),
        "n_rooms": n_rooms,
        "per_room_ceiling": {
            str(rid): round(h, 4) if h is not None else None
            for rid, h in per_room_ceiling.items()
        },
    }
    with open(out / "floor_ceiling.json", "w") as f:
        json.dump(floor_ceiling_data, f, indent=2)
    print(f"  Saved → {out / 'floor_ceiling.json'}")

    # room_segments.npz
    np.savez(
        str(out / "room_segments.npz"),
        slice_indices=slice_idx,       # indices into the original pts array
        point_labels=point_labels,     # room id per wall-slice point
        cell_labels=cell_labels,       # room id per occupied grid cell (aligns with np.where(grid==1))
        room_ids=np.array(sorted(room_bounds.keys())),
    )
    print(f"  Saved → {out / 'room_segments.npz'}")

    print("\n── Summary ──────────────────────────────────────────────────────")
    print(f"  Floor height  : {z_floor:.3f} m (world z)")
    print(f"  Ceiling height: {z_ceiling:.3f} m (above floor)")
    print(f"  Rooms found   : {n_rooms}")
    print("────────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module 3 — Room Segmentation")
    parser.add_argument("--point_cloud", default=os.getenv("OUTPUT_DIR", "outputs/") + "point_cloud_corrected.ply")
    parser.add_argument("--output_dir",  default=os.getenv("OUTPUT_DIR", "outputs/"))
    args = parser.parse_args()
    main(args.point_cloud, args.output_dir)
