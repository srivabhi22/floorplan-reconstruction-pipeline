"""
wall_fitting.py — Module 4 entry point: wall polygon fitting per room.

Steps:
  1  Extract per-room boundary contour from occupancy grid
  2  PCA dominant wall angle
  3  Snap segments to dominant axes + merge collinear runs
  4  Build rectilinear polygon
  5  Compute wall lengths and floor area
  6  Validate polygon
  7  Render floor plan and save outputs

Usage:
    python src/reconstruction/wall_fitting.py \
      --occupancy_grid outputs/occupancy_grid.npy \
      --room_segments outputs/room_segments.npz \
      --output_dir outputs/ \
      --floor_ceiling outputs/floor_ceiling.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.reconstruction.boundary_extraction import extract_all_boundaries
from src.reconstruction.orientation_detection import detect_dominant_angle
from src.reconstruction.segment_snapping import snap_segments, merge_collinear
from src.reconstruction.polygon_builder import build_polygon, is_closed
from src.reconstruction.measurements import wall_lengths, floor_area, validate_polygon
from src.reconstruction.render import render_floor_plan

load_dotenv()

SNAP_THRESH_DEG = float(os.getenv("SNAP_THRESH_DEG", 10.0))
COLLINEAR_TOL   = float(os.getenv("COLLINEAR_TOL",   0.02))


def main(
    occupancy_grid_path: str,
    room_segments_path: str,
    output_dir: str,
    floor_ceiling_path: str,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("── Module 4: Wall Polygon Fitting ───────────────────────────────")

    # ── Load inputs ───────────────────────────────────────────────────────────
    print("Loading occupancy grid and room segments …")
    grid_data     = np.load(occupancy_grid_path, allow_pickle=True).item()
    grid          = grid_data["grid"]
    origin        = grid_data["origin"]
    cell_size     = float(grid_data["cell_size"])

    seg_data    = np.load(room_segments_path, allow_pickle=True)
    cell_labels = seg_data["cell_labels"]   # room id per occupied grid cell

    # Recover (row, col) pairs in the same order segment_rooms produced them
    grid_rows, grid_cols = np.where(grid == 1)
    occ_grid_ij = np.stack([grid_rows, grid_cols], axis=1)   # (M, 2), aligns with cell_labels

    floor_ceiling = {}
    if floor_ceiling_path and Path(floor_ceiling_path).exists():
        with open(floor_ceiling_path) as f:
            floor_ceiling = json.load(f)

    per_room_ceiling = floor_ceiling.get("per_room_ceiling", {})

    # ── Step 1: Boundary extraction ───────────────────────────────────────────
    print("\nStep 1 — Extracting per-room boundaries …")
    boundaries = extract_all_boundaries(grid, occ_grid_ij, cell_labels, origin, cell_size)
    print(f"  {len(boundaries)} room(s) found")

    room_polygons  = {}
    room_metadata  = {}
    rooms_json     = {}

    for room_id, boundary in boundaries.items():
        print(f"\n  Room {room_id}: {len(boundary)} boundary points")

        if len(boundary) < 4:
            print(f"    SKIP — fewer than 4 boundary points (corridor/connector)")
            continue

        # ── Step 2: Dominant orientation ─────────────────────────────────────
        theta = detect_dominant_angle(boundary)
        print(f"    Step 2 — dominant angle: {np.degrees(theta):.1f}°")

        # ── Step 3 & 4: Snap + merge ──────────────────────────────────────────
        segments = snap_segments(boundary, theta, snap_thresh_deg=SNAP_THRESH_DEG)
        merged   = merge_collinear(segments, tol=COLLINEAR_TOL)
        print(f"    Step 3+4 — {len(segments)} segments → {len(merged)} after merge")

        if len(merged) < 4:
            print(f"    SKIP — fewer than 4 segments after merge")
            continue

        # ── Step 5: Build polygon ─────────────────────────────────────────────
        polygon = build_polygon(merged, theta)
        print(f"    Step 5 — polygon: {len(polygon)} vertices, closed={is_closed(polygon)}")

        # ── Step 6: Measurements ──────────────────────────────────────────────
        lengths = wall_lengths(polygon)
        area    = floor_area(polygon)
        print(f"    Step 6 — area: {area:.2f} m², walls: {[round(l,2) for l in lengths]}")

        # ── Validate ──────────────────────────────────────────────────────────
        warnings = validate_polygon(polygon)
        if warnings:
            for w in warnings:
                print(f"    WARNING: {w}")

        ceil_h = per_room_ceiling.get(str(room_id))

        room_polygons[room_id] = polygon
        room_metadata[room_id] = {
            "area_m2": area,
            "wall_lengths": lengths,
            "ceiling_height": ceil_h,
        }
        rooms_json[str(room_id)] = {
            "vertices": polygon.tolist(),
            "wall_lengths_m": [round(l, 4) for l in lengths],
            "floor_area_m2": round(area, 4),
            "ceiling_height_m": round(ceil_h, 4) if ceil_h is not None else None,
            "warnings": warnings,
        }

    # ── Step 7: Save outputs ──────────────────────────────────────────────────
    print("\nStep 7 — Saving outputs …")

    wall_polygons_path = out / "wall_polygons.json"
    with open(wall_polygons_path, "w") as f:
        json.dump({"rooms": rooms_json}, f, indent=2)
    print(f"  Saved → {wall_polygons_path}")

    png_path = out / "wall_polygons.png"
    render_floor_plan(room_polygons, room_metadata, str(png_path))
    print(f"  Saved → {png_path}")

    print("\n── Summary ──────────────────────────────────────────────────────")
    print(f"  Rooms fitted: {len(rooms_json)}")
    for rid, data in rooms_json.items():
        print(f"  Room {rid}: {data['floor_area_m2']:.2f} m²  "
              f"ceiling={data['ceiling_height_m']} m  "
              f"warnings={len(data['warnings'])}")
    print("────────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module 4 — Wall Polygon Fitting")
    parser.add_argument("--occupancy_grid",
                        default=os.getenv("OUTPUT_DIR", "outputs/") + "occupancy_grid.npy")
    parser.add_argument("--room_segments",
                        default=os.getenv("OUTPUT_DIR", "outputs/") + "room_segments.npz")
    parser.add_argument("--output_dir",
                        default=os.getenv("OUTPUT_DIR", "outputs/"))
    parser.add_argument("--floor_ceiling",
                        default=os.getenv("OUTPUT_DIR", "outputs/") + "floor_ceiling.json")
    args = parser.parse_args()
    main(args.occupancy_grid, args.room_segments, args.output_dir, args.floor_ceiling)
