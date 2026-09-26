"""
stitching.py — Module 6 entry point

Steps:
  1  All room polygons are already in world coordinates — verify no overlap
  2  Build adjacency graph from shared openings
  3  Detect and resolve overlaps (residual drift correction)
  4  Flag disconnected rooms
  5  Compute global footprint via Shapely unary_union
  6  Render stitched floor plan

Usage:
    python src/reconstruction/stitching.py \
      --wall_polygons outputs/wall_polygons.json \
      --openings outputs/openings.json \
      --output_dir outputs/
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.reconstruction.adjacency import build_adjacency_graph
from src.reconstruction.overlap_resolver import detect_overlaps, resolve_overlap
from src.reconstruction.footprint import compute_footprint
from src.reconstruction.stitch_render import render_stitched_plan

load_dotenv()

POSITION_TOL      = float(os.getenv("STITCH_POSITION_TOL",   0.10))
OVERLAP_WARN_M    = float(os.getenv("STITCH_OVERLAP_WARN_M", 0.05))


def main(wall_polygons_path: str, openings_path: str, output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("── Module 6: Multi-Room Stitching ───────────────────────────────")

    # ── Load inputs ───────────────────────────────────────────────────────────
    with open(wall_polygons_path) as f:
        wall_data = json.load(f)
    with open(openings_path) as f:
        openings_data = json.load(f)

    room_polygons = {rid: room["vertices"] for rid, room in wall_data["rooms"].items()}
    room_metadata = {
        rid: {
            "floor_area_m2":   room.get("floor_area_m2"),
            "ceiling_height_m": room.get("ceiling_height_m"),
            "wall_lengths_m":  room.get("wall_lengths_m", []),
        }
        for rid, room in wall_data["rooms"].items()
    }
    openings = openings_data["openings"]

    print(f"  {len(room_polygons)} room(s)  |  "
          f"{sum(len(v) for v in openings.values())} opening(s)")

    # ── Step 1: Verify co-registration ───────────────────────────────────────
    print("\nStep 1 — Verifying co-registration (all rooms in world frame) …")
    print("  All polygons already in ARKit world coordinates — no transform needed")

    # ── Step 2: Adjacency graph ───────────────────────────────────────────────
    print("\nStep 2 — Building adjacency graph …")
    edges, disconnected = build_adjacency_graph(openings, position_tol=POSITION_TOL)
    print(f"  {len(edges)} adjacency edge(s)")
    if disconnected:
        print(f"  WARNING: {len(disconnected)} disconnected room(s): {disconnected}")
    else:
        print("  All rooms connected")

    # ── Step 3: Overlap detection & resolution ────────────────────────────────
    print("\nStep 3 — Checking for polygon overlaps …")
    overlaps = detect_overlaps(room_polygons)

    if not overlaps:
        print("  No overlaps detected")
    else:
        print(f"  {len(overlaps)} overlap(s) found — resolving …")
        for ov in overlaps:
            # Find shared opening between the two rooms as anchor
            anchor = None
            for room_a, room_b, info in edges:
                if set([room_a, room_b]) == set([ov["room_a"], ov["room_b"]]):
                    anchor = info["position"]
                    break

            corrected, shift_m = resolve_overlap(
                room_polygons[ov["room_b"]], ov, anchor=anchor
            )
            room_polygons[ov["room_b"]] = corrected

            flag = "  ⚠ LARGE" if shift_m > OVERLAP_WARN_M else ""
            print(f"  Room {ov['room_a']} ↔ Room {ov['room_b']}: "
                  f"overlap {ov['overlap_area_m2']:.4f} m²  shift={shift_m*100:.1f} cm{flag}")

    # ── Step 4: Disconnected room report ─────────────────────────────────────
    print("\nStep 4 — Disconnected room check …")
    if disconnected:
        for rid in disconnected:
            print(f"  Room {rid}: no shared opening found — flagged as disconnected")
    else:
        print("  All rooms have at least one connection")

    # ── Step 5: Global footprint ──────────────────────────────────────────────
    print("\nStep 5 — Computing global footprint …")
    footprint_area, footprint_coords = compute_footprint(room_polygons)
    print(f"  Total footprint: {footprint_area:.2f} m²")

    # ── Step 6: Render ────────────────────────────────────────────────────────
    print("\nStep 6 — Rendering stitched floor plan …")
    render_stitched_plan(
        room_polygons  = room_polygons,
        room_metadata  = room_metadata,
        openings       = openings,
        edges          = edges,
        footprint_area = footprint_area,
        save_path      = str(out / "stitched_plan.png"),
    )

    # ── Save JSON ─────────────────────────────────────────────────────────────
    stitched = {
        "rooms": {
            rid: {
                "vertices":        room_polygons[rid],
                "floor_area_m2":   room_metadata[rid]["floor_area_m2"],
                "ceiling_height_m": room_metadata[rid]["ceiling_height_m"],
                "wall_lengths_m":  room_metadata[rid]["wall_lengths_m"],
            }
            for rid in room_polygons
        },
        "adjacency": [
            {"room_a": a, "room_b": b, **info}
            for a, b, info in edges
        ],
        "disconnected_rooms": disconnected,
        "footprint_m2": footprint_area,
        "footprint_boundary": footprint_coords,
        "overlap_corrections": [
            {"room_a": ov["room_a"], "room_b": ov["room_b"],
             "overlap_area_m2": ov["overlap_area_m2"]}
            for ov in overlaps
        ],
    }

    out_path = out / "stitched_plan.json"
    with open(out_path, "w") as f:
        json.dump(stitched, f, indent=2)
    print(f"  Saved → {out_path}")

    print("\n── Summary ──────────────────────────────────────────────────────")
    print(f"  Rooms          : {len(room_polygons)}")
    print(f"  Adjacency edges: {len(edges)}")
    print(f"  Overlaps fixed : {len(overlaps)}")
    print(f"  Disconnected   : {len(disconnected)}")
    print(f"  Footprint      : {footprint_area:.2f} m²")
    print("────────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module 6 — Multi-Room Stitching")
    base = os.getenv("OUTPUT_DIR", "outputs/")
    parser.add_argument("--wall_polygons", default=base + "wall_polygons.json")
    parser.add_argument("--openings",      default=base + "openings.json")
    parser.add_argument("--output_dir",    default=base)
    args = parser.parse_args()
    main(args.wall_polygons, args.openings, args.output_dir)
