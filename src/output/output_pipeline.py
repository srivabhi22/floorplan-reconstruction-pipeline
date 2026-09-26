"""
output_pipeline.py — Module 8 entry point

Steps:
  1  Load all intermediate outputs + corrected point cloud
  2  Compute confidence intervals per measurement
  3  Assemble unified floorplan.json
  4  Schema validation + gate checklist
  5  Render final annotated floor plan (floorplan.png)
  6  Save confidence_report.json

Usage:
    python src/output/output_pipeline.py \
      --outputs_dir outputs/ \
      --tier lidar
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import open3d as o3d
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.output.assembler import assemble
from src.output.validator import validate_schema, gate_checklist, print_checklist
from src.output.renderer import render_final_plan

load_dotenv()


def _load_json(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    print(f"  WARNING: {path} not found — using empty placeholder")
    return {}


def main(outputs_dir: str, tier: str) -> None:
    out = Path(outputs_dir)

    print("── Module 8: Output Assembly ────────────────────────────────────")

    # ── Step 1: Load all intermediate outputs ─────────────────────────────────
    print("\nStep 1 — Loading intermediate outputs …")
    wall_data        = _load_json(out / "wall_polygons.json")
    openings_data    = _load_json(out / "openings.json")
    stitched_data    = _load_json(out / "stitched_plan.json")
    floor_ceiling    = _load_json(out / "floor_ceiling.json")
    damage_data      = _load_json(out / "damage.json")

    pcd_path = out / "point_cloud_corrected.ply"
    if pcd_path.exists():
        pcd     = o3d.io.read_point_cloud(str(pcd_path))
        pcd_pts = np.asarray(pcd.points, dtype=np.float64)
    else:
        print(f"  WARNING: {pcd_path} not found — CI estimates will use defaults")
        pcd_pts = np.zeros((1, 3))

    print(f"  {len(wall_data.get('rooms', {}))} rooms  |  "
          f"{sum(len(v) for v in openings_data.get('openings', {}).values())} openings  |  "
          f"{len(damage_data.get('damage', []))} damage entries  |  "
          f"{len(pcd_pts):,} cloud points")

    # ── Steps 2–3: Assemble ───────────────────────────────────────────────────
    print("\nSteps 2-3 — Computing CIs and assembling floorplan.json …")
    doc = assemble(
        wall_data      = wall_data,
        openings_data  = openings_data,
        stitched_data  = stitched_data,
        floor_ceiling_data = floor_ceiling,
        damage_data    = damage_data,
        pcd_pts        = pcd_pts,
        tier           = tier,
        capture_timestamp = datetime.now(timezone.utc).isoformat(),
    )

    # ── Step 4: Schema validation ─────────────────────────────────────────────
    print("\nStep 4 — Schema validation …")
    try:
        validate_schema(doc)
        print("  Schema validation: PASSED")
    except Exception as e:
        print(f"  Schema validation: FAILED — {e}")

    checklist = gate_checklist(doc)
    print_checklist(checklist)

    # ── Step 5: Render ────────────────────────────────────────────────────────
    print("\nStep 5 — Rendering final floor plan …")
    render_final_plan(doc, save_path=str(out / "floorplan.png"))

    # ── Step 6: Save outputs ──────────────────────────────────────────────────
    print("\nStep 6 — Saving outputs …")

    fp_path = out / "floorplan.json"
    with open(fp_path, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"  Saved → {fp_path}")

    conf_report = {
        "tier":      tier,
        "timestamp": doc["capture_timestamp"],
        "checklist": checklist,
        "summary": {
            "total_checks": len(checklist),
            "passes": sum(1 for c in checklist if c["status"] == "pass"),
            "warns":  sum(1 for c in checklist if c["status"] == "warn"),
            "fails":  sum(1 for c in checklist if c["status"] == "fail"),
        },
    }
    cr_path = out / "confidence_report.json"
    with open(cr_path, "w") as f:
        json.dump(conf_report, f, indent=2)
    print(f"  Saved → {cr_path}")

    print("\n── Summary ──────────────────────────────────────────────────────")
    print(f"  Tier           : {tier}")
    print(f"  Rooms          : {len(doc['rooms'])}")
    print(f"  Footprint      : {doc['stitched_plan'].get('footprint_m2', 0):.2f} m²")
    print(f"  Scope items    : {len(doc['scope_line_items'])}")
    print(f"  Concealed flags: {len(doc['concealed_damage_flags'])}")
    print("────────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module 8 — Output Assembly")
    parser.add_argument("--outputs_dir", default=os.getenv("OUTPUT_DIR", "outputs/"))
    parser.add_argument("--tier",        default="lidar",
                        choices=["lidar", "video", "photo"])
    args = parser.parse_args()
    main(args.outputs_dir, args.tier)
