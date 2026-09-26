"""
pipeline.py — Top-level entry point

Runs Modules 1–8 in sequence for a single LiDAR or video capture.

Usage:
    python pipeline.py --input data/ --tier lidar --output outputs/
    python pipeline.py --input input/ --tier video --output outputs/
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.lidar.fuse import main as module1
from src.video.fuse import main as module9
from src.reconstruction.drift_correction import main as module2
from src.reconstruction.segment_pipeline import main as module3
from src.reconstruction.wall_fitting import main as module4
from src.reconstruction.opening_detection import main as module5
from src.reconstruction.stitching import main as module6
from src.damage.damage_pipeline import main as module7
from src.output.output_pipeline import main as module8


def run(input_dir: str, tier: str, output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("══════════════════════════════════════════════════════════════════")
    print("  Floorplan Pipeline")
    print(f"  Input : {input_dir}  |  Tier: {tier}  |  Output: {output_dir}")
    print("══════════════════════════════════════════════════════════════════\n")

    if tier == "video":
        module9(input_dir=input_dir, output_dir=output_dir)
    else:
        module1(data_dir=input_dir, output_dir=output_dir)
    print()
    module2(point_cloud_path=str(out / "point_cloud.ply"),
            data_dir=input_dir, output_dir=output_dir)
    print()
    module3(point_cloud_path=str(out / "point_cloud_corrected.ply"),
            output_dir=output_dir)
    print()
    module4(occupancy_grid_path=str(out / "occupancy_grid.npy"),
            room_segments_path=str(out / "room_segments.npz"),
            output_dir=output_dir,
            floor_ceiling_path=str(out / "floor_ceiling.json"))
    print()
    module5(wall_polygons_path=str(out / "wall_polygons.json"),
            occupancy_grid_path=str(out / "occupancy_grid.npy"),
            point_cloud_path=str(out / "point_cloud_corrected.ply"),
            output_dir=output_dir)
    print()
    module6(wall_polygons_path=str(out / "wall_polygons.json"),
            openings_path=str(out / "openings.json"),
            output_dir=output_dir)
    print()
    module7(data_dir=input_dir,
            wall_polygons_path=str(out / "wall_polygons.json"),
            stitched_plan_path=str(out / "stitched_plan.json"),
            output_dir=output_dir)
    print()
    module8(outputs_dir=output_dir, tier=tier)

    print("\n══════════════════════════════════════════════════════════════════")
    print(f"  Done.  floorplan.json + floorplan.png → {output_dir}")
    print("══════════════════════════════════════════════════════════════════")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Floorplan Pipeline — all modules")
    parser.add_argument("--input",  required=True, help="Raw data directory")
    parser.add_argument("--tier",   default="lidar", choices=["lidar", "video", "photo"])
    parser.add_argument("--output", default="outputs/")
    args = parser.parse_args()
    run(args.input, args.tier, args.output)
