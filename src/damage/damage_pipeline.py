"""
damage_pipeline.py — Module 7 entry point

Steps:
  1  Select keyframes (pose-change or stride)
  2  Damage segmentation per frame (SAM → SegFormer → HSV fallback)
  3  Back-project pixel masks to world-space 3D points
  4  Assign each damage cluster to a named surface
  5  Compute metric area (convex hull on plane projection)
  6  Concealed damage detection (depth anomaly + low confidence)
  7  Deduplicate across frames
  8  Save damage.json

Usage:
    python src/damage/damage_pipeline.py \
      --data_dir data/ \
      --wall_polygons outputs/wall_polygons.json \
      --stitched_plan outputs/stitched_plan.json \
      --output_dir outputs/
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.lidar.load_odometry import load_odometry, load_global_intrinsics
from src.damage.frame_selector import select_by_pose_change, select_by_stride, load_rgb_frames
from src.damage.segmentation import detect_damage
from src.damage.backproject import mask_to_world_points
from src.damage.surface_assignment import build_surface_planes, assign_to_surface
from src.damage.area_computation import damage_area_m2
from src.damage.concealed_detection import check_concealed
from src.damage.deduplication import deduplicate_detections

load_dotenv()

FRAME_STRIDE      = int(  os.getenv("DAMAGE_FRAME_STRIDE",    10))
MIN_TRANSLATION_M = float(os.getenv("DAMAGE_MIN_TRANSLATION", 0.3))
DEDUP_TOL_M       = float(os.getenv("DAMAGE_DEDUP_TOL",       0.10))
DEPTH_SCALE       = float(os.getenv("DEPTH_SCALE",            0.001))
MIN_CONFIDENCE    = int(  os.getenv("MIN_CONFIDENCE",         1))
USE_POSE_SELECT   = os.getenv("DAMAGE_USE_POSE_SELECT", "1") == "1"
PROCESS_WIDTH     = int(  os.getenv("DAMAGE_PROCESS_WIDTH",   320))


def main(data_dir: str, wall_polygons_path: str, stitched_plan_path: str,
         output_dir: str) -> None:

    data = Path(data_dir)
    out  = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("── Module 7: Damage Detection ────────────────────────────────────")

    # ── Load supporting data ──────────────────────────────────────────────────
    with open(wall_polygons_path) as f:
        wall_data = json.load(f)
    with open(stitched_plan_path) as f:
        stitched  = json.load(f)

    odom_df  = load_odometry(str(data / "odometry.csv"))
    fallback = load_global_intrinsics(str(data / "camera_matrix.csv"))

    # Global floor z from stitched plan (carry over if present, else 0)
    z_floor   = float(stitched.get("z_floor_world", 0.0))
    z_ceiling = float(stitched.get("z_ceiling_global", 2.5))

    surfaces = build_surface_planes(wall_data, z_floor=z_floor, z_ceiling=z_ceiling)
    print(f"  {len(surfaces)} named surfaces built from wall polygons")

    # ── Step 1: Frame selection ───────────────────────────────────────────────
    print("\nStep 1 — Selecting keyframes …")
    n_frames = len(odom_df)
    if USE_POSE_SELECT:
        frame_indices = select_by_pose_change(odom_df, min_translation_m=MIN_TRANSLATION_M)
    else:
        frame_indices = select_by_stride(n_frames, stride=FRAME_STRIDE)
    print(f"  {len(frame_indices)} keyframes selected from {n_frames} total")

    rgb_frames = load_rgb_frames(str(data / "rgb.mp4"), frame_indices)
    print(f"  {len(rgb_frames)} RGB frames loaded")

    # ── Steps 2–6: Per-frame processing ──────────────────────────────────────
    print("\nSteps 2-6 — Processing frames …")
    all_detections = []
    n_processed = 0

    for frame_idx, bgr in rgb_frames.items():
        row      = odom_df.iloc[frame_idx]
        frame_id = row["frame"]
        T        = row["T"]
        fx       = row.get("fx", fallback["fx"])
        fy       = row.get("fy", fallback["fy"])
        cx       = row.get("cx", fallback["cx"])
        cy       = row.get("cy", fallback["cy"])

        depth_path = str(data / "depth" / f"{frame_id}.png")
        conf_path  = str(data / "confidence" / f"{frame_id}.png")

        if not Path(depth_path).exists():
            continue

        # Step 2 — segmentation
        regions = detect_damage(bgr, process_width=PROCESS_WIDTH)
        if not regions:
            n_processed += 1
            continue

        for region in regions:
            mask = region["mask"]

            # Step 3 — back-project to world
            pts_world = mask_to_world_points(
                mask, depth_path, T, fx, fy, cx, cy,
                depth_scale=DEPTH_SCALE,
                min_confidence=MIN_CONFIDENCE,
                confidence_path=conf_path if Path(conf_path).exists() else None,
            )
            if len(pts_world) < 3:
                continue

            # Step 4 — surface assignment
            surf = assign_to_surface(pts_world, surfaces)
            if surf is None:
                continue

            # Step 5 — metric area
            area = damage_area_m2(pts_world)

            # Step 6 — concealed check
            surface_d = float(np.dot(surf["normal"], surf["point"]))
            concealed, rule_fired = check_concealed(
                depth_path, conf_path if Path(conf_path).exists() else depth_path,
                mask, surface_d, fx, fy, cx, cy, DEPTH_SCALE,
            )

            centroid = pts_world.mean(axis=0).tolist()
            all_detections.append({
                "room_id":         surf["room_id"],
                "surface":         surf["name"],
                "class":           region["class"] if not concealed else "concealed",
                "area_m2":         area,
                "centroid_world":  [round(v, 4) for v in centroid],
                "confidence":      round(region["confidence"], 3),
                "concealed":       concealed,
                "rule_fired":      rule_fired,
            })

        n_processed += 1
        print(f"  {n_processed}/{len(rgb_frames)} frames …", end="\r", flush=True)

    print()
    print(f"  Raw detections before dedup: {len(all_detections)}")

    # ── Step 7: Deduplicate ───────────────────────────────────────────────────
    print("\nStep 7 — Deduplicating across frames …")
    final = deduplicate_detections(all_detections, position_tol=DEDUP_TOL_M)
    print(f"  Final detections: {len(final)}")

    # ── Step 8: Save ──────────────────────────────────────────────────────────
    out_path = out / "damage.json"
    with open(out_path, "w") as f:
        json.dump({"damage": final}, f, indent=2)
    print(f"  Saved → {out_path}")

    # Summary
    print("\n── Summary ──────────────────────────────────────────────────────")
    from collections import Counter
    counts = Counter(d["class"] for d in final)
    for cls, n in counts.most_common():
        print(f"  {cls:20s}: {n}")
    concealed_n = sum(1 for d in final if d["concealed"])
    print(f"  Concealed flags  : {concealed_n}")
    print("────────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module 7 — Damage Detection")
    base = os.getenv("OUTPUT_DIR", "outputs/")
    parser.add_argument("--data_dir",       default=os.getenv("DATA_DIR", "data/"))
    parser.add_argument("--wall_polygons",  default=base + "wall_polygons.json")
    parser.add_argument("--stitched_plan",  default=base + "stitched_plan.json")
    parser.add_argument("--output_dir",     default=base)
    args = parser.parse_args()
    main(args.data_dir, args.wall_polygons, args.stitched_plan, args.output_dir)
