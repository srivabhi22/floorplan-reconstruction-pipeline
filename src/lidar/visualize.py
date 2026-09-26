"""
visualize.py — Point cloud interpretability tool

Colors the fused point cloud to make structure visible:
  1. Height colormap  : blue (floor) → green (mid/objects) → red (ceiling)
  2. RANSAC floor     : detected floor plane shown in a distinct flat colour
  3. Prints height stats so you know where floor/ceiling actually are

Usage:
    python src/lidar/visualize.py --input outputs/point_cloud.ply
    python src/lidar/visualize.py --input outputs/point_cloud.ply --mode ransac
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d


# ── Helpers ───────────────────────────────────────────────────────────────────

def height_colormap(heights: np.ndarray) -> np.ndarray:
    """
    Map a 1-D array of heights to RGB colours using a blue→green→red gradient.
    Low = blue (floor), mid = green (objects/furniture), high = red (ceiling).

    Returns (N, 3) float array in [0, 1].
    """
    lo, hi = heights.min(), heights.max()
    t = (heights - lo) / (hi - lo + 1e-9)   # normalise to [0, 1]

    r = np.clip(2 * t - 1, 0, 1)             # red ramps up in top half
    g = np.clip(1 - 2 * np.abs(t - 0.5), 0, 1)  # green peaks in the middle
    b = np.clip(1 - 2 * t, 0, 1)             # blue ramps down from bottom

    return np.stack([r, g, b], axis=1)


def print_stats(pts: np.ndarray) -> None:
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span   = hi - lo
    print("\n── Point cloud stats ───────────────────────────────────────────")
    print(f"  Total points  : {len(pts):,}")
    print(f"  X range       : {lo[0]:.2f} m  →  {hi[0]:.2f} m  (span {span[0]:.2f} m)")
    print(f"  Y range       : {lo[1]:.2f} m  →  {hi[1]:.2f} m  (span {span[1]:.2f} m)")
    print(f"  Z range       : {lo[2]:.2f} m  →  {hi[2]:.2f} m  (span {span[2]:.2f} m)")
    # guess which axis is vertical (largest span is usually height)
    vertical_axis = ["X", "Y", "Z"][np.argmax(span)]
    print(f"  Likely vertical axis: {vertical_axis}  (largest span)")
    print("────────────────────────────────────────────────────────────────\n")


# ── Modes ─────────────────────────────────────────────────────────────────────

def visualize_height(pcd: o3d.geometry.PointCloud) -> None:
    """Colour every point by its height using a smooth gradient."""
    pts    = np.asarray(pcd.points)
    # Use the axis with the largest range as the vertical axis
    spans  = pts.max(axis=0) - pts.min(axis=0)
    v_axis = int(np.argmax(spans))
    axis_name = ["X", "Y", "Z"][v_axis]

    heights = pts[:, v_axis]
    colors  = height_colormap(heights)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    print(f"  Colouring by {axis_name} axis  (blue=low / green=mid / red=high)")
    print("  Blue  → floor level")
    print("  Green → furniture / objects")
    print("  Red   → ceiling level\n")

    o3d.visualization.draw_geometries(
        [pcd],
        window_name="Height colourmap  (blue=floor · green=objects · red=ceiling)",
        width=1280, height=720,
    )


def visualize_ransac_floor(pcd: o3d.geometry.PointCloud) -> None:
    """
    Detect the dominant horizontal plane (floor) with RANSAC, colour it yellow,
    and colour the remaining points by height.
    """
    pts   = np.asarray(pcd.points)
    spans = pts.max(axis=0) - pts.min(axis=0)
    v_axis = int(np.argmax(spans))

    print("  Running RANSAC plane segmentation …")
    # distance_threshold in metres — 3 cm tolerance for floor flatness
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=0.03,
        ransac_n=3,
        num_iterations=1000,
    )
    a, b, c, d = plane_model
    print(f"  Detected plane equation: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")
    print(f"  Floor inliers: {len(inliers):,} / {len(pts):,} points")

    # Split into floor and non-floor clouds
    floor_cloud = pcd.select_by_index(inliers)
    rest_cloud  = pcd.select_by_index(inliers, invert=True)

    # Floor → bright yellow
    floor_cloud.paint_uniform_color([1.0, 0.85, 0.0])

    # Rest → height gradient
    rest_pts    = np.asarray(rest_cloud.points)
    rest_colors = height_colormap(rest_pts[:, v_axis])
    rest_cloud.colors = o3d.utility.Vector3dVector(rest_colors)

    print("\n  Yellow → detected floor plane")
    print("  Blue   → low non-floor points")
    print("  Green  → mid-height (furniture / objects)")
    print("  Red    → ceiling / high points\n")

    o3d.visualization.draw_geometries(
        [floor_cloud, rest_cloud],
        window_name="RANSAC floor (yellow) + height colourmap",
        width=1280, height=720,
    )


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise a point cloud with interpretable colours")
    parser.add_argument("--input", default="outputs/point_cloud.ply",
                        help="Path to the .ply file")
    parser.add_argument(
        "--mode",
        choices=["height", "ransac"],
        default="height",
        help=(
            "height : smooth blue→green→red gradient by elevation\n"
            "ransac : RANSAC floor detection (yellow) + height gradient for the rest"
        ),
    )
    args = parser.parse_args()

    ply_path = Path(args.input)
    if not ply_path.exists():
        print(f"ERROR: file not found — {ply_path}")
        sys.exit(1)

    print(f"Loading {ply_path} …")
    pcd = o3d.io.read_point_cloud(str(ply_path))

    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        print("ERROR: point cloud is empty.")
        sys.exit(1)

    print_stats(pts)

    if args.mode == "height":
        visualize_height(pcd)
    else:
        visualize_ransac_floor(pcd)


if __name__ == "__main__":
    main()
