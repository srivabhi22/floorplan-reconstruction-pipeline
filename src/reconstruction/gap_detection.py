"""
gap_detection.py — Steps 1 & 2
Find gaps in wall boundary segments and measure their metric width.
"""

import numpy as np
import open3d as o3d


def _sample_wall_segment(p1: np.ndarray, p2: np.ndarray,
                          origin: np.ndarray, cell_size: float,
                          grid: np.ndarray, n_samples: int = 200
                          ) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample `n_samples` evenly-spaced positions along a wall segment and return
    their grid occupancy (1/0) together with the metric positions.
    """
    t = np.linspace(0, 1, n_samples)
    pts = p1[None, :] + t[:, None] * (p2 - p1)          # (n_samples, 2) world xy

    col = ((pts[:, 0] - origin[0]) / cell_size).astype(int)
    row = ((pts[:, 1] - origin[1]) / cell_size).astype(int)
    H, W = grid.shape
    valid = (row >= 0) & (row < H) & (col >= 0) & (col < W)

    occ = np.zeros(n_samples, dtype=np.uint8)
    occ[valid] = grid[row[valid], col[valid]]
    return pts, occ


def find_gaps(
    polygon: list,
    grid: np.ndarray,
    origin: np.ndarray,
    cell_size: float,
    min_gap_m: float = 0.5,
    max_gap_m: float = 4.0,
    n_samples: int = 500,
) -> list[dict]:
    """
    Step 1 — Detect runs of empty cells along each wall edge.

    Returns list of gap dicts:
        {wall_idx, start_pt, end_pt, width_m, mid_pt, wall_normal}
    """
    verts = np.array(polygon)
    n = len(verts) - 1   # last vertex closes the polygon (==first)
    gaps = []

    for i in range(n):
        p1, p2 = verts[i], verts[i + 1]
        wall_len = float(np.linalg.norm(p2 - p1))
        if wall_len < min_gap_m:
            continue

        pts, occ = _sample_wall_segment(p1, p2, origin, cell_size, grid, n_samples)
        seg_len = wall_len / n_samples   # metres per sample step

        # Find runs of zeros (empty = gap)
        in_gap = False
        gap_start_idx = 0
        for j, o in enumerate(occ):
            if not in_gap and o == 0:
                in_gap = True
                gap_start_idx = j
            elif in_gap and (o == 1 or j == n_samples - 1):
                in_gap = False
                gap_end_idx = j
                width = (gap_end_idx - gap_start_idx) * seg_len
                if min_gap_m <= width <= max_gap_m:
                    start_pt = pts[gap_start_idx]
                    end_pt   = pts[gap_end_idx]
                    mid_pt   = (start_pt + end_pt) / 2

                    # Outward-facing normal of this wall edge (90° rotation)
                    direction = (p2 - p1) / (wall_len + 1e-9)
                    normal = np.array([-direction[1], direction[0]])

                    gaps.append({
                        "wall_idx":    i,
                        "start_pt":    start_pt.tolist(),
                        "end_pt":      end_pt.tolist(),
                        "width_m":     round(float(width), 4),
                        "mid_pt":      mid_pt.tolist(),
                        "wall_normal": normal.tolist(),
                    })
    return gaps


def refine_gap_width(
    gap: dict,
    wall_slice_pcd: o3d.geometry.PointCloud,
    search_radius: float = 0.15,
) -> float:
    """
    Step 2 — Sub-centimetre width refinement.

    Fit a line to wall-slice points on each side of the gap midpoint and
    interpolate the true wall endpoints. Falls back to the grid-based width
    if too few points exist near the gap.
    """
    pts = np.asarray(wall_slice_pcd.points)
    mid = np.array(gap["mid_pt"])
    half_w = gap["width_m"] / 2 + search_radius

    # Points near the gap (within a box)
    dist = np.linalg.norm(pts[:, :2] - mid, axis=1)
    nearby = pts[dist < half_w + 0.3, :2]

    if len(nearby) < 6:
        return gap["width_m"]

    # Split into left side and right side relative to midpoint along wall direction
    wall_dir = np.array(gap["end_pt"]) - np.array(gap["start_pt"])
    wall_dir /= np.linalg.norm(wall_dir) + 1e-9
    proj = (nearby - mid) @ wall_dir

    left_pts  = nearby[proj < -0.05]
    right_pts = nearby[proj >  0.05]

    if len(left_pts) < 3 or len(right_pts) < 3:
        return gap["width_m"]

    # Closest point on each side to the gap edge
    left_edge  = left_pts[np.argmax(proj[proj < -0.05])]
    right_edge = right_pts[np.argmin(proj[proj >  0.05])]
    refined    = float(np.linalg.norm(right_edge - left_edge))

    # Sanity: don't let refinement stray more than 10 cm from the grid estimate
    if abs(refined - gap["width_m"]) > 0.10:
        return gap["width_m"]
    return round(refined, 4)
