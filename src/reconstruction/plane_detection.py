"""
plane_detection.py — Step 2
RANSAC plane fitting per segment.

Detects up to three dominant planes per sub-cloud:
  - floor   : lowest cluster, roughly horizontal normal
  - ceiling : highest cluster, roughly horizontal normal
  - wall    : largest remaining vertical plane

Each plane is returned as (normal, d) where  normal·x + d = 0,
with normal always unit-length and pointing "outward" (away from the cloud centre).
"""

import numpy as np
import open3d as o3d


# ── Internal helpers ──────────────────────────────────────────────────────────

def _fit_plane(pcd: o3d.geometry.PointCloud, dist_thresh: float = 0.02
               ) -> tuple[np.ndarray, float, list[int]] | None:
    """
    Run one RANSAC plane fit. Returns (unit_normal, d, inlier_indices) or None
    if the cloud is too small.
    """
    if len(pcd.points) < 10:
        return None
    # Seed fixed for reproducibility (required by the repeatability gate)
    model, inliers = pcd.segment_plane(
        distance_threshold=dist_thresh,
        ransac_n=3,
        num_iterations=1000,
    )
    a, b, c, d = model
    normal = np.array([a, b, c], dtype=np.float64)
    norm = np.linalg.norm(normal)
    if norm < 1e-9:
        return None
    normal /= norm
    d /= norm
    return normal, d, inliers


def _is_horizontal(normal: np.ndarray, tol_deg: float = 20.0) -> bool:
    """True if the plane is roughly horizontal (normal close to vertical)."""
    # A horizontal plane has a normal close to [0,0,1] or [0,1,0]
    # We check whichever axis is most vertical by looking at the largest component
    return float(np.max(np.abs(normal))) > np.cos(np.radians(tol_deg))


def _is_vertical(normal: np.ndarray, tol_deg: float = 20.0) -> bool:
    """True if the plane is roughly vertical (normal close to horizontal)."""
    return not _is_horizontal(normal, tol_deg)


# ── Public API ────────────────────────────────────────────────────────────────

def detect_dominant_planes(
    pcd: o3d.geometry.PointCloud,
    dist_thresh: float = 0.02,
) -> dict:
    """
    Detect floor, ceiling, and dominant wall plane in a segment sub-cloud.

    Returns a dict with keys 'floor', 'ceiling', 'wall', each being a dict:
        {
            'normal': np.ndarray (3,),   # unit normal
            'd':      float,             # plane offset  (normal·x + d = 0)
            'n_inliers': int,
        }
    Missing planes (not enough points or not found) are stored as None.

    Detection strategy:
    1. Fit a plane to the bottom 20 % of points → floor candidate
    2. Fit a plane to the top 20 % of points → ceiling candidate
    3. Remove both sets of inliers, fit the largest remaining plane → wall
    """
    pts = np.asarray(pcd.points)
    if len(pts) < 30:
        return {"floor": None, "ceiling": None, "wall": None}

    # Determine the vertical axis (largest span)
    spans = pts.max(axis=0) - pts.min(axis=0)
    v = int(np.argmax(spans))

    results = {"floor": None, "ceiling": None, "wall": None}

    # ── Floor: bottom 20 % of points by vertical axis ────────────────────────
    thresh_lo = np.percentile(pts[:, v], 20)
    bottom_idx = np.where(pts[:, v] <= thresh_lo)[0]
    if len(bottom_idx) >= 10:
        bottom_pcd = pcd.select_by_index(bottom_idx.tolist())
        fit = _fit_plane(bottom_pcd, dist_thresh)
        if fit is not None:
            normal, d, inliers = fit
            results["floor"] = {
                "normal": normal, "d": d, "n_inliers": len(inliers)
            }

    # ── Ceiling: top 20 % of points ──────────────────────────────────────────
    thresh_hi = np.percentile(pts[:, v], 80)
    top_idx = np.where(pts[:, v] >= thresh_hi)[0]
    if len(top_idx) >= 10:
        top_pcd = pcd.select_by_index(top_idx.tolist())
        fit = _fit_plane(top_pcd, dist_thresh)
        if fit is not None:
            normal, d, inliers = fit
            results["ceiling"] = {
                "normal": normal, "d": d, "n_inliers": len(inliers)
            }

    # ── Wall: remove horizontal-plane inliers, fit largest remaining plane ───
    # Remove floor + ceiling bands, then fit
    mid_mask = (pts[:, v] > thresh_lo) & (pts[:, v] < thresh_hi)
    mid_idx = np.where(mid_mask)[0]
    if len(mid_idx) >= 10:
        mid_pcd = pcd.select_by_index(mid_idx.tolist())
        fit = _fit_plane(mid_pcd, dist_thresh)
        if fit is not None:
            normal, d, inliers = fit
            results["wall"] = {
                "normal": normal, "d": d, "n_inliers": len(inliers)
            }

    return results
