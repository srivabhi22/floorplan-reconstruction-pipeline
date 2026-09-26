"""
opening_classifier.py — Steps 3 & 4
Classify each gap as door, window, or archway using vertical point-cloud extent,
and flag concealed (glass) openings with low LiDAR density.
"""

import numpy as np
import open3d as o3d


# Thresholds in metres above floor (z_norm)
DOOR_FLOOR_MAX   = 0.15   # door starts at or near floor
DOOR_HEADER_MIN  = 1.80   # door header at least this high
DOOR_HEADER_MAX  = 2.30
WINDOW_SILL_MIN  = 0.50   # window sill lower bound
WINDOW_SILL_MAX  = 1.20
WINDOW_HEAD_MIN  = 1.60
WINDOW_HEAD_MAX  = 2.40
LOW_DENSITY_FRAC = 0.15   # fraction of expected points — below this = concealed


def _vertical_profile(
    pts_norm: np.ndarray,
    mid_xy: np.ndarray,
    radius: float = 0.25,
    z_bins: int = 40,
    z_max: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Count points in a vertical column above the gap midpoint and return
    a histogram (bin_centres, counts).
    """
    dist_xy = np.linalg.norm(pts_norm[:, :2] - mid_xy, axis=1)
    column  = pts_norm[dist_xy < radius]

    counts, edges = np.histogram(column[:, 2], bins=z_bins, range=(0, z_max))
    centres = (edges[:-1] + edges[1:]) / 2
    return centres, counts


def classify_opening(
    gap: dict,
    pts_norm: np.ndarray,
    search_radius: float = 0.25,
) -> dict:
    """
    Step 3 — Classify one gap using its vertical point density profile.

    Returns a dict with keys: type, sill_height_m, header_height_m, concealed.
    """
    mid_xy = np.array(gap["mid_pt"])
    z_centres, counts = _vertical_profile(pts_norm, mid_xy, radius=search_radius)

    # Smoothed occupancy: 1 where counts > 0, 0 where void
    occupied = (counts > 0).astype(float)

    # Find the lowest and highest void bands
    void_idx = np.where(counts == 0)[0]
    if len(void_idx) == 0:
        # No void at all — solid wall, shouldn't happen but handle gracefully
        return {"type": "unknown", "sill_height_m": None, "header_height_m": None,
                "concealed": False}

    void_low  = float(z_centres[void_idx[0]])
    void_high = float(z_centres[void_idx[-1]])

    # Check density for glass/concealed detection
    total_expected = len(pts_norm) * (np.pi * search_radius**2) / (
        (pts_norm[:, 0].max() - pts_norm[:, 0].min()) *
        (pts_norm[:, 1].max() - pts_norm[:, 1].min()) + 1e-6
    )
    actual_density = counts.sum()
    concealed = bool(actual_density < LOW_DENSITY_FRAC * max(total_expected, 1))

    # Classification rules
    if void_low <= DOOR_FLOOR_MAX and DOOR_HEADER_MIN <= void_high <= DOOR_HEADER_MAX:
        opening_type = "door"
        sill   = 0.0
        header = round(void_high, 3)
    elif WINDOW_SILL_MIN <= void_low <= WINDOW_SILL_MAX and WINDOW_HEAD_MIN <= void_high <= WINDOW_HEAD_MAX:
        opening_type = "window"
        sill   = round(void_low,  3)
        header = round(void_high, 3)
    elif void_low <= DOOR_FLOOR_MAX and void_high > DOOR_HEADER_MAX:
        # Tall opening, possibly archway
        opening_type = "archway"
        sill   = 0.0
        header = round(void_high, 3)
    else:
        opening_type = "unknown"
        sill   = round(void_low,  3)
        header = round(void_high, 3)

    return {
        "type":            opening_type,
        "sill_height_m":   sill,
        "header_height_m": header,
        "concealed":       concealed,
    }


def confidence_interval(width_m: float, opening_type: str) -> float:
    """
    Step 5 helper — assign a confidence interval based on opening type.
    Doors measured from clear LiDAR returns get tighter CI than windows.
    These values are conservative; a calibration pass would tighten them.
    """
    if opening_type == "door":
        return 0.012
    if opening_type == "window":
        return 0.018
    return 0.025
