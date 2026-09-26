"""
confidence.py — Step 1
Compute confidence intervals for every measurement type.

Formula:
    CI = base_error * tier_scalar / sqrt(point_count) * (1 + (1 - high_conf_fraction))

tier_scalar: LiDAR=1.0, video=1.5, photo=2.5
base_error:  wall_length=0.05 m, ceiling=0.03 m, opening=0.02 m, area=0.5 m²
"""

import math
import numpy as np

TIER_SCALAR = {"lidar": 1.0, "video": 1.5, "photo": 2.5}

BASE_ERROR = {
    "wall_length":    0.05,
    "ceiling_height": 0.03,
    "opening_width":  0.02,
    "floor_area":     0.50,
    "footprint_area": 1.00,
    "damage_area":    0.10,
}


def compute_ci(
    measurement_type: str,
    tier: str,
    point_count: int = 100,
    high_conf_fraction: float = 0.8,
) -> float:
    """
    Return a ± confidence interval in metres (or m² for areas).

    Args:
        measurement_type: key in BASE_ERROR dict
        tier:             'lidar', 'video', or 'photo'
        point_count:      number of 3D points contributing to this measurement
        high_conf_fraction: fraction of those points at LiDAR confidence level 2
    """
    base   = BASE_ERROR.get(measurement_type, 0.05)
    scalar = TIER_SCALAR.get(tier, 1.0)
    n      = max(point_count, 1)
    ci     = base * scalar / math.sqrt(n) * (1 + (1 - high_conf_fraction))
    return round(ci, 4)


def estimate_point_stats(
    pcd_pts: np.ndarray,
    region_centre: list,
    radius: float = 0.5,
) -> tuple[int, float]:
    """
    Count points near a measurement location and estimate high-confidence fraction.

    Since we don't track per-point confidence in the fused PLY, we use point
    density as a proxy — denser regions had more high-confidence frames contributing.
    Returns (point_count, high_conf_fraction_estimate).
    """
    centre = np.array(region_centre)
    dists  = np.linalg.norm(pcd_pts[:, :2] - centre[:2], axis=1)
    nearby = int((dists < radius).sum())

    # Density proxy: normalise to [0.5, 1.0] range
    max_density = 5000
    high_conf_frac = float(np.clip(0.5 + 0.5 * nearby / max_density, 0.5, 1.0))
    return nearby, high_conf_frac
