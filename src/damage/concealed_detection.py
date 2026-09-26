"""
concealed_detection.py — Step 6
Detect concealed (behind-surface) damage from depth anomalies and low LiDAR confidence.

Two rules:
  depth_anomaly        — a wall region where depth deviates > threshold from the fitted plane
  low_confidence_region — a wall region with a high fraction of low-confidence pixels
"""

import numpy as np
import cv2


def detect_depth_anomaly(
    depth_path: str,
    mask: np.ndarray,
    surface_plane_d: float,
    fx: float, fy: float, cx: float, cy: float,
    depth_scale: float = 0.001,
    anomaly_thresh_m: float = 0.05,
) -> bool:
    """
    True if the mean depth of masked pixels deviates more than anomaly_thresh_m
    from the expected surface plane distance.
    """
    raw = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
    if raw is None or not mask.any():
        return False

    depth = raw.astype(np.float64) * depth_scale

    if mask.shape != depth.shape:
        mask = cv2.resize(mask.astype(np.uint8), (depth.shape[1], depth.shape[0]),
                          interpolation=cv2.INTER_NEAREST).astype(bool)

    valid_depth = depth[mask & (depth > 0)]
    if len(valid_depth) == 0:
        return False

    mean_depth = float(valid_depth.mean())
    return abs(mean_depth - surface_plane_d) > anomaly_thresh_m


def detect_low_confidence(
    confidence_path: str,
    mask: np.ndarray,
    low_conf_frac_thresh: float = 0.4,
) -> bool:
    """
    True if more than low_conf_frac_thresh of the masked pixels have confidence == 0.
    """
    conf = cv2.imread(confidence_path, cv2.IMREAD_GRAYSCALE)
    if conf is None or not mask.any():
        return False

    if mask.shape != conf.shape:
        mask = cv2.resize(mask.astype(np.uint8), (conf.shape[1], conf.shape[0]),
                          interpolation=cv2.INTER_NEAREST).astype(bool)

    region_conf = conf[mask]
    frac_low    = float((region_conf == 0).sum()) / max(len(region_conf), 1)
    return frac_low > low_conf_frac_thresh


def check_concealed(
    depth_path: str,
    confidence_path: str,
    mask: np.ndarray,
    surface_plane_d: float,
    fx: float, fy: float, cx: float, cy: float,
    depth_scale: float = 0.001,
) -> tuple[bool, str | None]:
    """
    Run both concealed-damage rules on a mask region.

    Returns (is_concealed, rule_fired).
    rule_fired is None when not concealed.
    """
    if detect_depth_anomaly(depth_path, mask, surface_plane_d, fx, fy, cx, cy, depth_scale):
        return True, "depth_anomaly"
    if detect_low_confidence(confidence_path, mask):
        return True, "low_confidence_region"
    return False, None
