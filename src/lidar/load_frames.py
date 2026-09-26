"""
load_frames.py — Step 2
Loads a single depth + confidence frame pair from disk.
Returns a masked depth array (metres) ready for unprojection.
"""

import cv2
import numpy as np


def load_depth_frame(depth_path: str, depth_scale: float) -> np.ndarray:
    """
    Read a 16-bit depth PNG and convert pixel values to metres.

    Args:
        depth_path:  Path to the 16-bit PNG (pixel value = depth in mm).
        depth_scale: Multiplier to convert raw values to metres (typically 0.001).

    Returns:
        (H, W) float32 array of depth in metres. Zero means no measurement.
    """
    raw = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
    if raw is None:
        raise FileNotFoundError(f"Depth frame not found: {depth_path}")
    return raw.astype(np.float32) * depth_scale


def load_confidence_frame(confidence_path: str) -> np.ndarray:
    """
    Read an 8-bit confidence PNG.

    Returns:
        (H, W) uint8 array. Values: 0 = low, 1 = medium, 2 = high.
    """
    conf = cv2.imread(confidence_path, cv2.IMREAD_GRAYSCALE)
    if conf is None:
        raise FileNotFoundError(f"Confidence frame not found: {confidence_path}")
    return conf


def load_masked_depth(
    depth_path: str,
    confidence_path: str,
    depth_scale: float,
    min_confidence: int,
) -> np.ndarray:
    """
    Load depth and zero out pixels whose confidence is below min_confidence.

    Args:
        depth_path:      Path to 16-bit depth PNG.
        confidence_path: Path to 8-bit confidence PNG.
        depth_scale:     Raw-to-metres multiplier.
        min_confidence:  Minimum confidence level to keep (inclusive).

    Returns:
        (H, W) float32 depth array in metres; masked pixels are 0.
    """
    depth = load_depth_frame(depth_path, depth_scale)
    conf = load_confidence_frame(confidence_path)

    # Zero out low-confidence pixels so they are skipped during unprojection
    depth[conf < min_confidence] = 0.0
    return depth
