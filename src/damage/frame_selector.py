"""
frame_selector.py — Step 1
Select a sparse set of keyframes from the capture for damage detection.

Two strategies:
  - stride:   every N-th frame (default every 10)
  - pose:     frames where camera translation exceeds a threshold since last key
"""

import numpy as np
import pandas as pd


def select_by_stride(n_frames: int, stride: int = 10) -> list[int]:
    return list(range(0, n_frames, stride))


def select_by_pose_change(
    odom_df: pd.DataFrame,
    min_translation_m: float = 0.3,
) -> list[int]:
    """
    Pick frames where the camera moved at least min_translation_m since the last
    selected frame. Ensures good surface coverage without redundant near-identical views.
    """
    selected = [0]
    last_pos = odom_df.iloc[0][["x", "y", "z"]].values.astype(float)

    for i in range(1, len(odom_df)):
        pos = odom_df.iloc[i][["x", "y", "z"]].values.astype(float)
        if np.linalg.norm(pos - last_pos) >= min_translation_m:
            selected.append(i)
            last_pos = pos

    return selected


def load_rgb_frames(video_path: str, frame_indices: list[int]) -> dict[int, np.ndarray]:
    """
    Extract specific frames from rgb.mp4 by index.
    Returns dict frame_index -> (H, W, 3) uint8 BGR array.
    """
    import cv2
    cap = cv2.VideoCapture(video_path)
    frames = {}
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for idx in sorted(frame_indices):
        if idx >= total:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames[idx] = frame

    cap.release()
    return frames
