"""
load_odometry.py — Step 1
Reads odometry.csv and camera_matrix.csv, returns per-frame 4×4 world transforms
and intrinsics. Also provides a global intrinsic fallback from camera_matrix.csv.
"""

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


def load_odometry(odometry_path: str) -> pd.DataFrame:
    """
    Read odometry.csv and attach a 4×4 world-transform matrix to every row.

    Returns a DataFrame with the original columns plus:
      - 'T'  : (4, 4) float64 ndarray — camera-to-world transform for that frame
    """
    df = pd.read_csv(odometry_path, skipinitialspace=True)
    df.columns = df.columns.str.strip()

    # Zero-pad frame ids so they match the 6-digit filenames in depth/confidence
    df["frame"] = df["frame"].astype(str).str.zfill(6)

    transforms = []
    for _, row in df.iterrows():
        t = np.array([row["x"], row["y"], row["z"]], dtype=np.float64)
        q = np.array([row["qx"], row["qy"], row["qz"], row["qw"]], dtype=np.float64)

        # scipy convention: (x, y, z, w)
        R = Rotation.from_quat(q).as_matrix()  # (3, 3)

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        transforms.append(T)

    df["T"] = transforms
    return df


def load_global_intrinsics(camera_matrix_path: str) -> dict:
    """
    Read the 3×3 camera_matrix.csv and return {fx, fy, cx, cy} as a fallback
    when a frame's per-frame intrinsics are missing.
    """
    K = np.loadtxt(camera_matrix_path, delimiter=",")  # (3, 3)
    return {"fx": K[0, 0], "fy": K[1, 1], "cx": K[0, 2], "cy": K[1, 2]}
