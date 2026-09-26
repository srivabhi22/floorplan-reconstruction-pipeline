"""intrinsics.py — Step 2: load or estimate camera intrinsics."""

import csv
import subprocess
from pathlib import Path

import cv2
import numpy as np


def load_intrinsics(video_path: str, camera_matrix_csv: str | None = None) -> dict:
    """
    Return {fx, fy, cx, cy, width, height}.

    Priority:
    1. camera_matrix.csv (3×3 matrix)
    2. EXIF focal length from video metadata via ffprobe
    3. Fallback: assume 70° horizontal FOV
    """
    cap = cv2.VideoCapture(video_path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if camera_matrix_csv and Path(camera_matrix_csv).exists():
        K = _load_csv_matrix(camera_matrix_csv)
        print(f"  Intrinsics: loaded from {camera_matrix_csv}")
        return {"fx": float(K[0, 0]), "fy": float(K[1, 1]),
                "cx": float(K[0, 2]), "cy": float(K[1, 2]),
                "width": W, "height": H}

    # Try EXIF via ffprobe
    fx = _exif_focal(video_path, W)
    if fx:
        print(f"  Intrinsics: estimated from EXIF (fx={fx:.1f})")
        return {"fx": fx, "fy": fx, "cx": W / 2, "cy": H / 2, "width": W, "height": H}

    # Fallback: 70° horizontal FOV
    fx = W / (2 * np.tan(np.radians(70 / 2)))
    print(f"  Intrinsics: fallback 70° FOV (fx={fx:.1f})")
    return {"fx": fx, "fy": fx, "cx": W / 2, "cy": H / 2, "width": W, "height": H}


def _load_csv_matrix(path: str) -> np.ndarray:
    rows = []
    with open(path) as f:
        for row in csv.reader(f):
            rows.append([float(v) for v in row])
    return np.array(rows)


def _exif_focal(video_path: str, frame_width: int) -> float | None:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", video_path],
            capture_output=True, text=True, timeout=10,
        )
        import json
        data = json.loads(result.stdout)
        for stream in data.get("streams", []):
            tags = stream.get("tags", {})
            # Some devices embed focal_length in tags
            focal_str = tags.get("focal_length") or tags.get("com.android.capture.fps")
            if focal_str:
                try:
                    fl_mm = float(str(focal_str).split("/")[0])
                    # Assume ~6 mm sensor width (common smartphone)
                    fx = frame_width * fl_mm / 6.0
                    return fx
                except (ValueError, IndexError):
                    pass
    except Exception:
        pass
    return None
