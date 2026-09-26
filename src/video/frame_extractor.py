"""frame_extractor.py — Step 1: extract frames from walkthrough video."""

import cv2
import numpy as np
from pathlib import Path


def extract_frames(
    video_path: str,
    output_dir: str,
    frame_stride: int = 5,
    max_frames: int = 200,
) -> list[dict]:
    """
    Extract every `frame_stride`-th frame from a video.

    Returns list of dicts: {index, timestamp_s, path}
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Video: {total} frames @ {fps:.1f} fps")

    frames = []
    frame_idx = 0
    saved = 0

    while saved < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_stride == 0:
            ts = frame_idx / fps
            fname = out / f"frame_{saved:05d}.jpg"
            cv2.imwrite(str(fname), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            frames.append({"index": frame_idx, "timestamp_s": round(ts, 4), "path": str(fname)})
            saved += 1

        frame_idx += 1

    cap.release()
    print(f"  Extracted {saved} frames (stride={frame_stride})")
    return frames
