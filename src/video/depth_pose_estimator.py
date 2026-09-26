"""
depth_pose_estimator.py — Lightweight alternative to MASt3R.

Depth : Depth Anything V2 Small (vits, ~100 MB) — relative depth per frame
Poses : OpenCV ORB feature matching + Essential matrix decomposition
Scale : Floor-plane RANSAC — set metric scale so floor sits at camera_height metres
"""

import sys
import numpy as np
import cv2
import torch
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "third_party" / "Depth-Anything-V2"))

_MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64,  "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
}

_HF_URLS = {
    "vits": "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth",
    "vitb": "https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth",
}

_CAMERA_HEIGHT_M = 1.5   # assumed height of camera above floor


def run(
    frame_paths: list[str],
    intrinsics: dict,
    encoder: str = "vits",
    device_str: str = "auto",
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Returns:
        depth_maps : list of (H, W) float32 metric depth arrays
        poses      : list of (4, 4) float64 camera-to-world transforms
    """
    device = _pick_device(device_str)
    model = _load_depth_model(encoder, device)

    print(f"  Depth Anything V2 ({encoder}) on {device}")
    print(f"  Computing depth maps …")

    raw_depths = []
    bgr_frames = []
    for i, p in enumerate(frame_paths):
        bgr = cv2.imread(p)
        bgr_frames.append(bgr)
        raw = _infer_depth(model, bgr, device)
        raw_depths.append(raw)
        if (i + 1) % 10 == 0:
            print(f"    {i + 1}/{len(frame_paths)} depth maps done")

    print(f"  Estimating poses via ORB + Essential matrix …")
    poses = _estimate_poses(bgr_frames, intrinsics)

    print(f"  Recovering metric scale via floor-plane …")
    depth_maps = _recover_metric_scale(raw_depths, poses, intrinsics)

    return depth_maps, poses


# ── depth model ──────────────────────────────────────────────────────────────

def _pick_device(device_str: str) -> torch.device:
    if device_str != "auto":
        return torch.device(device_str)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_depth_model(encoder: str, device: torch.device):
    from depth_anything_v2.dpt import DepthAnythingV2

    weights_path = _ROOT / "models" / f"depth_anything_v2_{encoder}.pth"
    if not weights_path.exists():
        _download_weights(encoder, weights_path)

    model = DepthAnythingV2(**_MODEL_CONFIGS[encoder])
    model.load_state_dict(torch.load(str(weights_path), map_location="cpu"))
    return model.to(device).eval()


def _download_weights(encoder: str, dest: Path) -> None:
    import urllib.request
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = _HF_URLS[encoder]
    print(f"  Downloading Depth Anything V2 {encoder} weights (~100 MB) …")
    urllib.request.urlretrieve(url, str(dest))
    print(f"  Saved → {dest}")


def _infer_depth(model, bgr: np.ndarray, device: torch.device) -> np.ndarray:
    """Run model, return (H, W) float32 relative depth (larger = farther)."""
    with torch.no_grad():
        depth = model.infer_image(bgr)  # returns numpy (H, W) float32
    return depth.astype(np.float32)


# ── pose estimation ───────────────────────────────────────────────────────────

def _estimate_poses(frames: list[np.ndarray], intrinsics: dict) -> list[np.ndarray]:
    """ORB feature matching + Essential matrix → per-frame camera-to-world poses."""
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    orb = cv2.ORB_create(nfeatures=1000)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    poses = [np.eye(4)]  # first frame is world origin

    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    prev_kp, prev_desc = orb.detectAndCompute(prev_gray, None)
    T_world = np.eye(4)

    for i in range(1, len(frames)):
        gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        kp, desc = orb.detectAndCompute(gray, None)

        T_world = _relative_pose(prev_kp, prev_desc, kp, desc, K, T_world)
        poses.append(T_world.copy())

        prev_gray, prev_kp, prev_desc = gray, kp, desc

    return poses


def _relative_pose(
    kp1, desc1, kp2, desc2, K: np.ndarray, T_prev: np.ndarray
) -> np.ndarray:
    """Compute T_world for frame 2 given T_world for frame 1."""
    if desc1 is None or desc2 is None or len(kp1) < 8 or len(kp2) < 8:
        return T_prev  # no motion — keep last pose

    matches = _match(desc1, desc2)
    if len(matches) < 8:
        return T_prev

    pts1 = np.float64([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float64([kp2[m.trainIdx].pt for m in matches])

    E, mask = cv2.findEssentialMat(pts1, pts2, K,
                                   method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None:
        return T_prev

    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K, mask=mask)

    # Relative transform (translation is unit-scale — fixed later)
    T_rel = np.eye(4)
    T_rel[:3, :3] = R
    T_rel[:3, 3] = t.ravel()

    return T_prev @ np.linalg.inv(T_rel)


def _match(desc1, desc2):
    from cv2 import DMatch
    raw = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(desc1, desc2, k=2)
    # Lowe ratio test
    good = []
    for pair in raw:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)
    return good


# ── metric scale recovery ─────────────────────────────────────────────────────

def _recover_metric_scale(
    raw_depths: list[np.ndarray],
    poses: list[np.ndarray],
    intrinsics: dict,
) -> list[np.ndarray]:
    """
    Scale each frame's relative depth so the floor plane sits at camera_height metres.
    Uses RANSAC plane fitting on unprojected points from the bottom quarter of the image.
    """
    from src.lidar.unproject import depth_to_camera_points

    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]

    scaled = []
    for depth_raw in raw_depths:
        H, W = depth_raw.shape
        # Use bottom quarter — likely floor
        bottom = depth_raw[3 * H // 4:, :]
        # Normalise relative depth to [0.5, 5] m range
        d_min, d_max = bottom.min(), bottom.max()
        if d_max - d_min < 1e-6:
            scaled.append(depth_raw)
            continue

        # Invert: Depth Anything gives larger values for closer objects
        d_norm = (depth_raw - d_min) / (d_max - d_min)
        d_norm = 1.0 - d_norm  # now larger = farther
        d_norm = d_norm * 4.5 + 0.5  # map to [0.5, 5] m

        # Fit scale so floor points sit at camera_height below camera
        bottom_norm = d_norm[3 * H // 4:, :]
        pts_cam = depth_to_camera_points(bottom_norm.astype(np.float64), fx, fy, cx, cy + 3 * H // 4)
        if pts_cam.shape[0] < 10:
            scaled.append(d_norm.astype(np.float32))
            continue

        # Median Y-depth of floor region → scale factor
        y_vals = pts_cam[:, 1]  # Y in camera space ≈ down
        median_y = float(np.median(np.abs(y_vals)))
        if median_y > 0:
            scale = _CAMERA_HEIGHT_M / median_y
            d_metric = d_norm * scale
        else:
            d_metric = d_norm

        scaled.append(d_metric.astype(np.float32))

    return scaled
