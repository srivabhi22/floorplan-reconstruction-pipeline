"""mast3r_runner.py — Step 3: run MASt3R to get metric depth maps and camera poses."""

import sys
import os
import numpy as np
import torch
from pathlib import Path

# MASt3R lives in third_party — add to path
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "third_party" / "mast3r"))
sys.path.insert(0, str(_ROOT / "third_party" / "mast3r" / "dust3r"))

from mast3r.model import AsymmetricMASt3R
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from dust3r.inference import inference
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images

# HuggingFace model id — downloaded on first run (~1 GB)
_MODEL_HF = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
_WINDOW = 150   # max frames per MASt3R pass
_OVERLAP = 30   # overlap between windows


def run_mast3r(
    frame_paths: list[str],
    intrinsics: dict,
    output_dir: str,
    device_str: str = "auto",
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Run MASt3R on a list of frame image paths.

    Returns:
        depth_maps: list of (H, W) float32 arrays, one per frame
        poses:      list of (4, 4) float64 camera-to-world transforms
    """
    if device_str == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)

    print(f"  MASt3R device: {device}")
    model = _load_model(device)

    n = len(frame_paths)
    if n <= _WINDOW:
        depth_maps, poses = _run_window(model, frame_paths, intrinsics, device, output_dir)
    else:
        depth_maps, poses = _run_windowed(model, frame_paths, intrinsics, device, output_dir)

    return depth_maps, poses


def _load_model(device):
    model_path = Path(__file__).resolve().parents[2] / "models" / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
    if model_path.exists():
        print(f"  Loading model from {model_path}")
        return AsymmetricMASt3R.from_pretrained(str(model_path)).to(device)
    print(f"  Downloading model from HuggingFace ({_MODEL_HF}) …")
    return AsymmetricMASt3R.from_pretrained(_MODEL_HF).to(device)


def _run_window(model, paths, intrinsics, device, output_dir):
    """Single MASt3R pass over all frames using sequential pairs only."""
    imgs = load_images(paths, size=224, verbose=False)  # smaller resolution → less memory

    # sequential: only consecutive pairs — O(N) instead of O(N²)
    pairs = make_pairs(imgs, scene_graph="sequential-3", prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=1, verbose=False)

    # lightweight global aligner — no heavy sparse_ga
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
    scene = global_aligner(
        output,
        device=device,
        mode=GlobalAlignerMode.PointCloudOptimizer,
    )
    scene.compute_global_alignment(init="mst", niter=100, schedule="cosine", lr=0.01)

    return _extract_results(scene, paths, intrinsics)


def _run_windowed(model, paths, intrinsics, device, output_dir):
    """Process in overlapping windows then stitch poses."""
    n = len(paths)
    all_depths = [None] * n
    all_poses = [None] * n

    windows = []
    start = 0
    while start < n:
        end = min(start + _WINDOW, n)
        windows.append((start, end))
        if end == n:
            break
        start = end - _OVERLAP

    print(f"  Processing {len(windows)} window(s) of up to {_WINDOW} frames")

    anchor_pose = None  # world transform of the first frame of the previous window
    anchor_idx = 0

    for w_idx, (ws, we) in enumerate(windows):
        print(f"  Window {w_idx + 1}/{len(windows)}: frames {ws}–{we - 1}")
        window_paths = paths[ws:we]
        depths, poses = _run_window(model, window_paths, intrinsics, device, output_dir)

        if anchor_pose is not None:
            # Align this window's first frame to the anchor
            T_align = anchor_pose @ np.linalg.inv(poses[0])
            poses = [T_align @ p for p in poses]

        # Fill in non-overlap frames (and overlap frames for the first window)
        for local_i, global_i in enumerate(range(ws, we)):
            if all_depths[global_i] is None:
                all_depths[global_i] = depths[local_i]
                all_poses[global_i] = poses[local_i]

        # Next window's anchor = last non-overlap frame of this window
        anchor_idx = we - _OVERLAP - 1
        if anchor_idx < ws:
            anchor_idx = ws
        anchor_pose = poses[anchor_idx - ws]

    return all_depths, all_poses


def _extract_results(scene, paths, intrinsics):
    """Pull depth maps and poses out of a solved MASt3R scene."""
    depths = []
    poses = []

    # scene.get_depthmaps() returns one (H,W) tensor per image
    depthmaps = scene.get_depthmaps(raw=True)
    c2ws = scene.get_im_poses()   # (N, 4, 4) camera-to-world

    for i in range(len(paths)):
        d = depthmaps[i].detach().cpu().numpy().astype(np.float32)
        depths.append(d)

        T = c2ws[i].detach().cpu().numpy().astype(np.float64)
        poses.append(T)

    return depths, poses
