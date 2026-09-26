"""
ablation.py — Step 6
Saves the raw and corrected point clouds for ablation comparison,
computes the drift metric, and logs plane alignment residuals.
"""

import numpy as np
import open3d as o3d
import pandas as pd
from pathlib import Path


def plane_alignment_error(planes: list[dict | None]) -> float:
    """
    Mean pairwise d-value difference for segments that share the same plane type.
    Lower = better aligned across segments.
    Only considers segments where the plane was detected.
    """
    valid = [p for p in planes if p is not None]
    if len(valid) < 2:
        return 0.0
    d_vals = np.array([p["d"] for p in valid])
    # Mean absolute deviation from the median (robust measure)
    return float(np.mean(np.abs(d_vals - np.median(d_vals))))


def compute_drift_metric(odom_df: pd.DataFrame) -> float:
    """
    Closure error: Euclidean distance between first and last camera positions.
    Relevant when the scan forms a loop (camera returns near start).
    """
    first = odom_df.iloc[0][["x", "y", "z"]].values
    last  = odom_df.iloc[-1][["x", "y", "z"]].values
    return float(np.linalg.norm(last - first))


def save_ablation(
    raw_pcd: o3d.geometry.PointCloud,
    corrected_pcd: o3d.geometry.PointCloud,
    ablation_dir: Path,
) -> None:
    ablation_dir.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(ablation_dir / "raw.ply"),       raw_pcd)
    o3d.io.write_point_cloud(str(ablation_dir / "corrected.ply"), corrected_pcd)
    print(f"  Ablation clouds saved → {ablation_dir}")


def save_corrected_poses(
    odom_df: pd.DataFrame,
    segments: list[tuple[int, int]],
    correction_transforms: list[np.ndarray],
    output_path: Path,
) -> None:
    """
    Write corrected_poses.csv with the optimized per-frame 4×4 transform
    flattened to 16 columns (T_00 … T_33).
    """
    frame_to_seg = {}
    for seg_idx, (start, end) in enumerate(segments):
        for fi in range(start, end):
            frame_to_seg[fi] = seg_idx

    records = []
    for fi, (_, row) in enumerate(odom_df.iterrows()):
        seg_idx = frame_to_seg.get(fi, 0)
        C = correction_transforms[seg_idx]
        T_corrected = C @ row["T"]
        flat = T_corrected.flatten()
        rec = {"frame": row["frame"]}
        for k, v in enumerate(flat):
            rec[f"T_{k//4}{k%4}"] = v
        records.append(rec)

    pd.DataFrame(records).to_csv(output_path, index=False)
    print(f"  Corrected poses saved → {output_path}")


def print_ablation_report(
    floor_planes_before: list,
    floor_planes_after: list,
    odom_df: pd.DataFrame,
    n_seq_edges: int,
    n_plane_edges: int,
) -> None:
    err_before = plane_alignment_error(floor_planes_before)
    err_after  = plane_alignment_error(floor_planes_after)
    closure    = compute_drift_metric(odom_df)

    print("\n── Drift Correction Ablation Report ────────────────────────────")
    print(f"  Pose graph edges   : {n_seq_edges} sequential, {n_plane_edges} plane-match")
    print(f"  Floor plane error  : {err_before*100:.1f} cm  →  {err_after*100:.1f} cm")
    print(f"  Closure error      : {closure*100:.1f} cm  (first→last frame distance)")
    if closure < 0.3:
        print("  Loop detected      : yes (closure < 30 cm)")
    else:
        print("  Loop detected      : no  (open trajectory)")
    print("────────────────────────────────────────────────────────────────\n")
