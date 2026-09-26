"""
pose_graph.py — Step 3
Build a pose graph whose nodes are segments and whose edges encode constraints.

We use a lightweight pure-numpy/scipy representation instead of GTSAM:
  - Each node stores a 6-DOF correction delta [tx, ty, tz, rx, ry, rz] (initially zero)
  - Each edge stores the relative constraint and a scalar weight

This is sufficient for the single-session, single-room case where drift is small
and can be linearised. A GTSAM / g2o backend would be needed for large-scale
multi-session capture.
"""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass
class PoseGraphNode:
    seg_idx: int             # index into the segment list
    start_frame: int
    end_frame: int
    # Mean camera position over this segment's frames (used for loop detection)
    mean_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # Detected planes: dict with keys 'floor', 'ceiling', 'wall'
    planes: dict = field(default_factory=dict)
    # 6-DOF correction delta [tx, ty, tz, rx, ry, rz]; optimized in Step 4
    delta: np.ndarray = field(default_factory=lambda: np.zeros(6))


@dataclass
class PoseGraphEdge:
    src: int           # source node index
    dst: int           # destination node index
    # Relative constraint: the expected delta between src and dst
    relative_delta: np.ndarray  # (6,)
    weight: float      # higher = more trusted
    kind: str          # 'sequential' or 'plane_match'


def build_pose_graph(
    segments: list[tuple[int, int]],
    odom_df: pd.DataFrame,
    plane_results: list[dict],
    overlap: int,
    plane_normal_tol: float = 0.1,   # max angular difference (dot product deviation)
    plane_d_tol: float = 0.05,       # max offset difference in metres (5 cm)
) -> tuple[list[PoseGraphNode], list[PoseGraphEdge]]:
    """
    Build nodes and edges for the pose graph.

    Args:
        segments:       List of (start, end) frame ranges from segment.py.
        odom_df:        Odometry DataFrame with x, y, z columns.
        plane_results:  One dict per segment from plane_detection.detect_dominant_planes.
        overlap:        Number of frames shared between adjacent segments.
        plane_normal_tol: Maximum allowed |1 - |n1·n2|| for planes to be matched.
        plane_d_tol:    Maximum allowed |d1 - d2| for planes to be matched.

    Returns:
        (nodes, edges)
    """
    nodes: list[PoseGraphNode] = []
    edges: list[PoseGraphEdge] = []

    # ── Build nodes ───────────────────────────────────────────────────────────
    for i, (start, end) in enumerate(segments):
        rows = odom_df.iloc[start:end]
        mean_pos = rows[["x", "y", "z"]].mean().values
        node = PoseGraphNode(
            seg_idx=i,
            start_frame=start,
            end_frame=end,
            mean_position=mean_pos,
            planes=plane_results[i],
        )
        nodes.append(node)

    # ── Sequential edges (adjacent segments, high weight) ─────────────────────
    for i in range(len(nodes) - 1):
        # Relative pose in the overlap window: difference of mean ARKit translations
        start_overlap = segments[i + 1][0]
        end_overlap   = segments[i][1]
        if start_overlap < end_overlap:
            overlap_rows = odom_df.iloc[start_overlap:end_overlap]
        else:
            overlap_rows = odom_df.iloc[[segments[i + 1][0]]]

        # Mean relative translation; rotations stay zero (small drift assumption)
        rel_t = (
            odom_df.iloc[segments[i + 1][0]][["x", "y", "z"]].values
            - odom_df.iloc[segments[i][1] - 1][["x", "y", "z"]].values
        )
        rel_delta = np.concatenate([rel_t, np.zeros(3)])

        edges.append(PoseGraphEdge(
            src=i, dst=i + 1,
            relative_delta=rel_delta,
            weight=10.0,   # locally accurate
            kind="sequential",
        ))

    # ── Plane-match edges (non-adjacent segments sharing a surface) ───────────
    plane_keys = ["floor", "ceiling", "wall"]
    for i in range(len(nodes)):
        for j in range(i + 2, len(nodes)):   # skip adjacent (already have sequential)
            for key in plane_keys:
                pi = nodes[i].planes.get(key)
                pj = nodes[j].planes.get(key)
                if pi is None or pj is None:
                    continue

                ni, di = pi["normal"], pi["d"]
                nj, dj = pj["normal"], pj["d"]

                # Check that normals are nearly parallel (same surface orientation)
                dot = float(np.abs(np.dot(ni, nj)))
                if abs(1.0 - dot) > plane_normal_tol:
                    continue

                # Check that offset values agree (same physical plane)
                if abs(di - dj) > plane_d_tol:
                    continue

                # Constraint: the correction should bring d-values into agreement
                d_diff = dj - di
                # Project along the shared normal to get a translation constraint
                rel_t = (nj * d_diff)  # 3D shift
                rel_delta = np.concatenate([rel_t, np.zeros(3)])

                edges.append(PoseGraphEdge(
                    src=i, dst=j,
                    relative_delta=rel_delta,
                    weight=5.0,   # medium: plane match may be a false association
                    kind="plane_match",
                ))

    return nodes, edges
