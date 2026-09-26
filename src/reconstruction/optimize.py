"""
optimize.py — Step 4
Least-squares pose graph optimization (scipy fallback; no GTSAM required).

Each node holds a 6-DOF correction delta d_i = [tx, ty, tz, rx, ry, rz].
Every edge (i→j) contributes a residual:
    r = (d_j - d_i) - relative_delta_ij

We fix node 0 (the first segment) as the anchor so the system has a unique solution.
The weighted least-squares problem is solved with scipy.sparse.linalg.lsqr.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from src.reconstruction.pose_graph import PoseGraphNode, PoseGraphEdge


DOF = 6   # degrees of freedom per node


def optimize_pose_graph(
    nodes: list[PoseGraphNode],
    edges: list[PoseGraphEdge],
) -> list[np.ndarray]:
    """
    Solve for the correction delta of each node by minimising total weighted
    edge residuals using sparse least squares.

    The anchor node (index 0) is fixed at zero correction.

    Returns:
        List of (6,) correction deltas, one per node.
        Node 0 is always [0, 0, 0, 0, 0, 0].
    """
    n = len(nodes)
    m = len(edges)

    if m == 0 or n <= 1:
        return [np.zeros(DOF) for _ in nodes]

    # ── Build sparse Jacobian A and RHS b ────────────────────────────────────
    # Each edge contributes DOF rows to the system.
    # For edge (i→j):  d_j - d_i = relative_delta  →  A[row, j*DOF:(j+1)*DOF] = +I
    #                                                    A[row, i*DOF:(i+1)*DOF] = -I
    # Node 0 is anchored: d_0 = 0 (we just skip its columns and treat it as known).

    rows_data, cols_data, vals_data = [], [], []
    b_list = []
    row_idx = 0

    for edge in edges:
        i, j = edge.src, edge.dst
        w = np.sqrt(edge.weight)   # weight enters as sqrt in least-squares
        rel = edge.relative_delta  # (DOF,)

        for dof in range(DOF):
            r = row_idx + dof
            b_val = w * rel[dof]

            # Contribution from node j (unless it's the anchor)
            if j != 0:
                rows_data.append(r)
                cols_data.append(j * DOF + dof)
                vals_data.append(w)
                b_list.append(b_val)

            # Contribution from node i (unless it's the anchor)
            if i != 0:
                rows_data.append(r)
                cols_data.append(i * DOF + dof)
                vals_data.append(-w)
                if j == 0:
                    # b absorbs the known anchor value (zero)
                    b_list.append(b_val)

        row_idx += DOF

    n_rows = row_idx
    n_cols = n * DOF

    A = sp.csr_matrix(
        (vals_data, (rows_data, cols_data)),
        shape=(n_rows, n_cols),
    )
    b = np.array(b_list + [0.0] * (n_rows - len(b_list)))

    # ── Solve ────────────────────────────────────────────────────────────────
    x, *_ = spla.lsqr(A, b[:n_rows], atol=1e-8, btol=1e-8, iter_lim=5000)

    # ── Unpack deltas ────────────────────────────────────────────────────────
    deltas = []
    for k in range(n):
        if k == 0:
            deltas.append(np.zeros(DOF))
        else:
            deltas.append(x[k * DOF : (k + 1) * DOF])

    return deltas


def delta_to_transform(delta: np.ndarray) -> np.ndarray:
    """
    Convert a 6-DOF correction delta [tx, ty, tz, rx, ry, rz] to a 4×4 matrix.

    Small-angle approximation for rotations (valid for drift-level corrections).
    """
    tx, ty, tz, rx, ry, rz = delta
    # Small-angle rotation matrix
    R = np.array([
        [ 1,  -rz,  ry],
        [ rz,  1,  -rx],
        [-ry,  rx,   1],
    ], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3,  3] = [tx, ty, tz]
    return T
