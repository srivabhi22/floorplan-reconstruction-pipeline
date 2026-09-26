"""
room_segmentation.py — Step 5
DBSCAN clustering on occupied grid cells to label distinct rooms.

Each cluster = one room. Small clusters (< min_cluster_cells) are treated as
noise and labelled -1. The result maps every wall-slice point to a room id.
"""

import numpy as np
from sklearn.cluster import DBSCAN


def segment_rooms(
    grid: np.ndarray,
    origin: np.ndarray,
    cell_size: float,
    eps: float = 0.05,
    min_samples: int = 10,
    min_cluster_cells: int = 50,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Run DBSCAN on occupied cells and return room labels.

    Args:
        grid:              (H, W) binary occupancy grid.
        origin:            (2,) world origin of the grid [x_min, y_min].
        cell_size:         Cell edge length in metres.
        eps:               DBSCAN neighbourhood radius in metres.
        min_samples:       DBSCAN minimum neighbours to form a core point.
        min_cluster_cells: Clusters smaller than this are discarded as noise.

    Returns:
        (occupied_ij, labels, room_bounds)
        - occupied_ij:  (M, 2) row/col indices of occupied cells.
        - labels:       (M,) room id per occupied cell (-1 = noise).
        - room_bounds:  dict  room_id → (x_min, x_max, y_min, y_max) in world coords.
    """
    # Get indices of all occupied cells
    rows, cols = np.where(grid == 1)
    occupied_ij = np.stack([rows, cols], axis=1)   # (M, 2)

    if len(occupied_ij) == 0:
        return occupied_ij, np.array([], dtype=int), {}

    # Convert to metric coordinates for DBSCAN (so eps is in metres)
    occupied_xy = np.stack([
        origin[0] + cols * cell_size,
        origin[1] + rows * cell_size,
    ], axis=1)

    # eps in grid cells = eps_m / cell_size; we work in metres directly
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(occupied_xy)
    labels = db.labels_.copy()

    # Discard clusters that are too small
    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    small = unique[counts < min_cluster_cells]
    labels[np.isin(labels, small)] = -1

    # Re-index room ids to be consecutive from 0
    valid_ids = sorted(set(labels[labels >= 0]))
    remap = {old: new for new, old in enumerate(valid_ids)}
    relabelled = np.where(labels >= 0, np.vectorize(remap.get)(labels), -1)

    # Compute 2D bounding box per room for ceiling refinement (Step 6)
    room_bounds = {}
    for room_id in range(len(valid_ids)):
        mask = relabelled == room_id
        room_xy = occupied_xy[mask]
        room_bounds[room_id] = (
            float(room_xy[:, 0].min()),
            float(room_xy[:, 0].max()),
            float(room_xy[:, 1].min()),
            float(room_xy[:, 1].max()),
        )

    return occupied_ij, relabelled, room_bounds


def label_slice_points(
    slice_pts_xy: np.ndarray,
    occupied_ij: np.ndarray,
    labels: np.ndarray,
    origin: np.ndarray,
    cell_size: float,
) -> np.ndarray:
    """
    Map room labels from grid cells back to individual wall-slice points.

    For each point find its grid cell, then look up that cell's label.
    Points whose cell is unlabelled (noise) get label -1.

    Returns:
        (N,) int array of room ids, one per wall-slice point.
    """
    # Build a lookup dict: (row, col) → label
    cell_label = {(r, c): lab for (r, c), lab in zip(map(tuple, occupied_ij), labels)}

    col = ((slice_pts_xy[:, 0] - origin[0]) / cell_size).astype(int)
    row = ((slice_pts_xy[:, 1] - origin[1]) / cell_size).astype(int)

    point_labels = np.array([
        cell_label.get((r, c), -1) for r, c in zip(row, col)
    ], dtype=int)
    return point_labels
