"""boundary_extraction.py — Step 1: per-room contour extraction from occupancy grid."""

import cv2
import numpy as np


def extract_room_boundary(
    grid: np.ndarray,
    room_mask_cells: np.ndarray,
    origin: np.ndarray,
    cell_size: float,
) -> np.ndarray:
    """Extract the longest outer contour for one room and return (N,2) world-coord xy."""
    mask = np.zeros(grid.shape, dtype=np.uint8)
    mask[room_mask_cells[:, 0], room_mask_cells[:, 1]] = 255

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.empty((0, 2), dtype=np.float64)

    contour = max(contours, key=cv2.contourArea)
    pts_rc = contour.squeeze(1).astype(np.float64)  # (N, 2) col,row from cv2

    # cv2 contour gives (x=col, y=row); convert to world coords
    x = origin[0] + pts_rc[:, 0] * cell_size
    y = origin[1] + pts_rc[:, 1] * cell_size
    return np.stack([x, y], axis=1)


def extract_all_boundaries(
    grid: np.ndarray,
    occupied_ij: np.ndarray,
    cell_labels: np.ndarray,
    origin: np.ndarray,
    cell_size: float,
) -> dict:
    """Return dict room_id -> (N,2) boundary array for every labelled room."""
    boundaries = {}
    for room_id in np.unique(cell_labels[cell_labels >= 0]):
        mask = cell_labels == room_id
        room_cells = occupied_ij[mask]
        boundaries[int(room_id)] = extract_room_boundary(grid, room_cells, origin, cell_size)
    return boundaries
