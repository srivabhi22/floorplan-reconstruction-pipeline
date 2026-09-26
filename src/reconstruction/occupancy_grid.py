"""
occupancy_grid.py — Step 4
Project wall-slice points onto the XY plane and build a 2D binary occupancy grid.

The grid stores:
  - 1 where a wall point falls (occupied)
  - 0 everywhere else (free space)

Metadata (origin, cell_size) is saved alongside so downstream modules can
convert between grid cell indices and real-world metric coordinates.
"""

import numpy as np
import matplotlib.pyplot as plt


def build_occupancy_grid(
    pts_xy: np.ndarray,
    cell_size: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Build a 2D binary occupancy grid from XY wall-slice points.

    Args:
        pts_xy:    (N, 2) array of x, y coordinates (wall-slice, z dropped).
        cell_size: Edge length of each grid cell in metres (default 2 cm).

    Returns:
        (grid, origin, cell_size)
        - grid:      (H, W) uint8 binary array  (1 = occupied, 0 = free)
        - origin:    (2,) float array [x_min, y_min] in world coordinates
        - cell_size: the cell size used (echoed back for clarity)
    """
    x_min, y_min = pts_xy.min(axis=0)
    x_max, y_max = pts_xy.max(axis=0)

    W = int(np.ceil((x_max - x_min) / cell_size)) + 1
    H = int(np.ceil((y_max - y_min) / cell_size)) + 1

    grid = np.zeros((H, W), dtype=np.uint8)

    # Convert world coordinates to grid cell indices
    col = ((pts_xy[:, 0] - x_min) / cell_size).astype(int)
    row = ((pts_xy[:, 1] - y_min) / cell_size).astype(int)

    # Clip to grid bounds (floating point safety)
    col = np.clip(col, 0, W - 1)
    row = np.clip(row, 0, H - 1)

    grid[row, col] = 1
    origin = np.array([x_min, y_min], dtype=np.float64)
    return grid, origin, cell_size


def grid_to_world(cell_ij: np.ndarray, origin: np.ndarray, cell_size: float) -> np.ndarray:
    """Convert (row, col) grid indices → (x, y) world coordinates (cell centres)."""
    x = origin[0] + cell_ij[:, 1] * cell_size + cell_size / 2
    y = origin[1] + cell_ij[:, 0] * cell_size + cell_size / 2
    return np.stack([x, y], axis=1)


def world_to_grid(pts_xy: np.ndarray, origin: np.ndarray, cell_size: float) -> np.ndarray:
    """Convert (x, y) world coordinates → (row, col) grid indices."""
    col = ((pts_xy[:, 0] - origin[0]) / cell_size).astype(int)
    row = ((pts_xy[:, 1] - origin[1]) / cell_size).astype(int)
    return np.stack([row, col], axis=1)


def save_occupancy_grid(
    grid: np.ndarray,
    origin: np.ndarray,
    cell_size: float,
    path: str,
) -> None:
    """Save grid + metadata as a single .npy file (structured object array)."""
    np.save(path, {"grid": grid, "origin": origin, "cell_size": cell_size})


def visualise_occupancy_grid(
    grid: np.ndarray,
    origin: np.ndarray,
    cell_size: float,
    save_path: str | None = None,
) -> None:
    """
    Render the occupancy grid as a top-down architectural plan image.
    Walls are black, free space is white. Axis ticks are in metres.
    """
    H, W = grid.shape
    x_ticks = np.linspace(origin[0], origin[0] + W * cell_size, num=6)
    y_ticks = np.linspace(origin[1], origin[1] + H * cell_size, num=6)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(
        1 - grid,          # invert: walls black, free space white
        cmap="gray",
        origin="lower",
        extent=[origin[0], origin[0] + W * cell_size,
                origin[1], origin[1] + H * cell_size],
        interpolation="nearest",
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Occupancy Grid — top-down wall layout")
    ax.set_xticks(np.round(x_ticks, 2))
    ax.set_yticks(np.round(y_ticks, 2))
    ax.grid(True, color="lightblue", linewidth=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Occupancy grid image saved → {save_path}")
    else:
        plt.show()
    plt.close()
