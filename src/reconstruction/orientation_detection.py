"""orientation_detection.py — Step 2: PCA dominant wall angle detection."""

import numpy as np
from sklearn.decomposition import PCA


def compute_edge_vectors(boundary_pts: np.ndarray) -> np.ndarray:
    """Return (N,2) edge direction vectors between consecutive boundary points."""
    return np.diff(boundary_pts, axis=0)


def detect_dominant_angle(boundary_pts: np.ndarray) -> float:
    """Return angle (radians) of the first PCA component of edge vectors, or 0.0 if too few points."""
    if len(boundary_pts) < 4:
        return 0.0

    edges = compute_edge_vectors(boundary_pts)
    # Mirror edges so anti-parallel edges don't cancel in PCA
    edges = np.vstack([edges, -edges])

    lengths = np.linalg.norm(edges, axis=1, keepdims=True)
    edges = edges[lengths.squeeze() > 1e-9]
    if len(edges) < 2:
        return 0.0

    pca = PCA(n_components=1).fit(edges)
    v = pca.components_[0]
    return float(np.arctan2(v[1], v[0]))
