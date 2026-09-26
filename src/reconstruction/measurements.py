"""measurements.py — Step 6: wall lengths, floor area, polygon validation."""

import numpy as np


def wall_lengths(polygon: np.ndarray) -> list[float]:
    """Euclidean length of each wall segment (excluding closing duplicate vertex)."""
    verts = polygon[:-1] if len(polygon) > 1 else polygon
    n = len(verts)
    return [float(np.linalg.norm(verts[(i + 1) % n] - verts[i])) for i in range(n)]


def floor_area(polygon: np.ndarray) -> float:
    """Shoelace formula area in m²."""
    verts = polygon[:-1] if len(polygon) > 1 else polygon
    x, y = verts[:, 0], verts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def validate_polygon(
    polygon: np.ndarray,
    area_min: float = 5.0,
    area_max: float = 50.0,
    angle_tol_deg: float = 5.0,
) -> list[str]:
    """Return list of warning strings; empty = all checks passed."""
    warnings = []
    verts = polygon[:-1] if len(polygon) > 1 else polygon
    n = len(verts)

    area = floor_area(polygon)
    if area < area_min:
        warnings.append(f"Floor area {area:.2f} m² below minimum {area_min} m²")
    if area > area_max:
        warnings.append(f"Floor area {area:.2f} m² above maximum {area_max} m²")

    for i in range(n):
        p0 = verts[(i - 1) % n]
        p1 = verts[i]
        p2 = verts[(i + 1) % n]
        v1 = p0 - p1
        v2 = p2 - p1
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm1 < 1e-9 or norm2 < 1e-9:
            continue
        cos_a = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cos_a))
        if abs(angle_deg - 90.0) > angle_tol_deg:
            warnings.append(
                f"Vertex {i}: interior angle {angle_deg:.1f}° deviates from 90° "
                f"by {abs(angle_deg - 90.0):.1f}°"
            )

    return warnings
