"""segment_snapping.py — Steps 3 & 4: axis-snap and collinear-merge of boundary segments."""

import numpy as np


def rotate_points(pts: np.ndarray, angle: float) -> np.ndarray:
    """Rotate (N,2) points by angle radians about the origin."""
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])
    return (R @ pts.T).T


def snap_segments(
    boundary_pts: np.ndarray,
    theta: float,
    snap_thresh_deg: float = 10.0,
) -> list[dict]:
    """Rotate by -theta, classify each edge as H or V, snap coords to dominant axis."""
    rotated = rotate_points(boundary_pts, -theta)
    thresh_rad = np.radians(snap_thresh_deg)
    segments = []

    n = len(rotated)
    for i in range(n):
        p1 = rotated[i]
        p2 = rotated[(i + 1) % n]
        dx, dy = p2 - p1
        angle_from_h = abs(np.arctan2(abs(dy), abs(dx)))

        if angle_from_h <= thresh_rad:
            seg_type = 'H'
            y_mid = (p1[1] + p2[1]) / 2
            p1 = np.array([p1[0], y_mid])
            p2 = np.array([p2[0], y_mid])
        else:
            seg_type = 'V'
            x_mid = (p1[0] + p2[0]) / 2
            p1 = np.array([x_mid, p1[1]])
            p2 = np.array([x_mid, p2[1]])

        segments.append({'type': seg_type, 'p1': p1, 'p2': p2})

    return segments


def merge_collinear(segments: list[dict], tol: float = 0.02) -> list[dict]:
    """Merge consecutive same-type segments that share nearly the same axis coordinate."""
    if not segments:
        return []

    merged = []
    run = [segments[0]]

    def axis_coord(seg):
        return seg['p1'][1] if seg['type'] == 'H' else seg['p1'][0]

    def flush(run):
        if not run:
            return None
        t = run[0]['type']
        # Snap axis coordinate to mean of the run
        coord = float(np.mean([axis_coord(s) for s in run]))
        if t == 'H':
            xs = [run[0]['p1'][0]] + [s['p2'][0] for s in run]
            p1 = np.array([min(xs), coord])
            p2 = np.array([max(xs), coord])
        else:
            ys = [run[0]['p1'][1]] + [s['p2'][1] for s in run]
            p1 = np.array([coord, min(ys)])
            p2 = np.array([coord, max(ys)])
        return {'type': t, 'p1': p1, 'p2': p2}

    for seg in segments[1:]:
        prev = run[-1]
        same_type = seg['type'] == prev['type']
        same_axis = abs(axis_coord(seg) - axis_coord(prev)) <= tol
        if same_type and same_axis:
            run.append(seg)
        else:
            merged.append(flush(run))
            run = [seg]

    merged.append(flush(run))
    return [m for m in merged if m is not None]
