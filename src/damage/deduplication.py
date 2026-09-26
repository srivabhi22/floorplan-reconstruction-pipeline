"""
deduplication.py — Step 7
Cluster 3D damage detections that are within 10 cm of each other and share
the same class. Merge into one entry, keeping the largest area estimate.
"""

import numpy as np


def deduplicate_detections(
    detections: list[dict],
    position_tol: float = 0.10,
) -> list[dict]:
    """
    Merge detections with the same class whose world-space centroids are within
    position_tol metres of each other.

    Args:
        detections: list of detection dicts, each with keys:
                    class, centroid_world ([x,y,z]), area_m2, room_id, surface,
                    confidence, concealed, rule_fired.
        position_tol: merge radius in metres.

    Returns:
        Deduplicated list of detection dicts.
    """
    if not detections:
        return []

    used   = [False] * len(detections)
    merged = []

    for i, det_i in enumerate(detections):
        if used[i]:
            continue
        group = [det_i]
        used[i] = True
        ci = np.array(det_i["centroid_world"])

        for j, det_j in enumerate(detections):
            if used[j] or det_j["class"] != det_i["class"]:
                continue
            cj = np.array(det_j["centroid_world"])
            if np.linalg.norm(ci - cj) <= position_tol:
                group.append(det_j)
                used[j] = True

        # Keep the detection with the largest area as the representative
        best = max(group, key=lambda d: d.get("area_m2", 0))
        # Sum area across all merged detections for a better estimate
        total_area = sum(d.get("area_m2", 0) for d in group)
        best = dict(best)
        best["area_m2"] = round(total_area, 4)
        merged.append(best)

    return merged
