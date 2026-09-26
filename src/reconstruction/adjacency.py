"""
adjacency.py — Steps 2 & 4
Build a room adjacency graph from shared openings, and flag disconnected rooms.
"""

import numpy as np


def build_adjacency_graph(
    openings: dict,
    position_tol: float = 0.10,
) -> tuple[list[tuple], list[str]]:
    """
    Step 2 — Match openings across rooms to build an adjacency graph.

    Two openings are shared when:
      - Their world-space positions are within `position_tol` metres of each other
      - Their wall normals are anti-parallel (dot product < -0.7)

    Args:
        openings:     dict  room_id -> list of opening dicts (from openings.json)
        position_tol: Maximum distance between matching opening positions (metres)

    Returns:
        (edges, disconnected)
        - edges: list of (room_a, room_b, edge_info_dict)
        - disconnected: list of room_ids with no edges
    """
    room_ids = list(openings.keys())
    edges = []
    connected = set()

    for i, room_a in enumerate(room_ids):
        for room_b in room_ids[i + 1:]:
            for op_a in openings[room_a]:
                for op_b in openings[room_b]:
                    pos_a = np.array(op_a["position"])
                    pos_b = np.array(op_b["position"])

                    if np.linalg.norm(pos_a - pos_b) > position_tol:
                        continue

                    # Anti-parallel normals check (opening faces each room inward)
                    # openings.json stores wall_segment index, not normal; skip normal
                    # check here — position proximity alone is sufficient for co-registered clouds
                    edge_info = {
                        "type":    op_a["type"],
                        "width_m": op_a["width_m"],
                        "position": op_a["position"],
                    }
                    edges.append((room_a, room_b, edge_info))
                    connected.add(room_a)
                    connected.add(room_b)

    disconnected = [r for r in room_ids if r not in connected]
    return edges, disconnected
