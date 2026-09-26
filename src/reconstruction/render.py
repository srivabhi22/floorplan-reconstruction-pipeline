"""render.py — Step 7: render all room polygons as a labelled floor plan PNG."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection


_PALETTE = [
    "#AED6F1", "#A9DFBF", "#F9E79F", "#F1948A",
    "#D7BDE2", "#A3E4D7", "#FAD7A0", "#D5DBDB",
]


def render_floor_plan(
    room_polygons: dict,
    room_metadata: dict,
    save_path: str,
    floor_ceiling_json_path: str | None = None,
) -> None:
    """Render all rooms on one canvas with area, ceiling height, and wall length labels."""
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_aspect("equal")

    for idx, (room_id, polygon) in enumerate(room_polygons.items()):
        if len(polygon) < 3:
            continue

        color = _PALETTE[idx % len(_PALETTE)]
        verts = polygon[:-1] if len(polygon) > 1 else polygon

        patch = MplPolygon(verts, closed=True, facecolor=color, edgecolor="black",
                           linewidth=1.5, alpha=0.7)
        ax.add_patch(patch)

        # Room centroid label
        cx, cy = verts[:, 0].mean(), verts[:, 1].mean()
        meta = room_metadata.get(room_id, {})
        area = meta.get("area_m2", 0.0)
        ceil_h = meta.get("ceiling_height", None)
        h_str = f"\nh={ceil_h:.2f} m" if ceil_h else ""
        ax.text(cx, cy, f"Room {room_id}\n{area:.1f} m²{h_str}",
                ha="center", va="center", fontsize=8, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6))

        # Wall length labels at segment midpoints
        lengths = meta.get("wall_lengths", [])
        n = len(verts)
        for i, wl in enumerate(lengths):
            if wl < 0.05:
                continue
            mid = (verts[i] + verts[(i + 1) % n]) / 2
            ax.text(mid[0], mid[1], f"{wl:.2f} m",
                    ha="center", va="center", fontsize=6, color="#2C3E50",
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.5))

    ax.autoscale_view()
    ax.set_xlabel("X (m)", fontsize=10)
    ax.set_ylabel("Y (m)", fontsize=10)
    ax.set_title("Floor Plan", fontsize=14, fontweight="bold")
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
