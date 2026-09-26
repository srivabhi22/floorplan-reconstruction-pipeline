"""
stitch_render.py — Step 6
Render the stitched whole-property floor plan: all rooms, openings, labels.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch


ROOM_COLORS = [
    "#AED6F1", "#A9DFBF", "#F9E79F", "#F5CBA7",
    "#D2B4DE", "#A8D8EA", "#F0B27A", "#82E0AA",
]


def _polygon_centroid(vertices: list) -> tuple[float, float]:
    pts = np.array(vertices[:-1])   # drop closing duplicate
    return float(pts[:, 0].mean()), float(pts[:, 1].mean())


def render_stitched_plan(
    room_polygons: dict,
    room_metadata: dict,
    openings: dict,
    edges: list,
    footprint_area: float,
    save_path: str,
) -> None:
    """
    Draw all rooms, openings, room labels, wall lengths, and opening widths
    on one canvas and save as PNG.

    Args:
        room_polygons: dict room_id -> [[x,y], ...]  (closed)
        room_metadata: dict room_id -> {floor_area_m2, ceiling_height_m, wall_lengths_m}
        openings:      dict room_id -> list of opening dicts
        edges:         list of (room_a, room_b, edge_info) from adjacency
        footprint_area: total footprint m²
        save_path:     output PNG path
    """
    fig, ax = plt.subplots(figsize=(14, 14))
    ax.set_aspect("equal")

    color_map = {}
    for idx, (room_id, verts) in enumerate(room_polygons.items()):
        color = ROOM_COLORS[idx % len(ROOM_COLORS)]
        color_map[room_id] = color
        pts = np.array(verts)
        ax.fill(pts[:, 0], pts[:, 1], color=color, alpha=0.4, zorder=1)
        ax.plot(pts[:, 0], pts[:, 1], color="black", linewidth=1.5, zorder=2)

        # Wall length labels at each segment midpoint
        meta = room_metadata.get(room_id, {})
        wall_lengths = meta.get("wall_lengths_m", [])
        n = len(pts) - 1
        for i in range(min(n, len(wall_lengths))):
            mid = (pts[i] + pts[i + 1]) / 2
            ax.text(mid[0], mid[1], f"{wall_lengths[i]:.2f} m",
                    fontsize=6, ha="center", va="center",
                    color="dimgray", zorder=4,
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.6, ec="none"))

        # Room centroid label
        cx, cy = _polygon_centroid(verts)
        area    = meta.get("floor_area_m2", "?")
        ceiling = meta.get("ceiling_height_m", "?")
        label   = f"Room {room_id}\n{area:.1f} m²\nh={ceiling:.2f} m" if isinstance(area, float) else f"Room {room_id}"
        ax.text(cx, cy, label, fontsize=8, ha="center", va="center",
                fontweight="bold", zorder=5,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.75, ec="gray"))

    # Draw openings
    for room_id, room_openings in openings.items():
        for op in room_openings:
            pos = op["position"]
            w   = op.get("width_m", 0.9)
            otype = op.get("type", "door")

            if otype == "door":
                # Dashed gap marker
                ax.plot(pos[0], pos[1], "s", color="sienna", markersize=8, zorder=6)
                linestyle = "--"
            else:
                # Window: hatched marker
                ax.plot(pos[0], pos[1], "D", color="steelblue", markersize=7, zorder=6)
                linestyle = ":"

            ax.text(pos[0], pos[1] + 0.12, f"{w:.2f} m",
                    fontsize=6, ha="center", color="black", zorder=7)

    # Adjacency edges (faint lines between connected room centroids)
    for room_a, room_b, info in edges:
        if room_a not in room_polygons or room_b not in room_polygons:
            continue
        ca = _polygon_centroid(room_polygons[room_a])
        cb = _polygon_centroid(room_polygons[room_b])
        ax.plot([ca[0], cb[0]], [ca[1], cb[1]],
                color="gray", linewidth=0.8, linestyle="-.", alpha=0.5, zorder=3)

    ax.set_xlabel("X (m)", fontsize=10)
    ax.set_ylabel("Y (m)", fontsize=10)
    ax.set_title(f"Stitched Floor Plan  |  Total footprint: {footprint_area:.1f} m²", fontsize=12)

    # Legend
    legend_handles = [
        mpatches.Patch(color=color_map.get(rid, "white"), label=f"Room {rid}")
        for rid in room_polygons
    ]
    legend_handles += [
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="sienna",
                   markersize=8, label="Door"),
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="steelblue",
                   markersize=7, label="Window"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)
    ax.grid(True, linewidth=0.3, color="lightgray")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Rendered → {save_path}")
