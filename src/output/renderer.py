"""
renderer.py — Step 5
Render the final annotated floor plan: walls, openings, damage overlays,
room labels, north arrow, and scale bar.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrow


ROOM_COLORS = [
    "#D6EAF8", "#D5F5E3", "#FEF9E7", "#FDEDEC",
    "#F4ECF7", "#EAF2FF", "#FDF2E9", "#E8F8F5",
]

DAMAGE_COLORS = {
    "water_stain":   "#E67E22",
    "crack":         "#E74C3C",
    "mould":         "#27AE60",
    "peeling_paint": "#8E44AD",
    "efflorescence": "#2980B9",
    "concealed":     "#7F8C8D",
}


def _polygon_centroid(verts: list) -> np.ndarray:
    pts = np.array(verts[:-1])
    return pts.mean(axis=0)


def _draw_north_arrow(ax, x, y, size=0.3):
    ax.annotate("N", xy=(x, y + size), xytext=(x, y),
                fontsize=9, ha="center", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))


def _draw_scale_bar(ax, x, y, length_m=1.0):
    ax.plot([x, x + length_m], [y, y], color="black", lw=3, solid_capstyle="butt")
    ax.plot([x, x], [y - 0.05, y + 0.05], color="black", lw=2)
    ax.plot([x + length_m, x + length_m], [y - 0.05, y + 0.05], color="black", lw=2)
    ax.text(x + length_m / 2, y - 0.12, f"{length_m:.0f} m",
            ha="center", va="top", fontsize=7)


def render_final_plan(doc: dict, save_path: str) -> None:
    """
    Render the full annotated floor plan from the assembled floorplan.json document.
    """
    fig, ax = plt.subplots(figsize=(16, 16))
    ax.set_aspect("equal")

    all_x, all_y = [], []

    for idx, room in enumerate(doc.get("rooms", [])):
        rid   = room["id"]
        verts = np.array(room["walls"])
        color = ROOM_COLORS[idx % len(ROOM_COLORS)]

        ax.fill(verts[:, 0], verts[:, 1], color=color, alpha=0.5, zorder=1)
        ax.plot(verts[:, 0], verts[:, 1], color="black", linewidth=1.8, zorder=2)
        all_x.extend(verts[:, 0].tolist())
        all_y.extend(verts[:, 1].tolist())

        # Wall length labels
        wall_lengths = room.get("wall_lengths_m", {})
        n = len(verts) - 1
        names = list(wall_lengths.keys())
        for i in range(n):
            mid = (verts[i] + verts[i + 1]) / 2
            name = names[i] if i < len(names) else str(i)
            wl   = list(wall_lengths.values())[i] if i < len(wall_lengths) else None
            label = f"{wl:.2f} m" if wl else name
            ax.text(mid[0], mid[1], label, fontsize=6, ha="center", va="center",
                    color="dimgray", zorder=4,
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.7, ec="none"))

        # Room label
        cx, cy = _polygon_centroid(room["walls"])
        area    = room.get("floor_area_m2")
        ceiling = room.get("ceiling_height_m")
        lines   = [f"Room {rid}"]
        if area:    lines.append(f"{area:.1f} m²")
        if ceiling: lines.append(f"h={ceiling:.2f} m")
        ax.text(cx, cy, "\n".join(lines), fontsize=8, ha="center", va="center",
                fontweight="bold", zorder=5,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.8, ec="gray"))

        # Openings
        for op in room.get("openings", []):
            pos   = op["position"]
            w     = op.get("width_m", 0.9)
            otype = op.get("type", "door")
            marker = "s" if otype == "door" else "D"
            color_op = "sienna" if otype == "door" else "steelblue"
            ls = "--" if otype == "door" else ":"
            ax.plot(pos[0], pos[1], marker, color=color_op, markersize=9, zorder=6)
            ax.text(pos[0], pos[1] + 0.13, f"{w:.2f} m",
                    fontsize=6, ha="center", color=color_op, zorder=7)

        # Damage overlays
        for dmg in room.get("damage", []):
            centroid = dmg.get("centroid")
            if not centroid:
                continue
            cls   = dmg.get("class", "damage")
            area  = dmg.get("area_m2", 0)
            dcol  = DAMAGE_COLORS.get(cls, "#E74C3C")
            radius = max(0.1, float(np.sqrt(area / np.pi)))
            circle = plt.Circle((centroid[0], centroid[1]), radius,
                                 color=dcol, alpha=0.35, zorder=3)
            ax.add_patch(circle)
            ax.text(centroid[0], centroid[1], cls.replace("_", "\n"),
                    fontsize=5, ha="center", va="center", color=dcol, zorder=8)

    # North arrow and scale bar (bottom-left corner)
    if all_x:
        x_min, y_min = min(all_x), min(all_y)
        x_max, y_max = max(all_x), max(all_y)
        span_y = y_max - y_min
        _draw_north_arrow(ax, x_min - 0.5, y_min + 0.6)
        _draw_scale_bar(ax, x_min, y_min - 0.4)

    # Legend
    legend_handles = [mpatches.Patch(color=c, label=cls.replace("_", " ").title())
                      for cls, c in DAMAGE_COLORS.items()]
    legend_handles += [
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="sienna",
                   markersize=9, label="Door"),
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="steelblue",
                   markersize=8, label="Window"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=7, title="Legend")

    footprint = doc.get("stitched_plan", {}).get("footprint_m2", 0)
    tier      = doc.get("tier", "lidar").upper()
    ax.set_title(f"Floor Plan  [{tier}]  —  Total footprint: {footprint:.1f} m²",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("X (m)", fontsize=10)
    ax.set_ylabel("Y (m)", fontsize=10)
    ax.grid(True, linewidth=0.3, color="lightgray", zorder=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Rendered → {save_path}")
