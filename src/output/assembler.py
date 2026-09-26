"""
assembler.py — Steps 2 & 3
Merge all intermediate module outputs into the unified floorplan.json schema
and generate human-readable scope line items.
"""

import uuid
from datetime import datetime, timezone

import numpy as np

from src.output.confidence import compute_ci, estimate_point_stats


def _wall_name(idx: int) -> str:
    return ["north", "east", "south", "west"][idx % 4]


def assemble(
    wall_data: dict,
    openings_data: dict,
    stitched_data: dict,
    floor_ceiling_data: dict,
    damage_data: dict,
    pcd_pts: np.ndarray,
    tier: str,
    capture_timestamp: str | None = None,
) -> dict:
    """
    Build the unified floorplan.json document.

    All intermediate dicts are the parsed contents of their respective JSON files.
    pcd_pts is the (N,3) numpy array from point_cloud_corrected.ply.
    """
    ts = capture_timestamp or datetime.now(timezone.utc).isoformat()
    per_room_ceiling = floor_ceiling_data.get("per_room_ceiling", {})

    rooms_out = []
    for room_id_str, room in wall_data["rooms"].items():
        verts        = room["vertices"]
        wall_lengths = room.get("wall_lengths_m", [])
        floor_area   = room.get("floor_area_m2")
        ceiling_h    = per_room_ceiling.get(room_id_str) or room.get("ceiling_height_m")

        # Centroid for point-density lookup
        pts_arr  = np.array(verts[:-1])
        centroid = pts_arr.mean(axis=0).tolist()

        # Per-wall confidence
        wall_lengths_ci = {}
        wall_lengths_named = {}
        for i, wl in enumerate(wall_lengths):
            name = _wall_name(i)
            n, hcf = estimate_point_stats(pcd_pts, centroid)
            wall_lengths_named[name] = round(wl, 4)
            wall_lengths_ci[name]    = compute_ci("wall_length", tier, n, hcf)

        # Floor area CI
        n, hcf = estimate_point_stats(pcd_pts, centroid, radius=2.0)
        floor_area_ci = compute_ci("floor_area", tier, n, hcf)

        # Ceiling CI
        ceiling_ci = compute_ci("ceiling_height", tier, n, hcf)

        # Openings for this room
        room_openings = openings_data.get("openings", {}).get(room_id_str, [])
        openings_out = []
        for op in room_openings:
            n_op, hcf_op = estimate_point_stats(pcd_pts, op["position"])
            openings_out.append({
                "type":                  op["type"],
                "width_m":               op["width_m"],
                "width_ci_m":            compute_ci("opening_width", tier, n_op, hcf_op),
                "sill_height_m":         op.get("sill_height_m"),
                "header_height_m":       op.get("header_height_m"),
                "position":              op["position"],
                "wall_segment":          _wall_name(op.get("wall_segment", 0)),
                "confidence_interval_m": op.get("confidence_interval_m"),
                "concealed":             op.get("concealed", False),
            })

        # Damage for this room
        room_damage = [d for d in damage_data.get("damage", [])
                       if str(d.get("room_id")) == room_id_str]
        damage_out = []
        for dmg in room_damage:
            n_d, hcf_d = estimate_point_stats(pcd_pts, dmg.get("centroid_world", centroid))
            damage_out.append({
                "surface":     dmg.get("surface"),
                "class":       dmg.get("class"),
                "area_m2":     dmg.get("area_m2"),
                "area_ci_m2":  compute_ci("damage_area", tier, n_d, hcf_d),
                "centroid":    dmg.get("centroid_world"),
                "confidence":  dmg.get("confidence"),
                "concealed":   dmg.get("concealed", False),
                "rule_fired":  dmg.get("rule_fired"),
            })

        rooms_out.append({
            "id":                   room_id_str,
            "walls":                verts,
            "floor_area_m2":        floor_area,
            "floor_area_ci_m2":     floor_area_ci,
            "ceiling_height_m":     ceiling_h,
            "ceiling_height_ci_m":  ceiling_ci,
            "wall_lengths_m":       wall_lengths_named,
            "wall_lengths_ci_m":    wall_lengths_ci,
            "openings":             openings_out,
            "damage":               damage_out,
        })

    # Stitched plan
    footprint_m2 = stitched_data.get("footprint_m2", 0)
    n_fp, hcf_fp = estimate_point_stats(pcd_pts, pcd_pts[:, :2].mean(axis=0).tolist(), radius=5.0)
    footprint_ci = compute_ci("footprint_area", tier, n_fp, hcf_fp)

    # Concealed damage flags (all concealed=True entries)
    concealed_flags = [
        {
            "room_id":    d.get("room_id"),
            "surface":    d.get("surface"),
            "rule_fired": d.get("rule_fired"),
        }
        for d in damage_data.get("damage", []) if d.get("concealed")
    ]

    # Scope line items (Step 3)
    scope_items = _generate_scope_items(rooms_out, concealed_flags)

    return {
        "capture_id":        str(uuid.uuid4()),
        "tier":              tier,
        "capture_timestamp": ts,
        "rooms":             rooms_out,
        "stitched_plan": {
            "adjacency":       stitched_data.get("adjacency", []),
            "footprint_m2":    footprint_m2,
            "footprint_ci_m2": footprint_ci,
        },
        "concealed_damage_flags": concealed_flags,
        "scope_line_items":       scope_items,
    }


def _generate_scope_items(rooms: list, concealed_flags: list) -> list[str]:
    """Step 3 — human-readable scope line items keyed to surfaces."""
    items = []

    for room in rooms:
        rid = room["id"]
        for dmg in room.get("damage", []):
            if dmg.get("concealed"):
                continue
            surface = dmg.get("surface", "unknown")
            cls     = dmg.get("class", "damage")
            area    = dmg.get("area_m2", 0)
            items.append(
                f"room_{rid}/{surface}: {area:.2f} m² {cls} — remediation required"
            )

    for flag in concealed_flags:
        rid     = flag.get("room_id", "?")
        surface = flag.get("surface", "unknown")
        rule    = flag.get("rule_fired", "anomaly")
        items.append(
            f"room_{rid}/{surface}: concealed anomaly detected ({rule}) — inspect surface"
        )

    return items
