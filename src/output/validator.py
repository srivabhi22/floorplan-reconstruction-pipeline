"""
validator.py — Step 4
Validate the assembled floorplan.json against the published schema using jsonschema.
Prints a gate-relevant checklist and fails loudly on missing required fields.
"""

import jsonschema

FLOORPLAN_SCHEMA = {
    "type": "object",
    "required": ["capture_id", "tier", "capture_timestamp", "rooms",
                 "stitched_plan", "scope_line_items"],
    "properties": {
        "capture_id":        {"type": "string"},
        "tier":              {"type": "string", "enum": ["lidar", "video", "photo"]},
        "capture_timestamp": {"type": "string"},
        "rooms": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "walls", "floor_area_m2", "ceiling_height_m",
                             "wall_lengths_m", "openings", "damage"],
                "properties": {
                    "id":                  {"type": "string"},
                    "walls":               {"type": "array"},
                    "floor_area_m2":       {"type": ["number", "null"]},
                    "floor_area_ci_m2":    {"type": ["number", "null"]},
                    "ceiling_height_m":    {"type": ["number", "null"]},
                    "ceiling_height_ci_m": {"type": ["number", "null"]},
                    "wall_lengths_m":      {"type": "object"},
                    "wall_lengths_ci_m":   {"type": "object"},
                    "openings":            {"type": "array"},
                    "damage":              {"type": "array"},
                },
            },
        },
        "stitched_plan": {
            "type": "object",
            "required": ["footprint_m2"],
            "properties": {
                "footprint_m2":    {"type": "number"},
                "footprint_ci_m2": {"type": ["number", "null"]},
                "adjacency":       {"type": "array"},
            },
        },
        "concealed_damage_flags": {"type": "array"},
        "scope_line_items":       {"type": "array"},
    },
}

# Gate thresholds for the benchmark report
GATES = {
    "floor_area_m2_range":     (5.0, 200.0),
    "ceiling_height_m_range":  (2.0,   3.5),
    "wall_length_min_m":        0.5,
    "opening_width_range_m":   (0.5,   3.0),
}


def validate_schema(doc: dict) -> None:
    """Raise jsonschema.ValidationError if the document fails schema validation."""
    jsonschema.validate(instance=doc, schema=FLOORPLAN_SCHEMA)


def gate_checklist(doc: dict) -> list[dict]:
    """
    Return a checklist of gate-relevant fields with pass/warn/fail status.
    Each entry: {field, value, status, note}
    """
    results = []

    for room in doc.get("rooms", []):
        rid = room["id"]

        # Floor area
        fa = room.get("floor_area_m2")
        lo, hi = GATES["floor_area_m2_range"]
        results.append({
            "field": f"room_{rid}.floor_area_m2",
            "value": fa,
            "status": "pass" if fa and lo <= fa <= hi else "warn",
            "note": f"expected {lo}–{hi} m²",
        })

        # Ceiling height
        ch = room.get("ceiling_height_m")
        lo, hi = GATES["ceiling_height_m_range"]
        results.append({
            "field": f"room_{rid}.ceiling_height_m",
            "value": ch,
            "status": "pass" if ch and lo <= ch <= hi else "warn",
            "note": f"expected {lo}–{hi} m",
        })

        # Wall lengths
        for wall_name, wl in room.get("wall_lengths_m", {}).items():
            status = "pass" if wl >= GATES["wall_length_min_m"] else "warn"
            results.append({
                "field":  f"room_{rid}.wall_lengths_m.{wall_name}",
                "value":  wl,
                "status": status,
                "note":   f"min {GATES['wall_length_min_m']} m",
            })

        # Openings
        lo, hi = GATES["opening_width_range_m"]
        for op in room.get("openings", []):
            w = op.get("width_m")
            results.append({
                "field":  f"room_{rid}.opening.{op.get('wall_segment')}.width_m",
                "value":  w,
                "status": "pass" if w and lo <= w <= hi else "warn",
                "note":   f"expected {lo}–{hi} m",
            })

    # Footprint
    fp = doc.get("stitched_plan", {}).get("footprint_m2")
    results.append({
        "field":  "stitched_plan.footprint_m2",
        "value":  fp,
        "status": "pass" if fp and fp > 0 else "fail",
        "note":   "must be > 0",
    })

    return results


def print_checklist(checklist: list[dict]) -> None:
    print("\n── Gate Checklist ───────────────────────────────────────────────")
    for item in checklist:
        icon = {"pass": "✓", "warn": "⚠", "fail": "✗"}.get(item["status"], "?")
        print(f"  {icon} {item['field']:50s} {str(item['value']):>10}  ({item['note']})")
    fails = sum(1 for i in checklist if i["status"] == "fail")
    warns = sum(1 for i in checklist if i["status"] == "warn")
    print(f"\n  {len(checklist)} checks  —  {fails} fail(s)  {warns} warn(s)")
    print("────────────────────────────────────────────────────────────────")
