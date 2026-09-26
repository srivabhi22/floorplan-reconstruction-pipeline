# Floorplan Pipeline

Produces a dimensioned floor plan from three input tiers: photos, video, or LiDAR. One command per capture, same output contract for all tiers.

---

## Pipeline Structure

```
Input
  │
  ├── Photos  ──► Monocular Depth (Depth Anything v2)
  │               + SfM Poses (COLMAP / SuperPoint + LightGlue)
  │                                │
  ├── Video   ──► Monocular Depth (Depth Anything v2)
  │               + Visual Odometry (DROID-SLAM / ORB-SLAM3)
  │                                │
  └── LiDAR   ──► Confidence Masking
                  + ARKit Poses (from odometry.csv)
                                 │
                                 ▼
                    Point Cloud Construction       ← Module 1 (done)
                                 │
                                 ▼
                    Drift Correction (plane-anchor pose graph) ← Module 2 (done)
                                 │
                                 ▼
                    Floor / Ceiling Detection (RANSAC)         ← Module 3 (done)
                    Wall Slice · Occupancy Grid · Room Clustering
                                 │
                                 ▼
                    Wall Polygon Fitting (rectilinear snap)    ← Module 4 (done)
                                 │
                                 ▼
                    Opening Detection (door / window)          ← Module 5 (done)
                                 │
                                 ▼
                    Multi-Room Stitching                       ← Module 6 (done)
                                 │
                                 ▼
                    Wall Slice at 1.0–1.5 m
                                 │
                                 ▼
                    Room Segmentation (occupancy grid + DBSCAN)
                                 │
                                 ▼
                    Wall Polygon Fitting (rectilinear snap)
                                 │
                                 ▼
                    Opening Detection (door / window classification)
                                 │
                                 ▼
                    Multi-Room Stitching
                                 │
                                 ▼
                    Damage Detection (SAM + classifier)
                                 │
                                 ▼
                    Confidence Interval Estimation
                                 │
                                 ▼
                    Output: JSON + Rendered Floor Plan
```

---

## Running the Pipeline

```bash
python pipeline.py --input <input_dir> --tier <photo|video|lidar> --output <output_dir>
```

---

## Module 1 — LiDAR Preprocessing

**Status: complete**

Takes the raw `/data` folder and produces a single fused, cleaned world-space point cloud.

### Steps

1. Parse `odometry.csv` — extract per-frame pose (x, y, z, quaternion) and intrinsics (fx, fy, cx, cy), build 4×4 world transform per frame
2. Load depth frames (`depth/XXXXXX.png`, 16-bit, values in mm) — convert to metres
3. Load confidence frames (`confidence/XXXXXX.png`) — mask out pixels with confidence < 1
4. Unproject each valid pixel to camera-space 3D point using per-frame intrinsics
5. Transform camera-space points to world space using the frame's pose matrix
6. Voxel downsample at 2 cm resolution (Open3D) to keep memory tractable
7. Statistical outlier removal (`nb_neighbors=20`, `std_ratio=2.0`)
8. Save fused point cloud as `outputs/point_cloud.ply`

### Run Module 1 standalone

```bash
python src/lidar/fuse.py --data_dir data/ --output_dir outputs/
```

### Source files

```
src/lidar/
  load_odometry.py    # parse odometry.csv, build transform matrices
  load_frames.py      # load and mask depth + confidence frames
  unproject.py        # pixel → camera space → world space
  fuse.py             # entry point: batched fusion, downsample, save
```

### Output

```
outputs/
  point_cloud.ply     # fused world-space point cloud
```

---

## Module 2 — Drift Correction

**Status: complete**

Takes the fused point cloud from Module 1 and corrects accumulated ARKit pose drift so the scene is globally consistent. Uses a pose-graph optimisation (scipy sparse least-squares, no external graph library required).

### Steps

1. Divide 1715 frames into overlapping segments (150 frames, 30-frame overlap) and crop the point cloud to each segment's camera-path bounding box
2. RANSAC plane detection per segment — fit floor (bottom 20 % of points), ceiling (top 20 %), and dominant wall (mid band)
3. Build pose graph — sequential edges between adjacent segments (weight 10, locally accurate) and plane-match edges between non-adjacent segments that observe the same physical surface (weight 5)
4. Optimise pose graph with weighted sparse least-squares — produces a 6-DOF correction delta per segment, anchored at segment 0
5. Reproject all depth frames using corrected poses (`T_corrected = C_seg @ T_arkit`), re-downsample and clean
6. Save corrected cloud, ablation pair (raw vs corrected), corrected poses CSV, and print alignment report

### Run Module 2 standalone

```bash
python src/reconstruction/drift_correction.py \
  --point_cloud outputs/point_cloud.ply \
  --data_dir data/ \
  --output_dir outputs/
```

### Source files

```
src/reconstruction/
  segment.py          # split cloud into overlapping frame-range chunks
  plane_detection.py  # RANSAC floor / ceiling / wall per segment
  pose_graph.py       # build graph nodes and sequential + plane-match edges
  optimize.py         # scipy sparse least-squares solver, delta → 4×4 transform
  reproject.py        # reproject all frames with corrected poses
  ablation.py         # save ablation outputs, compute drift metrics
  drift_correction.py # entry point
```

### Outputs

```
outputs/
  point_cloud_corrected.ply      # drift-corrected world-space point cloud
  corrected_poses.csv            # optimized per-frame 4×4 transforms (T_00…T_33)
  drift_ablation/
    raw.ply                      # uncorrected cloud (for ablation)
    corrected.ply                # corrected cloud (for ablation)
```

### Ablation report (printed at end of run)

```
── Drift Correction Ablation Report ─────────────────────────
  Pose graph edges   : N sequential, M plane-match
  Floor plane error  : X.X cm  →  Y.Y cm
  Closure error      : Z.Z cm  (first→last frame distance)
  Loop detected      : yes / no
─────────────────────────────────────────────────────────────
```

---

## Module 3 — Room Segmentation

**Status: complete**

Takes the drift-corrected point cloud and produces a labelled 2D room layout with floor/ceiling heights.

### Steps

1. RANSAC on the lowest 20 % of points → `z_floor`; normalize all z so floor = 0
2. RANSAC on the top 20 % of normalized points → global `z_ceiling`
3. Filter to `1.0 m ≤ z_norm ≤ 1.5 m` — the wall-slice band that cuts through all walls cleanly
4. Project wall-slice points onto XY plane → 2D binary occupancy grid at 2 cm/cell
5. DBSCAN (`eps=5 cm`, `min_samples=10`) on occupied cells → one cluster per room; clusters < 50 cells discarded as noise
6. Per-room ceiling refinement: re-run RANSAC on points above 1.8 m within each room's 2D bounding box

### Run Module 3 standalone

```bash
python src/reconstruction/segment_pipeline.py \
  --point_cloud outputs/point_cloud_corrected.ply \
  --output_dir outputs/
```

### Source files

```
src/reconstruction/
  floor_ceiling.py      # RANSAC floor / ceiling detection, per-room ceiling
  wall_slice.py         # mid-height band extraction (1.0–1.5 m)
  occupancy_grid.py     # 2D binary grid + world↔grid coordinate helpers
  room_segmentation.py  # DBSCAN room clustering, point labelling
  segment_pipeline.py   # entry point
```

### Outputs

```
outputs/
  floor_ceiling.json         # floor z, global ceiling height, per-room ceiling heights
  wall_slice.ply             # mid-height point cloud (view with view.py)
  occupancy_grid.npy         # binary grid + origin / cell_size metadata
  occupancy_grid.png         # top-down plan image (walls black, free space white)
  room_segments.npz          # wall-slice point indices + room label per point
```

---

## Module 4 — Wall Polygon Fitting

**Status: complete**

Takes per-room occupancy grids from Module 3 and produces clean, metric, rectilinear wall polygons for each room.

### Steps

1. Per-room binary mask → `cv2.findContours` → raw metric boundary polygon
2. PCA on boundary edge vectors → dominant wall angle `theta` (handles non-axis-aligned buildings)
3. Rotate by `-theta`, snap each segment to nearest axis (H or V, threshold 10°)
4. Merge collinear consecutive segments → remove grid staircase artifacts
5. Reconstruct ordered rectilinear polygon, intersect segments at corners, rotate back by `+theta`
6. Shoelace formula → floor area; Euclidean distances → per-wall lengths
7. Render all rooms on one canvas with area, ceiling height, and wall-length labels

### Run Module 4 standalone

```bash
python src/reconstruction/wall_fitting.py \
  --occupancy_grid outputs/occupancy_grid.npy \
  --room_segments outputs/room_segments.npz \
  --floor_ceiling outputs/floor_ceiling.json \
  --output_dir outputs/
```

### Source files

```
src/reconstruction/
  boundary_extraction.py    # Step 1 — cv2 contour → metric boundary per room
  orientation_detection.py  # Step 2 — PCA dominant wall angle
  segment_snapping.py       # Steps 3+4 — axis snap + collinear merge
  polygon_builder.py        # Step 5 — rectilinear polygon assembly
  measurements.py           # Step 6 — wall lengths, floor area, validation
  render.py                 # Step 7 — matplotlib floor plan render
  wall_fitting.py           # entry point
```

### Outputs

```
outputs/
  wall_polygons.json         # per-room vertices, wall lengths, floor area, ceiling height
  wall_polygons.png          # rendered top-down floor plan with labels
```

### wall_polygons.json schema

```json
{
  "rooms": {
    "0": {
      "vertices": [[x, y], "..."],
      "wall_lengths_m": [3.2, 4.1, "..."],
      "floor_area_m2": 12.4,
      "ceiling_height_m": 2.61,
      "warnings": []
    }
  }
}
```

---

## Module 5 — Opening Detection

**Status: complete**

Detects doors, windows, and archways in wall polygon edges, measures widths with sub-centimetre refinement, and flags concealed (glass) openings.

### Steps

1. Sample each wall polygon edge against the occupancy grid → runs of empty cells = gap candidates; filter to 0.5–4.0 m width range
2. Refine gap width by fitting lines to 3D wall-slice points on each side of the gap (sub-2 cm precision)
3. Classify opening type from vertical point-density profile above the gap midpoint: door (floor → 1.9–2.1 m), window (sill 0.7–1.2 m → header 1.6–2.4 m), archway (tall)
4. Flag low-density gaps as concealed (glass) openings
5. Save `openings.json` with type, width, sill/header height, wall segment, and confidence interval

### Run Module 5 standalone

```bash
python src/reconstruction/opening_detection.py \
  --wall_polygons outputs/wall_polygons.json \
  --occupancy_grid outputs/occupancy_grid.npy \
  --point_cloud outputs/point_cloud_corrected.ply \
  --output_dir outputs/
```

### Source files

```
src/reconstruction/
  gap_detection.py        # Steps 1+2 — grid gap finding, 3D width refinement
  opening_classifier.py   # Steps 3+4 — vertical profile classification, CI
  opening_detection.py    # entry point
```

### Outputs

```
outputs/
  openings.json           # per-room list of openings with type, width, position, CI
```

### openings.json schema (per opening)

```json
{
  "room_id": "0",
  "type": "door",
  "width_m": 0.91,
  "sill_height_m": 0.0,
  "header_height_m": 2.05,
  "position": [x, y],
  "wall_segment": 2,
  "confidence_interval_m": 0.012,
  "concealed": false
}
```

---

## Module 6 — Multi-Room Stitching

**Status: complete**

Composes all per-room polygons and openings into a single whole-property floor plan with adjacency graph, overlap resolution, and global footprint.

### Steps

1. All room polygons are already in ARKit world coordinates — no relative transform needed
2. Match openings across rooms by position proximity (≤ 10 cm) → adjacency graph (nodes = rooms, edges = shared openings with type and width)
3. Detect polygon overlaps via Shapely intersection; resolve with minimal rigid translation anchored at the shared opening position; flag shifts > 5 cm
4. Flag any room with no adjacency edges as disconnected
5. `shapely.unary_union` of all room polygons → total footprint area m²
6. Render all rooms, openings (door = brown square, window = blue diamond), wall lengths, room labels, and adjacency edges on one canvas

### Run Module 6 standalone

```bash
python src/reconstruction/stitching.py \
  --wall_polygons outputs/wall_polygons.json \
  --openings outputs/openings.json \
  --output_dir outputs/
```

### Source files

```
src/reconstruction/
  adjacency.py          # Steps 2+4 — opening matching, disconnected room detection
  overlap_resolver.py   # Step 3 — Shapely intersection check + translation fix
  footprint.py          # Step 5 — Shapely unary_union, total area
  stitch_render.py      # Step 6 — whole-property floor plan render
  stitching.py          # entry point
```

### Outputs

```
outputs/
  stitched_plan.json    # all rooms, adjacency graph, footprint area, overlap log
  stitched_plan.png     # rendered whole-property floor plan
```

### stitched_plan.json top-level keys

```json
{
  "rooms": { "0": { "vertices": [...], "floor_area_m2": 12.4, ... } },
  "adjacency": [{ "room_a": "0", "room_b": "1", "type": "door", "width_m": 0.91 }],
  "disconnected_rooms": [],
  "footprint_m2": 38.6,
  "footprint_boundary": [[x, y], "..."],
  "overlap_corrections": []
}
```

---

## Viewer

View any `.ply` file with a world-space origin frame (X=red, Y=green, Z=blue):

```bash
python view.py outputs/point_cloud.ply
python view.py outputs/point_cloud_corrected.ply
```

---

## Input Format

### Photo Tier

```
input_dir/
  room_1/
    img_001.jpg
    img_002.jpg
    ...               # 2–8 images per room
  room_2/
    img_001.jpg
    ...
  camera_matrix.csv   # optional — read from device if available
```

### Video Tier

```
input_dir/
  walkthrough.mp4
  camera_matrix.csv   # optional
```

### LiDAR Tier

```
input_dir/
  rgb.mp4
  odometry.csv
  imu.csv
  camera_matrix.csv
  depth/
    000000.png        # 16-bit depth map per frame (mm)
    ...
  confidence/
    000000.png        # 0=low, 1=medium, 2=high
    ...
```

**`odometry.csv` columns:**
```
timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy, distortion_center_x, distortion_center_y
```

---

## Output Format

```
outputs/
  point_cloud.ply              # Module 1 — fused point cloud
  point_cloud_corrected.ply    # Module 2 — drift-corrected point cloud
  corrected_poses.csv          # Module 2 — optimized per-frame transforms
  drift_ablation/
    raw.ply                    # uncorrected cloud (ablation)
    corrected.ply              # corrected cloud (ablation)
  floor_ceiling.json           # Module 3 — floor z, ceiling heights per room
  wall_slice.ply               # Module 3 — mid-height wall cross-section
  occupancy_grid.npy           # Module 3 — 2D binary grid + metadata
  occupancy_grid.png           # Module 3 — top-down plan image
  room_segments.npz            # Module 3 — point indices + room labels
  wall_polygons.json           # Module 4 — per-room polygons + measurements
  wall_polygons.png            # Module 4 — rendered floor plan
  openings.json                # Module 5 — detected openings with widths and types
  stitched_plan.json           # Module 6 — full property plan with adjacency graph
  stitched_plan.png            # Module 6 — rendered whole-property floor plan
  floorplan.json               # final structured output
  floorplan.png                # rendered floor plan
```

**`floorplan.json` schema (per room):**
```json
{
  "rooms": [
    {
      "id": "room_1",
      "walls": [[x1,y1], [x2,y2], "..."],
      "floor_area_m2": 12.4,
      "ceiling_height_m": 2.61,
      "openings": [
        {
          "type": "door",
          "width_m": 0.91,
          "position": [x, y],
          "confidence_interval_m": 0.015
        }
      ],
      "damage": [
        {
          "class": "water_stain",
          "surface": "north_wall",
          "area_m2": 0.3
        }
      ],
      "measurements": {
        "wall_lengths_m": {"north": 3.2, "east": 4.1, "south": 3.2, "west": 4.1},
        "confidence_intervals_m": {"north": 0.01, "east": 0.01, "south": 0.01, "west": 0.01}
      }
    }
  ],
  "stitched_plan": {
    "adjacency": [["room_1", "room_2", "door"]],
    "footprint_m2": 38.6
  },
  "tier": "lidar",
  "capture_timestamp": "2026-09-26T10:00:00Z"
}
```

---

## Device Matrix

| Tier   | Hardware              | Depth Source                   | Pose Source     | Expected Wall Accuracy      |
|--------|-----------------------|--------------------------------|-----------------|-----------------------------|
| Photo  | Any Android / iPhone  | Depth Anything v2 (estimated)  | SfM (COLMAP)    | ±8% with calibrated CI      |
| Video  | Any Android / iPhone  | Depth Anything v2 (estimated)  | Visual Odometry | ±3% with calibrated CI      |
| LiDAR  | iPhone Pro (sample)   | ARKit LiDAR depth              | ARKit poses     | ±1 cm with calibrated CI    |

---

## Dependencies

```
numpy
pandas
opencv-python
scipy
open3d
scikit-learn
matplotlib
shapely
python-dotenv
torch
```

Install:
```bash
uv pip install -r requirements.txt
```
