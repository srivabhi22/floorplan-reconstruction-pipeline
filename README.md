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
                    Point Cloud Construction       ← Module 1 (LiDAR done)
                                 │
                                 ▼
                    Drift Correction (plane-anchor pose graph)
                                 │
                                 ▼
                    Floor / Ceiling Detection (RANSAC)
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
  point_cloud.ply     # intermediate — fused point cloud (Module 1 output)
  floorplan.json      # final structured output
  floorplan.png       # rendered floor plan
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
torch
```

Install:
```bash
pip install -r requirements.txt
```
