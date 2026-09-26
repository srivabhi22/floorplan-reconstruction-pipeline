"""
view.py — General point cloud viewer

Opens any .ply file and draws it with an origin axis frame (X=red, Y=green, Z=blue).

Usage:
    python view.py outputs/point_cloud.ply
    python view.py outputs/point_cloud_corrected.ply
"""

import sys
import open3d as o3d

path = sys.argv[1] if len(sys.argv) > 1 else "outputs/point_cloud.ply"

pcd    = o3d.io.read_point_cloud(path)
origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])

o3d.visualization.draw_geometries(
    [pcd, origin],
    window_name=path,
    width=1280,
    height=720,
)
