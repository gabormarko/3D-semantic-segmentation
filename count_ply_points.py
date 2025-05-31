#!/usr/bin/env python3

import sys
import open3d as o3d

if __name__ == "__main__":
    if len(sys.argv) != 2:
         print("Usage: python count_ply_points.py <path_to_ply_file>")
         sys.exit(1)
    ply_path = sys.argv[1]
    pcd = o3d.io.read_point_cloud(ply_path)
    print("PLY file has", len(pcd.points), "points.") 