import torch
import numpy as np
import open3d as o3d

# Load dummy occupancy grid
dummy_occ = torch.load("dummy_occupancy.pt")  # (Z, Y, X)

# Find all occupied voxel indices (z, y, x)
occupied = (dummy_occ > 0).nonzero()
occupied_indices = [(x, y, z) for z, y, x in occupied.tolist()]
occupied_indices = np.array(occupied_indices, dtype=np.int32)

# Use grid_origin and voxel_size for a simple test grid
# (Assume origin at (0,0,0) and voxel_size=1.0 for dummy grid)
grid_origin = np.zeros(3)
voxel_size = 1.0

# Convert to world coordinates
occupied_world = occupied_indices * voxel_size + grid_origin

# Create Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(occupied_world)
pcd.paint_uniform_color([1, 0, 0])  # Red for dummy

# Save to PLY
o3d.io.write_point_cloud("dummy_occupancy_points_ascii.ply", pcd, write_ascii=True)
print(f"Saved all dummy_occupancy.pt points to dummy_occupancy_points_ascii.ply ({occupied_world.shape[0]} points)")
