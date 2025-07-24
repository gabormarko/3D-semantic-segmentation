import torch
import numpy as np
import open3d as o3d

# Load occupancy grid
data = torch.load("tensor_data.pt")
occ = data["occupancy_3D"]  # (Z, Y, X)

# Find all occupied voxel indices (z, y, x)
occupied = (occ > 0).nonzero()
occupied_indices = [(x, y, z) for z, y, x in occupied.tolist()]
occupied_indices = np.array(occupied_indices, dtype=np.int32)

# Get grid_origin and voxel_size from tensor_data.pt if available
grid_origin = data.get("grid_origin", np.zeros(3))
if isinstance(grid_origin, torch.Tensor):
    grid_origin = grid_origin.numpy()
voxel_size = float(data.get("voxel_size", 1.0))

# Convert to world coordinates
occupied_world = occupied_indices * voxel_size + grid_origin

# Create Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(occupied_world)
pcd.paint_uniform_color([0, 0, 1])  # Blue

# Save to PLY
o3d.io.write_point_cloud("occupancy3D_points_ascii.ply", pcd, write_ascii=True)
print(f"Saved all occupancy_3D points to occupancy3D_points_ascii.ply ({occupied_world.shape[0]} points)")
