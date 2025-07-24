import torch
import open3d as o3d
import numpy as np

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

# Find the minimum x value among occupied voxels
min_x = occupied_indices[:, 0].min()
min_x_indices = occupied_indices[occupied_indices[:, 0] == min_x]
min_x_world = min_x_indices * voxel_size + grid_origin

print(f"Occupied voxels with smallest x={min_x} (x, y, z) and world coordinates:")
for idx_xyz, world_xyz in zip(min_x_indices, min_x_world):
    print(f"  idx: (x={idx_xyz[0]}, y={idx_xyz[1]}, z={idx_xyz[2]})  world: ({world_xyz[0]:.4f}, {world_xyz[1]:.4f}, {world_xyz[2]:.4f})")

occupied_xyz = occupied_indices.astype(np.float32)

# Load existing PLY
pcd = o3d.io.read_point_cloud("combined_frustum_with_ray_line_ascii.ply")
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

# Add blue points for occupied voxels (use all occupied, not just N)
blue = np.array([[0, 0, 1]] * occupied_xyz.shape[0], dtype=np.float32)

# Add the origin as a big pink point
origin_point = grid_origin.reshape(1, 3)
pink = np.array([[1.0, 0.0, 1.0]], dtype=np.float32)

# Combine all points and colors
all_points = np.vstack([points, occupied_xyz, origin_point])
all_colors = np.vstack([colors, blue, pink])

# Save new PLY
pcd.points = o3d.utility.Vector3dVector(all_points)
pcd.colors = o3d.utility.Vector3dVector(all_colors)
o3d.io.write_point_cloud("combined_frustum_with_ray_line_and_voxels_ascii.ply", pcd, write_ascii=True)

print("Added all occupied voxels as blue points and the origin as a pink point to combined_frustum_with_ray_line_and_voxels_ascii.ply")
