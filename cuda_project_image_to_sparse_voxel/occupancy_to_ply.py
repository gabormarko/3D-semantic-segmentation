import torch
import numpy as np
from plyfile import PlyData, PlyElement

# Load occupancy tensor
occ = torch.load("occupancy.pt")
if isinstance(occ, dict) and 'occupancy_3D' in occ:
    occ = occ['occupancy_3D']
occ_np = occ.cpu().numpy() if hasattr(occ, 'cpu') else occ.numpy()

# Get occupied indices (z, y, x)
occupied = np.stack(np.nonzero(occ_np), axis=1)
print(f"Found {occupied.shape[0]} occupied voxels.")

# Try to get grid_origin and voxel_size from tensor_data.pt if available
try:
    data = torch.load("tensor_data.pt")
    grid_origin = data.get("grid_origin", np.zeros(3))
    if isinstance(grid_origin, torch.Tensor):
        grid_origin = grid_origin.numpy()
    voxel_size = float(data.get("voxel_size", 1.0))
    print(f"Loaded grid_origin from tensor_data.pt: {grid_origin}")
    print(f"Loaded voxel_size from tensor_data.pt: {voxel_size}")
except Exception as e:
    print(f"Could not load tensor_data.pt: {e}")
    grid_origin = np.zeros(3)
    voxel_size = 1.0

# Convert to (x, y, z) and world coordinates
occupied_indices = np.array([(x, y, z) for z, y, x in occupied], dtype=np.float32)
occupied_world = occupied_indices * voxel_size + grid_origin

# Prepare PLY vertex array
vertex = np.array([
    (x, y, z) for x, y, z in occupied_world
], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

el = PlyElement.describe(vertex, 'vertex')
PlyData([el], text=True).write('occupancy_voxels_ascii.ply')
print("Wrote occupancy_voxels_ascii.ply with all occupied voxel centers.")
