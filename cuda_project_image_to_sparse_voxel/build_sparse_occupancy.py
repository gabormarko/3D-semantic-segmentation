"""
Script to build a dense [Z,Y,X] occupancy tensor from a Mitsu voxel-grid .ply.
Each occupied voxel receives a unique 1-based ID, empty cells = 0.
Usage:
  python build_sparse_occupancy.py \
    --voxel_ply /path/to/grid.ply \
    --voxel_size 0.05 \
    --grid_origin 0 0 0 \
    --out_tensor occupancy.pt
"""
import argparse
import numpy as np
import torch
from plyfile import PlyData

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--voxel_ply', required=True, help='path to Mitsu grid .ply')
    p.add_argument('--voxel_size', type=float, required=True, help='voxel size')
    p.add_argument('--grid_origin', nargs=3, type=float, default=[0,0,0], help='grid origin (x y z)')
    p.add_argument('--out_tensor', required=True, help='output path for occupancy tensor (.pt)')
    args = p.parse_args()

    print(f"Reading PLY via plyfile: {args.voxel_ply}")
    plydata = PlyData.read(args.voxel_ply)
    vertex = plydata['vertex']
    pts = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T.astype(np.float32)
    print(f"Loaded {pts.shape[0]} points from PLY")

    origin = np.array(args.grid_origin, dtype=np.float32)
    print(f"[DEBUG] Using grid_origin for occupancy: {origin}")
    coords = np.round((pts - origin) / args.voxel_size).astype(np.int64)
    print(f"Computed integer voxel coords, sample: {coords[:5]}")

    # Compute grid dimensions
    min_coord = coords.min(axis=0)
    if not np.all(min_coord >= 0):
        print(f"Warning: negative min coords {min_coord}, will offset to zero")
        coords -= min_coord
    max_coord = coords.max(axis=0)
    dims = max_coord + 1  # sizes along x,y,z
    print(f"Grid dims (x,y,z): {dims}")
    # Assign unique IDs to occupied voxels
    occ = np.zeros(dims[::-1], dtype=np.int32)  # [Z,Y,X]
    for i, c in enumerate(coords):
        occ[tuple(c[::-1])] = i + 1  # 1-based
    print(f"Assigning IDs to {coords.shape[0]} voxels")
    print(f"Occupancy tensor shape: {occ.shape}")
    print(f"Max ID: {occ.max()}")
    # Debug: print only the number of occupied voxels
    num_occupied = (occ > 0).sum()
    print(f"Number of occupied voxels: {num_occupied}")
    torch.save(torch.from_numpy(occ), args.out_tensor)
    print(f"Saved occupancy tensor to {args.out_tensor}")

if __name__ == '__main__':
    main()
