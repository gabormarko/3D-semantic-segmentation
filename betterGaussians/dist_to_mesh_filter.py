import numpy as np
import argparse
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree

def load_ply_xyz(ply_path):
    ply = PlyData.read(ply_path)
    vertex = ply['vertex']
    xyz = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1)
    return xyz, vertex

def save_ply_xyz(vertex, mask, out_path):
    filtered_vertex = vertex[mask]
    PlyData([PlyElement.describe(filtered_vertex, 'vertex')]).write(out_path)

def main():
    parser = argparse.ArgumentParser(description="Filter Gaussians by distance to closest voxel")
    parser.add_argument('--gaussian_ply', type=str, required=True, help='Input Gaussian .ply file')
    parser.add_argument('--voxel_ply', type=str, required=True, help='Input voxel grid .ply file')
    parser.add_argument('--max_dist', type=float, required=True, help='Max allowed distance to closest voxel')
    parser.add_argument('--out_ply', type=str, required=True, help='Output filtered Gaussian .ply file')
    args = parser.parse_args()

    # Load Gaussian centers
    gauss_xyz, gauss_vertex = load_ply_xyz(args.gaussian_ply)
    # Load voxel centers
    voxel_xyz, _ = load_ply_xyz(args.voxel_ply)

    # Build KD-tree for fast nearest neighbor search
    tree = cKDTree(voxel_xyz)
    dists, _ = tree.query(gauss_xyz, k=1)
    mask = dists < args.max_dist

    print(f"Filtered {np.sum(mask)} / {len(mask)} Gaussians within {args.max_dist} of a voxel.")

    # Save filtered Gaussians
    save_ply_xyz(gauss_vertex, mask, args.out_ply)
    print(f"Saved filtered Gaussians to {args.out_ply}")

if __name__ == "__main__":
    main()