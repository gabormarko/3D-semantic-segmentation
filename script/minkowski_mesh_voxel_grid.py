
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import argparse
import torch
from utils.hash_grid import MinkowskiVoxelGrid

def parse_args():
    parser = argparse.ArgumentParser(description="Sparse voxel grid from mesh surface")
    parser.add_argument("--mesh_file", required=True, help="Path to input mesh PLY file")
    parser.add_argument("--output_dir", default="output/minkowski_mesh_grid", help="Output directory")
    parser.add_argument("--cell_size", type=float, default=0.05, help="Voxel grid cell size")
    parser.add_argument("--density_eps", type=float, default=0.05, help="Epsilon for density filtering")
    parser.add_argument("--density_min_neighbors", type=int, default=10, help="Min neighbors for density filtering")
    return parser.parse_args()

def filter_by_local_density(points, eps=0.1, min_neighbors=100):
    tree = cKDTree(points)
    counts = tree.query_ball_point(points, r=eps, return_length=True)
    mask = counts > min_neighbors
    return mask

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[INFO] Loading mesh: {args.mesh_file}")
    mesh = o3d.io.read_triangle_mesh(args.mesh_file)
    if not mesh.has_vertices():
        print("[ERROR] Mesh has no vertices.")
        return
    # Sample points near the surface
    print("[INFO] Sampling points near mesh surface...")
    points = mesh.sample_points_poisson_disk(number_of_points=200000)
    surface_points = np.asarray(points.points)
    print(f"[INFO] Sampled {surface_points.shape[0]} points.")
    # Optionally use mesh vertex colors if available
    if mesh.has_vertex_colors():
        mesh_colors = np.asarray(mesh.vertex_colors)
        # Map sampled points to nearest mesh vertex for color
        tree = cKDTree(np.asarray(mesh.vertices))
        _, idx = tree.query(surface_points)
        colors = mesh_colors[idx]
    else:
        colors = np.ones_like(surface_points) * 0.5
    # Density filtering
    print(f"[INFO] Filtering by local density: eps={args.density_eps}, min_neighbors={args.density_min_neighbors}")
    density_mask = filter_by_local_density(surface_points, eps=args.density_eps, min_neighbors=args.density_min_neighbors)
    filtered_points = surface_points[density_mask]
    filtered_colors = colors[density_mask]
    print(f"[INFO] Kept {filtered_points.shape[0]} / {surface_points.shape[0]} points after density filtering.")
    # Voxel grid creation
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    surface_points_tensor = torch.from_numpy(filtered_points).to(device)
    colors_tensor = torch.from_numpy(filtered_colors).to(device)
    print(f"[INFO] Creating MinkowskiVoxelGrid...")
    minkowski_grid = MinkowskiVoxelGrid(surface_points_tensor, colors=colors_tensor, voxel_size=args.cell_size, device=device)
    voxel_centers = minkowski_grid.get_voxel_centers().detach().cpu().numpy()
    feats = minkowski_grid.get_features().cpu().numpy()
    # Save voxel grid as PLY
    # Determine number of voxels for filename
    num_voxels = voxel_centers.shape[0]
    ply_path = os.path.join(args.output_dir, f"mesh_minkowski_grid_vox{num_voxels}.ply")
    # Save voxel grid with header comments for voxel_size and grid_origin
    min_corner = np.min(filtered_points, axis=0)
    grid_origin = min_corner
    grid_shape = None
    if hasattr(minkowski_grid, 'grid_shape'):
        grid_shape = getattr(minkowski_grid, 'grid_shape', None)
        if isinstance(grid_shape, (list, tuple, np.ndarray)) and len(grid_shape) == 3:
            grid_shape = tuple(int(x) for x in grid_shape)
        else:
            grid_shape = None
    def write_ply_with_comments(filename, points, colors, voxel_size, grid_origin, grid_shape=None):
        with open(filename, 'wb') as f:
            header = (
                f"ply\n"
                f"format binary_little_endian 1.0\n"
                f"comment voxel_size {voxel_size}\n"
                f"comment grid_origin {grid_origin[0]} {grid_origin[1]} {grid_origin[2]}\n"
                + (f"comment grid_shape {grid_shape[0]} {grid_shape[1]} {grid_shape[2]}\n" if grid_shape is not None else "")
                + f"element vertex {points.shape[0]}\n"
                f"property double x\n"
                f"property double y\n"
                f"property double z\n"
                f"property uchar red\n"
                f"property uchar green\n"
                f"property uchar blue\n"
                f"end_header\n"
            )
            f.write(header.encode('utf-8'))
            xyz = points.astype(np.float64)
            rgb = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
            for i in range(xyz.shape[0]):
                f.write(xyz[i].tobytes())
                f.write(rgb[i].tobytes())
    write_ply_with_comments(
        ply_path,
        voxel_centers,
        feats if feats.shape[1] == 3 else np.ones_like(voxel_centers),
        args.cell_size,
        grid_origin,
        grid_shape=grid_shape
    )
    print(f"[INFO] Saved sparse voxel grid to {ply_path} (with voxel_size, grid_origin, and grid_shape in header)")

if __name__ == "__main__":
    main()
