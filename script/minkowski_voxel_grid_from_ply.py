import os
import sys
import numpy as np
from scipy.spatial import cKDTree
import argparse
from plyfile import PlyData

def parse_args():
    parser = argparse.ArgumentParser(description="Minkowski voxel grid generator from .ply file with density filtering")
    parser.add_argument("--ply", required=True, help="Input .ply file with Gaussian properties")
    parser.add_argument("--output_dir", default="output/minkowski_grid", help="Output directory for visualizations")
    parser.add_argument("--cell_size", type=float, default=0.05, help="Size of voxel grid cells")
    parser.add_argument("--density_eps", type=float, default=0.05, help="Epsilon radius for density filtering")
    parser.add_argument("--density_min_neighbors", type=int, default=10, help="Minimum neighbors for density filtering")
    parser.add_argument("--opacity_threshold", type=float, default=0.9, help="Minimum opacity for a gaussian to be considered part of the surface.")
    return parser.parse_args()

def filter_by_local_density(points, eps=0.1, min_neighbors=100):
    tree = cKDTree(points)
    counts = tree.query_ball_point(points, r=eps, return_length=True)
    mask = counts > min_neighbors
    return mask

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    ply = PlyData.read(args.ply)
    vertex = ply['vertex']
    xyz = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1)
    # Extract RGB from f_dc_0, f_dc_1, f_dc_2
    r = np.clip(vertex['f_dc_0'], 0, 1) * 255
    g = np.clip(vertex['f_dc_1'], 0, 1) * 255
    b = np.clip(vertex['f_dc_2'], 0, 1) * 255
    colors = np.stack([r, g, b], axis=1).astype(np.uint8)
    # Extract opacity
    opacity = vertex['opacity']
    # Rank-based opacity filtering: keep top (1-opacity_threshold) fraction by opacity
    keep_fraction = 1.0 - args.opacity_threshold  # e.g. 0.8 means keep top 20%%
    N_total = opacity.shape[0]
    N_keep = int(N_total * keep_fraction)
    if N_keep < 1:
        N_keep = 1
    # Get indices of top N_keep opacities
    top_indices = np.argpartition(opacity, -N_keep)[-N_keep:]
    # Sort indices for consistent output
    top_indices = top_indices[np.argsort(-opacity[top_indices])]
    xyz = xyz[top_indices]
    colors = colors[top_indices]
    opacity = opacity[top_indices]
    print(f"[INFO] Filtered to {xyz.shape[0]} points with top {int(100*keep_fraction)}%% opacity (N={N_keep} of {N_total})")
    # Density filtering
    print(f"[INFO] Filtering by local density: eps={args.density_eps}, min_neighbors={args.density_min_neighbors}")
    density_mask = filter_by_local_density(xyz, eps=args.density_eps, min_neighbors=args.density_min_neighbors)
    filtered_points = xyz[density_mask]
    filtered_colors = colors[density_mask]
    print(f"[INFO] Kept {filtered_points.shape[0]} / {xyz.shape[0]} gaussians after density filtering.")
    # --- Sparse voxel grid creation (after opacity and density filtering) ---
    min_corner = np.min(filtered_points, axis=0)
    voxel_size = args.cell_size
    print(f"Using specified voxel size: {voxel_size:.6f}")
    voxel_indices = np.floor((filtered_points - min_corner) / voxel_size).astype(int)
    unique_indices, inverse = np.unique(voxel_indices, axis=0, return_inverse=True)
    voxel_centers = unique_indices * voxel_size + min_corner + voxel_size / 2.0
    print(f"[INFO] Sparse voxel grid: {voxel_centers.shape[0]} voxels")
    voxel_colors = np.zeros((voxel_centers.shape[0], 3), dtype=np.float32)
    for i in range(voxel_centers.shape[0]):
        pts_in_voxel = (inverse == i)
        if np.any(pts_in_voxel):
            voxel_colors[i] = np.mean(filtered_colors[pts_in_voxel], axis=0)
        else:
            voxel_colors[i] = [127, 127, 127]
    voxel_colors = voxel_colors.astype(np.uint8)
    # Export sparse voxel grid as PLY
    voxel_grid_data = np.zeros(voxel_centers.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    voxel_grid_data['x'] = voxel_centers[:, 0]
    voxel_grid_data['y'] = voxel_centers[:, 1]
    voxel_grid_data['z'] = voxel_centers[:, 2]
    voxel_grid_data['red'] = voxel_colors[:, 0]
    voxel_grid_data['green'] = voxel_colors[:, 1]
    voxel_grid_data['blue'] = voxel_colors[:, 2]
    ply_parts = args.ply.split(os.sep)
    try:
        scene_name = ply_parts[ply_parts.index('point_cloud') - 1]
    except Exception:
        scene_name = ply_parts[-3] if len(ply_parts) > 3 else ply_parts[0]
    ply_base = os.path.splitext(os.path.basename(args.ply))[0]
    import re
    iter_match = re.search(r'iteration_(\d+)', ply_base)
    iter_str = iter_match.group(1) if iter_match else ""
    params_str = f"_minkowski_{voxel_centers.shape[0]}vox_iter{iter_str}_opac{args.opacity_threshold}_cell{args.cell_size}_eps{args.density_eps}_neig{args.density_min_neighbors}_grid"
    voxel_grid_name = f"{scene_name}{params_str}.ply"
    voxel_grid_path = os.path.join(args.output_dir, voxel_grid_name)
    from plyfile import PlyElement
    PlyData([PlyElement.describe(voxel_grid_data, 'vertex')]).write(voxel_grid_path)
    print(f"[INFO] Saved sparse voxel grid to {voxel_grid_path}")

if __name__ == "__main__":
    main()
