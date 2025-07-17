import os
import sys

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import numpy as np
import torch
from scene import Scene, GaussianModel
from arguments import ModelParams
from utils.hash_grid import MinkowskiVoxelGrid
from scipy.spatial import cKDTree

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Standalone MinkowskiEngine voxel grid generator with density filtering")
    parser.add_argument("--model_path", required=True, help="Path to the model directory")
    parser.add_argument("--iteration", type=int, default=-1, help="Model iteration to load (-1 for latest)")
    parser.add_argument("--source_path", default="", help="Path to the source data")
    parser.add_argument("--images", default="", help="Path to the images")
    parser.add_argument("--eval", action="store_true", help="Evaluation mode")
    parser.add_argument("--object_path", default="", help="Path to object data")
    parser.add_argument("--n_views", type=int, default=0, help="Number of views")
    parser.add_argument("--random_init", action="store_true", help="Random initialization")
    parser.add_argument("--train_split", type=float, default=0.8, help="Training split ratio")
    parser.add_argument("--cell_size", type=float, default=0.05, help="Size of voxel grid cells")
    parser.add_argument("--output_dir", default="output/minkowski_grid", help="Output directory for visualizations")
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
    gaussians = GaussianModel(0)
    if args.iteration == -1:
        from utils.system_utils import searchForMaxIteration
        chkpnt_dir = os.path.join(args.model_path, "chkpnts")
        args.iteration = searchForMaxIteration(chkpnt_dir)
        checkpoint_path = os.path.join(chkpnt_dir, f"iteration_{args.iteration}.pth")
    else:
        chkpnt_dir = os.path.join(args.model_path, "chkpnts")
        checkpoint_path = os.path.join(chkpnt_dir, f"iteration_{args.iteration}.pth")
    import argparse
    model_parser = argparse.ArgumentParser()
    model_params = ModelParams(model_parser)
    model_params.model_path = args.model_path
    model_params.source_path = args.source_path
    model_params.images = args.images
    model_params.eval = args.eval
    model_params.object_path = args.object_path
    model_params.n_views = args.n_views
    model_params.random_init = args.random_init
    model_params.train_split = args.train_split
    scene = Scene(model_params, gaussians, load_iteration=args.iteration)
    surface_points = gaussians.get_xyz
    device = surface_points.device if hasattr(surface_points, 'device') else ('cuda' if torch.cuda.is_available() else 'cpu')
    # Use DC color if available, else gray
    try:
        colors = gaussians._features_dc.detach().cpu().numpy()  # shape [N, 1, 3]
        if colors.shape[1] == 1 and colors.shape[2] == 3:
            colors = colors[:, 0, :]  # Now shape [N, 3]
        else:
            print("Warning: Unexpected DC color shape, using gray.")
            colors = np.ones_like(surface_points.cpu().numpy()) * 0.5
        colors = np.clip(colors, 0, 1)
        colors = torch.from_numpy(colors).to(device)
    except Exception as e:
        print(f"Warning: Could not extract DC color ({{e}}). Exported points will be gray.")
        colors = torch.ones_like(surface_points) * 0.5
    # --- Filter for dense surface regions (if available) ---
    if hasattr(gaussians, "get_opacity"):
        print(f"[INFO] Original point count: {surface_points.shape[0]}")
        print(f"[INFO] Filtering by opacity > {args.opacity_threshold}")
        opacity = gaussians.get_opacity
        mask = (opacity > args.opacity_threshold).squeeze()
        surface_points = surface_points[mask]
        colors = colors[mask]
        print(f"Filtered to {surface_points.shape[0]} high-opacity points.")
        # Export filtered high-opacity points as PLY
        import open3d as o3d
        high_op_ply_path = os.path.join(args.output_dir, f"high_opacity_points_opac{args.opacity_threshold}_iter{args.iteration}.ply")
        points_np = surface_points.detach().cpu().numpy() if torch.is_tensor(surface_points) else np.asarray(surface_points)
        colors_np = colors.detach().cpu().numpy() if torch.is_tensor(colors) else np.asarray(colors)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_np)
        if colors_np.shape[1] == 3:
            pcd.colors = o3d.utility.Vector3dVector(np.clip(colors_np, 0, 1))
        else:
            pcd.colors = o3d.utility.Vector3dVector(np.ones_like(points_np))
        o3d.io.write_point_cloud(high_op_ply_path, pcd, write_ascii=True)
        print(f"[INFO] Saved high-opacity points to {high_op_ply_path}")
    # --- Density filtering ---
    print(f"[INFO] Filtering by local density: eps={args.density_eps}, min_neighbors={args.density_min_neighbors}")
    # CHANGE: Use numpy array for means
    means = surface_points.detach().cpu().numpy() if torch.is_tensor(surface_points) else np.asarray(surface_points)
    density_mask = filter_by_local_density(means, eps=args.density_eps, min_neighbors=args.density_min_neighbors)
    filtered_points = means[density_mask]
    filtered_colors = colors.cpu().numpy()[density_mask] if torch.is_tensor(colors) else colors[density_mask]
    print(f"[INFO] Kept {filtered_points.shape[0]} / {means.shape[0]} gaussians after density filtering.")
    # --- Set voxel size ---
    min_corner = np.min(filtered_points, axis=0)
    max_corner = np.max(filtered_points, axis=0)
    voxel_size = args.cell_size
    print(f"Using specified voxel size: {voxel_size:.6f}")

    # Debug print for grid center
    grid_center = np.mean([min_corner, max_corner], axis=0)
    print(f"[DEBUG] grid center: {grid_center}")
    surface_points_tensor = torch.from_numpy(filtered_points).to(device)
    colors_tensor = torch.from_numpy(filtered_colors).to(device)
    print(f"[DEBUG] surface_points_tensor type: {type(surface_points_tensor)}, device: {surface_points_tensor.device}")
    print(f"[DEBUG] colors_tensor type: {type(colors_tensor)}, device: {colors_tensor.device}")
    print(f"[DEBUG] surface_points_tensor shape: {surface_points_tensor.shape}")
    print(f"[DEBUG] colors_tensor shape: {colors_tensor.shape}")
    minkowski_grid = MinkowskiVoxelGrid(surface_points_tensor, colors=colors_tensor, voxel_size=voxel_size, device=device)
    voxel_centers_raw = minkowski_grid.get_voxel_centers().detach().cpu().numpy()
    print(f"[DEBUG] voxel_centers_raw shape: {voxel_centers_raw.shape}")
    # Debug print for actual voxel size if available
    if hasattr(minkowski_grid, 'voxel_size'):
        print(f"[DEBUG] MinkowskiVoxelGrid.voxel_size: {getattr(minkowski_grid, 'voxel_size', voxel_size)}")
    else:
        print(f"[DEBUG] Used voxel_size: {voxel_size}")
    # Debug print for grid shape if available
    grid_shape_dbg = None
    if hasattr(minkowski_grid, 'grid_shape'):
        grid_shape_dbg = getattr(minkowski_grid, 'grid_shape', None)
        print(f"[DEBUG] grid_shape: {grid_shape_dbg}")
    scene_name = os.path.basename(os.path.normpath(args.model_path))
    
    # Create a more descriptive filename
    params_str = f"_opac{args.opacity_threshold}_cell{args.cell_size}_eps{args.density_eps}_neig{args.density_min_neighbors}"
    minkowski_base = f"{scene_name}_minkowski_{len(minkowski_grid)}vox_iter{args.iteration}{params_str}"
    
    minkowski_points_path = os.path.join(args.output_dir, minkowski_base + "_filt_points.ply")
    minkowski_grid_path = os.path.join(args.output_dir, minkowski_base + "_grid.ply")
    import open3d as o3d
    voxel_centers = minkowski_grid.get_voxel_centers().detach().cpu().numpy()
    voxel_centers = np.asarray(voxel_centers)
    if voxel_centers.ndim == 1:
        if voxel_centers.size == 0:
            print("[Warning] No voxel centers to export.")
            return
        if voxel_centers.size % 3 != 0:
            print(f"[Error] Unexpected voxel_centers size: {voxel_centers.size}. Cannot reshape to (N,3). Skipping export.")
            return
        voxel_centers = voxel_centers.reshape(-1, 3)
    elif voxel_centers.ndim == 2 and voxel_centers.shape[1] != 3:
        print(f"[Error] Unexpected voxel_centers shape: {voxel_centers.shape}. Skipping export.")
        return
    if not voxel_centers.flags['C_CONTIGUOUS']:
        voxel_centers = np.ascontiguousarray(voxel_centers)
    voxel_centers = voxel_centers.astype(np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(voxel_centers)
    feats = minkowski_grid.get_features().cpu().numpy()
    if feats.shape[1] == 3:
        feats = np.asarray(feats)
        if feats.ndim != 2 or feats.shape[1] != 3:
            feats = feats.reshape(-1, 3)
        if not feats.flags['C_CONTIGUOUS']:
            feats = np.ascontiguousarray(feats)
        feats = feats.astype(np.float64)
        pcd.colors = o3d.utility.Vector3dVector(np.clip(feats, 0, 1))
    else:
        pcd.colors = o3d.utility.Vector3dVector(np.ones_like(voxel_centers))
    # o3d.io.write_point_cloud(minkowski_points_path, pcd)
    # print(f"Saved MinkowskiEngine voxel centers to {minkowski_points_path}")
    # Save voxel grid with header comments for voxel_size and grid_origin
    grid_origin = min_corner
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
    grid_shape = None
    if hasattr(minkowski_grid, 'grid_shape'):
        grid_shape = getattr(minkowski_grid, 'grid_shape', None)
        if isinstance(grid_shape, (list, tuple, np.ndarray)) and len(grid_shape) == 3:
            grid_shape = tuple(int(x) for x in grid_shape)
        else:
            grid_shape = None
    write_ply_with_comments(
        minkowski_grid_path,
        voxel_centers,
        feats if feats.shape[1] == 3 else np.ones_like(voxel_centers),
        voxel_size,
        grid_origin,
        grid_shape=grid_shape
    )
    print(f"Saved MinkowskiEngine voxel grid to {minkowski_grid_path} (with voxel_size, grid_origin, and grid_shape in header)")

if __name__ == "__main__":
    main()
