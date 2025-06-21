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

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Standalone MinkowskiEngine voxel grid generator")
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
    return parser.parse_args()

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
        opacity = gaussians.get_opacity
        mask = (opacity > 0.8).squeeze()
        surface_points = surface_points[mask]
        colors = colors[mask]
        print(f"Filtered to {surface_points.shape[0]} high-opacity points.")

    # --- Estimate voxel size for ~10,000 voxels ---
    if torch.is_tensor(surface_points):
        min_corner = surface_points.min(dim=0)[0]
        max_corner = surface_points.max(dim=0)[0]
        bbox = max_corner - min_corner
        bbox_prod = bbox.prod().item()
    else:
        min_corner = np.min(surface_points, axis=0)
        max_corner = np.max(surface_points, axis=0)
        bbox = max_corner - min_corner
        bbox_prod = np.prod(bbox)
    target_voxels = 500000000*1/2 # Adjusted target voxels for larger scenes
    voxel_size = (bbox_prod / target_voxels)
    print(f"Auto-tuned voxel size for ~{target_voxels} voxels: {voxel_size:.3f}")

    if not isinstance(surface_points, torch.Tensor):
        surface_points_tensor = torch.from_numpy(np.asarray(surface_points)).to(device)
    else:
        surface_points_tensor = surface_points.to(device)
    if not isinstance(colors, torch.Tensor):
        colors = torch.from_numpy(np.asarray(colors)).to(device)
    else:
        colors = colors.to(device)
    print(f"[DEBUG] surface_points_tensor type: {type(surface_points_tensor)}, device: {surface_points_tensor.device}")
    print(f"[DEBUG] colors type: {type(colors)}, device: {colors.device}")
    print(f"[DEBUG] surface_points_tensor shape: {surface_points_tensor.shape}")
    print(f"[DEBUG] colors shape: {colors.shape}")
    minkowski_grid = MinkowskiVoxelGrid(surface_points_tensor, colors=colors, voxel_size=voxel_size, device=device)
    voxel_centers_raw = minkowski_grid.get_voxel_centers().detach().cpu().numpy()
    print(f"[DEBUG] voxel_centers_raw shape: {voxel_centers_raw.shape}")
    scene_name = os.path.basename(os.path.normpath(args.model_path))
    minkowski_base = f"{scene_name}_minkowski_{len(minkowski_grid)}vox_iter{args.iteration}"
    minkowski_points_path = os.path.join(args.output_dir, minkowski_base + "_points.ply")
    minkowski_grid_path = os.path.join(args.output_dir, minkowski_base + "_grid.ply")
    import open3d as o3d
    # Ensure voxel_centers is 2D, float64, and contiguous for Open3D
    voxel_centers = minkowski_grid.get_voxel_centers().detach().cpu().numpy()
    voxel_centers = np.asarray(voxel_centers)
    # Robust shape handling
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
    o3d.io.write_point_cloud(minkowski_points_path, pcd)
    print(f"Saved MinkowskiEngine voxel centers to {minkowski_points_path}")
    o3d.io.write_point_cloud(minkowski_grid_path, pcd)
    print(f"Saved MinkowskiEngine voxel grid to {minkowski_grid_path}")

if __name__ == "__main__":
    main()
