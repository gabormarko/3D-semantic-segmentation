import os
import sys
import time
import numpy as np

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import torch
import argparse
from tqdm import tqdm

from utils.surface_detection import SurfaceDetector
from utils.hash_grid import HashGrid, MinkowskiVoxelGrid
from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams
from utils.system_utils import searchForMaxIteration

def parse_args():
    # First create a parser for ModelParams to get its default values
    model_parser = argparse.ArgumentParser()
    ModelParams(model_parser)
    model_defaults = vars(model_parser.parse_args([]))
    
    # Now create our main parser
    parser = argparse.ArgumentParser(description="Test hash grid implementation with surface detection")
    
    # Create argument groups
    model_group = parser.add_argument_group("Model Parameters")
    grid_group = parser.add_argument_group("Hash Grid Parameters")
    io_group = parser.add_argument_group("Input/Output Parameters")
    
    # Model parameters (using defaults from ModelParams)
    model_group.add_argument("--model_path", required=True, help="Path to the model directory")
    model_group.add_argument("--iteration", type=int, default=-1, help="Model iteration to load (-1 for latest)")
    model_group.add_argument("--source_path", default=model_defaults.get("source_path", ""), help="Path to the source data")
    model_group.add_argument("--images", default=model_defaults.get("images", ""), help="Path to the images")
    model_group.add_argument("--eval", action="store_true", default=model_defaults.get("eval", False), help="Evaluation mode")
    model_group.add_argument("--object_path", default=model_defaults.get("object_path", ""), help="Path to object data")
    model_group.add_argument("--n_views", type=int, default=model_defaults.get("n_views", 0), help="Number of views")
    model_group.add_argument("--random_init", action="store_true", default=model_defaults.get("random_init", False), help="Random initialization")
    model_group.add_argument("--train_split", type=float, default=model_defaults.get("train_split", 0.8), help="Training split ratio")
    
    # Hash grid parameters
    grid_group.add_argument("--cell_size", type=float, default=0.05, help="Size of hash grid cells")
    grid_group.add_argument("--hash_size", type=int, default=2**20, help="Size of hash table")
    grid_group.add_argument("--max_points_per_cell", type=int, default=32, help="Maximum points per cell")
    grid_group.add_argument("--test_queries", type=int, default=1000, help="Number of test queries to perform")
    grid_group.add_argument("--reg_grid", action="store_true", help="Use regular (structured) grid instead of adaptive hash grid")
    grid_group.add_argument("--target_voxel_count", type=int, default=50000, help="Target number of voxels for regular (structured) grid")
    
    # I/O parameters
    io_group.add_argument("--output_dir", default="output/hash_grid", help="Output directory for visualizations")
    io_group.add_argument("--save_ply", action="store_true", help="Save visualization as PLY files")
    
    # Add new arguments
    parser.add_argument("--query_batch_size", type=int, default=100,
                      help="Number of queries to process in each batch")    
    parser.add_argument("--minkowski", action="store_true", help="Only run the MinkowskiEngine voxel grid output and skip other processing.")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    if args.minkowski:
        # Minimal model/scene loading for MinkowskiEngine output only
        gaussians = GaussianModel(0)
        if args.iteration == -1:
            args.iteration = searchForMaxIteration(os.path.join(args.model_path, "point_cloud"))
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
        # Use all points for Minkowski grid
        surface_points = gaussians.get_xyz
        # Use DC color if available, else white
        device = surface_points.device if hasattr(surface_points, 'device') else ('cuda' if torch.cuda.is_available() else 'cpu')
        if hasattr(gaussians, "get_features_dc"):
            colors = gaussians.get_features_dc.detach()
            if colors.shape[1] == 1 and colors.shape[-1] == 3:
                colors = colors[:, 0, :]
            else:
                colors = torch.ones_like(surface_points)  # fallback
        else:
            colors = torch.ones_like(surface_points)
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
        minkowski_grid = MinkowskiVoxelGrid(surface_points_tensor, colors=colors, voxel_size=args.cell_size, device=device)
        scene_name = os.path.basename(os.path.normpath(args.model_path))
        minkowski_base = f"{scene_name}_minkowski_{len(minkowski_grid)}vox_iter{args.iteration}"
        minkowski_points_path = os.path.join(args.output_dir, minkowski_base + "_points.ply")
        minkowski_grid_path = os.path.join(args.output_dir, minkowski_base + "_grid.ply")
        import open3d as o3d
        voxel_centers = minkowski_grid.get_voxel_centers().cpu().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(voxel_centers)
        feats = minkowski_grid.get_features().cpu().numpy()
        if feats.shape[1] == 3:
            pcd.colors = o3d.utility.Vector3dVector(np.clip(feats, 0, 1))
        else:
            pcd.colors = o3d.utility.Vector3dVector(np.ones_like(voxel_centers))
        o3d.io.write_point_cloud(minkowski_points_path, pcd)
        print(f"Saved MinkowskiEngine voxel centers to {minkowski_points_path}")
        o3d.io.write_point_cloud(minkowski_grid_path, pcd)
        print(f"Saved MinkowskiEngine voxel grid to {minkowski_grid_path}")
        return

    # Create a new parser for ModelParams with only the required arguments
    model_parser = argparse.ArgumentParser()
    model_params = ModelParams(model_parser)
    
    # Set model parameters from our args
    model_params.model_path = args.model_path
    model_params.source_path = args.source_path
    model_params.images = args.images
    model_params.eval = args.eval
    model_params.object_path = args.object_path
    model_params.n_views = args.n_views
    model_params.random_init = args.random_init
    model_params.train_split = args.train_split
    
    # Load model
    gaussians = GaussianModel(0)  # sh_degree=0 for visualization
    
    # Find latest iteration if not specified
    if args.iteration == -1:
        args.iteration = searchForMaxIteration(os.path.join(args.model_path, "point_cloud"))
    
    # Load scene
    scene = Scene(model_params, gaussians, load_iteration=args.iteration)
    
    # Initialize surface detector with more aggressive filtering for dense geometry
    detector = SurfaceDetector(
        opacity_threshold=0.8,    # High opacity for solid surfaces
        scale_threshold=0.02,     # Smaller scale threshold to focus on detailed geometry
        density_threshold=0.4,    # Higher density threshold to focus on concentrated areas
        k_neighbors=16,          # More neighbors for better density estimation
        spatial_concentration_threshold=0.4,  # Higher threshold for spatial concentration
        min_cluster_size=50      # Smaller minimum cluster size to allow for smaller dense regions
    )
    
    # Get Gaussian parameters
    xyz = gaussians.get_xyz
    opacity = gaussians.get_opacity
    scaling = gaussians.get_scaling
    
    # Compute scene extent before surface detection
    scene_extent = torch.max(torch.norm(xyz, dim=1)).item()
    print(f"Scene extent: {scene_extent:.3f}")
    
    # Extract surface points
    print("Extracting surface points...")
    surface_points, surface_normals = detector.extract_surface_points(
        xyz, opacity, scaling, scene_extent
    )
    print(f"Found {len(surface_points)} surface points")
    
    # Calculate cell sizes
    base_cell_size = scene_extent / 50  # Base size for sparse regions
    min_cell_size = base_cell_size * 0.2  # Much smaller cells in dense regions
    max_cell_size = base_cell_size * 1.5  # Larger cells in sparse regions
    
    print(f"\nCell Size Parameters:")
    print(f"Base cell size: {base_cell_size:.3f}")
    print(f"Minimum cell size (dense regions): {min_cell_size:.3f}")
    print(f"Maximum cell size (sparse regions): {max_cell_size:.3f}")
    
    # Create grid
    grid = HashGrid(
        min_cell_size=min_cell_size,
        max_cell_size=max_cell_size,
        hash_size=args.hash_size,
        max_points_per_cell=args.max_points_per_cell,
        confidence_threshold=0.5,
        curvature_threshold=0.1,
        concentration_weight=0.5,
        density_weight=0.4,
        curvature_weight=0.1
    )
    
    # Build grid
    if args.reg_grid:
        print("Building regular (structured) grid...")
        # Compute bounding box to estimate cell size for ~50,000 voxels
        min_corner = surface_points.min(dim=0)[0]
        max_corner = surface_points.max(dim=0)[0]
        bbox = max_corner - min_corner
        target_voxels = args.target_voxel_count
        cell_size = (bbox.prod().item() / target_voxels) ** (1/3)
        cell_size = cell_size * 0.1  # Make grid even finer to increase number of non-empty voxels
        print(f"Auto-tuned cell size for ~{target_voxels} voxels (after refinement): {cell_size:.3f}")
        grid.build_structured_grid(
            points=surface_points,
            cell_size=cell_size,
            confidence=None,
            target_voxel_count=args.target_voxel_count
        )
        grid_type = "reg_grid"
        voxel_count = len(grid.hash_table)


    # --- BEGIN: Filter out voxels with less than average points ---
    if args.minkowski:
        # Count points per voxel
        point_counts = [len(indices) for indices in grid.hash_table.values()]
        if len(point_counts) == 0:
            avg_points = 0
        else:
            avg_points = sum(point_counts) / len(point_counts)

        # Only keep voxels with at least the average number of points
        filtered_hash_table = {}
        for cell_hash, indices in grid.hash_table.items():
            if len(indices) >= avg_points:
                filtered_hash_table[cell_hash] = indices
        grid.hash_table = filtered_hash_table

        print(f"Filtered voxels: {len(grid.hash_table)} remain with >= average ({avg_points:.1f}) points per voxel")
    # --- END: Filter out voxels with less than average points ---
    else:
        print("Building adaptive hash grid...")
        grid.build(
            points=surface_points,
            normals=surface_normals,
            confidence=None
        )
        grid_type = "hash_grid"
        voxel_count = len(grid.hash_table)
    
    # Test queries
    print(f"\nPerforming {args.test_queries} test queries...")
    query_batch_size = args.query_batch_size
    total_queries = args.test_queries
    total_time = 0
    total_success = 0
    
    for batch_start in tqdm(range(0, total_queries, query_batch_size), desc="Testing queries"):
        batch_end = min(batch_start + query_batch_size, total_queries)
        batch_size = batch_end - batch_start
        
        # Generate random query points within scene bounds
        query_points = torch.rand((batch_size, 3), device=surface_points.device) * scene_extent
        
        try:
            # Time the query
            start_time = time.time()
            indices, distances = grid.query_points(query_points, k=8)
            batch_time = time.time() - start_time
            total_time += batch_time
            
            # Count successful queries (those with at least one valid neighbor)
            valid_queries = (indices != -1).any(dim=1).sum().item()
            total_success += valid_queries
            
            # Print batch statistics
            print(f"\nBatch {batch_start//query_batch_size + 1}:")
            print(f"  Queries: {batch_size}")
            print(f"  Valid queries: {valid_queries}")
            print(f"  Average time per query: {batch_time/batch_size*1000:.2f}ms")
            
        except Exception as e:
            print(f"\nWarning: Error in batch {batch_start//query_batch_size + 1}: {str(e)}")
            continue
    
    # Print final statistics
    print("\nFinal Statistics:")
    print(f"Total queries: {total_queries}")
    print(f"Successful queries: {total_success}")
    print(f"Success rate: {total_success/total_queries*100:.1f}%")
    print(f"Average time per query: {total_time/total_queries*1000:.2f}ms")
    
    # Visualization and output
    if args.save_ply:
        scene_name = os.path.basename(os.path.normpath(args.model_path))
        if args.reg_grid:
            base_name = f"{scene_name}_reg_grid_{voxel_count}vox_iter{args.iteration}_op{detector.opacity_threshold}_sc{detector.scale_threshold}_de{detector.density_threshold}_k{detector.k_neighbors}"
        else:
            base_name = f"{scene_name}_hash_grid_iter{args.iteration}_op{detector.opacity_threshold}_sc{detector.scale_threshold}_de{detector.density_threshold}_k{detector.k_neighbors}"
        points_path = os.path.join(args.output_dir, base_name + "_points.ply")
        grid_path = os.path.join(args.output_dir, base_name + "_grid.ply")
        grid.visualize_points(points_path)
        grid.visualize_grid(grid_path)
        print(f"Saved point cloud to {points_path}")
        print(f"Saved grid visualization to {grid_path}")
    else:
        print(f"Visualizing {grid_type} (this may take a while)...")
        grid.visualize(None)

    # After extracting surface points, create Minkowski voxel grid for debugging
    try:
        # Ensure surface_points and colors are torch tensors on the correct device
        device = surface_points.device if hasattr(surface_points, 'device') else ('cuda' if torch.cuda.is_available() else 'cpu')
        if not isinstance(surface_points, torch.Tensor):
            surface_points_tensor = torch.from_numpy(np.asarray(surface_points)).to(device)
        else:
            surface_points_tensor = surface_points.to(device)
        if hasattr(gaussians, "get_features_dc"):
            colors = gaussians.get_features_dc.detach()
            if colors.shape[1] == 1 and colors.shape[-1] == 3:
                colors = colors[:, 0, :]
            else:
                colors = torch.ones_like(surface_points_tensor)  # fallback
        else:
            colors = torch.ones_like(surface_points_tensor)
        if not isinstance(colors, torch.Tensor):
            colors = torch.from_numpy(np.asarray(colors)).to(device)
        else:
            colors = colors.to(device)
        # Debug: print types and devices
        print(f"[DEBUG] surface_points_tensor type: {type(surface_points_tensor)}, device: {surface_points_tensor.device}")
        print(f"[DEBUG] colors type: {type(colors)}, device: {colors.device}")
        minkowski_grid = MinkowskiVoxelGrid(surface_points_tensor, colors=colors, voxel_size=args.cell_size, device=device)
        scene_name = os.path.basename(os.path.normpath(args.model_path))
        minkowski_base = f"{scene_name}_minkowski_{len(minkowski_grid)}vox_iter{args.iteration}_op{detector.opacity_threshold}_sc{detector.scale_threshold}_de{detector.density_threshold}_k{detector.k_neighbors}"
        minkowski_points_path = os.path.join(args.output_dir, minkowski_base + "_points.ply")
        minkowski_grid_path = os.path.join(args.output_dir, minkowski_base + "_grid.ply")
        # Save voxel centers as point cloud
        import open3d as o3d
        voxel_centers = minkowski_grid.get_voxel_centers().cpu().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(voxel_centers)
        # Use features as colors if available
        feats = minkowski_grid.get_features().cpu().numpy()
        if feats.shape[1] == 3:
            pcd.colors = o3d.utility.Vector3dVector(np.clip(feats, 0, 1))
        else:
            pcd.colors = o3d.utility.Vector3dVector(np.ones_like(voxel_centers))
        o3d.io.write_point_cloud(minkowski_points_path, pcd)
        print(f"Saved MinkowskiEngine voxel centers to {minkowski_points_path}")
        # Optionally, save a voxel grid mesh (as points only for now)
        o3d.io.write_point_cloud(minkowski_grid_path, pcd)
        print(f"Saved MinkowskiEngine voxel grid to {minkowski_grid_path}")
    except Exception as e:
        print(f"[MinkowskiEngine] Skipping Minkowski voxel grid output: {e}")

if __name__ == "__main__":
    main()