import os
import sys
import time

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import torch
import argparse
from tqdm import tqdm

from utils.surface_detection import SurfaceDetector
from utils.hash_grid import HashGrid
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
    
    # I/O parameters
    io_group.add_argument("--output_dir", default="output/hash_grid", help="Output directory for visualizations")
    io_group.add_argument("--save_ply", action="store_true", help="Save visualization as PLY files")
    
    # Add new arguments
    parser.add_argument("--query_batch_size", type=int, default=100,
                      help="Number of queries to process in each batch")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
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
    
    # Calculate adaptive cell sizes based on density
    # Use smaller cells in dense regions, larger in sparse regions
    base_cell_size = scene_extent / 50  # Base size for sparse regions
    min_cell_size = base_cell_size * 0.2  # Much smaller cells in dense regions
    max_cell_size = base_cell_size * 1.5  # Larger cells in sparse regions
    
    print(f"\nCell Size Parameters:")
    print(f"Base cell size: {base_cell_size:.3f}")
    print(f"Minimum cell size (dense regions): {min_cell_size:.3f}")
    print(f"Maximum cell size (sparse regions): {max_cell_size:.3f}")
    
    # Create hash grid with adaptive cell sizing
    grid = HashGrid(
        min_cell_size=min_cell_size,    # Very small cells in dense regions
        max_cell_size=max_cell_size,    # Larger cells in sparse regions
        hash_size=args.hash_size,
        max_points_per_cell=args.max_points_per_cell,
        confidence_threshold=0.5,
        curvature_threshold=0.1,
        concentration_weight=0.5,        # Increased weight for spatial concentration
        density_weight=0.4,             # Increased weight for local density
        curvature_weight=0.1            # Reduced weight for curvature
    )
    
    # Build grid with points and normals
    grid.build(
        points=surface_points,
        normals=surface_normals,
        confidence=None  # No confidence scores available yet
    )
    
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
    
    # Visualize with memory optimization
    if args.save_ply:
        # Extract scene name from model path
        scene_name = os.path.basename(os.path.normpath(args.model_path))
        # Create filename with scene name and surface detection parameters
        param_str = f"op{detector.opacity_threshold:.1f}_sc{detector.scale_threshold:.2f}_de{detector.density_threshold:.1f}_k{detector.k_neighbors}"
        save_path = os.path.join(args.output_dir, f"{scene_name}_hash_grid_iter{args.iteration}_{param_str}")
        print("Saving visualization (this may take a while)...")
        # Save points and grid separately to reduce memory usage
        print("Saving point cloud...")
        grid.visualize_points(save_path + "_points.ply")
        print("Saving grid visualization...")
        grid.visualize_grid(save_path + "_grid.ply")
    else:
        print("Visualizing hash grid (this may take a while)...")
        grid.visualize(None)

if __name__ == "__main__":
    main() 