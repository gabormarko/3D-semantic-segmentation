import os
import sys

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
    
    # Initialize surface detector
    detector = SurfaceDetector(
        opacity_threshold=0.5,
        scale_threshold=0.1,
        density_threshold=0.1,
        k_neighbors=8
    )
    
    # Get Gaussian parameters
    xyz = gaussians.get_xyz
    opacity = gaussians.get_opacity
    scaling = gaussians.get_scaling
    
    # Compute scene extent
    scene_extent = torch.max(torch.norm(xyz, dim=1)).item()
    
    # Extract surface points
    print("Extracting surface points...")
    surface_points, surface_normals = detector.extract_surface_points(
        xyz, opacity, scaling, scene_extent
    )
    print(f"Found {len(surface_points)} surface points")
    
    # Initialize hash grid
    print("Building hash grid...")
    grid = HashGrid(
        cell_size=args.cell_size,
        hash_size=args.hash_size,
        max_points_per_cell=args.max_points_per_cell
    )
    grid.build(surface_points, surface_normals)
    
    # Test queries
    print(f"Performing {args.test_queries} test queries...")
    query_times = []
    
    # Generate random query points within the scene bounds
    min_coords = surface_points.min(dim=0)[0]
    max_coords = surface_points.max(dim=0)[0]
    query_points = torch.rand(args.test_queries, 3, device=surface_points.device)
    query_points = query_points * (max_coords - min_coords) + min_coords
    
    # Perform queries
    for query_point in tqdm(query_points):
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        indices, distances = grid.query_points(query_point.unsqueeze(0), k=8)
        end_time.record()
        
        torch.cuda.synchronize()
        query_times.append(start_time.elapsed_time(end_time))
    
    # Print statistics
    query_times = torch.tensor(query_times)
    print(f"\nQuery statistics:")
    print(f"Mean query time: {query_times.mean():.2f} ms")
    print(f"Min query time: {query_times.min():.2f} ms")
    print(f"Max query time: {query_times.max():.2f} ms")
    print(f"Std query time: {query_times.std():.2f} ms")
    
    # Visualize
    if args.save_ply:
        save_path = os.path.join(args.output_dir, f"hash_grid_iter{args.iteration}")
    else:
        save_path = None
    
    print("Visualizing hash grid...")
    grid.visualize(save_path)

if __name__ == "__main__":
    main() 