import os
import sys

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import torch
import argparse
from scene import Scene, GaussianModel
from utils.surface_detection import SurfaceDetector
from arguments import ModelParams, PipelineParams
from utils.system_utils import searchForMaxIteration

def parse_args():
    # First create a parser for ModelParams to get its default values
    model_parser = argparse.ArgumentParser()
    ModelParams(model_parser)
    model_defaults = vars(model_parser.parse_args([]))
    
    # Now create our main parser
    parser = argparse.ArgumentParser(description="Detect surfaces from Gaussian splatting model")
    
    # Create argument groups
    model_group = parser.add_argument_group("Model Parameters")
    detector_group = parser.add_argument_group("Surface Detector Parameters")
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
    
    # Surface detector parameters
    detector_group.add_argument("--opacity_threshold", type=float, default=0.5, help="Minimum opacity threshold")
    detector_group.add_argument("--scale_threshold", type=float, default=0.1, help="Maximum scale threshold")
    detector_group.add_argument("--density_threshold", type=float, default=0.1, help="Minimum density threshold")
    detector_group.add_argument("--k_neighbors", type=int, default=8, help="Number of neighbors for density estimation")
    
    # I/O parameters
    io_group.add_argument("--output_dir", default="output/surface", help="Output directory for visualizations")
    io_group.add_argument("--save_ply", action="store_true", help="Save surface as PLY file")
    
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
        opacity_threshold=args.opacity_threshold,
        scale_threshold=args.scale_threshold,
        density_threshold=args.density_threshold,
        k_neighbors=args.k_neighbors
    )
    
    # Get Gaussian parameters
    xyz = gaussians.get_xyz
    opacity = gaussians.get_opacity
    scaling = gaussians.get_scaling
    
    # Compute scene extent
    scene_extent = torch.max(torch.norm(xyz, dim=1)).item()
    
    # Extract surface points
    surface_points, surface_normals = detector.extract_surface_points(
        xyz, opacity, scaling, scene_extent
    )
    
    print(f"Found {len(surface_points)} surface points")
    
    # Visualize surface
    if args.save_ply:
        # Create filename with surface detection parameters
        param_str = f"op{detector.opacity_threshold:.1f}_sc{detector.scale_threshold:.2f}_de{detector.density_threshold:.1f}_k{detector.k_neighbors}"
        save_path = os.path.join(args.output_dir, f"surface_iter{args.iteration}_{param_str}.ply")
    else:
        save_path = None
    
    detector.visualize_surface(surface_points, surface_normals, save_path)

    # For hash grid testing
if __name__ == "__main__":
    main() 