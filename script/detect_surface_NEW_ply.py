import os
import sys

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)



import torch
import argparse
import psutil
import os
from scene.gaussian_model import GaussianModel
from utils.surface_detection import SurfaceDetector

def parse_args():
    # Only create our main parser (no ModelParams needed)
    parser = argparse.ArgumentParser(description="Detect surfaces from Gaussian splatting model")
    
    # Create argument groups
    model_group = parser.add_argument_group("Model Parameters")
    detector_group = parser.add_argument_group("Surface Detector Parameters")
    io_group = parser.add_argument_group("Input/Output Parameters")
    

    # Model parameters
    model_group.add_argument("--ply_path", required=True, help="Path to the input .ply file with Gaussian properties")
    
    # Surface detector parameters
    detector_group.add_argument("--opacity_threshold", type=float, default=0.2, help="Minimum opacity threshold")
    detector_group.add_argument("--scale_threshold", type=float, default=0.2, help="Maximum scale threshold")
    detector_group.add_argument("--density_threshold", type=float, default=0.02, help="Minimum density threshold")
    detector_group.add_argument("--k_neighbors", type=int, default=2, help="Number of neighbors for density estimation")
    detector_group.add_argument("--max_points", type=int, default=1500000, help="Maximum number of Gaussians to process (default: 100000)")
    detector_group.add_argument("--spikiness_threshold", type=float, default=10.0, help="Maximum allowed ratio of largest to smallest scale (spikiness filter)")
    
    # I/O parameters
    io_group.add_argument("--output_dir", default="output/surface", help="Output directory for visualizations")
    io_group.add_argument("--save_ply", action="store_true", default=True, help="Save surface as PLY file (default: True)")
    
    return parser.parse_args()

def main():
    args = parse_args()

    def print_mem_usage(msg):
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / 1024 / 1024
        print(f"[MEM] {msg}: {mem_mb:.2f} MB")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print_mem_usage("Before loading Gaussian model")
    gaussians = GaussianModel(sh_degree=3)
    gaussians.load_ply(args.ply_path)
    print_mem_usage("After loading Gaussian model")

    # Get Gaussian parameters
    xyz = gaussians.get_xyz
    opacity = gaussians.get_opacity
    scaling = gaussians.get_scaling

    # Limit number of points
    N = xyz.shape[0]
    if N > args.max_points:
        print(f"Limiting to first {args.max_points} of {N} points for surface detection.")
        idx = torch.arange(N)[:args.max_points]
        xyz = xyz[idx]
        opacity = opacity[idx]
        scaling = scaling[idx]

    # Filter out spiky Gaussians (clamp very small scale values to avoid division by zero)
    scaling_clamped = scaling.clone()
    scaling_clamped[scaling_clamped < 1e-6] = 1e-6
    ratios = scaling_clamped.max(dim=1).values / scaling_clamped.min(dim=1).values
    mask = ratios < args.spikiness_threshold
    num_spiky = (~mask).sum().item()
    if num_spiky > 0:
        print(f"Filtered out {num_spiky} spiky Gaussians (ratio > {args.spikiness_threshold})")
    xyz = xyz[mask]
    opacity = opacity[mask]
    scaling = scaling[mask]

    # Plot and save spikiness distribution after spiky filtering
    import matplotlib.pyplot as plt
    import numpy as np
    plt.figure(figsize=(8, 5))
    plt.xlim(0, 20)
    plt.hist(ratios.detach().cpu().numpy(), bins=np.linspace(0, 20, 101), color='royalblue', alpha=0.8)
    plt.axvline(args.spikiness_threshold, color='red', linestyle='--', label=f'Threshold ({args.spikiness_threshold})')
    plt.xlabel('Spikiness Ratio (max(scale)/min(scale))')
    plt.ylabel('Number of Gaussians')
    plt.title('Spikiness Distribution of Gaussians')    
    plt.legend()
    spiky_plot_path = os.path.join(args.output_dir, 'spikiness_distribution.png')
    plt.tight_layout()
    plt.savefig(spiky_plot_path)
    plt.close()
    print(f"Spikiness distribution plot saved to: {spiky_plot_path}")

    print_mem_usage("Before surface detection")

    # Initialize surface detector
    detector = SurfaceDetector(
        opacity_threshold=args.opacity_threshold,
        scale_threshold=args.scale_threshold,
        density_threshold=args.density_threshold,
        k_neighbors=args.k_neighbors
    )

    # Compute scene extent
    scene_extent = torch.max(torch.norm(xyz, dim=1)).item()

    # Extract surface points
    surface_points, surface_normals = detector.extract_surface_points(
        xyz, opacity, scaling, scene_extent
    )

    print_mem_usage("After surface detection")

    print(f"Found {len(surface_points)} surface points")

    # Visualize surface
    if args.save_ply:
        # Extract scene name from ply path
        scene_name = os.path.splitext(os.path.basename(args.ply_path))[0]
        param_str = f"op{detector.opacity_threshold:.1f}_sc{detector.scale_threshold:.2f}_de{detector.density_threshold:.1f}_k{detector.k_neighbors}"
        save_path = os.path.join(args.output_dir, f"{scene_name}_surface_{param_str}.ply")
    else:
        save_path = None

    detector.visualize_surface(surface_points, surface_normals, save_path)
    if save_path is not None:
        print(f"Surface PLY file saved to: {save_path}")

    # For hash grid testing
if __name__ == "__main__":
    main() 