import argparse
import numpy as np
import matplotlib.pyplot as plt
from plyfile import PlyData
from scipy.spatial import cKDTree
import os

def main():
    parser = argparse.ArgumentParser(description="Plot local density distribution from a Gaussian Splatting PLY file.")
    parser.add_argument("--input_ply", required=True, help="Path to the input PLY file containing Gaussian data.")
    parser.add_argument("--eps", type=float, default=0.05, help="Neighborhood radius for density calculation.")
    parser.add_argument("--high_density", type=int, default=20, help="Threshold for high-density plot.")
    args = parser.parse_args()

    # --- Load PLY data ---
    try:
        plydata = PlyData.read(args.input_ply)
        vertices = plydata['vertex']
        print(f"Successfully read {args.input_ply}")
    except Exception as e:
        print(f"Error reading PLY file: {e}")
        return

    # --- Extract positions ---
    if all(x in vertices.data.dtype.names for x in ['x', 'y', 'z']):
        xyz = np.stack([vertices['x'], vertices['y'], vertices['z']], axis=1)
    else:
        print("Error: 'x', 'y', 'z' properties not found in the PLY file.")
        return

    print(f"Loaded {xyz.shape[0]} Gaussian centers.")

    # --- Compute local densities ---
    print(f"Building KDTree and computing densities with eps={args.eps} ...")
    tree = cKDTree(xyz)
    densities = tree.query_ball_point(xyz, r=args.eps, return_length=True)
    densities = np.array(densities)
    print(f"Density stats: min={densities.min()}, max={densities.max()}, mean={densities.mean():.2f}, median={np.median(densities):.2f}")

    # --- Plot 1: Full Density Distribution ---
    plt.figure(figsize=(12, 6))
    plt.hist(densities, bins=100, log=True)
    plt.title(f'Full Local Density Distribution\n({os.path.basename(args.input_ply)})')
    plt.xlabel(f'Number of neighbors within {args.eps}m')
    plt.ylabel('Number of Gaussians (Log Scale)')
    plt.grid(True)
    output_filename_full = os.path.splitext(args.input_ply)[0] + f'_density_dist_full_eps{args.eps}.png'
    plt.savefig(output_filename_full)
    print(f"Saved full density distribution plot to: {output_filename_full}")
    plt.close()

    # --- Plot 2: High Density Distribution ---
    high_density_mask = densities >= args.high_density
    high_densities = densities[high_density_mask]
    print(f"\nFound {len(high_densities)} Gaussians with density >= {args.high_density}.")
    if len(high_densities) > 0:
        plt.figure(figsize=(12, 6))
        plt.hist(high_densities, bins=50)
        plt.title(f'High Local Density (>= {args.high_density}) Distribution\n({os.path.basename(args.input_ply)})')
        plt.xlabel(f'Number of neighbors within {args.eps}m')
        plt.ylabel('Number of Gaussians')
        plt.grid(True)
        output_filename_high = os.path.splitext(args.input_ply)[0] + f'_density_dist_high_eps{args.eps}_thresh{args.high_density}.png'
        plt.savefig(output_filename_high)
        print(f"Saved high density distribution plot to: {output_filename_high}")
        plt.close()
    else:
        print(f"No Gaussians with density >= {args.high_density} found, skipping high density plot.")

    # --- Plot 3: Zoomed-In Low Density Distribution (0-100 neighbors) ---
    low_density_mask = (densities >= 0) & (densities <= 100)
    low_densities = densities[low_density_mask]
    print(f"\nFound {len(low_densities)} Gaussians with density in [0, 100].")
    if len(low_densities) > 0:
        plt.figure(figsize=(12, 6))
        plt.hist(low_densities, bins=100, range=(0, 100))
        plt.title(f'Zoomed-In Local Density Distribution (0-100 neighbors)\n({os.path.basename(args.input_ply)})')
        plt.xlabel(f'Number of neighbors within {args.eps}m')
        plt.ylabel('Number of Gaussians')
        plt.grid(True)
        output_filename_low = os.path.splitext(args.input_ply)[0] + f'_density_dist_zoom0-100_eps{args.eps}.png'
        plt.savefig(output_filename_low)
        print(f"Saved zoomed-in low density distribution plot to: {output_filename_low}")
        plt.close()
    else:
        print(f"No Gaussians with density in [0, 100] found, skipping zoomed-in plot.")

if __name__ == "__main__":
    main()
