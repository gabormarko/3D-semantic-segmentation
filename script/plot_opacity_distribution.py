import argparse
import numpy as np
import matplotlib.pyplot as plt
from plyfile import PlyData
import os

def sigmoid(x):
    """Converts logits to probabilities."""
    return 1 / (1 + np.exp(-x))

def main():
    parser = argparse.ArgumentParser(description="Plot opacity distribution from a Gaussian Splatting PLY file.")
    parser.add_argument("--input_ply", required=True, help="Path to the input PLY file containing Gaussian data.")
    args = parser.parse_args()

    # --- Load PLY data ---
    try:
        plydata = PlyData.read(args.input_ply)
        vertices = plydata['vertex']
        print(f"Successfully read {args.input_ply}")
    except Exception as e:
        print(f"Error reading PLY file: {e}")
        return

    if 'opacity' not in vertices.data.dtype.names:
        print("Error: 'opacity' property not found in the PLY file. Cannot generate plots.")
        return

    # --- Extract and convert opacities from logits to probabilities ---
    opacity_logits = vertices['opacity']
    opacities = sigmoid(opacity_logits)
    
    print(f"Loaded {len(opacities)} opacities.")
    print(f"Min opacity: {opacities.min():.4f}, Max opacity: {opacities.max():.4f}, Mean opacity: {opacities.mean():.4f}")

    # --- Plot 1: Full Opacity Distribution ---
    plt.figure(figsize=(12, 6))
    plt.hist(opacities, bins=100, range=(0, 1))
    plt.title(f'Full Opacity Distribution\n({os.path.basename(args.input_ply)})')
    plt.xlabel('Opacity (0.0 to 1.0)')
    plt.ylabel('Number of Gaussians (Log Scale)')
    plt.grid(True)
    plt.yscale('log')  # Log scale is useful for seeing the full range of counts
    
    output_filename_full = os.path.splitext(args.input_ply)[0] + '_opacity_dist_full.png'
    plt.savefig(output_filename_full)
    print(f"Saved full distribution plot to: {output_filename_full}")
    plt.close()

    # --- Plot 2: High Opacity (>= 0.99) Distribution ---
    high_opacity_mask = opacities >= 0.99
    high_opacities = opacities[high_opacity_mask]

    print(f"\nFound {len(high_opacities)} Gaussians with opacity >= 0.99.")

    if len(high_opacities) > 0:
        plt.figure(figsize=(12, 6))
        # Use fine-grained bins for the small 0.99-1.0 range
        plt.hist(high_opacities, bins=50, range=(0.99, 1.0))
        plt.title(f'High Opacity (>= 0.99) Distribution\n({os.path.basename(args.input_ply)})')
        plt.xlabel('Opacity (0.99 to 1.0)')
        plt.ylabel('Number of Gaussians')
        plt.grid(True)
        
        output_filename_high = os.path.splitext(args.input_ply)[0] + '_opacity_dist_high.png'
        plt.savefig(output_filename_high)
        print(f"Saved high opacity distribution plot to: {output_filename_high}")
        plt.close()
    else:
        print("No Gaussians with opacity >= 0.99 found, skipping second plot.")

    # --- Plot 3: Very High Opacity (>= 0.999) Distribution ---
    very_high_opacity_mask = opacities >= 0.999
    very_high_opacities = opacities[very_high_opacity_mask]

    print(f"\nFound {len(very_high_opacities)} Gaussians with opacity >= 0.999.")

    if len(very_high_opacities) > 0:
        plt.figure(figsize=(12, 6))
        # Use fine-grained bins for the small 0.999-1.0 range
        plt.hist(very_high_opacities, bins=50, range=(0.999, 1.0))
        plt.title(f'Very High Opacity (>= 0.999) Distribution\n({os.path.basename(args.input_ply)})')
        plt.xlabel('Opacity (0.999 to 1.0)')
        plt.ylabel('Number of Gaussians')
        plt.grid(True)
        
        output_filename_very_high = os.path.splitext(args.input_ply)[0] + '_opacity_dist_very_high.png'
        plt.savefig(output_filename_very_high)
        print(f"Saved very high opacity distribution plot to: {output_filename_very_high}")
        plt.close()
    else:
        print("No Gaussians with opacity >= 0.999 found, skipping third plot.")

    # --- Plot 4: Ultra High Opacity (>= 0.99999) Distribution ---
    ultra_high_opacity_mask = opacities >= 0.99999
    ultra_high_opacities = opacities[ultra_high_opacity_mask]

    print(f"\nFound {len(ultra_high_opacities)} Gaussians with opacity >= 0.99999.")

    if len(ultra_high_opacities) > 0:
        plt.figure(figsize=(12, 6))
        # Use fine-grained bins for the small 0.99999-1.0 range
        plt.hist(ultra_high_opacities, bins=50, range=(0.99999, 1.0))
        plt.title(f'Ultra High Opacity (>= 0.99999) Distribution\n({os.path.basename(args.input_ply)})')
        plt.xlabel('Opacity (0.99999 to 1.0)')
        plt.ylabel('Number of Gaussians')
        plt.grid(True)
        
        output_filename_ultra_high = os.path.splitext(args.input_ply)[0] + '_opacity_dist_ultra_high.png'
        plt.savefig(output_filename_ultra_high)
        print(f"Saved ultra high opacity distribution plot to: {output_filename_ultra_high}")
        plt.close()
    else:
        print("No Gaussians with opacity >= 0.99999 found, skipping fourth plot.")


if __name__ == "__main__":
    main()
