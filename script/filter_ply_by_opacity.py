import argparse
import numpy as np
from plyfile import PlyData, PlyElement
import os

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def main():
    parser = argparse.ArgumentParser(description="Filter a PLY file by opacity and save the result.")
    parser.add_argument('--input_ply', required=True, help='Input PLY file with Gaussians.')
    parser.add_argument('--output_ply', required=True, help='Output PLY file after opacity filtering.')
    parser.add_argument('--opacity_threshold', type=float, default=0.99, help='Keep only Gaussians with opacity >= threshold (after sigmoid).')
    args = parser.parse_args()

    plydata = PlyData.read(args.input_ply)
    vertices = plydata['vertex']
    if 'opacity' not in vertices.data.dtype.names:
        print("Error: 'opacity' property not found in the PLY file.")
        return

    opacity = vertices['opacity']
    # If opacity is logits, apply sigmoid. If already in [0,1], this is idempotent.
    opacities = sigmoid(opacity)
    mask = opacities >= args.opacity_threshold
    print(f"Filtering: {np.sum(mask)} / {len(opacities)} Gaussians kept (opacity >= {args.opacity_threshold})")

    # Apply mask to all properties
    filtered_verts = vertices.data[mask]
    filtered_element = PlyElement.describe(filtered_verts, 'vertex')
    PlyData([filtered_element], text=plydata.text).write(args.output_ply)
    print(f"Saved filtered PLY to: {args.output_ply}")

if __name__ == "__main__":
    main()
