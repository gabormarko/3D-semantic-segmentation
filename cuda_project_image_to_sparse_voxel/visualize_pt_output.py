import torch
import numpy as np
import argparse
import os

def visualize_pt_to_ply(pt_file_path):
    """
    Loads a .pt file containing voxel data, extracts world coordinates,
    and saves them as a .ply file for visualization.
    """
    if not os.path.exists(pt_file_path):
        print(f"Error: File not found at {pt_file_path}")
        return

    print(f"Loading data from {pt_file_path}...")
    try:
        data = torch.load(pt_file_path, map_location='cpu')
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    print("Available keys in the file:", list(data.keys()))

    if 'xyz' not in data:
        print("Error: 'xyz' key not found in the loaded data.")
        return

    world_coords = data['xyz']
    print(f"Found 'xyz' data with shape: {world_coords.shape}")

    if not isinstance(world_coords, torch.Tensor):
        print(f"Error: 'xyz' data is not a torch.Tensor, but {type(world_coords)}")
        return
        
    world_coords_np = world_coords.numpy()

    # Define the output path for the .ply file
    output_dir = os.path.dirname(pt_file_path)
    base_filename = os.path.basename(pt_file_path).replace('.pt', '')
    ply_output_path = os.path.join(output_dir, f"{base_filename}_world_coords.ply")

    print(f"Saving world coordinates to {ply_output_path}...")
    try:
        with open(ply_output_path, 'w') as f:
            f.write('ply\n')
            f.write('format ascii 1.0\n')
            f.write(f'element vertex {world_coords_np.shape[0]}\n')
            f.write('property float x\n')
            f.write('property float y\n')
            f.write('property float z\n')
            f.write('property uchar red\n')
            f.write('property uchar green\n')
            f.write('property uchar blue\n')
            f.write('end_header\n')
            for i in range(world_coords_np.shape[0]):
                pt = world_coords_np[i]
                f.write(f"{pt[0]} {pt[1]} {pt[2]} 255 0 0\n")
        print("Successfully created .ply file.")
    except Exception as e:
        print(f"Error writing .ply file: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize world coordinates from a .pt file by converting to .ply.')
    parser.add_argument('input_file', type=str, help='Path to the input .pt file.')
    args = parser.parse_args()

    visualize_pt_to_ply(args.input_file)
