import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, default='/home/neural_fields/Unified-Lift-Gabor/cuda_project_image_to_sparse_voxel/voxel_feature_checkpoints_vox41759/ALL_nonzero_voxel_features_232.pt', help='Path to checkpoint .pt file or raw hit_count tensor file.')
parser.add_argument('--topk', type=int, default=10, help='Number of top-hit voxels to display.')
args = parser.parse_args()

data = torch.load(args.input_file, map_location='cpu')
xyz_coords = None

# Check if it's a dictionary from the new format or a raw tensor from the old
if isinstance(data, dict):
    print("Detected dictionary format. Extracting 'hit_count' and 'xyz'.")
    if 'hit_count' in data:
        hit_count = data['hit_count']
    else:
        raise KeyError("File is a dictionary but does not contain a 'hit_count' key.")
    if 'xyz' in data:
        xyz_coords = data['xyz']
    else:
        print("Warning: 'xyz' key not found in dictionary. Cannot show world coordinates.")
else:
    print("Detected raw tensor format.")
    hit_count = data


print(f"Loaded hit_count tensor: shape={hit_count.shape}, dtype={hit_count.dtype}")
print(f"Max hits: {hit_count.max().item()}")
print(f"Nonzero voxels: {(hit_count > 0).sum().item()} / {hit_count.numel()}")
print(f"Mean hits (nonzero): {hit_count[hit_count > 0].float().mean().item() if (hit_count > 0).any() else 0:.2f}")

# Histogram
plt.figure()
plt.hist(hit_count[hit_count > 0].cpu().numpy(), bins=30)
plt.title('Histogram of voxel hit counts (nonzero only)')
plt.xlabel('Number of hits')
plt.ylabel('Number of voxels')
plt.grid(True)

# Save the figure
output_filename = os.path.splitext(args.input_file)[0] + '_histogram.png'
plt.savefig(output_filename)
print(f"Histogram saved to {output_filename}")
plt.close() # Close the plot to prevent it from displaying in interactive environments

# Optionally, print a few of the most hit voxels
if (hit_count > 0).any():
    topk = min(args.topk, (hit_count > 0).sum().item())
    top_indices = hit_count.argsort(descending=True)[:topk]
    print(f"\nTop {topk} voxels by hit count:")
    for i, idx in enumerate(top_indices):
        count = hit_count[idx].item()
        if xyz_coords is not None:
            coord = xyz_coords[idx].numpy()
            print(f"Voxel at world coord {coord}: count={count}")
        else:
            print(f"Voxel index {idx.item()}: count={count}")
else:
    print("No voxels were hit.")
