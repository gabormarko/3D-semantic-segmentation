import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--hit_counts', type=str, required=True, help='Path to voxel_hit_counts_XX.pt file')
args = parser.parse_args()

hit_count = torch.load(args.hit_counts, map_location='cpu')

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
plt.show()

# Optionally, print a few of the most hit voxels
if (hit_count > 0).any():
    topk = min(10, (hit_count > 0).sum().item())
    top_indices = hit_count.argsort(descending=True)[:topk]
    print(f"\nTop {topk} voxels by hit count:")
    for i, idx in enumerate(top_indices):
        print(f"Voxel {idx.item()}: count={hit_count[idx].item()}")
else:
    print("No voxels were hit.")
