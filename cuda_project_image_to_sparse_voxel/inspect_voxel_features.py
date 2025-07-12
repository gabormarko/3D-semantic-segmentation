import torch
import numpy as np

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--proj_output', type=str, default='proj_output.pt', help='Path to projection output file')
    parser.add_argument('--topk', type=int, default=10, help='Show stats for top-k most hit voxels')
    args = parser.parse_args()

    out = torch.load(args.proj_output, map_location='cpu')
    feats = out['projected_feats']  # [num_voxels, feature_dim]
    mapping = out['mapping2dto3d_num']  # [num_voxels]

    print(f"projected_feats shape: {feats.shape}")
    print(f"mapping2dto3d_num shape: {mapping.shape}")

    # Find nonzero voxels
    nonzero = (mapping > 0).nonzero(as_tuple=True)[0]
    print(f"Number of voxels with features: {len(nonzero)} / {mapping.shape[0]}")

    if len(nonzero) == 0:
        print("No voxels with features.")
        return

    # Stats for all nonzero voxels
    all_feats = feats[nonzero]
    print(f"All nonzero voxel features: min={all_feats.min().item():.4f}, max={all_feats.max().item():.4f}, mean={all_feats.mean().item():.4f}, std={all_feats.std().item():.4f}")

    # Top-k voxels by mapping count
    topk = min(args.topk, len(nonzero))
    topk_indices = mapping[nonzero].argsort(descending=True)[:topk]
    print(f"\nTop {topk} voxels by number of hits:")
    for rank, idx in enumerate(topk_indices):
        voxel_idx = nonzero[idx].item()
        count = mapping[voxel_idx].item()
        feat = feats[voxel_idx]
        print(f"Voxel {voxel_idx}: count={count}, feat[:10]={feat[:10].cpu().numpy()}")

    # Optionally, print histogram of mapping counts
    import matplotlib.pyplot as plt
    plt.figure()
    plt.hist(mapping[mapping > 0].cpu().numpy(), bins=30)
    plt.title('Histogram of voxel hit counts')
    plt.xlabel('Number of hits')
    plt.ylabel('Number of voxels')
    plt.show()

if __name__ == '__main__':
    main()
