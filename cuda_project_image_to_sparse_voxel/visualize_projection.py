import torch
import matplotlib.pyplot as plt
import numpy as np

# Load the output
out = torch.load("proj_output.pt")
mapping2dto3d = out['mapping2dto3d_num']  # (num_voxels,)
projected_feats = out['projected_feats']  # (num_voxels, feature_dim)

print("=== Projection Output Statistics ===")
print(f"mapping2dto3d shape: {mapping2dto3d.shape}, dtype: {mapping2dto3d.dtype}")
print(f"projected_feats shape: {projected_feats.shape}, dtype: {projected_feats.dtype}")

# Debug: print unique values and counts in mapping2dto3d
unique_vals, unique_counts = torch.unique(mapping2dto3d, return_counts=True)
print("mapping2dto3d unique values and counts:")
for val, count in zip(unique_vals.tolist(), unique_counts.tolist()):
    print(f"  value {val}: count {count}")

# Debug: print min/max of projected_feats
print("projected_feats stats: min", projected_feats.min().item(), "max", projected_feats.max().item())

# Nonzero voxels
nonzero_voxels = (mapping2dto3d > 0).sum().item()
total_voxels = mapping2dto3d.numel()
print(f"Nonzero voxels: {nonzero_voxels} / {total_voxels} ({100*nonzero_voxels/total_voxels:.2f}%)")

# Histogram: how many 2D points contributed to each voxel
plt.figure(figsize=(8,4))
plt.hist(mapping2dto3d[mapping2dto3d > 0].cpu().numpy(), bins=50, log=True)
plt.title("Histogram: 2D points per nonzero voxel")
plt.xlabel("Number of 2D points projected to voxel")
plt.ylabel("Voxel count (log scale)")
plt.tight_layout()
plt.show()

# Feature statistics
if projected_feats.ndim == 2 and nonzero_voxels > 0:
    mean_feat = projected_feats[mapping2dto3d > 0].mean(dim=0)
    std_feat = projected_feats[mapping2dto3d > 0].std(dim=0)
    print(f"Mean of projected features (first 10 dims): {mean_feat[:10].cpu().numpy()}")
    print(f"Std of projected features (first 10 dims): {std_feat[:10].cpu().numpy()}")

    # Visualize the distribution of the first feature dimension
    plt.figure()
    plt.hist(projected_feats[mapping2dto3d > 0, 0].cpu().numpy(), bins=50)
    plt.title("Distribution of projected feature[0] (nonzero voxels)")
    plt.xlabel("Feature[0] value")
    plt.ylabel("Voxel count")
    plt.tight_layout()
    plt.show()
else:
    print("projected_feats is not 2D or there are no nonzero voxels, skipping feature stats.")

# Optionally, save a summary
summary = {
    "nonzero_voxels": nonzero_voxels,
    "total_voxels": total_voxels,
    "mean_feat_first10": mean_feat[:10].cpu().numpy().tolist() if nonzero_voxels > 0 else [],
    "std_feat_first10": std_feat[:10].cpu().numpy().tolist() if nonzero_voxels > 0 else [],
}
with open("proj_output_summary.json", "w") as f:
    import json
    json.dump(summary, f, indent=2)
print("Summary saved to cuda_project_image_to_sparse_voxel/proj_output_summary.json")