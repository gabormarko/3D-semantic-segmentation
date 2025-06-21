import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Paths (edit as needed)
VOXEL_FEATURES = "/home/neural_fields/Unified-Lift-Gabor/output/minkowski_grid/teatime_voxel_features.npy"

# Load voxel features
voxel_features = np.load(VOXEL_FEATURES)  # [N, C]
N_vox, C = voxel_features.shape

# Only consider nonzero voxels (those with any nonzero feature)
nonzero_mask = np.any(voxel_features != 0, axis=1)
nonzero_features = voxel_features[nonzero_mask]

print(f"Loaded {N_vox} voxels, {np.sum(nonzero_mask)} nonzero voxels.")

# 1. Plot histogram of all feature values
plt.figure(figsize=(8,4))
plt.hist(nonzero_features.flatten(), bins=100, color='blue', alpha=0.7)
plt.title('Histogram of Voxel Feature Values (nonzero voxels)')
plt.xlabel('Feature Value')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('voxel_feature_histogram.png')
print('Saved voxel feature histogram as voxel_feature_histogram.png')

# 2. Plot per-channel statistics
means = np.mean(nonzero_features, axis=0)
stds = np.std(nonzero_features, axis=0)
plt.figure(figsize=(10,4))
plt.plot(means, label='mean')
plt.plot(stds, label='std')
plt.title('Per-channel Mean and Std of Voxel Features')
plt.xlabel('Channel')
plt.ylabel('Value')
plt.legend()
plt.tight_layout()
plt.savefig('voxel_feature_channel_stats.png')
print('Saved per-channel stats as voxel_feature_channel_stats.png')

# 3. Try alternative aggregation: median
median_features = np.median(nonzero_features, axis=0)
print('Median of each channel:', median_features)

# 4. Try normalization: min-max and z-score
min_vals = np.min(nonzero_features, axis=0)
max_vals = np.max(nonzero_features, axis=0)
zscore_features = (nonzero_features - means) / (stds + 1e-8)
minmax_features = (nonzero_features - min_vals) / (max_vals - min_vals + 1e-8)

# Plot histogram of z-score normalized features
plt.figure(figsize=(8,4))
plt.hist(zscore_features.flatten(), bins=100, color='green', alpha=0.7)
plt.title('Histogram of Z-score Normalized Voxel Features')
plt.xlabel('Z-score')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('voxel_feature_zscore_histogram.png')
print('Saved z-score histogram as voxel_feature_zscore_histogram.png')

# Plot histogram of min-max normalized features
plt.figure(figsize=(8,4))
plt.hist(minmax_features.flatten(), bins=100, color='orange', alpha=0.7)
plt.title('Histogram of Min-Max Normalized Voxel Features')
plt.xlabel('Min-Max Value')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('voxel_feature_minmax_histogram.png')
print('Saved min-max histogram as voxel_feature_minmax_histogram.png')

# 5. Save normalized features for further experiments
np.save('voxel_features_zscore.npy', zscore_features)
np.save('voxel_features_minmax.npy', minmax_features)
print('Saved normalized feature arrays.')
