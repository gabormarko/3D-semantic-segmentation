import numpy as np
import collections
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Path to the voxel features (N_voxels, N_features)
FEATURE_PATH = "/home/neural_fields/Unified-Lift-Gabor/output/minkowski_grid/teatime_voxel_features.npy"
N_CLUSTERS = 10  # Should match the clustering used in your PLY script

# Load features
features = np.load(FEATURE_PATH)  # (N, C)

# K-means clustering
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=0)
labels = kmeans.fit_predict(features)  # (N,)

# Print cluster distribution
label_counts = collections.Counter(labels)
print("Cluster distribution (cluster: count):")
for label, count in label_counts.most_common():
    print(f"{label}: {count}")

# Plot cluster distribution
plt.figure(figsize=(10,4))
plt.bar(label_counts.keys(), label_counts.values())
plt.xlabel('Cluster')
plt.ylabel('Voxel count')
plt.title('Voxel Cluster Distribution (KMeans)')
plt.show()

# Print feature statistics (mean, std, min, max per channel)
print("\nFeature statistics per channel:")
means = features.mean(axis=0)
stds = features.std(axis=0)
mins = features.min(axis=0)
maxs = features.max(axis=0)
for i in range(features.shape[1]):
    print(f"Channel {i}: mean={means[i]:.4f}, std={stds[i]:.4f}, min={mins[i]:.4f}, max={maxs[i]:.4f}")
