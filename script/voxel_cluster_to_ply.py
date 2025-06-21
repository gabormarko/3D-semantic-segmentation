import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# === User parameters ===
# Path to the voxel grid PLY file (positions)
PLY_PATH = "/home/neural_fields/Unified-Lift-Gabor/output/minkowski_grid/teatime_minkowski_68458vox_iter30000_grid.ply"
# Path to the voxel features (N_voxels, N_features)
FEATURE_PATH = "/home/neural_fields/Unified-Lift-Gabor/output/minkowski_grid/teatime_voxel_features.npy"
# Output path for colored point cloud
OUTPUT_PLY = "/home/neural_fields/Unified-Lift-Gabor/output/minkowski_grid/clustered_voxel_labels.ply"
# Number of clusters
N_CLUSTERS = 10

# === Load voxel positions from PLY ===
pcd = o3d.io.read_point_cloud(PLY_PATH)
positions = np.asarray(pcd.points)  # (N, 3)

# === Load features ===
features = np.load(FEATURE_PATH)    # (N, C)
if features.shape[0] != positions.shape[0]:
    raise ValueError(f"Mismatch: {features.shape[0]} features vs {positions.shape[0]} positions")

# === K-means clustering ===
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=0)
labels = kmeans.fit_predict(features)  # (N,)

# === Assign a color to each cluster (using matplotlib colormap) ===
cmap = plt.get_cmap('tab10' if N_CLUSTERS <= 10 else 'tab20')
colors = np.array([cmap(l % cmap.N)[:3] for l in labels])  # (N, 3), values in [0,1]

# === Create new point cloud with cluster colors ===
pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.io.write_point_cloud(OUTPUT_PLY, pcd)
print(f"Clustered voxel label point cloud written to {OUTPUT_PLY}")
