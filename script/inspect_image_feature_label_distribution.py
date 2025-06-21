import numpy as np
import collections
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

FEATURE_DIR = "/home/neural_fields/Unified-Lift-Gabor/data/lerf/teatime/lseg_features"
N_CLUSTERS = 10  # or use argmax for class prediction

feature_files = sorted([f for f in os.listdir(FEATURE_DIR) if f.endswith('.npy') and not f.endswith('_depth.npy')])

for fname in feature_files:
    path = os.path.join(FEATURE_DIR, fname)
    arr = np.load(path)  # shape: (C, H, W) or (H, W, C)
    if arr.shape[0] < 10:  # likely (C, H, W)
        arr = arr.transpose(1, 2, 0)  # to (H, W, C)
    flat = arr.reshape(-1, arr.shape[-1])  # (N_pixels, C)

    # Option 1: KMeans clustering
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=0)
    labels = kmeans.fit_predict(flat)

    # Option 2: Argmax (if features are class scores)
    # labels = np.argmax(flat, axis=1)

    label_counts = collections.Counter(labels)
    print(f"{fname} label/cluster distribution:")
    for label, count in label_counts.most_common():
        print(f"  {label}: {count}")
    print()

    # Optional: plot
    plt.figure(figsize=(8,3))
    plt.bar(label_counts.keys(), label_counts.values())
    plt.title(f"{fname} label/cluster distribution")
    plt.xlabel("Label/Cluster")
    plt.ylabel("Pixel count")
    plt.show()
