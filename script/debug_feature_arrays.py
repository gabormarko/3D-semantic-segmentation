import os
import numpy as np

# FEATURE_DIR = "/home/neural_fields/Unified-Lift-Gabor/data/lerf/teatime/lseg_features"
FEATURE_DIR = "/home/neural_fields/Unified-Lift-Gabor/output/minkowski_grid"


feature_files = [f for f in os.listdir(FEATURE_DIR) if f.endswith('.npy') and not f.endswith('_depth.npy')]

print(f"Found {len(feature_files)} feature files.")

for fname in sorted(feature_files):
    path = os.path.join(FEATURE_DIR, fname)
    arr = np.load(path)
    arr = arr.astype(np.float32)
    # print(arr)
    # Transpose if needed
    if arr.shape[0] < 10:  # [C, H, W] -> [H, W, C]
        arr = arr.transpose(1, 2, 0)
    flat = arr.flatten()
    n_nan = np.isnan(flat).sum()
    n_inf = np.isinf(flat).sum()
    unique_vals = np.unique(flat)
    print(f"{fname}: shape={arr.shape}, mean={np.mean(flat):.4f}, std={np.std(flat):.4f}, min={np.min(flat):.4f}, max={np.max(flat):.4f}, NaN={n_nan}, Inf={n_inf}, unique_vals={len(unique_vals)}")
    if n_nan > 0 or n_inf > 0:
        print(f"  WARNING: NaN or Inf detected in {fname}")
    if len(unique_vals) == 1:
        print(f"  WARNING: All values are constant ({unique_vals[0]}) in {fname}")

print("Feature array check complete.")