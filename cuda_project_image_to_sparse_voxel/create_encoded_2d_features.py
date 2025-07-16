import os
import numpy as np
import torch

# Paths
features_dir = '/home/neural_fields/Unified-Lift-Gabor/data/scannetpp/officescene/lseg_features'
images_dir = '/home/neural_fields/Unified-Lift-Gabor/data/scannetpp/officescene/images'
output_dir = '/home/neural_fields/Unified-Lift-Gabor/cuda_project_image_to_sparse_voxel/input'
os.makedirs(output_dir, exist_ok=True)

# List all .npy feature files and corresponding images
feature_files = sorted([f for f in os.listdir(features_dir) if f.endswith('.npy')])
image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

# Map feature files to image files by name (without extension)
feature_to_image = {os.path.splitext(f)[0]: f for f in feature_files}
image_to_feature = {os.path.splitext(f)[0]: f for f in image_files if os.path.splitext(f)[0] in feature_to_image}

print("feature_files:", feature_files)
print("image_files:", image_files)
print("feature_to_image:", feature_to_image)
print("image_to_feature:", image_to_feature)

# Collect features and check shapes
features_list = []
for img_base, img_file in image_to_feature.items():
    feat_file = feature_to_image[img_base]
    feat_path = os.path.join(features_dir, feat_file)
    feat = np.load(feat_path)  # shape: (H, W, C) or (C, H, W)
    # Ensure shape is (H, W, C)
    if feat.ndim == 3 and feat.shape[0] < 10:  # likely (C, H, W)
        feat = np.transpose(feat, (1, 2, 0))
    features_list.append(feat)
    print(f"Loaded {feat_file}: shape {feat.shape}")

# Stack into (view_num, H, W, C)
features_arr = np.stack(features_list, axis=0)  # (V, H, W, C)
print(f"Stacked features: {features_arr.shape}")

# Add batch dimension: (B, V, H, W, C)
features_arr = features_arr[None, ...]  # (1, V, H, W, C)
print(f"Final encoded_2d_features shape: {features_arr.shape}")

# Save as torch tensor
encoded_2d_features = torch.from_numpy(features_arr).float()
torch.save(encoded_2d_features, os.path.join(output_dir, 'encoded_2d_features.pt'))
print(f"Saved encoded_2d_features.pt to {output_dir}")
