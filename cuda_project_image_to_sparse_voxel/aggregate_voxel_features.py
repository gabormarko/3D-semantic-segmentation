import torch
import glob
import os

# Directory containing per-image projection outputs
dir_proj = "proj_output"
output_avg = "average_voxel_features.pt"
output_hits = "voxel_hit_counts.pt"

# Find all projection output files
glob_pattern = os.path.join(dir_proj, "proj_output_*.pt")
proj_files = sorted(glob.glob(glob_pattern))

if not proj_files:
    raise RuntimeError(f"No projection output files found in {dir_proj}")

# Load the first file to get shapes
data0 = torch.load(proj_files[0])
# Expecting a dict with keys: 'proj_feats' (N_vox, F), 'mapping2dto3d' (N_vox,)
proj_feats = data0['proj_feats']
num_voxels, feat_dim = proj_feats.shape

# Accumulators
feature_sum = torch.zeros((num_voxels, feat_dim), dtype=proj_feats.dtype)
hit_count = torch.zeros((num_voxels,), dtype=torch.long)

for f in proj_files:
    d = torch.load(f)
    feats = d['proj_feats']  # (N_vox, F)
    # A voxel is hit if any feature is nonzero (or use mapping2dto3d if available)
    if 'mapping2dto3d' in d:
        hit_mask = (d['mapping2dto3d'] > 0)
    else:
        hit_mask = feats.abs().sum(dim=1) > 0
    feature_sum[hit_mask] += feats[hit_mask]
    hit_count[hit_mask] += 1

# Avoid division by zero
avg_feats = torch.zeros_like(feature_sum)
nonzero = hit_count > 0
avg_feats[nonzero] = feature_sum[nonzero] / hit_count[nonzero].unsqueeze(1)

torch.save({'avg_feats': avg_feats, 'hit_count': hit_count}, output_avg)
torch.save(hit_count, output_hits)
print(f"Saved average features to {output_avg} and hit counts to {output_hits}")
