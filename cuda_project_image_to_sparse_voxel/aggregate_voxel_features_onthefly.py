import os
import glob
import torch
import subprocess
import shutil

# Create output directory for checkpoints and final outputs
CHECKPOINT_DIR = "voxel_feature_checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Paths and config
LSEG_DIR = "../data/scannetpp/officescene/lseg_features"
CAM_PARAMS = "camera_params/camera_params.json"
OCCUPANCY = "ALL_occupancy.pt"
VOXEL_PLY = "/home/neural_fields/Unified-Lift-Gabor/output/minkowski_grid/officescene/officescene_minkowski_9434vox_iter50000_grid.ply"
TENSOR_DATA_TMP = "tensor_data/tmp_tensor_data.pt"

# Extract voxel size and grid origin from the PLY file (as in bash script)
def extract_voxel_params(ply_path):
    voxel_size = None
    grid_origin = None
    with open(ply_path, 'rb') as f:
        for line in f:
            try:
                line = line.decode('ascii')
            except:
                break
            if 'comment voxel_size' in line:
                voxel_size = float(line.split()[-1])
            if 'comment grid_origin' in line:
                grid_origin = [float(x) for x in line.split()[-3:]]
            if 'end_header' in line:
                break
    if voxel_size is None or grid_origin is None:
        raise RuntimeError("Could not extract voxel_size or grid_origin from PLY header")
    return voxel_size, grid_origin

VOXEL_SIZE, GRID_ORIGIN = extract_voxel_params(VOXEL_PLY)

# Find all LSEG feature files
feature_files = sorted(glob.glob(os.path.join(LSEG_DIR, '*.npy')))
if not feature_files:
    raise RuntimeError(f"No .npy feature files found in {LSEG_DIR}")

# Prepare the first tensor data to get voxel/feature shape
import tempfile
with tempfile.TemporaryDirectory() as tmpdir:
    first_npy = feature_files[0]
    shutil.copy(first_npy, tmpdir)
    subprocess.run([
        'python', 'prepare_tensor_data.py',
        '--lseg_dir', tmpdir,
        '--scaled_camera_params', CAM_PARAMS,
        '--occupancy', OCCUPANCY,
        '--voxel_size', str(VOXEL_SIZE),
        '--grid_origin', str(GRID_ORIGIN[0]), str(GRID_ORIGIN[1]), str(GRID_ORIGIN[2]),
        '--max_images', '1',
        '--output', TENSOR_DATA_TMP
    ], check=True)

# Run projection kernel on the first image to get output shape

# Run projection kernel on the first image to get output shape
    subprocess.run([
        'python', 'debug_project_features.py',
        '--tensor_data', TENSOR_DATA_TMP,
        '--output', 'proj_output.pt'
    ], check=True)
    output_dict = torch.load('proj_output.pt')
    if 'projected_feats' not in output_dict:
        print("[ERROR] 'projected_feats' not found in proj_output.pt. Available keys:", list(output_dict.keys()))
        raise KeyError("'projected_feats' not found in proj_output.pt")
    proj_feats = output_dict['projected_feats']
    num_voxels, feat_dim = proj_feats.shape
    feature_sum = torch.zeros((num_voxels, feat_dim), dtype=proj_feats.dtype)
    hit_count = torch.zeros((num_voxels,), dtype=torch.long)

# Clean up temp tensor data
os.remove(TENSOR_DATA_TMP)

# Aggregate over all images
for fpath in feature_files:
    img_name = os.path.basename(fpath)[:-4]
    print(f"[INFO] Processing {img_name} ...")
    with tempfile.TemporaryDirectory() as tmpdir:
        shutil.copy(fpath, tmpdir)
        # Copy camera_params.json into temp dir
        cam_params_dst = os.path.join(tmpdir, os.path.basename(CAM_PARAMS))
        shutil.copy(CAM_PARAMS, cam_params_dst)
        subprocess.run([
            'python', 'prepare_tensor_data.py',
            '--lseg_dir', tmpdir,
            '--scaled_camera_params', cam_params_dst,
            '--occupancy', OCCUPANCY,
            '--voxel_size', str(VOXEL_SIZE),
            '--grid_origin', str(GRID_ORIGIN[0]), str(GRID_ORIGIN[1]), str(GRID_ORIGIN[2]),
            '--max_images', '1',
            '--output', TENSOR_DATA_TMP
        ], check=True)
        # Run projection kernel and get output
        subprocess.run([
            'python', 'debug_project_features.py',
            '--tensor_data', TENSOR_DATA_TMP,
            '--output', 'proj_output.pt'
        ], check=True)
        output_dict = torch.load('proj_output.pt')
        if 'projected_feats' not in output_dict:
            print(f"[ERROR] 'projected_feats' not found in proj_output.pt for {img_name}. Available keys:", list(output_dict.keys()))
            raise KeyError("'projected_feats' not found in proj_output.pt")
        feats = output_dict['projected_feats']
        if 'mapping2dto3d' in output_dict:
            hit_mask = (output_dict['mapping2dto3d'] > 0)
        else:
            hit_mask = feats.abs().sum(dim=1) > 0
        feature_sum[hit_mask] += feats[hit_mask]
        hit_count[hit_mask] += 1
        os.remove(TENSOR_DATA_TMP)
        os.remove('proj_output.pt')

    # Periodic checkpointing every 25 images
    idx = feature_files.index(fpath) + 1
    if idx % 25 == 0:
        avg_feats = torch.zeros_like(feature_sum)
        nonzero = hit_count > 0
        avg_feats[nonzero] = feature_sum[nonzero] / hit_count[nonzero].unsqueeze(1)
        torch.save({'avg_feats': avg_feats, 'hit_count': hit_count}, os.path.join(CHECKPOINT_DIR, f'average_voxel_features_{idx}.pt'))
        torch.save(hit_count, os.path.join(CHECKPOINT_DIR, f'voxel_hit_counts_{idx}.pt'))
        print(f"[CHECKPOINT] Saved {CHECKPOINT_DIR}/average_voxel_features_{idx}.pt and voxel_hit_counts_{idx}.pt after {idx} images.")

# Compute final average and save
avg_feats = torch.zeros_like(feature_sum)
nonzero = hit_count > 0
avg_feats[nonzero] = feature_sum[nonzero] / hit_count[nonzero].unsqueeze(1)

torch.save({'avg_feats': avg_feats, 'hit_count': hit_count}, os.path.join(CHECKPOINT_DIR, 'average_voxel_features.pt'))
torch.save(hit_count, os.path.join(CHECKPOINT_DIR, 'voxel_hit_counts.pt'))
print(f"[DONE] Aggregated average features and hit counts saved in {CHECKPOINT_DIR}.")
