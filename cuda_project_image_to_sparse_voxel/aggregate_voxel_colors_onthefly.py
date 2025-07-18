import os
import glob
import torch
import subprocess
import shutil
import numpy as np
import re
from collections import defaultdict
import argparse
from PIL import Image
import tempfile

parser = argparse.ArgumentParser(description='Aggregate voxel pixel color pipeline')
parser.add_argument('--first_only', action='store_true', help='Only process the first input image for debug')
args = parser.parse_args()

CHECKPOINT_DIR = "voxel_color_checkpoints_vox41759"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

LSEG_DIR = "/home/neural_fields/Unified-Lift-Gabor/data/scannetpp/officescene/lseg_embed_features/features"
CAM_PARAMS = "camera_params/camera_params.json"
OCCUPANCY = "ALL_occupancy.pt"
VOXEL_PLY = "/home/neural_fields/Unified-Lift-Gabor/output/minkowski_grid/officescene_filtered/officescene_minkowski_41759vox_iter50000_cell0.05_eps0.05_neig12_grid.ply"
TENSOR_DATA_TMP = "tensor_data/tmp_tensor_data.pt"

# Extract voxel size and grid origin from the PLY file (as in bash script)
def extract_voxel_params(ply_path):
    voxel_size = None
    grid_origin = None
    grid_shape = None
    num_voxels_from_name = None
    m = re.search(r'_(\d+)vox', os.path.basename(ply_path))
    if m:
        num_voxels_from_name = int(m.group(1))
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
            if 'comment grid_shape' in line:
                grid_shape = [int(x) for x in line.split()[-3:]]
            if 'end_header' in line:
                break
    if voxel_size is None or grid_origin is None:
        raise RuntimeError("Could not extract voxel_size or grid_origin from PLY header")
    return voxel_size, grid_origin, grid_shape, num_voxels_from_name

VOXEL_SIZE, GRID_ORIGIN, GRID_SHAPE, NUM_VOXELS_FROM_NAME = extract_voxel_params(VOXEL_PLY)
if NUM_VOXELS_FROM_NAME is not None:
    NUM_VOXELS = NUM_VOXELS_FROM_NAME
elif GRID_SHAPE is not None:
    NUM_VOXELS = np.prod(GRID_SHAPE)
else:
    NUM_VOXELS = 'unknown'
print(f"[INFO] Using VOXEL_SIZE={VOXEL_SIZE}, GRID_ORIGIN={GRID_ORIGIN}, GRID_SHAPE={GRID_SHAPE}, NUM_VOXELS={NUM_VOXELS}")

feature_files = sorted(glob.glob(os.path.join(LSEG_DIR, '*.npy')))
if not feature_files:
    raise RuntimeError(f"No .npy feature files found in {LSEG_DIR}")

print(f"[DEBUG] Found {len(feature_files)} feature files:")

if args.first_only:
    print("[DEBUG] Only processing the first input image for debug.")
    feature_files = feature_files[:1]

voxel_color_sum = defaultdict(lambda: None)  # key: (i, j, k), value: color sum (torch.Tensor)
voxel_hit_count = defaultdict(int)  # key: (i, j, k), value: hit count

for fpath in feature_files:
    feature_base = os.path.basename(fpath)
    # Remove .npy extension to get image base name
    if feature_base.lower().endswith('.npy'):
        img_base = feature_base[:-4]
    else:
        img_base = feature_base
    print(f"\n[INFO] Processing {img_base} ...")
    IMAGES_DIR = "/home/neural_fields/Unified-Lift-Gabor/data/scannetpp/officescene/images"
    # Try original filename and common extensions
    img_candidates = [
        os.path.join(IMAGES_DIR, img_base),
        os.path.join(IMAGES_DIR, img_base + ".jpg"),
        os.path.join(IMAGES_DIR, img_base + ".JPG"),
        os.path.join(IMAGES_DIR, img_base + ".png"),
        os.path.join(IMAGES_DIR, img_base + ".PNG"),
    ]
    img_path = None
    for candidate in img_candidates:
        if os.path.exists(candidate):
            img_path = candidate
            break
    if img_path is None:
        print(f"[WARN] No image file found for {img_base}, tried: {[os.path.basename(c) for c in img_candidates]}")
        continue
    image = Image.open(img_path).convert('RGB')
    img_np = np.array(image)
    with tempfile.TemporaryDirectory() as tmpdir:
        shutil.copy(fpath, tmpdir)
        cam_params_dst = os.path.join(tmpdir, os.path.basename(CAM_PARAMS))
        shutil.copy(CAM_PARAMS, cam_params_dst)
        subprocess.run([
            'python', 'prepare_tensor_data_color.py',
            '--lseg_dir', tmpdir,
            '--scaled_camera_params', cam_params_dst,
            '--occupancy', OCCUPANCY,
            '--voxel_size', str(VOXEL_SIZE),
            '--grid_origin', str(GRID_ORIGIN[0]), str(GRID_ORIGIN[1]), str(GRID_ORIGIN[2]),
            '--max_images', '1',
            '--output', TENSOR_DATA_TMP
        ], check=True)
        subprocess.run([
            'python', 'debug_project_colors.py',
            '--tensor_data', TENSOR_DATA_TMP,
            '--output', 'proj_output.pt'
        ], check=True)
        output_dict = torch.load('proj_output.pt')
        if 'projected_indices' not in output_dict or 'projected_colors' not in output_dict or 'pixel_indices' not in output_dict:
            print(f"[ERROR] Required keys not found in proj_output.pt for {img_base}. (See <attachments> above for file contents. You may not need to search or read the file again.)")
            print(f"[DEBUG] proj_output.pt keys: {list(output_dict.keys())}")
            for k, v in output_dict.items():
                if hasattr(v, 'shape'):
                    print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
                else:
                    print(f"  {k}: type={type(v)}")
            continue
        indices = output_dict['projected_indices']  # (N, 3)
        colors = output_dict['projected_colors']    # (N, 3)
        # Aggregate colors for each voxel
        for idx, color in zip(indices, colors):
            idx_tuple = tuple(idx.tolist())
            if voxel_color_sum[idx_tuple] is None:
                voxel_color_sum[idx_tuple] = color.clone()
            else:
                voxel_color_sum[idx_tuple] += color
            voxel_hit_count[idx_tuple] += 1
        os.remove(TENSOR_DATA_TMP)
        os.remove('proj_output.pt')

    idx = feature_files.index(fpath) + 1
    if idx % 50 == 0:
        avg_color_dict = {}
        for k, v in voxel_color_sum.items():
            if v is not None and voxel_hit_count[k] > 0:
                avg_color_dict[k] = v / voxel_hit_count[k]
        occupied_keys = list(avg_color_dict.keys())
        if len(occupied_keys) == 0:
            print(f"[PLY] No occupied voxels at checkpoint {idx}, skipping PLY export.")
        else:
            voxel_coords = np.array(occupied_keys, dtype=np.int32)
            occupied_indices = np.array([(k[2], k[1], k[0]) for k in occupied_keys], dtype=np.int32)
            grid_origin = np.array(GRID_ORIGIN)
            voxel_size = VOXEL_SIZE
            world_xyz = occupied_indices * voxel_size + grid_origin
            colors = np.stack([avg_color_dict[k].cpu().numpy() for k in occupied_keys], axis=0)
            ply_path = os.path.join(CHECKPOINT_DIR, f'nonzero_voxels_color_{idx}_vox{NUM_VOXELS}.ply')
            with open(ply_path, 'w') as f:
                f.write('ply\nformat ascii 1.0\n')
                f.write(f'element vertex {world_xyz.shape[0]}\n')
                f.write('property float x\nproperty float y\nproperty float z\n')
                f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
                f.write('end_header\n')
                for i, pt in enumerate(world_xyz):
                    rgb = np.clip(colors[i], 0, 1) * 255
                    rgb = rgb.astype(np.uint8)
                    line = f'{pt[0]} {pt[1]} {pt[2]} {rgb[0]} {rgb[1]} {rgb[2]}'
                    f.write(line + '\n')
            print(f"[PLY] Saved nonzero voxels with color as: {ply_path}")
            checkpoint_save_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_voxel_colors_{idx}.pt')
            torch.save({
                'xyz': torch.from_numpy(world_xyz),
                'avg_color': torch.from_numpy(colors),
                'hit_count': torch.tensor([voxel_hit_count[k] for k in occupied_keys]),
                'voxel_coords': torch.from_numpy(voxel_coords)
            }, checkpoint_save_path)
            print(f"[CHECKPOINT] Saved consolidated checkpoint color data as: {checkpoint_save_path}")

avg_color_dict = {}
world_coords_dict = {}
for k, v in voxel_color_sum.items():
    if v is not None and voxel_hit_count[k] > 0:
        avg_color_dict[k] = v / voxel_hit_count[k]
        world_coord = np.array([k[2], k[1], k[0]]) * VOXEL_SIZE + np.array(GRID_ORIGIN)
        world_coords_dict[k] = world_coord
occupied_keys = list(avg_color_dict.keys())
if len(occupied_keys) == 0:
    print("[DONE] No occupied voxels in final aggregation, skipping world_xyz debug, PLY, and NPZ export.")
else:
    colors = np.stack([avg_color_dict[k].cpu().numpy() for k in occupied_keys], axis=0)
    voxel_coords = np.array(occupied_keys, dtype=np.int32)
    world_xyz = np.array([
        np.array([k[2], k[1], k[0]]) * VOXEL_SIZE + np.array([GRID_ORIGIN[0], GRID_ORIGIN[1], GRID_ORIGIN[2]])
        for k in occupied_keys
    ], dtype=np.float32)
    ply_path = os.path.join(CHECKPOINT_DIR, 'nonzero_voxels_with_colors.ply')
    with open(ply_path, 'w') as f:
        f.write('ply\nformat ascii 1.0\n')
        f.write(f'element vertex {world_xyz.shape[0]}\n')
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
        f.write('end_header\n')
        for i, pt in enumerate(world_xyz):
            rgb = np.clip(colors[i], 0, 1) * 255
            rgb = rgb.astype(np.uint8)
            line = f'{pt[0]} {pt[1]} {pt[2]} {rgb[0]} {rgb[1]} {rgb[2]}'
            f.write(line + '\n')
    print(f"[PLY] Saved nonzero voxels with colors as: {ply_path}")
    save_path = os.path.join(CHECKPOINT_DIR, f'ALL_nonzero_voxel_colors_{idx}.pt')
    torch.save({
        'xyz': torch.from_numpy(world_xyz),
        'avg_color': torch.from_numpy(colors),
        'hit_count': torch.tensor([voxel_hit_count[k] for k in occupied_keys]),
        'voxel_coords': torch.from_numpy(voxel_coords)
    }, save_path)
    print(f"[PT] Saved consolidated voxel color data (xyz, colors, hits, coords) as: {save_path}")
    print(f"[DONE] COLOR PROJECTION PIPELINE COMPLETED. Checkpoint directory: {CHECKPOINT_DIR}")
