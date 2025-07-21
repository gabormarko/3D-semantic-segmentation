import os
import glob
import torch
import subprocess
import shutil
import numpy as np
import re
import cv2
from collections import defaultdict
import argparse

# Option to only process the first input image for debug
parser = argparse.ArgumentParser(description='Aggregate voxel features pipeline')
parser.add_argument('--first_only', action='store_true', help='Only process the first input image for debug')
args = parser.parse_args()

# Create output directory for checkpoints and final outputs
CHECKPOINT_DIR = "voxel_feature_checkpoints_vox96741_filtered"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Paths and config

LSEG_DIR = "/home/neural_fields/Unified-Lift-Gabor/data/scannetpp/officescene/lseg_embed_features/features"
CAM_PARAMS_ORIG = "camera_params/camera_params.json"
CAM_PARAMS_DS = "camera_params/camera_params_downsampled.json"
OCCUPANCY = "ALL_occupancy.pt"
VOXEL_PLY = "/home/neural_fields/Unified-Lift-Gabor/output/minkowski_grid/officescene_filtered_ply_adv/cc946f8b-f_minkowski_96741vox_iter_opac0.8_cell0.04_eps0.06_neig8_grid.ply"
TENSOR_DATA_TMP = "tensor_data/tmp_tensor_data.pt"

# Extract voxel size and grid origin from the PLY file (as in bash script)
# --- Create downsampled camera params JSON ---
import json
downsample_factor = 1
if not os.path.exists(CAM_PARAMS_DS):
    with open(CAM_PARAMS_ORIG, 'r') as f:
        cam_data = json.load(f)
    # Update intrinsics and image size for each camera
    for cam in cam_data.get('cameras', []):
        if 'intrinsic' in cam:
            cam['intrinsic'][0] *= downsample_factor  # fx
            cam['intrinsic'][1] *= downsample_factor  # fy
            cam['intrinsic'][2] *= downsample_factor  # cx
            cam['intrinsic'][3] *= downsample_factor  # cy
        if 'width' in cam:
            cam['width'] = int(cam['width'] * downsample_factor)
        if 'height' in cam:
            cam['height'] = int(cam['height'] * downsample_factor)
    with open(CAM_PARAMS_DS, 'w') as f:
        json.dump(cam_data, f, indent=2)
    print(f"[INFO] Created downsampled camera params: {CAM_PARAMS_DS}")

# --- DEBUG: Print camera intrinsics from downsampled JSON ---
if os.path.exists(CAM_PARAMS_DS):
    with open(CAM_PARAMS_DS, 'r') as f:
        cam_data_ds = json.load(f)
    print("[DEBUG] Camera intrinsics from downsampled JSON:")
    for i, cam in enumerate(cam_data_ds.get('cameras', [])):
        if 'intrinsic' in cam:
            fx, fy, cx, cy = cam['intrinsic']
            print(f"  Camera {i}: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
        if 'width' in cam and 'height' in cam:
            print(f"    Image size: width={cam['width']}, height={cam['height']}")
def extract_voxel_params(ply_path):
    voxel_size = None
    grid_origin = None
    grid_shape = None
    num_voxels_from_name = None
    # Try to extract NUM_VOXELS from filename (e.g., ..._6680vox_...)
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
    # print(f"[INFO] Using VOXEL_SIZE={VOXEL_SIZE}, GRID_ORIGIN={GRID_ORIGIN}, GRID_SHAPE={GRID_SHAPE}, NUM_VOXELS={NUM_VOXELS}")

feature_files = sorted(glob.glob(os.path.join(LSEG_DIR, '*.npy')))
if not feature_files:
    raise RuntimeError(f"No .npy feature files found in {LSEG_DIR}")

# Use downsampled camera params for all subsequent steps
CAM_PARAMS = CAM_PARAMS_DS

# If debug mode, only keep the first file
if args.first_only:
    # print("[DEBUG] Only processing the first input image for debug.")
    feature_files = feature_files[:1]

# Prepare for aggregation using 3D voxel indices
# --- Always (re)create ALL_occupancy.pt before aggregation ---
print(f"[INFO] Creating (or overwriting) occupancy tensor using build_sparse_occupancy.py ...")
build_occ_cmd = [
    'python', 'build_sparse_occupancy.py',
    '--voxel_ply', VOXEL_PLY,
    '--voxel_size', str(VOXEL_SIZE),
    '--grid_origin', str(GRID_ORIGIN[0]), str(GRID_ORIGIN[1]), str(GRID_ORIGIN[2]),
    '--out_tensor', OCCUPANCY
]
result = subprocess.run(build_occ_cmd)
if result.returncode != 0 or not os.path.exists(OCCUPANCY):
    raise RuntimeError(f"Failed to create {OCCUPANCY} using build_sparse_occupancy.py")
    # print(f"[INFO] Created {OCCUPANCY} successfully.")

import tempfile
voxel_feature_sum = defaultdict(lambda: None)  # key: (i, j, k), value: feature sum (torch.Tensor)
voxel_hit_count = defaultdict(int)  # key: (i, j, k), value: hit count
feat_dim = None

# --- DEBUG: Save initial occupied voxels as .ply before aggregation ---
# --- Use Open3D and occupancy3D_to_ply.py logic for robust debug PLY export ---
try:
    import open3d as o3d
    if os.path.exists(OCCUPANCY):
        occ_data = torch.load(OCCUPANCY)
        occ_grid = None
        if isinstance(occ_data, dict):
            for k in occ_data:
                if 'occupancy' in k and isinstance(occ_data[k], torch.Tensor) and occ_data[k].ndim == 3:
                    occ_grid = occ_data[k]
                    break
        elif isinstance(occ_data, torch.Tensor) and occ_data.ndim == 3:
            occ_grid = occ_data
        if occ_grid is not None:
            occupied = (occ_grid > 0).nonzero()
            occupied_indices = np.array([(x, y, z) for z, y, x in occupied.tolist()], dtype=np.int32)
            grid_origin = np.array(GRID_ORIGIN)
            voxel_size = VOXEL_SIZE
            occupied_world = occupied_indices * voxel_size + grid_origin
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(occupied_world)
            pcd.paint_uniform_color([1, 1, 1])  # white
            ply_path = os.path.join(CHECKPOINT_DIR, f'debug_initial_occupied_voxels_vox{NUM_VOXELS}.ply')
            o3d.io.write_point_cloud(ply_path, pcd, write_ascii=True)
            # print(f"[DEBUG] Saved initial occupied voxels as: {ply_path} ({occupied_world.shape[0]} points)")
        else:
            pass
    else:
        pass
except ImportError:
    pass


# Aggregate over all images using 3D voxel indices
for fpath in feature_files:
    # Print original camera intrinsics before update
    with open(CAM_PARAMS_ORIG, 'r') as f:
        orig_cam_data = json.load(f)
    for cam in orig_cam_data.get('cameras', []):
        if 'intrinsic' in cam:
            print(f"[INTRINSICS] Original: fx={cam['intrinsic'][0]}, fy={cam['intrinsic'][1]}, cx={cam['intrinsic'][2]}, cy={cam['intrinsic'][3]}")
        if 'width' in cam and 'height' in cam:
            print(f"[INTRINSICS] Original image size: width={cam['width']}, height={cam['height']}")

    img_name = os.path.basename(fpath)[:-4]
    # Print original camera intrinsics and image size for this image before downsampling
    with open(CAM_PARAMS, 'r') as f:
        cam_data_pre = json.load(f)
    for cam in cam_data_pre.get('cameras', []):
        if 'intrinsic' in cam:
            print(f"[INTRINSICS][PRE-DS] {img_name}: fx={cam['intrinsic'][0]}, fy={cam['intrinsic'][1]}, cx={cam['intrinsic'][2]}, cy={cam['intrinsic'][3]}")
        if 'width' in cam and 'height' in cam:
            print(f"[INTRINSICS][PRE-DS] {img_name}: width={cam['width']}, height={cam['height']}")
    import cv2
    with tempfile.TemporaryDirectory() as tmpdir:
        # Find corresponding image file
        img_name = os.path.basename(fpath)[:-4]
        images_dir = '/home/neural_fields/Unified-Lift-Gabor/data/scannetpp/officescene/images'
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            candidate = os.path.join(images_dir, img_name + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break
        if img_path is None:
            candidate = os.path.join(images_dir, img_name)
            if os.path.exists(candidate):
                img_path = candidate
        if img_path is None:
            print(f"[ERROR] Could not find image for {img_name}")
            continue

        # Downsample image and update camera intrinsics
        downsample_factor = 0.5  # Set this factor as needed
        img = cv2.imread(img_path)
        if img is None:
            print(f"[ERROR] Could not load image: {img_path}")
            continue
        H_orig, W_orig = img.shape[:2]
        H_new, W_new = int(H_orig * downsample_factor), int(W_orig * downsample_factor)
        img_ds = cv2.resize(img, (W_new, H_new), interpolation=cv2.INTER_AREA)
        img_ds_path = os.path.join(tmpdir, img_name + '_ds.png')
        cv2.imwrite(img_ds_path, img_ds)

        # Update camera params JSON for this image
        with open(CAM_PARAMS, 'r') as f:
            cam_data = json.load(f)
        for cam in cam_data.get('cameras', []):
            if 'intrinsic' in cam:
                cam['intrinsic'][0] *= downsample_factor
                cam['intrinsic'][1] *= downsample_factor
                cam['intrinsic'][2] *= downsample_factor
                cam['intrinsic'][3] *= downsample_factor
            if 'width' in cam:
                cam['width'] = int(cam['width'] * downsample_factor)
            if 'height' in cam:
                cam['height'] = int(cam['height'] * downsample_factor)
        cam_params_dst = os.path.join(tmpdir, 'camera_params_downsampled.json')
        with open(cam_params_dst, 'w') as f:
            json.dump(cam_data, f, indent=2)
        # Print updated camera intrinsics after downsampling
        for cam in cam_data.get('cameras', []):
            if 'intrinsic' in cam:
                print(f"[INTRINSICS] Updated: fx={cam['intrinsic'][0]}, fy={cam['intrinsic'][1]}, cx={cam['intrinsic'][2]}, cy={cam['intrinsic'][3]}")
            if 'width' in cam and 'height' in cam:
                print(f"[INTRINSICS] Updated image size: width={cam['width']}, height={cam['height']}")

        # Save feature map as-is
        shutil.copy(fpath, tmpdir)
        ds_fpath = os.path.join(tmpdir, os.path.basename(fpath))

        # Run feature extraction, passing new image size
        subprocess.run([
            'python', 'prepare_tensor_data.py',
            '--lseg_dir', tmpdir,
            '--scaled_camera_params', cam_params_dst,
            '--occupancy', OCCUPANCY,
            '--voxel_size', str(VOXEL_SIZE),
            '--grid_origin', str(GRID_ORIGIN[0]), str(GRID_ORIGIN[1]), str(GRID_ORIGIN[2]),
            '--max_images', '1',
            '--output', TENSOR_DATA_TMP,
            '--image_size', str(H_new), str(W_new),
            '--downsample_factor', str(downsample_factor)
        ], check=True)

        # --- STEP 2: Debug feature map dimensions after prepare_tensor_data.py ---
        if os.path.exists(TENSOR_DATA_TMP) and os.path.getsize(TENSOR_DATA_TMP) > 0:
            tensor_data = torch.load(TENSOR_DATA_TMP, map_location='cpu')
            # Print intrinsics from tensor_data (used in CUDA)
            intr = tensor_data.get('intrinsicParams', None)
            if intr is not None:
                print(f"[INTRINSICS] Used in CUDA: {intr.flatten().tolist()}")
            feats = tensor_data.get('encoded_2d_features', None)
            print(f"[STEP 2] Encoded feature tensor shape: {feats.shape if feats is not None else None}")
            if feats is not None:
                _, _, Hf, Wf, Cf = feats.shape
                print(f"[STEP 2] Encoded feature map spatial size: H={Hf}, W={Wf}")
                print(f"[STEP 2] Feature map channel count: C={Cf}")
                # No longer referencing feat_arr, just check shape consistency
                if Hf != H_new or Wf != W_new:
                    print(f"[STEP 2][WARN] Encoded feature map size does not match expected image size! (Expected H={H_new}, W={W_new})")
                else:
                    print(f"[STEP 2][OK] Encoded feature map size matches expected image size.")
                # Print camera intrinsics from tensor_data
                intr = tensor_data.get('intrinsicParams', None)
                if intr is not None:
                    print(f"[STEP 2] Camera intrinsics from tensor_data: {intr.flatten().tolist()}")
            else:
                print(f"[STEP 2][ERROR] Feature map not found in tensor_data!")
        else:
            print(f"[STEP 2][ERROR] tensor_data not found or empty after feature extraction!")

        # --- STEP 3: Project features to voxels and print CUDA projection inputs ---
        print(f"[STEP 3] Running debug_project_features.py ...")
        subprocess.run([
            'python', 'debug_project_features.py',
            '--tensor_data', TENSOR_DATA_TMP,
            '--output', 'proj_output.pt'
        ], check=True)
        output_dict = torch.load('proj_output.pt')
        if 'projected_feats' not in output_dict or 'projected_indices' not in output_dict:
            raise KeyError("'projected_feats' or 'projected_indices' not found in proj_output.pt")
        feats = output_dict['projected_feats']
        indices = output_dict['projected_indices']  # shape: (N, 3) int tensor, 3D voxel indices
        print(f"[STEP 3] CUDA projection input: feature tensor shape={feats.shape}, indices shape={indices.shape}")
        if feats.shape[0] == 0:
            print(f"[STEP 3][ERROR] No features projected for {img_name}!")
        if torch.isnan(feats).any() or torch.isinf(feats).any():
            print(f"[STEP 3][ERROR] NaN or Inf detected in projected features for {img_name}")
        if feat_dim is None:
            feat_dim = feats.shape[1]
        for idx, feat in zip(indices, feats):
            idx_tuple = tuple(idx.tolist())
            if voxel_feature_sum[idx_tuple] is None:
                voxel_feature_sum[idx_tuple] = feat.clone()
            else:
                voxel_feature_sum[idx_tuple] += feat
            voxel_hit_count[idx_tuple] += 1
        os.remove(TENSOR_DATA_TMP)
        os.remove('proj_output.pt')

    idx = feature_files.index(fpath) + 1
    if idx % 20 == 0:
        # Save checkpoint: average features and hit counts for all occupied voxels
        avg_feats_dict = {}
        for k, v in voxel_feature_sum.items():
            if v is not None and voxel_hit_count[k] > 0:
                avg_feats_dict[k] = v / voxel_hit_count[k]
        # torch.save({'avg_feats': avg_feats_dict, 'hit_count': dict(voxel_hit_count)}, os.path.join(CHECKPOINT_DIR, f'average_voxel_features_{idx}.pt'))
        # print(f"[CHECKPOINT] Saved {CHECKPOINT_DIR}/average_voxel_features_{idx}.pt after {idx} images.")

        # Also save .ply for visualization with color as hit count
        # Compute world coordinates for all occupied voxels using the same logic as initial occupied voxels
        import open3d as o3d
        occupied_keys = list(avg_feats_dict.keys())
        if len(occupied_keys) == 0:
            print(f"[PLY] No occupied voxels at checkpoint {idx}, skipping PLY export and world_xyz debug.")
        else:
            # Consolidate checkpoint data into tensors
            avg_feats = np.stack([avg_feats_dict[k].cpu().numpy() for k in occupied_keys], axis=0)
            hit_counts = np.array([voxel_hit_count[k] for k in occupied_keys], dtype=np.int32)
            voxel_coords = np.array(occupied_keys, dtype=np.int32)

            # Convert (z, y, x) to (x, y, z) for visualization, then scale and add grid_origin
            occupied_indices = np.array([(k[2], k[1], k[0]) for k in occupied_keys], dtype=np.int32)
            grid_origin = np.array(GRID_ORIGIN)
            voxel_size = VOXEL_SIZE
            world_xyz = occupied_indices * voxel_size + grid_origin
            
            # Save consolidated checkpoint data as a single .pt file
            checkpoint_save_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_features_{idx}.pt')
            torch.save({
                'xyz': torch.from_numpy(world_xyz),
                'avg_feats': torch.from_numpy(avg_feats),
                'hit_count': torch.from_numpy(hit_counts),
                'voxel_coords': torch.from_numpy(voxel_coords)
            }, checkpoint_save_path)
            print(f"[CHECKPOINT] Saved consolidated checkpoint data as: {checkpoint_save_path}")

            # Swap Y and Z axes to match camera coordinate system
            #print("[DEBUG] Swapping Y and Z axes of world coordinates for visualization.")
            # world_xyz = world_xyz_orig[:, [0, 2, 1]]
            """
            # Debug: print a few mappings from (z, y, x) to world coordinates
            print("[DEBUG] First 5 voxel index mappings (z, y, x) -> world_xyz:")
            for i in range(min(5, len(occupied_keys))):
                k = occupied_keys[i]
                print(f"  Voxel index (z={k[0]}, y={k[1]}, x={k[2]}) -> world = {world_xyz[i]}")
            # Debug: print min/max for each world axis
            print("[DEBUG] world_xyz axis ranges:")
            for axis, name in enumerate(['x', 'y', 'z']):
                print(f"  {name}: min={world_xyz[:, axis].min():.3f}, max={world_xyz[:, axis].max():.3f}")
            """
            # Use pink color for all hit voxels
            colors = np.tile(np.array([255, 0, 255], dtype=np.uint8), (world_xyz.shape[0], 1))
            # Save as Open3D point cloud for robust PLY export
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(world_xyz)
            pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
            ply_path = os.path.join(CHECKPOINT_DIR, f'nonzero_voxels_hitcount_{idx}_vox{NUM_VOXELS}.ply')
            o3d.io.write_point_cloud(ply_path, pcd, write_ascii=True)
            print(f"[PLY] Saved nonzero voxels with hit count color as: {ply_path}")

# Save final average features and hit counts for all occupied voxels

avg_feats_dict = {}
world_coords_dict = {}
for k, v in voxel_feature_sum.items():
    if v is not None and voxel_hit_count[k] > 0:
        avg_feats_dict[k] = v / voxel_hit_count[k]
        # Compute world coordinates for this voxel index (z, y, x)
        world_coord = np.array([k[2], k[1], k[0]]) * VOXEL_SIZE + np.array(GRID_ORIGIN)
        world_coords_dict[k] = world_coord
# torch.save({'avg_feats': avg_feats_dict, 'hit_count': dict(voxel_hit_count), 'world_coords': world_coords_dict}, os.path.join(CHECKPOINT_DIR, 'average_voxel_features.pt'))
# print(f"[DONE] Aggregated average features, hit counts, and world coordinates saved in {CHECKPOINT_DIR}.")

# Save nonzero voxel world coordinates and features for visualization/analysis
occupied_keys = list(avg_feats_dict.keys())
if len(occupied_keys) == 0:
    print("[DONE] No occupied voxels in final aggregation, skipping world_xyz debug, PLY, and NPZ export.")
else:
    # Consolidate all data into tensors
    avg_feats = np.stack([avg_feats_dict[k].cpu().numpy() for k in occupied_keys], axis=0)
    hit_counts = np.array([voxel_hit_count[k] for k in occupied_keys], dtype=np.int32)
    voxel_coords = np.array(occupied_keys, dtype=np.int32) # (z, y, x)

    # Map (z, y, x) -> (x, y, z) for world coordinates
    # CUDA uses (z, y, x) order for indices
    world_xyz_orig = np.array([
        np.array([k[2], k[1], k[0]]) * VOXEL_SIZE + np.array([GRID_ORIGIN[0], GRID_ORIGIN[1], GRID_ORIGIN[2]])
        for k in occupied_keys
    ], dtype=np.float32)
    world_xyz = world_xyz_orig # Assign world_xyz from the original coordinates
    # Swap Y and Z axes to match camera coordinate system
    #print("[DEBUG] Swapping Y and Z axes of final world coordinates.")
    #world_xyz = world_xyz_orig[:, [0, 2, 1]]
    """
    # Debug: print a few mappings from (z, y, x) to world coordinates
    print("[DEBUG] First 5 voxel index mappings (z, y, x) -> world_xyz:")
    for i in range(min(5, len(occupied_keys))):
        k = occupied_keys[i]
        print(f"  Voxel index (z={k[0]}, y={k[1]}, x={k[2]}) -> world = {world_xyz[i]}")

    # Debug: print min/max for each world axis
    print("[DEBUG] world_xyz axis ranges:")
    for axis, name in enumerate(['x', 'y', 'z']):
        print(f"  {name}: min={world_xyz[:, axis].min():.3f}, max={world_xyz[:, axis].max():.3f}")
    """
    # Save as .ply for visualization
    ply_path = os.path.join(CHECKPOINT_DIR, f'ALL_nonzero_voxels_with_features_{idx}_vox{NUM_VOXELS}.ply')
    with open(ply_path, 'w') as f:
        f.write('ply\nformat ascii 1.0\n')
        f.write(f'element vertex {world_xyz.shape[0]}\n')
        f.write('property float x\nproperty float y\nproperty float z\n')
        if avg_feats.shape[1] >= 3:
            f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
        f.write('end_header\n')
        for i, pt in enumerate(world_xyz):
            line = f'{pt[0]} {pt[1]} {pt[2]}'
            if avg_feats.shape[1] >= 3:
                rgb = np.clip(avg_feats[i, :3], 0, 1) * 255
                rgb = rgb.astype(np.uint8)
                line += f' {rgb[0]} {rgb[1]} {rgb[2]}'
            f.write(line + '\n')
    print(f"[PLY] Saved nonzero voxels with features as: {ply_path}")

    # Save only features and indices for occupied voxels, compressing before saving
    save_path = os.path.join(CHECKPOINT_DIR, f'ALL_nonzero_voxel_features_{idx}_vox{NUM_VOXELS}.pt')
    # Convert features to float16 for compression
    avg_feats_compressed = torch.from_numpy(avg_feats).to(torch.float16)
    # Save only world coordinates and compressed features
    torch.save({
        'xyz': torch.from_numpy(world_xyz),
        'avg_feats': avg_feats_compressed,
        'voxel_coords': torch.from_numpy(voxel_coords)
    }, save_path)
    print(f"[PT] Saved filtered and compressed voxel data (xyz, features, coords) as: {save_path}")
    print(f"[DONE] PROJECTION PIPELINE COMPLETED. Checkpoint directory: {CHECKPOINT_DIR}")