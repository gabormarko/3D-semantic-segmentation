import os
import glob
import torch
import subprocess
import shutil
import numpy as np
import re
from collections import defaultdict
import argparse

# Option to only process the first input image for debug
parser = argparse.ArgumentParser(description='Aggregate voxel features pipeline')
parser.add_argument('--first_only', action='store_true', help='Only process the first input image for debug')
args = parser.parse_args()

# Create output directory for checkpoints and final outputs
CHECKPOINT_DIR = "voxel_feature_checkpoints_vox59293"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Paths and config
LSEG_DIR = "/home/neural_fields/Unified-Lift-Gabor/data/scannetpp/officescene/lseg_embed_features/features"
CAM_PARAMS = "camera_params/camera_params.json"
OCCUPANCY = "ALL_occupancy.pt"
VOXEL_PLY = "/home/neural_fields/Unified-Lift-Gabor/output/minkowski_grid/officescene_filtered_ply_adv/cc946f8b-f_minkowski_59293vox_iter_opac0.85_cell0.05_eps0.06_neig8_grid.ply"
TENSOR_DATA_TMP = "tensor_data/tmp_tensor_data.pt"

# Extract voxel size and grid origin from the PLY file (as in bash script)
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
    if grid_shape is None:
        print("[WARN] Could not extract grid_shape from PLY header.")
    else:
        print(f"[INFO] Grid shape: {grid_shape} (z, y, x)")
    if num_voxels_from_name is not None:
        print(f"[INFO] Number of voxels (from filename): {num_voxels_from_name}")
    elif grid_shape is not None:
        print(f"[INFO] Number of voxels (from grid_shape): {np.prod(grid_shape)}")
    else:
        print(f"[INFO] Number of voxels: unknown")
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

# Debug: print all feature files found
print(f"[DEBUG] Found {len(feature_files)} feature files:")
#for f in feature_files:
#    print(f"  {f}")

# If debug mode, only keep the first file
if args.first_only:
    print("[DEBUG] Only processing the first input image for debug.")
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
print(f"[INFO] Created {OCCUPANCY} successfully.")

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
        # Try to get occupancy grid (3D tensor)
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
            # (z, y, x) order, convert to (x, y, z)
            occupied_indices = np.array([(x, y, z) for z, y, x in occupied.tolist()], dtype=np.int32)
            # Use grid_origin and voxel_size from PLY header
            grid_origin = np.array(GRID_ORIGIN)
            voxel_size = VOXEL_SIZE
            occupied_world = occupied_indices * voxel_size + grid_origin
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(occupied_world)
            pcd.paint_uniform_color([1, 1, 1])  # white
            ply_path = os.path.join(CHECKPOINT_DIR, f'debug_initial_occupied_voxels_vox{NUM_VOXELS}.ply')
            o3d.io.write_point_cloud(ply_path, pcd, write_ascii=True)
            print(f"[DEBUG] Saved initial occupied voxels as: {ply_path} ({occupied_world.shape[0]} points)")

            """ DEBUG TESTS
            # --- EXTRA DEBUG: Print grid/voxel conventions ---
            print("[DEBUG] PYTHON GRID CONVENTION:")
            print(f"  occ_grid shape: {occ_grid.shape} (z, y, x)")
            print(f"  grid_origin: {grid_origin}")
            print(f"  voxel_size: {voxel_size}")
            print(f"  Example: world = [x, y, z] * voxel_size + grid_origin")
            print(f"  Example: voxel = round((world - grid_origin) / voxel_size) -> (x, y, z)")

            # --- EXTRA DEBUG: Print world-to-voxel and voxel-to-world mapping ---
            print("[DEBUG] First 5 occupied voxel indices (z, y, x) and world coordinates:")
            for i in range(min(5, occupied.shape[0])):
                z, y, x = occupied[i].tolist()
                world = np.array([x, y, z]) * voxel_size + grid_origin
                print(f"  Voxel (z={z}, y={y}, x={x}) -> world: {world}")
                print(f"    Occupancy value: {occ_grid[z, y, x].item()}")
            
            # Test: Map camera center to voxel index and print occupancy
            # (Assume camera center is available from previous debug, else set manually)
            cam_center = np.array([0.80659677, 0.97706404, 1.61907633])
            voxel_idx = np.round((cam_center - grid_origin) / voxel_size).astype(int)
            zc, yc, xc = voxel_idx[2], voxel_idx[1], voxel_idx[0]
            print(f"[DEBUG] Camera center: {cam_center} -> voxel index (z={zc}, y={yc}, x={xc})")
            if 0 <= zc < occ_grid.shape[0] and 0 <= yc < occ_grid.shape[1] and 0 <= xc < occ_grid.shape[2]:
                print(f"    Occupancy value at camera center voxel: {occ_grid[zc, yc, xc].item()}")
            else:
                print("    Camera center voxel index out of bounds!")

            # Test: Print occupancy along a ray from camera center in view direction
            view_dir = np.array([-0.04455666, 0.9986632, -0.0262016])
            for step in range(5):
                pt = cam_center + view_dir * (step * 0.2)
                voxel_idx = np.round((pt - grid_origin) / voxel_size).astype(int)
                z, y, x = voxel_idx[2], voxel_idx[1], voxel_idx[0]
                print(f"[DEBUG] Ray step {step}: pt={pt} -> voxel (z={z}, y={y}, x={x})", end='')
                if 0 <= z < occ_grid.shape[0] and 0 <= y < occ_grid.shape[1] and 0 <= x < occ_grid.shape[2]:
                    print(f", occupancy={occ_grid[z, y, x].item()}")
                else:
                    print(", out of bounds!")
            """
        else:
            print("[DEBUG] OCCUPANCY file found but no 3D occupancy grid present.")
    else:
        print("[DEBUG] OCCUPANCY file not found, skipping initial occupied voxels debug ply.")
except ImportError:
    print("[DEBUG] Open3D not installed, skipping robust debug PLY export.")


# Aggregate over all images using 3D voxel indices
for fpath in feature_files:
    img_name = os.path.basename(fpath)[:-4]
    print(f"\n[INFO] Processing {img_name} ...")
    
    #print(f"[DEBUG] Feature file: {fpath}")
    #print(f"[DEBUG] Using camera params for image: {img_name}")
    with tempfile.TemporaryDirectory() as tmpdir:
        shutil.copy(fpath, tmpdir)
        cam_params_dst = os.path.join(tmpdir, os.path.basename(CAM_PARAMS))
        shutil.copy(CAM_PARAMS, cam_params_dst)
        # --- Print all relevant parameters before CUDA call ---
        """
        print("\n[PYTHON DEBUG] Parameters passed to CUDA kernel:")
        # Load tensor_data for direct access
        tensor_data = torch.load(TENSOR_DATA_TMP, map_location='cpu') if os.path.exists(TENSOR_DATA_TMP) else None
        if tensor_data is not None:
            feats = tensor_data.get('encoded_2d_features', None)
            occ = tensor_data.get('occupancy_3D', None)
            intr = tensor_data.get('intrinsicParams', None)
            extr = tensor_data.get('viewMatrixInv', None)
            grid_origin = tensor_data.get('grid_origin', None)
            voxel_size = tensor_data.get('voxel_size', None)
            print(f"  encoded_2d_features shape: {feats.shape if feats is not None else None}")
            print(f"  occupancy_3D shape: {occ.shape if occ is not None else None}")
            print(f"  intrinsicParams: {intr[0,0].tolist() if intr is not None else None}")
            print(f"  viewMatrixInv (first view):\n{extr[0,0] if extr is not None else None}")
            print(f"  grid_origin: {grid_origin}")
            print(f"  voxel_size: {voxel_size}")
        else:
            print("  [WARN] tensor_data not found for CUDA debug printout!")
        """
        # --- End debug printout ---
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

        # --- [FIXED] Moved debug printout to after file creation ---
        print("\n[PYTHON DEBUG] Parameters passed to CUDA kernel:")
        if os.path.exists(TENSOR_DATA_TMP) and os.path.getsize(TENSOR_DATA_TMP) > 0:
            tensor_data = torch.load(TENSOR_DATA_TMP, map_location='cpu')
            feats = tensor_data.get('encoded_2d_features', None)
            occ = tensor_data.get('occupancy_3D', None)
            intr = tensor_data.get('intrinsicParams', None)
            extr = tensor_data.get('viewMatrixInv', None)
            grid_origin = tensor_data.get('grid_origin', None)
            voxel_size = tensor_data.get('voxel_size', None)
            print(f"  encoded_2d_features shape: {feats.shape if feats is not None else None}")
            print(f"  occupancy_3D shape: {occ.shape if occ is not None else None}")
            print(f"  intrinsicParams: {intr[0,0].tolist() if intr is not None else None}")
            print(f"  viewMatrixInv (first view):\n{extr[0,0] if extr is not None else None}")
            print(f"  grid_origin: {grid_origin}")
            print(f"  voxel_size: {voxel_size}")
        else:
            print(f"  [WARN] {TENSOR_DATA_TMP} not found or is empty. Cannot print CUDA debug info.")
        # --- End debug printout ---

        subprocess.run([
            'python', 'debug_project_features.py',
            '--tensor_data', TENSOR_DATA_TMP,
            '--output', 'proj_output.pt'
        ], check=True)
        output_dict = torch.load('proj_output.pt')
        if 'projected_feats' not in output_dict or 'projected_indices' not in output_dict:
            print(f"[ERROR] 'projected_feats' or 'projected_indices' not found in proj_output.pt for {img_name}. Available keys:", list(output_dict.keys()))
            raise KeyError("'projected_feats' or 'projected_indices' not found in proj_output.pt")
        feats = output_dict['projected_feats']
        indices = output_dict['projected_indices']  # shape: (N, 3) int tensor, 3D voxel indices
        print(f"[DEBUG] Projected feats shape: {feats.shape}, Projected indices shape: {indices.shape}")
        if feats.shape[0] == 0:
            print(f"[ERROR] No features projected for {img_name}!")
        if torch.isnan(feats).any() or torch.isinf(feats).any():
            print(f"[ERROR] NaN or Inf detected in projected features for {img_name}")
        """
        # Debug: print a few voxel index to world mappings
        for i in range(min(5, indices.shape[0])):
            idx = indices[i].tolist()
            world = np.array([idx[2], idx[1], idx[0]]) * VOXEL_SIZE + np.array(GRID_ORIGIN)
            print(f"[DEBUG] Voxel idx (z={idx[0]}, y={idx[1]}, x={idx[2]}) -> world: {world}")
        """
        if feat_dim is None:
            feat_dim = feats.shape[1]
        if 'mapping2dto3d' in output_dict:
            hit_mask = (output_dict['mapping2dto3d'] > 0)
        else:
            hit_mask = feats.abs().sum(dim=1) > 0
        feats = feats[hit_mask]
        indices = indices[hit_mask]
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
    if idx % 50 == 0:
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


        # --- Camera Center and View Direction Extraction (COLMAP logic or fallback) ---
        cam_center = np.zeros(3)
        view_dir = np.array([0, 0, 1], dtype=np.float32)

        import json
        cam_params_path = CAM_PARAMS
        # The feature file's base name (img_name) should match the camera param entry's 'name' field (including extension)
        try:
            with open(cam_params_path, 'r') as f:
                cam_params = json.load(f)
            img_entry = None
            for v in cam_params.get('images', {}).values():
                if v['name'] == img_name:
                    img_entry = v
                    break
            if img_entry is not None:
                qvec = np.array(img_entry['qvec'])
                tvec = np.array(img_entry['tvec'])
                # qvec_to_R as in visualize_frustum.py
                qw, qx, qy, qz = qvec
                R = np.array([
                    [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
                    [2*(qx*qy+qz*qw),   1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
                    [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),   1-2*(qx*qx+qy*qy)]
                ], dtype=np.float32)
                cam_center = -(R.T @ tvec)
                view_dir = R.T @ np.array([0, 0, 1], dtype=np.float32)  # camera z axis in world
                """
                print("[PYTHON DEBUG] Intrinsics: fx=%.6f fy=%.6f cx=%.6f cy=%.6f" % (cam_params['cameras']['1']['params'][0], cam_params['cameras']['1']['params'][1], cam_params['cameras']['1']['params'][2], cam_params['cameras']['1']['params'][3]))
                print("[PYTHON DEBUG] Extrinsic (R|t):")
                print(R)
                print(tvec)
                """
            else:
                # Fallback: use extrinsic from tensor_data.pt
                tensor_data = torch.load(TENSOR_DATA_TMP, map_location='cpu') if os.path.exists(TENSOR_DATA_TMP) else None
                if tensor_data is not None and 'viewMatrixInv' in tensor_data:
                    extr = tensor_data['viewMatrixInv']
                    if extr.ndim == 4:
                        extr = extr[0, 0]  # [4,4]
                    elif extr.ndim == 3:
                        extr = extr[0]     # [4,4]
                    else:
                        extr = extr        # [4,4]
                    cam_center = extr[:3, 3].cpu().numpy()
                    view_dir = extr[:3, 2].cpu().numpy()  # third column is camera z axis in world
                    """
                    print("[PYTHON DEBUG] Intrinsics: unknown (using tensor_data)")
                    print("[PYTHON DEBUG] Extrinsic (viewMatrixInv):")
                    print(extr)
                    """
        except Exception as e:
            print(f"[WARN] Could not extract camera center: {e}")
            cam_center = np.zeros(3)
            view_dir = np.array([0, 0, 1], dtype=np.float32)

        """
        # Print debug info
        print(f"[PYTHON DEBUG] Camera center: {cam_center}")
        print(f"[PYTHON DEBUG] View direction (world): {view_dir}")
        """
        if len(occupied_keys) == 0:
            print(f"[PYTHON DEBUG] No occupied voxels, skipping world_xyz debug and visualization.")
            # --- Extra debug: visualize camera, frustum, and a few rays ---
            try:
                import open3d as o3d
                import json
                # Load intrinsics for this image
                with open(CAM_PARAMS, 'r') as f:
                    cam_params = json.load(f)
                cam = cam_params['cameras']['1']
                fx, fy, cx, cy = cam['params'][:4]
                img_w, img_h = cam.get('width', 1752), cam.get('height', 1168)
                # Use feature tensor shape if available
                if 'images' in cam_params and len(cam_params['images']) > 0:
                    for v in cam_params['images'].values():
                        if v['name'] == img_name:
                            img_w = v.get('width', img_w)
                            img_h = v.get('height', img_h)
                            break
                # Compute frustum corners in pixel space
                corners_px = np.array([
                    [0, 0], [img_w-1, 0], [img_w-1, img_h-1], [0, img_h-1], [cx, cy]
                ])
                # Project to camera space (z=1)
                corners_cam = np.stack([
                    np.array([(u-cx)/fx, (v-cy)/fy, 1.0]) for u, v in corners_px
                ], axis=0)
                # Transform to world space
                # Use R, t from above
                if img_entry is not None:
                    Rw = R.T
                    tw = cam_center
                else:
                    Rw = np.eye(3)
                    tw = cam_center
                corners_world = (Rw @ corners_cam.T).T + tw
                # Rays: from camera center through center and corners
                ray_dirs = corners_world - cam_center[None, :]
                ray_dirs = ray_dirs / (np.linalg.norm(ray_dirs, axis=1, keepdims=True) + 1e-8)
                ray_len = 2.0  # meters
                ray_ends = cam_center[None, :] + ray_dirs * ray_len
                # Build geometry for camera/frustum/rays as points only (sampled along lines)
                points = [cam_center] + [c for c in corners_world]
                colors = [[1,1,1]] + [[0,1,0]]*4 + [[0,0,1]]  # cam center white, corners green, center blue
                # Sampled points for lines
                N_line = 50  # number of points per line
                # Frustum edges (cam center to corners)
                for i in range(4):
                    p0 = cam_center
                    p1 = corners_world[i]
                    for t in np.linspace(0, 1, N_line):
                        pt = (1-t)*p0 + t*p1
                        points.append(pt)
                        colors.append([0,1,0])
                # Frustum base edges (between corners)
                for i in range(4):
                    p0 = corners_world[i]
                    p1 = corners_world[(i+1)%4]
                    for t in np.linspace(0, 1, N_line):
                        pt = (1-t)*p0 + t*p1
                        points.append(pt)
                        colors.append([0,1,0])
                # Rays (center and corners)
                for i in range(5):
                    p0 = cam_center
                    p1 = ray_ends[i]
                    for t in np.linspace(0, 1, N_line):
                        pt = (1-t)*p0 + t*p1
                        points.append(pt)
                        colors.append([1,0,0])
                # Create Open3D point cloud for all points
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(np.array(points))
                pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
                # --- Overlay occupied voxel grid ---
                vox_ply = os.path.join(CHECKPOINT_DIR, f'debug_initial_occupied_vox{NUM_VOXELS}.ply')
                vox_pcd = None
                if os.path.exists(vox_ply):
                    vox_pcd = o3d.io.read_point_cloud(vox_ply)
                    # Optionally recolor voxels to white
                    vox_pcd.paint_uniform_color([1,1,1])
                # Save combined point cloud (camera/frustum/rays + voxels)
                all_pcds = [pcd]
                if vox_pcd is not None:
                    all_pcds.append(vox_pcd)
                combined_pcd = all_pcds[0]
                for extra in all_pcds[1:]:
                    combined_pcd += extra
                ply_path = os.path.join(CHECKPOINT_DIR, f'camera_frustum_rays_voxels_{img_name}.ply')
                o3d.io.write_point_cloud(ply_path, combined_pcd, write_ascii=True)
                print(f"[DEBUG] Saved camera+frustum+rays+voxels as: {ply_path}")
            except Exception as e:
                print(f"[DEBUG] Could not save camera/ray/frustum/voxel visualization: {e}")
        else:
            # print(f"[PYTHON DEBUG] First 5 voxel world coords: {world_xyz[:5]}")

            # Compute and print mean direction from camera to voxels
            mean_voxel = world_xyz.mean(axis=0)
            mean_dir = mean_voxel - cam_center
            mean_dir_norm = mean_dir / (np.linalg.norm(mean_dir) + 1e-8)
            dot = np.dot(view_dir / (np.linalg.norm(view_dir) + 1e-8), mean_dir_norm)
            #print(f"[DEBUG] Mean direction from camera to voxels: {mean_dir_norm}")
            #print(f"[DEBUG] Dot product (view_dir · mean_dir_norm): {dot}")
            if dot < 0.5:
                print(f"[WARNING] Camera may not be facing the voxels!")

            """
            # Write PLY with rays as edges and a view direction arrow
            ply_path = os.path.join(CHECKPOINT_DIR, f'nonzero_voxels_hitcount_{idx}_rays.ply')
            num_points = world_xyz.shape[0]
            arrow_length = max(0.2, 0.1 * np.linalg.norm(world_xyz.mean(axis=0) - cam_center))
            arrow_tip = cam_center + view_dir / np.linalg.norm(view_dir) * arrow_length
            num_vertices = num_points + 2  # +1 for camera center, +1 for arrow tip
            num_edges = num_points + 1     # +1 for view direction arrow
            with open(ply_path, 'w') as f:
                f.write('ply\nformat ascii 1.0\n')
                f.write(f'element vertex {num_vertices}\n')
                f.write('property float x\nproperty float y\nproperty float z\n')
                f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
                f.write(f'element edge {num_edges}\n')
                f.write('property int vertex1\nproperty int vertex2\n')
                f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
                f.write('end_header\n')
                # Write voxel vertices
                for i, pt in enumerate(world_xyz):
                    rgb = colors[i]
                    f.write(f'{pt[0]} {pt[1]} {pt[2]} {rgb[0]} {rgb[1]} {rgb[2]}\n')
                # Write camera center vertex (white)
                f.write(f'{cam_center[0]} {cam_center[1]} {cam_center[2]} 255 255 255\n')
                # Write arrow tip vertex (blue)
                f.write(f'{arrow_tip[0]} {arrow_tip[1]} {arrow_tip[2]} 0 0 255\n')
                # Write edges: from camera center (last-1 vertex) to each voxel
                for i in range(num_points):
                    f.write(f'{num_points} {i} 255 0 0\n')  # red lines
                # Write view direction arrow (last-1 to last vertex)
                f.write(f'{num_points} {num_points+1} 0 0 255\n')  # blue arrow
            print(f"[PLY] Saved nonzero voxels, rays, and view direction as: {ply_path}")
            """

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

    # Save all consolidated data as a single .pt file
    save_path = os.path.join(CHECKPOINT_DIR, f'ALL_nonzero_voxel_features_{idx}_vox{NUM_VOXELS}.pt')
    torch.save({
        'xyz': torch.from_numpy(world_xyz),
        'avg_feats': torch.from_numpy(avg_feats),
        'hit_count': torch.from_numpy(hit_counts),
        'voxel_coords': torch.from_numpy(voxel_coords)
    }, save_path)
    print(f"[PT] Saved consolidated voxel data (xyz, features, hits, coords) as: {save_path}")
    print(f"[DONE] PROJECTION PIPELINE COMPLETED. Checkpoint directory: {CHECKPOINT_DIR}")
