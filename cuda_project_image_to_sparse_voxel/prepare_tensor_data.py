"""
Script to prepare tensor_data.pt for ScanNet++ feature projection (moved here).
Collects:
 - encoded_2d_features: Tensor[B=1, V, H, W, C]
 - occupancy_3D: Tensor[Z, Y, X] (with 1-based IDs)
 - intrinsicParams: Tensor[B=1, V, 4]
 - viewMatrixInv: Tensor[B=1, V, 4, 4]
 - grid_origin: Tensor[3]
 - voxel_size: float

Usage:
  python prepare_tensor_data.py \
    --lseg_dir /path/to/lseg_features \
    --scaled_camera_params camera_params/scaled_camera_params.json \
    --occupancy occupancy.pt \
    --voxel_size 0.05 \
    --grid_origin 0 0 0 \
    --max_images 10 \
    --output tensor_data.pt
"""
import argparse
import json
import os
import numpy as np
import torch


def qvec2rotmat(q):
    # from COLMAP: q = [qw, qx, qy, qz]
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qw*qz,     2*qx*qz + 2*qw*qy    ],
        [2*qx*qy + 2*qw*qz,     1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qw*qx    ],
        [2*qx*qz - 2*qw*qy,     2*qy*qz + 2*qw*qx,     1 - 2*qx*qx - 2*qy*qy]
    ], dtype=np.float32)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lseg_dir', required=True, help='Folder of .npy LSeg features')
    parser.add_argument('--scaled_camera_params', required=True, help='Path to scaled camera params JSON')
    parser.add_argument('--occupancy', required=True, help='Path to occupancy.pt')
    parser.add_argument('--voxel_size', type=float, required=True, help='Voxel size')
    parser.add_argument('--grid_origin', nargs=3, type=float, required=True, help='Grid origin (x y z)')
    parser.add_argument('--max_images', type=int, default=10, help='Max images to use')
    parser.add_argument('--output', required=True, help='Output tensor_data.pt')
    parser.add_argument('--image_size', nargs=2, type=int, help='Target image size (H W) for upsampling feature maps')
    parser.add_argument('--downsample_factor', type=float, default=None, help='Downsampling factor used for image and camera params')
    args = parser.parse_args()

    print(f"Loading occupancy from: {args.occupancy}")
    occ = torch.load(args.occupancy)
    print(f"Occupancy shape: {occ.shape}, max ID: {occ.max().item()}")

    print(f"Loading camera params from: {args.scaled_camera_params}")
    with open(args.scaled_camera_params, 'r') as f:
        cam_params = json.load(f)
    imgs = cam_params['images']
    cams = cam_params['cameras']
    print(f"Found {len(imgs)} images, {len(cams)} cameras in JSON")

    # Build mapping from image filename to entry
    filename_to_entry = {}
    for k, v in imgs.items() if isinstance(imgs, dict) else enumerate(imgs):
        # v may be dict with 'name' or just the filename string
        if isinstance(v, dict) and 'name' in v:
            filename_to_entry[v['name']] = v
        elif isinstance(v, str):
            filename_to_entry[v] = v  # fallback
        else:
            continue
    print(f"Indexed {len(filename_to_entry)} image filenames from JSON")

    # Gather LSeg .npy features
    files = sorted([p for p in os.listdir(args.lseg_dir) if p.endswith('.npy')])
    if args.max_images:
        files = files[:args.max_images]
    V = len(files)
    print(f"Using {V} feature files from {args.lseg_dir}")

    # Print unmatched names
    feature_basenames = set([fname[:-4] if fname.endswith('.npy') else fname for fname in files])
    img_names = set(filename_to_entry.keys())
    unmatched_features = feature_basenames - img_names
    unmatched_imgs = img_names - feature_basenames
    if unmatched_features:
        print("[DEBUG] Feature files with no matching image entry:")
        for name in sorted(unmatched_features):
            print("  ", name)
    if unmatched_imgs:
        print(f"[DEBUG] Image entries with no matching feature file: {len(unmatched_imgs)}")

    import cv2
    feats_list, intr_list, ext_list = [], [], []
    H = W = C = None
    used = 0
    images_dir = '/home/neural_fields/Unified-Lift-Gabor/data/scannetpp/officescene/images'
    # Parse image size from args if provided
    H_new = W_new = None
    if args.image_size is not None:
        H_new = int(args.image_size[0])
        W_new = int(args.image_size[1])
    # Use downsample_factor for any scaling logic if provided
    downsample_factor = args.downsample_factor if args.downsample_factor is not None else None

    for fname in files:
        base = fname[:-4]  # strip '.npy'
        entry = filename_to_entry.get(base, None)
        if entry is None:
            print(f"[WARN] No camera entry for feature file: {fname} (expected name: {base}), skipping.")
            continue
        # Debug: print feature/camera alignment
        if isinstance(entry, dict) and 'name' in entry:
            print(f"[ALIGNMENT DEBUG] Feature file: {fname} <-> Camera entry: {entry['name']}")
        arr = np.load(os.path.join(args.lseg_dir, fname))

        # Upsample feature map to match new image size if needed
        C, H_, W_ = arr.shape
        if H_new is not None and W_new is not None and (H_ != H_new or W_ != W_new):
            print(f"[UPSAMPLE] Feature map shape before: {arr.shape}, target: (C={C}, H={H_new}, W={W_new})")
            arr_upsampled = np.zeros((C, H_new, W_new), dtype=np.float32)
            for c in range(C):
                # Convert to float32 and ensure contiguous for OpenCV
                channel = np.ascontiguousarray(arr[c].astype(np.float32))
                arr_upsampled[c] = cv2.resize(channel, (W_new, H_new), interpolation=cv2.INTER_LINEAR)
            arr = arr_upsampled.astype(arr.dtype)
            print(f"[UPSAMPLE] Feature map shape after: {arr.shape}")
        else:
            print(f"[UPSAMPLE] No upsampling needed for feature map: {arr.shape}")

        # If downsample_factor is provided, scale camera intrinsics accordingly
        if downsample_factor is not None and isinstance(entry, dict):
            cam_id = str(entry['camera_id'])
            params = cams[cam_id]['params']
            if len(params) == 4:
                fx, fy, cx, cy = params
            else:
                fx, cx, cy = params; fy = fx
            fx *= downsample_factor
            fy *= downsample_factor
            cx *= downsample_factor
            cy *= downsample_factor
            intr_list.append(torch.tensor([fx, fy, cx, cy], dtype=torch.float32))
        elif isinstance(entry, dict):
            cam_id = str(entry['camera_id'])
            params = cams[cam_id]['params']
            if len(params) == 4:
                fx, fy, cx, cy = params
            else:
                fx, cx, cy = params; fy = fx
            intr_list.append(torch.tensor([fx, fy, cx, cy], dtype=torch.float32))
        feats_list.append(torch.from_numpy(arr).float())
        # entry may be dict or str
        if isinstance(entry, dict):
            # Use string key for camera_id to avoid KeyError
            cam_id = str(entry['camera_id'])
            params = cams[cam_id]['params']
            if len(params) == 4:
                fx, fy, cx, cy = params
            else:
                fx, cx, cy = params; fy = fx
            intr_list.append(torch.tensor([fx, fy, cx, cy], dtype=torch.float32))
            
            # Use the pre-computed rotation matrix 'R' from the JSON file
            R = np.array(entry['R'], dtype=np.float32)
            t = np.array(entry['tvec'], dtype=np.float32)

            # Create the camera-to-world matrix (the inverse of the view matrix [R|t])
            # The inverse is [R.T | -R.T @ t]
            cam_to_world = np.eye(4, dtype=np.float32)
            cam_to_world[:3, :3] = R.T
            cam_to_world[:3, 3] = -R.T @ t
            
            ext_list.append(torch.from_numpy(cam_to_world))
            used += 1
        else:
            print(f"[WARN] Entry for {base} is not a dict, skipping.")
            continue

    if not feats_list:
        raise RuntimeError("No valid feature/camera pairs found!")

    encoded_2d = torch.stack(feats_list, 0).unsqueeze(0)  # [1, V, C, H, W]
    # Convert to channels-last: [1, V, H, W, C]
    encoded_2d = encoded_2d.permute(0, 1, 3, 4, 2).contiguous()
    print(f"[DEBUG] Final encoded_2d_features shape: {encoded_2d.shape} (should be [1, V, H, W, C])")
    intrinsicParams = torch.stack(intr_list, 0).unsqueeze(0)
    viewMatrixInv = torch.stack(ext_list, 0).unsqueeze(0)
    grid_origin = torch.tensor(args.grid_origin, dtype=torch.float32)
    voxel_size = float(args.voxel_size)

    out = {
        'encoded_2d_features': encoded_2d,  # [B, V, H, W, C]
        'occupancy_3D': occ,
        'intrinsicParams': intrinsicParams,
        'viewMatrixInv': viewMatrixInv,
        'grid_origin': grid_origin,
        'voxel_size': voxel_size,
    }
    print(f"Saving tensor_data to: {args.output} (encoded_2d_features shape: {encoded_2d.shape})")
    torch.save(out, args.output)
    print("Done.")
    # Debug: print keys and shapes in saved tensor_data
    loaded = torch.load(args.output, map_location='cpu')
    print("Saved tensor_data.pt keys and shapes:")
    for k, v in loaded.items():
        if hasattr(v, 'shape'):
            print(f"  {k}: shape={tuple(v.shape)}, dtype={v.dtype}")
        else:
            print(f"  {k}: {type(v)}, value={v}")
