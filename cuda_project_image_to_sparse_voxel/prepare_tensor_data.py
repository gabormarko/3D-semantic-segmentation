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

    feats_list, intr_list, ext_list = [], [], []
    H = W = C = None
    used = 0
    images_dir = '/home/neural_fields/Unified-Lift-Gabor/data/scannetpp/officescene/images'
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

        # Robustly find the original image file
        orig_img_path = None
        checked_candidates = []
        if isinstance(entry, dict) and 'name' in entry:
            # First, check for the file with the base name and no extra extension
            candidate = os.path.join(images_dir, base)
            checked_candidates.append(candidate)
            if os.path.exists(candidate):
                orig_img_path = candidate
                print(f"[find_original_image] Found exact match (no extension added): {candidate}")
            # Then try with common extensions if not found
            if orig_img_path is None:
                for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                    candidate = os.path.join(images_dir, base + ext)
                    checked_candidates.append(candidate)
                    if os.path.exists(candidate):
                        orig_img_path = candidate
                        print(f"[find_original_image] Found exact match: {candidate}")
                        break
            # Fallback: case-insensitive and extension-insensitive search
            if orig_img_path is None:
                try:
                    for fname_img in os.listdir(images_dir):
                        fbase, fext = os.path.splitext(fname_img)
                        if fbase.lower() == base.lower():
                            candidate = os.path.join(images_dir, fname_img)
                            checked_candidates.append(candidate)
                            print(f"[find_original_image] Fallback match: {fname_img} in {images_dir}")
                            orig_img_path = candidate
                            break
                except Exception as e:
                    print(f"[find_original_image] Error listing directory {images_dir}: {e}")
            if orig_img_path is None:
                print(f"[find_original_image] Checked candidates:")
                for c in checked_candidates:
                    print(f"  {c}")

        if orig_img_path is not None:
            from PIL import Image
            orig_img = Image.open(orig_img_path)
            orig_w, orig_h = orig_img.size  # PIL: (width, height)
            print(f"[DEBUG] Original image size for {base}: {orig_h}x{orig_w} (height x width)")
            # arr: [C, H, W] or [H, W, C]
            if arr.shape[0] < 10:  # [C, H, W]
                arr_torch = torch.from_numpy(arr).unsqueeze(0).float()  # [1, C, H, W]
            else:  # [H, W, C]
                arr_torch = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
            upsampled = torch.nn.functional.interpolate(
                arr_torch, size=(orig_h, orig_w), mode='bilinear', align_corners=False
            )
            arr = upsampled.squeeze(0).cpu().numpy()
            print(f"[DEBUG] Upsampled feature map shape for {base}: {arr.shape}")
            # Print image size in (height x width) order for clarity
            if arr.shape[0] < 10:
                C, H_, W_ = arr.shape
            else:
                H_, W_, C = arr.shape
            print(f"[DEBUG] Image size for frustum: {H_}x{W_} (height x width)")
            if arr.shape[0] < 10:
                C, H_, W_ = arr.shape
            else:
                H_, W_, C = arr.shape
                arr = np.transpose(arr, (2, 0, 1))  # [C, H, W]
            if H is None:
                H, W, C = H_, W_, arr.shape[0]
                print(f"Upsampled feature map to: {H}x{W}x{C}")
            feats_list.append(torch.from_numpy(arr).float())
        else:
            print(f"[DEBUG] No original image found for {base}, using feature shape as is: {arr.shape}")
            if arr.shape[0] < 10:
                C, H_, W_ = arr.shape
            else:
                H_, W_, C = arr.shape
                arr = np.transpose(arr, (2, 0, 1))  # [C, H, W]
            if H is None:
                H, W, C = H_, W_, arr.shape[0]
                print(f"Detected feature map size: {H}x{W}x{C}")
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
