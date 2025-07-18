"""
Script to prepare tensor_data.pt for image-color projection pipeline.
Collects:
 - encoded_2d_features: Tensor[B=1, V, H, W, C]
 - occupancy_3D: Tensor[Z, Y, X] (with 1-based IDs)
 - intrinsicParams: Tensor[B=1, V, 4]
 - viewMatrixInv: Tensor[B=1, V, 4, 4]
 - grid_origin: Tensor[3]
 - voxel_size: float
 - image: np.ndarray (H, W, 3) for color projection

Usage:
  python prepare_tensor_data_color.py \
    --lseg_dir /path/to/lseg_features \
    --scaled_camera_params camera_params/scaled_camera_params.json \
    --occupancy occupancy.pt \
    --voxel_size 0.05 \
    --grid_origin 0 0 0 \
    --max_images 1 \
    --output tensor_data.pt
"""
import argparse
import json
import os
import numpy as np
import torch
from PIL import Image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lseg_dir', required=True, help='Folder of .npy LSeg features')
    parser.add_argument('--scaled_camera_params', required=True, help='Path to scaled camera params JSON')
    parser.add_argument('--occupancy', required=True, help='Path to occupancy.pt')
    parser.add_argument('--voxel_size', type=float, required=True, help='Voxel size')
    parser.add_argument('--grid_origin', nargs=3, type=float, required=True, help='Grid origin (x y z)')
    parser.add_argument('--max_images', type=int, default=1, help='Max images to use (should be 1 for color pipeline)')
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
        if isinstance(v, dict) and 'name' in v:
            filename_to_entry[v['name']] = v
        elif isinstance(v, str):
            filename_to_entry[v] = v
        else:
            continue
    print(f"Indexed {len(filename_to_entry)} image filenames from JSON")

    # Gather LSeg .npy features
    files = sorted([p for p in os.listdir(args.lseg_dir) if p.endswith('.npy')])
    if args.max_images:
        files = files[:args.max_images]
    V = len(files)
    print(f"Using {V} feature files from {args.lseg_dir}")

    images_dir = '/home/neural_fields/Unified-Lift-Gabor/data/scannetpp/officescene/images'
    feats_list, intr_list, ext_list = [], [], []
    image_array = None
    for fname in files:
        base = fname[:-4]
        entry = filename_to_entry.get(base, None)
        if entry is None:
            print(f"[WARN] No camera entry for feature file: {fname} (expected name: {base}), skipping.")
            continue
        arr = np.load(os.path.join(args.lseg_dir, fname))
        # Find the original image file
        orig_img_path = None
        if isinstance(entry, dict) and 'name' in entry:
            candidate = os.path.join(images_dir, base)
            if os.path.exists(candidate):
                orig_img_path = candidate
            if orig_img_path is None:
                for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                    candidate = os.path.join(images_dir, base + ext)
                    if os.path.exists(candidate):
                        orig_img_path = candidate
                        break
            if orig_img_path is None:
                for fname_img in os.listdir(images_dir):
                    fbase, fext = os.path.splitext(fname_img)
                    if fbase.lower() == base.lower():
                        orig_img_path = os.path.join(images_dir, fname_img)
                        break
        if orig_img_path is not None:
            orig_img = Image.open(orig_img_path).convert('RGB')
            orig_w, orig_h = orig_img.size
            image_array = np.array(orig_img)  # (H, W, 3)
            arr_torch = torch.from_numpy(arr).unsqueeze(0).float()
            upsampled = torch.nn.functional.interpolate(
                arr_torch, size=(orig_h, orig_w), mode='bilinear', align_corners=False
            )
            arr = upsampled.squeeze(0).cpu().numpy()
            C, H_, W_ = arr.shape
            feats_list.append(torch.from_numpy(arr).float())
        else:
            print(f"[DEBUG] No original image found for {base}, using feature shape as is: {arr.shape}")
            C, H_, W_ = arr.shape
            feats_list.append(torch.from_numpy(arr).float())
        if isinstance(entry, dict):
            cam_id = str(entry['camera_id'])
            params = cams[cam_id]['params']
            if len(params) == 4:
                fx, fy, cx, cy = params
            else:
                fx, cx, cy = params; fy = fx
            intr_list.append(torch.tensor([fx, fy, cx, cy], dtype=torch.float32))
            R = np.array(entry['R'], dtype=np.float32)
            t = np.array(entry['tvec'], dtype=np.float32)
            cam_to_world = np.eye(4, dtype=np.float32)
            cam_to_world[:3, :3] = R.T
            cam_to_world[:3, 3] = -R.T @ t
            ext_list.append(torch.from_numpy(cam_to_world))
        else:
            print(f"[WARN] Entry for {base} is not a dict, skipping.")
            continue
    if not feats_list:
        raise RuntimeError("No valid feature/camera pairs found!")
    encoded_2d = torch.stack(feats_list, 0).unsqueeze(0)
    encoded_2d = encoded_2d.permute(0, 1, 3, 4, 2).contiguous()
    intrinsicParams = torch.stack(intr_list, 0).unsqueeze(0)
    viewMatrixInv = torch.stack(ext_list, 0).unsqueeze(0)
    grid_origin = torch.tensor(args.grid_origin, dtype=torch.float32)
    voxel_size = float(args.voxel_size)
    out = {
        'encoded_2d_features': encoded_2d,
        'occupancy_3D': occ,
        'intrinsicParams': intrinsicParams,
        'viewMatrixInv': viewMatrixInv,
        'grid_origin': grid_origin,
        'voxel_size': voxel_size,
        'image': image_array if image_array is not None else np.zeros((H_, W_, 3), dtype=np.uint8),
    }
    print(f"Saving tensor_data to: {args.output} (encoded_2d_features shape: {encoded_2d.shape})")
    torch.save(out, args.output)
    print("Done.")
    loaded = torch.load(args.output, map_location='cpu')
    print("Saved tensor_data.pt keys and shapes:")
    for k, v in loaded.items():
        if hasattr(v, 'shape'):
            print(f"  {k}: shape={tuple(v.shape)}, dtype={v.dtype}")
        else:
            print(f"  {k}: {type(v)}, value={v}")
