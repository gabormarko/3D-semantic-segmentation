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
    parser.add_argument('--scaled_camera_params', required=True, help='Path to scaled_camera_params.json')
    parser.add_argument('--occupancy', required=True, help='Path to occupancy.pt (Z,Y,X) int64 IDs')
    parser.add_argument('--voxel_size', type=float, default=1.0, help='Voxel size')
    parser.add_argument('--grid_origin', nargs=3, type=float, default=[0,0,0], help='Grid origin')
    parser.add_argument('--max_images', type=int, default=None, help='Max number of images (views)')
    parser.add_argument('--output', default='tensor_data.pt', help='Output .pt file')
    args = parser.parse_args()

    # Validate input paths
    if not os.path.isdir(args.lseg_dir):
        print(f"Error: lseg_dir not found: {args.lseg_dir}")
        exit(1)
    if not os.path.isfile(args.occupancy):
        print(f"Error: occupancy tensor not found: {args.occupancy}")
        exit(1)
    if not os.path.isfile(args.scaled_camera_params):
        print(f"Error: scaled_camera_params JSON not found: {args.scaled_camera_params}")
        exit(1)

    # Load occupancy
    print(f"Loading occupancy from: {args.occupancy}")
    occupancy = torch.load(args.occupancy, map_location='cpu')
    print(f"Occupancy shape: {occupancy.shape}, max ID: {int(occupancy.max().item())}")

    # Load camera params
    print(f"Loading camera params from: {args.scaled_camera_params}")
    with open(args.scaled_camera_params, 'r') as f:
        cam_data = json.load(f)
    imgs = cam_data['images']
    # Normalize images: could be dict of id->info or list of dicts
    if isinstance(imgs, dict):
        imgs = list(imgs.values())
    cams = {int(k): v for k,v in cam_data['cameras'].items()}
    print(f"Found {len(imgs)} images, {len(cams)} cameras in JSON")

    # Gather LSeg .npy features
    files = sorted([p for p in os.listdir(args.lseg_dir) if p.endswith('.npy')])
    if args.max_images:
        files = files[:args.max_images]
    V = len(files)
    print(f"Using {V} feature files from {args.lseg_dir}")

    feats_list, intr_list, ext_list = [], [], []
    H = W = C = None
    for fname in files:
        # remove .npy suffix to match image names exactly
        base = fname[:-4]  # strip '.npy'
        # match entry by exact image name
        entry = next((im for im in imgs if im.get('name','') == base), None)
        if entry is None:
            raise ValueError(f"No camera entry for feature file: {fname} (expected name: {base})")
        arr = np.load(os.path.join(args.lseg_dir, fname))
        if H is None:
            H, W, C = arr.shape
            print(f"Detected feature map size: {H}x{W}x{C}")
        feats_list.append(torch.from_numpy(arr).float())
        params = cams[int(entry['camera_id'])]['params']
        if len(params) == 4:
            fx, fy, cx, cy = params
        else:
            fx, cx, cy = params; fy = fx
        intr_list.append(torch.tensor([fx, fy, cx, cy], dtype=torch.float32))
        q, t = entry['qvec'], entry['tvec']
        R = qvec2rotmat(q)
        T = np.eye(4, dtype=np.float32)
        T[:3,:3], T[:3,3] = R, t
        invT = np.linalg.inv(T)
        ext_list.append(torch.from_numpy(invT))

    encoded_2d = torch.stack(feats_list, 0).unsqueeze(0)
    intrinsicParams = torch.stack(intr_list, 0).unsqueeze(0)
    viewMatrixInv = torch.stack(ext_list, 0).unsqueeze(0)
    grid_origin = torch.tensor(args.grid_origin, dtype=torch.float32)
    voxel_size = float(args.voxel_size)

    out = {
        'encoded_2d_features': encoded_2d,
        'occupancy_3D': occupancy,
        'intrinsicParams': intrinsicParams,
        'viewMatrixInv': viewMatrixInv,
        'grid_origin': grid_origin,
        'voxel_size': voxel_size,
    }
    print(f"Saving tensor_data to: {args.output}")
    torch.save(out, args.output)
    print("Done.")
