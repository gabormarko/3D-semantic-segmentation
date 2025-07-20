"""
Debug script to run the CUDA project_features_cuda kernel for color projection on prepared tensor_data.pt.
Usage:
  cd cuda_project_image_to_sparse_voxel
  python debug_project_colors.py \
      --tensor_data tensor_data.pt \
      --output proj_output.pt
"""
import argparse
import torch
import numpy as np
import os
import project_features_cuda
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tensor_data', required=True,
                        help='Path to tensor_data.pt with encoded_2d_features, occupancy_3D, intrinsicParams, viewMatrixInv, grid_origin, voxel_size.')
    parser.add_argument('--output', default='proj_output.pt', help='Output path for projected colors and counts.')
    args = parser.parse_args()

    assert os.path.isfile(args.tensor_data), f"tensor_data not found: {args.tensor_data}"
    data = torch.load(args.tensor_data, map_location='cpu')
    occ  = data['occupancy_3D']          # [Z, Y, X]
    intr = data['intrinsicParams']       # [1, V, 4]
    extr = data['viewMatrixInv']         # [1, V, 4, 4]
    grid_origin = data['grid_origin']    # [3]
    voxel_size  = data['voxel_size']

    # If image is present, load it
    if 'image' in data:
        img_np = data['image']
    else:
        raise RuntimeError("Image array not found in tensor_data.pt. Please include the image for color projection.")

    # --- Build a reverse map from voxel ID to (z,y,x) coordinate ---
    occ_cpu = data['occupancy_3D'].cpu()
    max_id_from_occ = int(occ_cpu.max().item())
    id_to_zyx = torch.full((max_id_from_occ + 1, 3), -1, dtype=torch.long)
    nonzero_coords = occ_cpu.nonzero(as_tuple=False).long()
    if nonzero_coords.numel() > 0:
        nonzero_ids = occ_cpu[nonzero_coords[:, 0], nonzero_coords[:, 1], nonzero_coords[:, 2]].long()
        id_to_zyx[nonzero_ids] = nonzero_coords
    else:
        print("[WARN] Occupancy grid is empty. No reverse map built.")

    # Compute occ_indices for all debug projection code
    occ_cpu = occ.cpu().numpy() if occ.dim() == 3 else occ[0].cpu().numpy()
    occ_indices = (occ_cpu > 0).nonzero()
    _, _, H, W, _ = data['encoded_2d_features'].shape
    img_h, img_w = img_np.shape[0], img_np.shape[1]

    # --- Project all occupied voxel centers to (u, v) and collect color ---
    projected_colors = []
    projected_indices = []
    pixel_indices = []
    for i in range(len(occ_indices[0])):
        z, y, x = occ_indices[0][i], occ_indices[1][i], occ_indices[2][i]
        world = grid_origin.cpu().numpy() + voxel_size * np.array([x, y, z])
        R = extr.cpu().numpy().reshape(4, 4)[:3, :3]
        t = extr.cpu().numpy().reshape(4, 4)[:3, 3]
        cam = np.dot(R.T, (world - t))
        fx, fy, cx, cy = intr[0, 0, :].cpu().numpy()
        if cam[2] > 0:
            u = fx * (cam[0] / cam[2]) + cx
            v = fy * (cam[1] / cam[2]) + cy
            u_int, v_int = int(round(u)), int(round(v))
            if 0 <= u_int < img_w and 0 <= v_int < img_h:
                color = img_np[v_int, u_int] / 255.0  # RGB normalized
                projected_colors.append(color)
                projected_indices.append([z, y, x])
                pixel_indices.append([u_int, v_int])
    if projected_colors:
        projected_colors = torch.tensor(projected_colors, dtype=torch.float32)
        projected_indices = torch.tensor(projected_indices, dtype=torch.int32)
        pixel_indices = torch.tensor(pixel_indices, dtype=torch.int32)
    else:
        projected_colors = torch.empty((0, 3), dtype=torch.float32)
        projected_indices = torch.empty((0, 3), dtype=torch.int32)
        pixel_indices = torch.empty((0, 2), dtype=torch.int32)

    out = {
        'projected_colors': projected_colors,
        'projected_indices': projected_indices,
        'pixel_indices': pixel_indices,
    }
    torch.save(out, args.output)
    print(f"Saved color projection output to {args.output}")

if __name__ == '__main__':
    main()