"""
Debug script to run the CUDA project_features_cuda kernel on prepared tensor_data.pt.
Usage:
  cd cuda_project_image_to_sparse_voxel
  python debug_project_features.py \
      --tensor_data tensor_data.pt \
      --output proj_output.pt
"""
import argparse
import torch
import os
import project_features_cuda
# Force synchronous kernel launches for easier backtraces
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tensor_data', required=True,
                        help='Path to tensor_data.pt with encoded_2d_features, occupancy_3D, intrinsicParams, viewMatrixInv, grid_origin, voxel_size.')
    parser.add_argument('--output', default='proj_output.pt', help='Output path for projected_feats and counts.')
    args = parser.parse_args()

    assert os.path.isfile(args.tensor_data), f"tensor_data not found: {args.tensor_data}"
    data = torch.load(args.tensor_data, map_location='cpu')
    feats = data['encoded_2d_features']  # [1, V, H, W, C]
    occ  = data['occupancy_3D']          # [Z, Y, X]
    intr = data['intrinsicParams']       # [1, V, 4]
    extr = data['viewMatrixInv']         # [1, V, 4, 4]
    grid_origin = data['grid_origin']    # [3]
    voxel_size  = data['voxel_size']

    print("=== Loaded tensor_data ===")
    for name, t in [('feats', feats), ('occ', occ), ('intr', intr), ('extr', extr), ('grid_origin', grid_origin)]:
        print(f"{name}: shape={tuple(t.shape)}, dtype={t.dtype}, device={t.device}")
    print(f"voxel_size: {voxel_size}")

    # Print a slice of each tensor to inspect values
    print("\n--- Tensor Value Inspection ---")
    print("feats (slice):", feats[0, 0, 0, 0, :10].cpu().numpy())
    print("occ (unique values):", torch.unique(occ).cpu().numpy())
    print("intr (first view):", intr[0, 0, :].cpu().numpy())
    print("extr (first view):", extr[0, 0, :, :].cpu().numpy())
    print("grid_origin:", grid_origin.cpu().numpy())
    print("-----------------------------\n")

    # move to cuda
    feats = feats.cuda().contiguous()  # [1, V, H, W, C]
    occ    = occ.unsqueeze(0).cuda().contiguous().long()  # [1, Z, Y, X]
    # Restrict to first view only for debug
    feats = feats[:, 0:1, ...]         # [1, 1, H, W, C]
    intr = intr[:, 0, :].cuda().contiguous()  # [1, 4]
    extr = extr[:, 0, :, :].contiguous().view(-1).cuda()  # flatten and move to CUDA
    grid_origin = grid_origin.cuda().contiguous()

    # Print shapes for debug
    print("[DEBUG] feats shape:", feats.shape)
    print("[DEBUG] occ shape:", occ.shape)
    print("[DEBUG] intr shape:", intr.shape)
    print("[DEBUG] extr shape:", extr.shape)
    print("[DEBUG] grid_origin shape:", grid_origin.shape)

    # compute max ID and allocate outputs
    max_id = int(occ.max().item())
    num_ids = max_id + 1  # include zero ID
    C = feats.shape[-1]
    mapping2dto3d = torch.zeros((num_ids,), dtype=torch.int32, device='cuda')
    proj_feats    = torch.zeros((num_ids, C), dtype=torch.float32, device='cuda')

    # build opts: [W, H, dmin, dmax, step]
    # Use correct dims for opts
    _, V, H, W, C = feats.shape
    dmin, dmax = 0.0, 10.0
    step = voxel_size * 0.5
    opts = torch.tensor([W, H, dmin, dmax, step], dtype=torch.float32, device='cpu')
    pred_mode = torch.tensor([False], dtype=torch.bool, device='cpu')

    print("=== Pre-kernel shapes ===")
    print(f"feats: {feats.shape}, occ: {occ.shape}, intr: {intr.shape}, extr: {extr.shape}")
    print(f"opts: {opts.tolist()}, pred_mode: {pred_mode.item()}")
    print(f"mapping2dto3d: {mapping2dto3d.shape}, proj_feats: {proj_feats.shape}")
    # DEBUG: dump strides and total elements
    print("DEBUG STRIDES & SIZES:")
    print("feats.stride=", feats.stride(), "numel=", feats.numel())
    print("occ.stride=", occ.stride(), "numel=", occ.numel())
    print("intr.stride=", intr.stride(), "numel=", intr.numel())
    print("extr.stride=", extr.stride(), "numel=", extr.numel())
    print("opts.stride=", opts.stride(), "numel=", opts.numel())
    print("pred_mode.stride=", pred_mode.stride(), "numel=", pred_mode.numel())
    # verify expected sizes
    assert feats.dim()==5
    assert occ.dim()==4
    # assert intr.dim()==3  # Removed: intr is now 2D as required by the kernel
    # assert extr.dim()==4  # Removed: extr is now 1D as required by the kernel
    assert opts.numel()==5
    assert pred_mode.numel()==1
    # run kernel
    print("Calling project_features_cuda...")
    project_features_cuda.project_features_cuda(
        feats, occ, extr, intr,
        opts, mapping2dto3d, proj_feats,
        pred_mode
    )
    torch.cuda.synchronize()
    print("Kernel done.")

    # Debug: print unique values and stats after kernel
    print("mapping2dto3d unique values:", torch.unique(mapping2dto3d, return_counts=True))
    print("proj_feats stats: min", proj_feats.min().item(), "max", proj_feats.max().item())

    # move to cpu and save
    out = {
        'mapping2dto3d_num': mapping2dto3d.cpu(),
        'projected_feats': proj_feats.cpu(),
    }
    torch.save(out, args.output)
    print(f"Saved projection output to {args.output}")

if __name__ == '__main__':
    main()
