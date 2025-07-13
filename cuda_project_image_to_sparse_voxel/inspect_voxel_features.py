import torch
import numpy as np

import argparse

VOXEL_PLY = "/home/neural_fields/Unified-Lift-Gabor/output/minkowski_grid/officescene/officescene_minkowski_9434vox_iter50000_grid.ply"

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--proj_output', type=str, default='proj_output.pt', help='Path to projection output file')
    parser.add_argument('--topk', type=int, default=10, help='Show stats for top-k most hit voxels')
    parser.add_argument('--ply_header', type=str, default=None, help='Path to reference PLY file for voxel_size and grid_origin')

    args = parser.parse_args()
    # If --ply_header is not provided, use VOXEL_PLY
    if args.ply_header is None:
        args.ply_header = VOXEL_PLY
        print(f"[INFO] --ply_header not specified, using default: {VOXEL_PLY}")

    def extract_voxel_params_and_shape(ply_path):
        voxel_size = None
        grid_origin = None
        grid_shape = None
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
                    grid_shape = tuple(int(x) for x in line.split()[-3:])
                if 'end_header' in line:
                    break
        if voxel_size is None or grid_origin is None:
            raise RuntimeError("Could not extract voxel_size or grid_origin from PLY header")
        return voxel_size, np.array(grid_origin, dtype=np.float32), grid_shape


    out = torch.load(args.proj_output, map_location='cpu')
    # Support both per-image and aggregation outputs
    if 'projected_feats' in out and 'mapping2dto3d_num' in out:
        feats = out['projected_feats']
        mapping = out['mapping2dto3d_num']
        print(f"projected_feats shape: {feats.shape}")
        print(f"mapping2dto3d_num shape: {mapping.shape}")
    elif 'avg_feats' in out and 'hit_count' in out:
        feats = out['avg_feats']
        mapping = out['hit_count']
        print(f"avg_feats shape: {feats.shape}")
        print(f"hit_count shape: {mapping.shape}")
    else:
        print("[ERROR] Unrecognized file format. Keys:", list(out.keys()))
        return

    # Find nonzero voxels
    nonzero = (mapping > 0).nonzero(as_tuple=True)[0]
    print(f"Number of voxels with features: {len(nonzero)} / {mapping.shape[0]}")

    if len(nonzero) == 0:
        print("No voxels with features.")
        return

    # Stats for all nonzero voxels
    all_feats = feats[nonzero]
    print(f"All nonzero voxel features: min={all_feats.min().item():.4f}, max={all_feats.max().item():.4f}, mean={all_feats.mean().item():.4f}, std={all_feats.std().item():.4f}")

    # Top-k voxels by mapping count
    topk = min(args.topk, len(nonzero))
    topk_indices = mapping[nonzero].argsort(descending=True)[:topk]
    print(f"\nTop {topk} voxels by number of hits:")
    for rank, idx in enumerate(topk_indices):
        voxel_idx = nonzero[idx].item()
        count = mapping[voxel_idx].item()
        feat = feats[voxel_idx]
        print(f"Voxel {voxel_idx}: count={count}, feat[:10]={feat[:10].cpu().numpy()}")


    # Histogram of mapping counts
    import matplotlib.pyplot as plt
    plt.figure()
    plt.hist(mapping[mapping > 0].cpu().numpy(), bins=30)
    plt.title('Histogram of voxel hit counts')
    plt.xlabel('Number of hits')
    plt.ylabel('Number of voxels')
    plt.grid(True)
    plt.show()

    # 3D scatter plot of nonzero voxels (indices only)
    # Try to extract grid_shape from PLY if needed
    grid_shape = feats.shape[:-1] if feats.ndim == 2 else mapping.shape
    ply_voxel_size, ply_grid_origin, ply_grid_shape = None, None, None
    if len(grid_shape) == 1 and args.ply_header is not None:
        try:
            ply_voxel_size, ply_grid_origin, ply_grid_shape = extract_voxel_params_and_shape(args.ply_header)
            if ply_grid_shape is not None:
                grid_shape = ply_grid_shape
                print(f"[INFO] Using grid_shape from PLY header: {grid_shape}")
        except Exception as e:
            print(f"[WARN] Could not extract grid_shape from PLY: {e}")

    if len(nonzero) > 0:
        if len(grid_shape) == 3:
            zyx = np.stack(np.unravel_index(nonzero.cpu().numpy(), grid_shape), axis=1)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(zyx[:, 2], zyx[:, 1], zyx[:, 0], c=mapping[nonzero].cpu().numpy(), cmap='viridis', s=2)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('Nonzero voxel indices (colored by hit count)')
            plt.show()
        else:
            print(f"[WARN] Grid shape is not 3D (shape={grid_shape}), skipping 3D scatter plot.")

    # Save nonzero voxels as a .ply file (in world coordinates if possible)
    ply_path = args.proj_output + '_nonzero_voxels.ply'
    # grid_shape may have been updated above
    if len(grid_shape) == 1:
        print(f"[WARN] Could not infer 3D grid shape, saving flat indices as x.")
        xyz = np.zeros((len(nonzero), 3), dtype=np.float32)
        xyz[:, 0] = nonzero.cpu().numpy()
        world_xyz = xyz
    else:
        zyx = np.stack(np.unravel_index(nonzero.cpu().numpy(), grid_shape), axis=1)
        xyz = zyx[:, [2, 1, 0]].astype(np.float32)
        # Convert to world coordinates if possible
        # Use voxel_size and grid_origin from PLY if available
        try:
            if ply_voxel_size is None or ply_grid_origin is None:
                ply_voxel_size, ply_grid_origin, _ = extract_voxel_params_and_shape(args.ply_header)
            world_xyz = xyz * ply_voxel_size + ply_grid_origin
            print(f"Saving world coordinates using voxel_size={ply_voxel_size}, grid_origin={ply_grid_origin.tolist()}")
        except Exception as e:
            print(f"[WARN] Could not extract voxel_size/grid_origin: {e}. Saving grid indices instead.")
            world_xyz = xyz
    # Write PLY
    with open(ply_path, 'w') as f:
        f.write('ply\nformat ascii 1.0\n')
        f.write(f'element vertex {world_xyz.shape[0]}\n')
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write('end_header\n')
        for pt in world_xyz:
            f.write(f'{pt[0]} {pt[1]} {pt[2]}\n')
    print(f"Saved nonzero voxels as PLY: {ply_path}")

if __name__ == '__main__':
    main()
