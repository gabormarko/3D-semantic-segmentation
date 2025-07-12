import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def show_feature_map(feature_tensor, title="Feature Map (first channel)", save_path=None):
    # feature_tensor: [H, W, C] or [C, H, W]
    if feature_tensor.ndim == 3:
        if feature_tensor.shape[0] < 10:  # [C, H, W]
            img = feature_tensor[0].cpu().numpy()
        else:  # [H, W, C]
            img = feature_tensor[..., 0].cpu().numpy()
    else:
        img = feature_tensor.cpu().numpy()
    plt.figure()
    plt.imshow(img, cmap='viridis')
    plt.title(title)
    plt.colorbar()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved feature visualization to {save_path}")
    else:
        plt.show()
    plt.close()

def show_projection(mapping2dto3d, occ, title="Projection Mapping (nonzero IDs)", save_path=None):
    # mapping2dto3d: [num_voxels], occ: [Z, Y, X]
    mapping = mapping2dto3d.cpu().numpy().reshape(-1)
    occ_np = occ.cpu().numpy()
    nonzero_voxels = np.argwhere(occ_np > 0)
    mapped = mapping[occ_np.flatten()]
    plt.figure()
    plt.hist(mapped[mapped > 0], bins=50)
    plt.title(title)
    plt.xlabel('2D pixel index')
    plt.ylabel('Voxel count')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved projection visualization to {save_path}")
    else:
        plt.show()
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tensor_data', required=True)
    parser.add_argument('--projection', required=False)
    parser.add_argument('--show_feature', action='store_true')
    parser.add_argument('--show_projection', action='store_true')
    parser.add_argument('--save_feature', type=str, default=None, help='Path to save feature visualization')
    parser.add_argument('--save_projection', type=str, default=None, help='Path to save projection visualization')
    args = parser.parse_args()

    data = torch.load(args.tensor_data, map_location='cpu')
    if args.show_feature or args.save_feature:
        feats = data['encoded_2d_features']  # [1, V, H, W, C]
        print(f"Feature tensor shape: {feats.shape}")
        # Show first view, first channel
        show_feature_map(feats[0, 0, ...], title="Feature Map (first view, first channel)", save_path=args.save_feature)

    if (args.show_projection or args.save_projection) and args.projection:
        proj = torch.load(args.projection, map_location='cpu')
        mapping2dto3d = proj['mapping2dto3d_num']
        occ = data['occupancy_3D']
        print(f"Projection mapping shape: {mapping2dto3d.shape}, occupancy shape: {occ.shape}")
        show_projection(mapping2dto3d, occ, save_path=args.save_projection)

if __name__ == "__main__":
    main()
