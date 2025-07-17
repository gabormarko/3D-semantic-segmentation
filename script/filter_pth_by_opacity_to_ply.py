import argparse
import torch
import numpy as np
import open3d as o3d

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def main():
    parser = argparse.ArgumentParser(description="Filter a .pth checkpoint by opacity and save as .ply.")
    parser.add_argument('--input_pth', required=True, help='Input .pth checkpoint file with Gaussians.')
    parser.add_argument('--output_ply', required=True, help='Output PLY file after opacity filtering.')
    parser.add_argument('--opacity_threshold', type=float, default=0.99, help='Keep only Gaussians with opacity >= threshold (after sigmoid).')
    args = parser.parse_args()

    data = torch.load(args.input_pth, map_location='cpu')
    # The checkpoint is a tuple; positions at index 1, opacity logits at index 6
    if not (isinstance(data, tuple) and len(data) > 6):
        print("Unexpected checkpoint structure. Expected a tuple with at least 7 elements.")
        return
    xyz = data[1].cpu().numpy() if hasattr(data[1], 'cpu') else np.asarray(data[1])  # [N, 3]
    opacity_logits = data[6].cpu().numpy() if hasattr(data[6], 'cpu') else np.asarray(data[6])  # [N, 1]
    # Squeeze last dim if needed
    if opacity_logits.ndim == 2 and opacity_logits.shape[1] == 1:
        opacity_logits = opacity_logits.squeeze(1)
    opacities = sigmoid(opacity_logits)
    mask = opacities >= args.opacity_threshold
    print(f"Filtering: {np.sum(mask)} / {len(opacities)} Gaussians kept (opacity >= {args.opacity_threshold})")
    filtered_xyz = xyz[mask]
    # Save as PLY (no color, just points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_xyz)
    o3d.io.write_point_cloud(args.output_ply, pcd, write_ascii=True)
    print(f"Saved filtered PLY to: {args.output_ply}")

if __name__ == "__main__":
    main()
