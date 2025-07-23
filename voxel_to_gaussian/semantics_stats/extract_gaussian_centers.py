#!/usr/bin/env python3
# Extract Gaussian centers (xyz) from a .ply file and save as .npy or .npz
import numpy as np
import sys
import os
import argparse

try:
    from plyfile import PlyData, PlyElement
except ImportError:
    print("Please install plyfile: pip install plyfile")
    sys.exit(1)

def extract_gaussian_centers(ply_path, out_path, npz=False):
    ply = PlyData.read(ply_path)
    # Assume 'vertex' element contains 'x', 'y', 'z'
    vertex = ply['vertex']
    xyz = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1)  # shape (N,3)
    print(f"[INFO] Extracted {xyz.shape[0]} centers from {ply_path}")
    if npz:
        np.savez_compressed(out_path, mu=xyz)
        print(f"[INFO] Saved centers to {out_path} (npz, key='mu')")
    else:
        np.save(out_path, xyz)
        print(f"[INFO] Saved centers to {out_path} (npy)")

    # Print stats for f_dc_0 and f_dc_1 if present
    for dc_idx in [0, 1]:
        dc_field = f'f_dc_{dc_idx}'
        if dc_field in vertex.data.dtype.names:
            dc_data = vertex[dc_field]
            print(f"[INFO] {dc_field} shape: {dc_data.shape}, min: {dc_data.min():.4f}, max: {dc_data.max():.4f}")
    # Always use f_dc_0, f_dc_1, f_dc_2 as RGB if present
    if all(f'f_dc_{i}' in vertex.data.dtype.names for i in range(3)):
        r = vertex['f_dc_0']
        g = vertex['f_dc_1']
        b = vertex['f_dc_2']
        # Clamp to [0,1], scale to [0,255]
        rgb = np.stack([
            np.clip(r, 0, 1) * 255,
            np.clip(g, 0, 1) * 255,
            np.clip(b, 0, 1) * 255
        ], axis=1)
        colors = rgb.astype(np.uint8)
        print(f"[INFO] Used f_dc_0, f_dc_1, f_dc_2 as RGB for .ply export.")
    else:
        # Default to white
        colors = np.full((xyz.shape[0], 3), 255, dtype=np.uint8)
        print(f"[INFO] f_dc_0, f_dc_1, f_dc_2 not found, using white for .ply export.")

    # Save colored ply
    ply_out_path = out_path.replace('.npy', '_colored.ply').replace('.npz', '_colored.ply')
    vertex_data = np.zeros(xyz.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    vertex_data['x'] = xyz[:, 0]
    vertex_data['y'] = xyz[:, 1]
    vertex_data['z'] = xyz[:, 2]
    vertex_data['red'] = colors[:, 0]
    vertex_data['green'] = colors[:, 1]
    vertex_data['blue'] = colors[:, 2]
    PlyData([PlyElement.describe(vertex_data, 'vertex')]).write(ply_out_path)
    print(f"[INFO] Saved colored centers to {ply_out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Gaussian centers from .ply file")
    parser.add_argument('--ply', type=str, required=True, help='Input .ply file')
    parser.add_argument('--out', type=str, required=True, help='Output .npy or .npz file')
    parser.add_argument('--npz', action='store_true', help='Save as .npz with key mu')
    args = parser.parse_args()
    extract_gaussian_centers(args.ply, args.out, npz=args.npz)
