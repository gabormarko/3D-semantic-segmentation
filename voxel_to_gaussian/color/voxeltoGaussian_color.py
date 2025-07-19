#!/usr/bin/env python3
# Gaussian Color Assignment Pipeline
# Assigns each Gaussian the color of its nearest voxel.

import argparse
import pathlib
import numpy as np
import torch
from sklearn.neighbors import KDTree

def load_voxels(path: pathlib.Path):
    """Returns (pos, color) as *CPU* float tensors."""
    if path.suffix in {".pt", ".pth"}:
        d = torch.load(path, map_location="cpu")
        # Use 'xyz' and 'avg_color' keys as produced by voxel color checkpoint
        if "xyz" in d and "avg_color" in d:
            return d["xyz"].float(), d["avg_color"].float()
        elif "pos" in d and "color" in d:
            return d["pos"].float(), d["color"].float()
        else:
            raise KeyError(f"Voxel file {path} missing required keys. Found: {list(d.keys())}")
    elif path.suffix == ".npz":
        d = np.load(path)
        return torch.from_numpy(d["pos"]).float(), torch.from_numpy(d["color"]).float()
    raise ValueError(f"Unsupported voxel file format: {path}")

def load_gaussians_mu(path: pathlib.Path):
    """Load Gaussian means from .npy/.npz file."""
    if path.suffix in {".npy", ".npz"}:
        data = np.load(path)
        if hasattr(data, "files") and "mu" in data.files:
            mu = data["mu"]
        elif hasattr(data, "files") and len(data.files) > 0:
            mu = data[data.files[0]]
        else:
            mu = data
        return torch.from_numpy(mu).float()
    else:
        raise ValueError(f"Unsupported Gaussian file format: {path}")

@torch.inference_mode()
def map_gaussians_to_voxels(voxel_pos, gaussian_mu, batch_size=200_000):
    tree = KDTree(voxel_pos.numpy(), leaf_size=16)
    idx_chunks = []
    M = gaussian_mu.shape[0]
    for s in range(0, M, batch_size):
        e = min(s + batch_size, M)
        idx = tree.query(gaussian_mu[s:e].numpy(), k=1, return_distance=False)
        idx_chunks.append(torch.from_numpy(idx[:, 0]))
    return torch.cat(idx_chunks, dim=0)

def assign_colors_to_gaussians(voxel_colors, g2v_idx):
    # voxel_colors: (N, 3) or (N, 3, ...)
    # g2v_idx: (M,)
    colors = voxel_colors[g2v_idx]
    # If colors are not uint8, convert to [0,255] and uint8
    if colors.max() <= 1.0:
        colors = (colors * 255.0).round().clamp(0,255).to(torch.uint8)
    else:
        colors = colors.to(torch.uint8)
    return colors.cpu().numpy()  # (M,3)

def _cli_build_map(args):
    voxel_pos, _ = load_voxels(args.vox)
    gaussian_mu = load_gaussians_mu(args.gauss)
    idx = map_gaussians_to_voxels(voxel_pos, gaussian_mu, batch_size=args.batch)
    np.save(args.out, idx.numpy())
    print(f"[✓] Map saved: {args.out}  shape={idx.shape}")

def _cli_assign_color(args):
    _, voxel_colors = load_voxels(args.vox)
    g2v_idx = torch.from_numpy(np.load(args.map))
    colors = assign_colors_to_gaussians(voxel_colors, g2v_idx)
    np.savez(args.out, colors=colors)
    print(f"[✓] Colors saved: {args.out}  shape={colors.shape}")

if __name__ == "__main__":
    p = argparse.ArgumentParser("Gaussian Color Assignment Pipeline")
    sp = p.add_subparsers(dest="cmd", required=True)

    bld = sp.add_parser("build_map", help="Gaussian → voxel 1-NN index map")
    bld.add_argument("--vox", type=pathlib.Path, required=True, help=".pt/.npz voxel color file")
    bld.add_argument("--gauss", type=pathlib.Path, required=True, help=".npy/.npz with 'mu'")
    bld.add_argument("--out", type=pathlib.Path, required=True, help=".npy output of shape (M,)")
    bld.add_argument("--batch", type=int, default=200_000)
    bld.set_defaults(func=_cli_build_map)

    col = sp.add_parser("assign_color", help="Assign color to each Gaussian")
    col.add_argument("--vox", type=pathlib.Path, required=True)
    col.add_argument("--map", type=pathlib.Path, required=True, help=".npy produced by build_map")
    col.add_argument("--out", type=pathlib.Path, required=True, help=".npz output with colors")
    col.set_defaults(func=_cli_assign_color)

    args = p.parse_args()
    args.func(args)
