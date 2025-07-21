import numpy as np
import argparse
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree

def load_ply_xyz(ply_path):
    ply = PlyData.read(ply_path)
    vertex = ply['vertex']
    xyz = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1)
    return xyz, vertex

def save_ply_xyz(vertex, mask, out_path):
    filtered_vertex = vertex[mask]
    PlyData([PlyElement.describe(filtered_vertex, 'vertex')]).write(out_path)

def main():
    parser = argparse.ArgumentParser(description="Filter Gaussians by scale and sparsity (spikiness)")
    parser.add_argument('--gaussian_ply', type=str, required=True, help='Input Gaussian .ply file')
    parser.add_argument('--scale_min', type=float, default=0.0, help='Minimum allowed scale (max of scale vector)')
    parser.add_argument('--scale_max', type=float, default=1e3, help='Maximum allowed scale (max of scale vector)')
    parser.add_argument('--spikiness_threshold', type=float, default=10.0, help='Maximum allowed ratio of largest to smallest scale (spikiness filter)')
    parser.add_argument('--out_ply', type=str, required=True, help='Output filtered Gaussian .ply file')
    args = parser.parse_args()

    # Load Gaussian centers and vertex
    gauss_xyz, gauss_vertex = load_ply_xyz(args.gaussian_ply)
    # Extract scale vectors
    scale_fields = [f'scale_{i}' for i in range(3) if f'scale_{i}' in gauss_vertex.data.dtype.names]
    if not scale_fields:
        print("No scale fields found in input PLY. Exiting.")
        return
    scales_vec = np.stack([gauss_vertex[f] for f in scale_fields], axis=1)
    # Clamp very small scale values to avoid division by zero
    scales_clamped = np.copy(scales_vec)
    scales_clamped[scales_clamped < 1e-6] = 1e-6
    max_scale = np.max(scales_clamped, axis=1)
    min_scale = np.min(scales_clamped, axis=1)
    mean_scale = np.mean(scales_clamped, axis=1)
    print(f"[STATS] Scale values across all Gaussians:")
    print(f"  max_scale: min={max_scale.min():.4f}, max={max_scale.max():.4f}, mean={max_scale.mean():.4f}, std={max_scale.std():.4f}")
    print(f"  min_scale: min={min_scale.min():.4f}, max={min_scale.max():.4f}, mean={min_scale.mean():.4f}, std={min_scale.std():.4f}")
    print(f"  mean_scale: min={mean_scale.min():.4f}, max={mean_scale.max():.4f}, mean={mean_scale.mean():.4f}, std={mean_scale.std():.4f}")
    spikiness_ratios = max_scale / min_scale
    # Filtering logic
    mask = (max_scale >= args.scale_min) & (max_scale <= args.scale_max) & (spikiness_ratios < args.spikiness_threshold)
    print(f"Filtered {np.sum(mask)} / {len(mask)} Gaussians with scale in [{args.scale_min}, {args.scale_max}] and spikiness < {args.spikiness_threshold}.")
    # Save filtered Gaussians
    save_ply_xyz(gauss_vertex, mask, args.out_ply)
    print(f"Saved filtered Gaussians to {args.out_ply}")

if __name__ == "__main__":
    main()