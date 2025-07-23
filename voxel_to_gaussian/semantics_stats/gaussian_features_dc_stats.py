import numpy as np
import torch
import argparse
from plyfile import PlyData

def get_features_dc_from_ply(ply_path):
    ply = PlyData.read(ply_path)
    vertex_fields = ply['vertex'].data.dtype.names
    # Try to get features_dc from PLY
    if 'features_dc' in vertex_fields:
        features_dc = ply['vertex']['features_dc']
        if features_dc.ndim == 2 and features_dc.shape[1] >= 3:
            rgb = np.array(features_dc[:, :3])
        else:
            rgb = np.array(features_dc).reshape(-1, 3)
        return rgb
    # Try to get as separate fields
    for prefix in ['features_dc', 'feature', 'color']:
        try:
            rgb = np.stack([
                ply['vertex'][f'{prefix}_0'],
                ply['vertex'][f'{prefix}_1'],
                ply['vertex'][f'{prefix}_2']
            ], axis=1)
            return rgb
        except Exception:
            continue
    # Try f_dc_0, f_dc_1, f_dc_2 fields
    if all(f in vertex_fields for f in ['f_dc_0', 'f_dc_1', 'f_dc_2']):
        rgb = np.stack([
            ply['vertex']['f_dc_0'],
            ply['vertex']['f_dc_1'],
            ply['vertex']['f_dc_2']
        ], axis=1)
        return rgb
    # Try standard RGB fields
    if all(f in vertex_fields for f in ['red', 'green', 'blue']):
        rgb = np.stack([
            ply['vertex']['red'],
            ply['vertex']['green'],
            ply['vertex']['blue']
        ], axis=1)
        return rgb
    # Print available fields for debugging
    print("Available vertex fields:", vertex_fields)
    raise ValueError("No features_dc, f_dc_0/1/2, or RGB fields found in PLY file.")

def get_rgb_stats_from_features_dc(rgb, bins=32):
    # features_dc[:,0] is red, features_dc[:,1] is green, features_dc[:,2] is blue
    stats = {}
    channel_map = {0: 'Red', 1: 'Green', 2: 'Blue'}
    for i in range(3):
        vals = rgb[:, i]
        stats[channel_map[i]] = {
            'min': float(np.min(vals)),
            'max': float(np.max(vals)),
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals)),
            '25_percentile': float(np.percentile(vals, 25)),
            '75_percentile': float(np.percentile(vals, 75)),
            'histogram': np.histogram(vals, bins=bins)[0].tolist()
        }
    return stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get statistics of RGB color values from features_dc in Gaussian PLY file.")
    parser.add_argument('--ply', type=str, required=True, help='Path to Gaussian .ply file')
    parser.add_argument('--bins', type=int, default=32, help='Number of bins for histogram')
    args = parser.parse_args()

    rgb = get_features_dc_from_ply(args.ply)
    stats = get_rgb_stats_from_features_dc(rgb, bins=args.bins)
    print(f"Statistics for {args.ply} (features_dc):")
    for channel in ['Red', 'Green', 'Blue']:
        print(f"\n{channel} channel (features_dc[{['Red','Green','Blue'].index(channel)}]):")
        for k, v in stats[channel].items():
            if k == 'histogram':
                print(f"  {k}: {v}")
            else:
                print(f"  {k}: {v:.4f}")

    # Plot and save histograms as PNG
    import matplotlib.pyplot as plt
    for i, channel in enumerate(['Red', 'Green', 'Blue']):
        vals = rgb[:, i]
        plt.figure(figsize=(8, 4))
        plt.hist(vals, bins=args.bins, color=channel.lower(), alpha=0.7)
        plt.title(f"Histogram of {channel} values")
        plt.xlabel(f"{channel} value")
        plt.ylabel("Count")
        plt.grid(True)
        png_path = f"{args.ply}_{channel}_hist.png"
        plt.savefig(png_path)
        plt.close()
        print(f"Saved histogram PNG: {png_path}")
