import numpy as np
import argparse
from plyfile import PlyData
import os

def get_rgb_stats(ply_path, bins=32):
    ply = PlyData.read(ply_path)
    # Try to get RGB fields
    if 'red' in ply['vertex'].data.dtype.names:
        red = ply['vertex']['red']
        green = ply['vertex']['green']
        blue = ply['vertex']['blue']
        colors = np.stack([red, green, blue], axis=1)
    elif 'color' in ply['vertex'].data.dtype.names:
        colors = np.array(ply['vertex']['color'])
    else:
        raise ValueError("No RGB color fields found in PLY file.")

    stats = {}
    for i, channel in enumerate(['R', 'G', 'B']):
        vals = colors[:, i]
        stats[channel] = {
            'min': float(np.min(vals)),
            'max': float(np.max(vals)),
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals)),
            '25_percentile': float(np.percentile(vals, 25)),
            '75_percentile': float(np.percentile(vals, 75)),
            'histogram': np.histogram(vals, bins=bins, range=(0,255))[0].tolist()
        }
    return stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get statistics of RGB color values from Gaussian PLY file.")
    parser.add_argument('--ply', type=str, required=True, help='Path to Gaussian .ply file')
    parser.add_argument('--bins', type=int, default=32, help='Number of bins for histogram')
    args = parser.parse_args()

    stats = get_rgb_stats(args.ply, bins=args.bins)
    print(f"Statistics for {args.ply}:")
    for channel in ['R', 'G', 'B']:
        print(f"\n{channel} channel:")
        for k, v in stats[channel].items():
            if k == 'histogram':
                print(f"  {k}: {v}")
            else:
                print(f"  {k}: {v:.4f}")
