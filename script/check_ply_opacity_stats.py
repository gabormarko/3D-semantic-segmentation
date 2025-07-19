#!/usr/bin/env python3
from plyfile import PlyData
import numpy as np
import sys

if len(sys.argv) < 2:
    print("Usage: python check_ply_opacity_stats.py <path_to_ply>")
    sys.exit(1)

ply_path = sys.argv[1]
ply = PlyData.read(ply_path)
vertex = ply['vertex']
opacity = vertex['opacity']
print(f"[INFO] Opacity stats for {ply_path}:")
print(f"  min: {np.min(opacity):.6f}")
print(f"  max: {np.max(opacity):.6f}")
print(f"  mean: {np.mean(opacity):.6f}")

# Plot and save histogram of opacity distribution
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 5))
plt.hist(opacity, bins=100, color='blue', alpha=0.7)
plt.xlabel('Opacity')
plt.ylabel('Count')
plt.title(f'Opacity Distribution\n{ply_path}')
plt.grid(True)
png_path = ply_path.replace('.ply', '_opacity_hist.png')
plt.savefig(png_path)
print(f"[INFO] Saved opacity histogram to {png_path}")
