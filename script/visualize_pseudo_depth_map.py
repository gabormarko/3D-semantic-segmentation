import numpy as np
import matplotlib.pyplot as plt

DEPTH_PATH = "/home/neural_fields/Unified-Lift-Gabor/data/lerf/teatime/lseg_features_pseudodepth/test_2_pseudodepth.npy"
PNG_PATH = "/home/neural_fields/Unified-Lift-Gabor/data/lerf/teatime/lseg_features_pseudodepth/test_2_pseudodepth.png"

DEPTH_PATH = "/home/neural_fields/Unified-Lift-Gabor/data/lerf/teatime/lseg_features_pseudodepth/frame_00175_pseudodepth.npy"
PNG_PATH = "/home/neural_fields/Unified-Lift-Gabor/data/lerf/teatime/lseg_features_pseudodepth/frame_00175_pseudodepth.png"

# Load depth map
D = np.load(DEPTH_PATH)

# Mask invalid (zero) depths for visualization
D_vis = np.where(D > 0, D, np.nan)

plt.figure(figsize=(10, 8))
plt.imshow(D_vis, cmap='plasma')
plt.colorbar(label='Depth (meters)')
plt.title('Pseudo Depth Map: test_0_pseudodepth.npy')
plt.axis('off')
plt.tight_layout()
plt.savefig(PNG_PATH, bbox_inches='tight', pad_inches=0.1)
plt.show()
print(f"Saved depth map visualization as {PNG_PATH}")
