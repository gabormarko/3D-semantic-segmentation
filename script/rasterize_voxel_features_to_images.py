import os
import numpy as np
import open3d as o3d
from tqdm import tqdm
import matplotlib.pyplot as plt
from colmap_read_utils import read_cameras_binary, read_images_binary

# Paths (edit as needed)
VOXEL_PLY = "/home/neural_fields/Unified-Lift-Gabor/output/minkowski_grid/teatime_minkowski_68458vox_iter30000_grid.ply"
VOXEL_FEATURES = "/home/neural_fields/Unified-Lift-Gabor/output/minkowski_grid/teatime_voxel_features.npy"
COLMAP_SPARSE_DIR = "/home/neural_fields/Unified-Lift-Gabor/data/lerf/teatime/sparse/0"
OUTPUT_DIR = "/home/neural_fields/Unified-Lift-Gabor/output/minkowski_grid/rasterized_lseg/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Load voxel grid and features
print(f"Loading voxel grid: {VOXEL_PLY}")
pcd = o3d.io.read_point_cloud(VOXEL_PLY)
vox_coords = np.asarray(pcd.points)  # [N, 3]
voxel_features = np.load(VOXEL_FEATURES)  # [N, C]
N_vox, C = voxel_features.shape

# Print statistics of loaded voxel features
print("Loaded voxel feature statistics:")
print("  mean:", np.mean(voxel_features))
print("  std:", np.std(voxel_features))
print("  min:", np.min(voxel_features))
print("  max:", np.max(voxel_features))

# 2. Load COLMAP cameras and images
cameras = read_cameras_binary(os.path.join(COLMAP_SPARSE_DIR, "cameras.bin"))
images = read_images_binary(os.path.join(COLMAP_SPARSE_DIR, "images.bin"))

# 3. Create a color map for features (use PCA to reduce to 3D, then colormap)
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
feat_rgb = pca.fit_transform(voxel_features)
feat_rgb = (feat_rgb - feat_rgb.min(0)) / (feat_rgb.ptp(0) + 1e-8)  # Normalize to [0,1]
feat_rgb = (feat_rgb * 255).astype(np.uint8)
print("feat_rgb min/max:", feat_rgb.min(), feat_rgb.max())

# 4. Rasterize for each image
for img in tqdm(images.values()):
    cam = cameras[img.camera_id]
    if cam.model != "PINHOLE":
        print(f"Skipping camera model {cam.model}")
        continue
    fx, fy, cx, cy = cam.params[:4]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    R = np.array([
        [1-2*img.qvec[2]**2-2*img.qvec[3]**2, 2*img.qvec[1]*img.qvec[2]-2*img.qvec[3]*img.qvec[0], 2*img.qvec[1]*img.qvec[3]+2*img.qvec[2]*img.qvec[0]],
        [2*img.qvec[1]*img.qvec[2]+2*img.qvec[3]*img.qvec[0], 1-2*img.qvec[1]**2-2*img.qvec[3]**2, 2*img.qvec[2]*img.qvec[3]-2*img.qvec[1]*img.qvec[0]],
        [2*img.qvec[1]*img.qvec[3]-2*img.qvec[2]*img.qvec[0], 2*img.qvec[2]*img.qvec[3]+2*img.qvec[1]*img.qvec[0], 1-2*img.qvec[1]**2-2*img.qvec[2]**2]
    ])
    t = img.tvec.reshape(3, 1)
    H, W = cam.height, cam.width  # Use camera intrinsics for image size
    # Project all voxels to this image
    vox_cam = R @ (vox_coords.T - t)  # [3, N]
    pix = K @ vox_cam  # [3, N]
    pix = pix[:2] / pix[2:3]  # [2, N]
    pix = pix.T  # [N, 2]
    # Create blank image
    img_out = np.zeros((H, W, 3), dtype=np.uint8)
    # Only keep voxels in front of camera and inside image
    valid = (vox_cam[2] > 0) & (pix[:,0] >= 0) & (pix[:,0] < W) & (pix[:,1] >= 0) & (pix[:,1] < H)
    print(f"{img.name}: {np.sum(valid)} voxels projected in front of camera and inside image")
    pix_valid = pix[valid].astype(int)
    colors_valid = feat_rgb[valid]
    print(f"{img.name}: {pix_valid.shape[0]} pixels to rasterize")
    # Rasterize: assign color to pixel (nearest voxel wins, no z-buffer)
    splat_radius = 6  # Splat with a 13x13 patch (radius=6)
    for (x, y), color in zip(pix_valid, colors_valid):
        for dx in range(-splat_radius, splat_radius+1):
            for dy in range(-splat_radius, splat_radius+1):
                xx, yy = x+dx, y+dy
                if 0 <= xx < W and 0 <= yy < H:
                    img_out[yy, xx] = color
    # Debug: check if any pixel is not black
    non_black = np.sum(np.any(img_out != 0, axis=2))
    print(f"{img.name}: {non_black} non-black pixels in output image")
    # Save
    out_path = os.path.join(OUTPUT_DIR, os.path.splitext(img.name)[0] + "_rasterized.png")
    plt.imsave(out_path, img_out)
    print(f"Saved: {out_path}")
