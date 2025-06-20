import os
import numpy as np
import open3d as o3d
from tqdm import tqdm
from colmap_read_utils import read_cameras_binary, read_images_binary

# Paths (edit as needed)
VOXEL_PLY = "/home/neural_fields/Unified-Lift-Gabor/output/minkowski_grid/teatime_minkowski_68458vox_iter30000_grid.ply"
FEATURE_DIR = "/home/neural_fields/Unified-Lift-Gabor/data/lerf/teatime/lseg_features"
OUTPUT_FEATURES = "/home/neural_fields/Unified-Lift-Gabor/output/minkowski_grid/teatime_voxel_features.npy"
COLMAP_SPARSE_DIR = "/home/neural_fields/Unified-Lift-Gabor/data/lerf/teatime/sparse/0"

# 1. Load voxel grid (PLY)
print(f"Loading voxel grid: {VOXEL_PLY}")
pcd = o3d.io.read_point_cloud(VOXEL_PLY)
vox_coords = np.asarray(pcd.points)  # [N, 3]
N_vox = vox_coords.shape[0]

# 1b. Load COLMAP cameras and images
cameras = read_cameras_binary(os.path.join(COLMAP_SPARSE_DIR, "cameras.bin"))
images = read_images_binary(os.path.join(COLMAP_SPARSE_DIR, "images.bin"))
# Map image name to (K, R, t)
img_pose_dict = {}
for img in images.values():
    cam = cameras[img.camera_id]
    # Intrinsics
    if cam.model == "PINHOLE":
        fx, fy, cx, cy = cam.params[:4]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    else:
        raise NotImplementedError(f"Camera model {cam.model} not supported")
    # Extrinsics (COLMAP: world2cam)
    # Convert quaternion to rotation matrix
    def qvec2rotmat(qvec):
        w, x, y, z = qvec
        return np.array([
            [1-2*y**2-2*z**2, 2*x*y-2*z*w, 2*x*z+2*y*w],
            [2*x*y+2*z*w, 1-2*x**2-2*z**2, 2*y*z-2*x*w],
            [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x**2-2*y**2]
        ])
    R = qvec2rotmat(img.qvec)
    t = img.tvec.reshape(3, 1)
    img_pose_dict[img.name] = (K, R, t)

# 2. Prepare to aggregate features for each voxel
feature_dim = None
voxel_feat_sum = None
voxel_feat_count = np.zeros(N_vox, dtype=np.int32)

# 3. For each image feature file, project features to voxels
for fname in tqdm(sorted(os.listdir(FEATURE_DIR))):
    if not fname.endswith('.npy'):
        continue
    feat_path = os.path.join(FEATURE_DIR, fname)
    features = np.load(feat_path)  # [C, H, W] or [H, W, C]
    if features.shape[0] < 10:  # likely [C, H, W], transpose to [H, W, C]
        features = features.transpose(1, 2, 0)
    H, W, C = features.shape
    if feature_dim is None:
        feature_dim = C
        voxel_feat_sum = np.zeros((N_vox, C), dtype=np.float32)
    # Find corresponding camera pose for this image
    # Assume feature file is named like 'IMG_0001.npy' and image is 'IMG_0001.jpg' or similar
    img_base = os.path.splitext(fname)[0]
    # Try to find matching image name in COLMAP
    colmap_img_name = None
    for k in img_pose_dict:
        if img_base in k:
            colmap_img_name = k
            break
    if colmap_img_name is None:
        print(f"Warning: No COLMAP pose for {fname}")
        continue
    K, R, t = img_pose_dict[colmap_img_name]
    # --- DEMO: Use a constant depth for all pixels (replace with real depth if available) ---
    const_depth = 2.0  # meters (adjust as needed)
    # Precompute inverse intrinsics and camera-to-world
    Kinv = np.linalg.inv(K)
    Rcw = R.T  # COLMAP: world2cam, so cam2world is R.T
    tcw = -R.T @ t  # cam2world translation
    # Project each pixel to 3D and assign to nearest voxel
    for y in range(H):
        for x in range(W):
            # 1. Backproject pixel to camera coordinates
            pix = np.array([x, y, 1.0])
            p_cam = Kinv @ pix * const_depth  # [3]
            # 2. Transform to world coordinates
            p_world = Rcw @ p_cam + tcw.squeeze()
            # 3. Find nearest voxel
            dists = np.linalg.norm(vox_coords - p_world, axis=1)
            voxel_idx = np.argmin(dists)
            # 4. Aggregate feature
            voxel_feat_sum[voxel_idx] += features[y, x]
            voxel_feat_count[voxel_idx] += 1

# 4. Average features for each voxel
voxel_feat_avg = np.zeros_like(voxel_feat_sum)
nonzero = voxel_feat_count > 0
voxel_feat_avg[nonzero] = voxel_feat_sum[nonzero] / voxel_feat_count[nonzero][:, None]

# 5. Save voxel features
np.save(OUTPUT_FEATURES, voxel_feat_avg)
print(f"Saved voxel features: {OUTPUT_FEATURES}")
