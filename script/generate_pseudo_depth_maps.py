import os
import numpy as np
import open3d as o3d
from tqdm import tqdm
from colmap_read_utils import read_cameras_binary, read_images_binary

# --- User parameters ---
VOXEL_PLY = "/home/neural_fields/Unified-Lift-Gabor/output/minkowski_grid/teatime_minkowski_68458vox_iter30000_grid.ply"
COLMAP_SPARSE_DIR = "/home/neural_fields/Unified-Lift-Gabor/data/lerf/teatime/sparse/0"
OUTPUT_DEPTH_DIR = "/home/neural_fields/Unified-Lift-Gabor/data/lerf/teatime/lseg_features_pseudodepth"
os.makedirs(OUTPUT_DEPTH_DIR, exist_ok=True)

# --- Load voxel grid ---
pcd = o3d.io.read_point_cloud(VOXEL_PLY)
vox_coords = np.asarray(pcd.points)  # [N, 3]

# --- Load COLMAP cameras and images ---
cameras = read_cameras_binary(os.path.join(COLMAP_SPARSE_DIR, "cameras.bin"))
images = read_images_binary(os.path.join(COLMAP_SPARSE_DIR, "images.bin"))

# --- For each image, create a pseudo-depth map ---
def qvec2rotmat(qvec):
    w, x, y, z = qvec
    return np.array([
        [1-2*y**2-2*z**2, 2*x*y-2*z*w, 2*x*z+2*y*w],
        [2*x*y+2*z*w, 1-2*x**2-2*z**2, 2*y*z-2*x*w],
        [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x**2-2*y**2]
    ])

for img in tqdm(images.values()):
    cam = cameras[img.camera_id]
    if cam.model != "PINHOLE":
        print(f"Skipping {img.name}: unsupported camera model {cam.model}")
        continue
    fx, fy, cx, cy = cam.params[:4]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    Kinv = np.linalg.inv(K)
    R = qvec2rotmat(img.qvec)
    t = img.tvec.reshape(3, 1)
    Rcw = R.T
    tcw = -R.T @ t
    width, height = cam.width, cam.height
    depth_map = np.zeros((height, width), dtype=np.float32)
    cam_center = tcw.squeeze()
    for y in range(height):
        for x in range(width):
            pix = np.array([x, y, 1.0])
            ray_dir = Rcw @ (Kinv @ pix)
            ray_dir = ray_dir / np.linalg.norm(ray_dir)
            # Find the closest voxel center along the ray
            # Project all voxels into the ray direction
            rel_vox = vox_coords - cam_center
            t_vals = rel_vox @ ray_dir
            mask = t_vals > 0  # only in front of camera
            if not np.any(mask):
                depth_map[y, x] = 0.0
                continue
            proj_points = cam_center + t_vals[:, None] * ray_dir[None, :]
            dists = np.linalg.norm(proj_points - vox_coords, axis=1)
            # Only consider voxels close to the ray
            close_mask = (dists < 0.02) & mask  # 2cm threshold
            if not np.any(close_mask):
                depth_map[y, x] = 0.0
                continue
            # Use the closest voxel along the ray
            idx = np.argmin(t_vals + (~close_mask)*1e6)
            depth_map[y, x] = t_vals[idx]
    # Save depth map
    out_path = os.path.join(OUTPUT_DEPTH_DIR, os.path.splitext(img.name)[0] + "_pseudodepth.npy")
    np.save(out_path, depth_map)
    print(f"Saved pseudo-depth map: {out_path}")
