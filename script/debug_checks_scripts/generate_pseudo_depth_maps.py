import os
import numpy as np
import open3d as o3d
from tqdm import tqdm
from colmap_read_utils import read_cameras_binary, read_images_binary
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

# --- User parameters ---
VOXEL_PLY = "/home/neural_fields/Unified-Lift-Gabor/output/minkowski_grid/teatime_minkowski_249921vox_iter30000_grid.ply"
COLMAP_SPARSE_DIR = "/home/neural_fields/Unified-Lift-Gabor/data/lerf/teatime/sparse/0"
OUTPUT_DEPTH_DIR = "/home/neural_fields/Unified-Lift-Gabor/data/lerf/teatime/lseg_features_pseudodepth"
os.makedirs(OUTPUT_DEPTH_DIR, exist_ok=True)

# --- Load voxel grid ---
pcd = o3d.io.read_point_cloud(VOXEL_PLY)
vox_coords = np.asarray(pcd.points)  # [N, 3]

# --- Build KD-tree for fast nearest neighbor search ---
voxel_tree = cKDTree(vox_coords)

# --- Load COLMAP cameras and images ---
cameras = read_cameras_binary(os.path.join(COLMAP_SPARSE_DIR, "cameras.bin"))
images = read_images_binary(os.path.join(COLMAP_SPARSE_DIR, "images.bin"))

def qvec2rotmat(qvec):
    w, x, y, z = qvec
    return np.array([
        [1-2*y**2-2*z**2, 2*x*y-2*z*w, 2*x*z+2*y*w],
        [2*x*y+2*z*w, 1-2*x**2-2*z**2, 2*y*z-2*x*w],
        [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x**2-2*y**2]
    ])

def save_depth_png(depth_map, png_path, title=None):
    D_vis = np.where(depth_map > 0, depth_map, np.nan)
    plt.figure(figsize=(10, 8))
    plt.imshow(D_vis, cmap='plasma')
    plt.colorbar(label='Depth (meters)')
    if title:
        plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(png_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"Saved depth map visualization as {png_path}")

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
    # Parameters for ray sampling
    max_depth = 10.0  # meters
    step = 0.05      # meters (should match voxel size)
    t_samples = np.arange(0.1, max_depth, step)
    for y in range(height):
        for x in range(width):
            pix = np.array([x, y, 1.0])
            ray_dir = Rcw @ (Kinv @ pix)
            ray_dir = ray_dir / np.linalg.norm(ray_dir)
            points = cam_center[None, :] + t_samples[:, None] * ray_dir[None, :]
            dists, idxs = voxel_tree.query(points, distance_upper_bound=step*1.5)
            valid = np.isfinite(dists)
            if np.any(valid):
                first = np.argmax(valid)
                depth_map[y, x] = t_samples[first]
            else:
                depth_map[y, x] = 0.0
    # Save depth map
    base = os.path.splitext(img.name)[0]
    out_path = os.path.join(OUTPUT_DEPTH_DIR, base + "_pseudodepth.npy")
    np.save(out_path, depth_map)
    print(f"Saved pseudo-depth map: {out_path}")
    # Save PNG visualization
    png_path = os.path.join(OUTPUT_DEPTH_DIR, base + "_pseudodepth.png")
    save_depth_png(depth_map, png_path, title=f"Pseudo Depth Map: {base}")
