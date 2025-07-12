import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import json
from mpl_toolkits.mplot3d import Axes3D

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tensor_data', required=True, help='Path to tensor_data.pt')
    parser.add_argument('--occupancy', required=True, help='Path to occupancy.pt')
    parser.add_argument('--camera_params', required=True, help='Path to camera_params.json')
    parser.add_argument('--image_name', default='DSC03423.JPG', help='Image name to use for camera pose')
    parser.add_argument('--output', default='occupancy_grid_view.png', help='Output image file')
    args = parser.parse_args()

    # Load occupancy grid
    occ = torch.load(args.occupancy)
    if hasattr(occ, 'numpy'):
        occ = occ.numpy()
    print(f"Occupancy grid shape: {occ.shape}, max: {occ.max()}, min: {occ.min()}")

    # Load tensor_data for grid origin and voxel size
    tensor_data = torch.load(args.tensor_data, map_location='cpu')
    grid_origin = tensor_data['grid_origin'].cpu().numpy()
    voxel_size = float(tensor_data['voxel_size'])
    print(f"Grid origin: {grid_origin}, voxel size: {voxel_size}")

    # Load camera pose from camera_params.json
    with open(args.camera_params, 'r') as f:
        cam_params = json.load(f)
    img_entry = None
    for v in cam_params['images'].values():
        if v['name'] == args.image_name:
            img_entry = v
            break
    if img_entry is None:
        raise ValueError(f"Image {args.image_name} not found in camera_params.json")
    qvec = np.array(img_entry['qvec'])
    tvec = np.array(img_entry['tvec'])
    # COLMAP qvec: [qw, qx, qy, qz]
    def qvec_to_R(qvec):
        qw, qx, qy, qz = qvec
        return np.array([
            [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
            [2*(qx*qy+qz*qw),   1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
            [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),   1-2*(qx*qx+qy*qy)]
        ], dtype=np.float32)
    R = qvec_to_R(qvec)
    C = -(R.T @ tvec)
    print(f"Camera center: {C}")
    R_c2w = R.T

    # Get all occupied voxel coordinates (z, y, x)
    occ_coords = np.argwhere(occ > 0)
    print(f"Number of occupied voxels: {len(occ_coords)}")
    # Convert to world coordinates (x, y, z)
    occ_coords_xyz = occ_coords[:, [2, 1, 0]]
    world_coords = occ_coords_xyz * voxel_size + grid_origin

    # Project voxel centers to camera image plane
    cam_entry = cam_params['cameras'][str(img_entry['camera_id'])]
    fx = cam_entry['params'][0]
    fy = cam_entry['params'][1]
    cx = cam_entry['params'][2]
    cy = cam_entry['params'][3]
    img_height = cam_entry['height']
    img_width = cam_entry['width']

    # Transform world points to camera coordinates
    rel_coords = world_coords - C[None, :]
    cam_coords = (R_c2w.T @ rel_coords.T).T  # [N, 3]
    # Only keep points in front of the camera
    in_front = cam_coords[:, 2] > 0.1
    cam_coords = cam_coords[in_front]
    # Project to image plane
    u = fx * (cam_coords[:, 0] / cam_coords[:, 2]) + cx
    v = fy * (cam_coords[:, 1] / cam_coords[:, 2]) + cy
    # Only keep points inside the image
    valid = (u >= 0) & (u < img_width) & (v >= 0) & (v < img_height)
    u = u[valid]
    v = v[valid]
    print(f"Projected {len(u)} voxels into the image.")

    # Plot the image
    plt.figure(figsize=(10, 8))
    plt.scatter(u, v, s=0.1, c='red', alpha=0.5, label='Projected Voxels')
    plt.xlim([0, img_width])
    plt.ylim([img_height, 0])
    plt.xlabel('u (pixels)')
    plt.ylabel('v (pixels)')
    plt.title(f'Occupancy Grid Projected to Camera: {args.image_name}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=200)
    print(f"Saved projection image to {args.output}")

    # Optionally, plot a 3D view
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(world_coords[:, 0], world_coords[:, 1], world_coords[:, 2], s=0.1, c='green', alpha=0.2)
    ax.scatter(C[0], C[1], C[2], c='red', s=50, label='Camera Center')
    ax.set_title('3D Occupancy Grid and Camera')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.tight_layout()
    plt.savefig('occupancy_grid_3d.png', dpi=200)
    print("Saved 3D occupancy grid visualization to occupancy_grid_3d.png")

if __name__ == '__main__':
    main()
