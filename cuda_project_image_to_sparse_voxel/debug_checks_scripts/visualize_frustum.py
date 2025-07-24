import torch
import open3d as o3d
import numpy as np
import argparse
import json

def qvec_to_R(qvec):
    qw, qx, qy, qz = qvec
    return np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),   1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),   1-2*(qx*qx+qy*qy)]
    ], dtype=np.float32)

def visualize_frustum_and_voxels(tensor_data_path, occupancy_path, output_ply_path, camera_params_path=None, image_name=None):
    """
    Loads tensor data and an occupancy grid, then visualizes the camera frustum
    and the voxel grid in 3D space, saving the result to a PLY file.
    """
    # Load the data needed for visualization
    tensor_data = torch.load(tensor_data_path)
    occupancy_data = torch.load(occupancy_path)

    # --- Voxel Grid Preparation ---
    # The occupancy tensor is a dense grid with voxel IDs. We need to find the coordinates.
    occ_coords = occupancy_data.nonzero(as_tuple=False).cpu().numpy() # Get (z, y, x) indices
    
    # Swap columns to get (x, y, z)
    occ_coords = occ_coords[:, [2, 1, 0]]

    # Convert from grid coordinates to world coordinates
    grid_origin = tensor_data['grid_origin'].cpu().numpy()
    voxel_size = tensor_data['voxel_size']
    world_coords = occ_coords * voxel_size + grid_origin

    voxel_pcd = o3d.geometry.PointCloud()
    voxel_pcd.points = o3d.utility.Vector3dVector(world_coords)
    voxel_pcd.paint_uniform_color([0.0, 1.0, 0.0])  # Green

    # --- Camera Frustum Preparation ---
    # If camera_params_path and image_name are provided, use COLMAP logic for intrinsics and image size
    if camera_params_path and image_name:
        with open(camera_params_path, 'r') as f:
            cam_params = json.load(f)
        # Find the image entry by name
        img_entry = None
        for v in cam_params['images'].values():
            if v['name'] == image_name:
                img_entry = v
                break
        if img_entry is None:
            raise ValueError(f"Image {image_name} not found in camera_params.json")
        qvec = np.array(img_entry['qvec'])
        tvec = np.array(img_entry['tvec'])
        R = qvec_to_R(qvec)
        C = -(R.T @ tvec)
        R_c2w = R.T
        # Get intrinsics and image size from COLMAP camera entry
        cam_entry = cam_params['cameras'][str(img_entry['camera_id'])]
        fx = cam_entry['params'][0]
        fy = cam_entry['params'][1]
        cx = cam_entry['params'][2]
        cy = cam_entry['params'][3]
        img_height = cam_entry['height']
        img_width = cam_entry['width']
        # Features: [B, V, H, W, C]
        features = tensor_data['encoded_2d_features']
        # If needed, upsample features to (img_height, img_width)
        import torch.nn.functional as F
        if features.shape[2:4] != (img_height, img_width):
            print(f"[DEBUG] Upsampling feature map from {features.shape[2:4]} to ({img_height}, {img_width})")
            B, V, H, W, C = features.shape
            features = features.permute(0, 1, 4, 2, 3).contiguous().view(B*V*C, H, W)
            features = F.interpolate(features.unsqueeze(0), size=(img_height, img_width), mode='bilinear', align_corners=False).squeeze(0)
            features = features.view(B, V, C, img_height, img_width).permute(0, 1, 3, 4, 2).contiguous()
            tensor_data['encoded_2d_features'] = features
    else:
        # Fallback: use intrinsics and image size from tensor_data
        intrinsics = tensor_data['intrinsicParams'].squeeze().cpu().numpy()
        img_height = tensor_data['encoded_2d_features'].shape[2]
        img_width = tensor_data['encoded_2d_features'].shape[3]
        fx, fy, cx, cy = intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3]
        cam_pose = tensor_data['viewMatrixInv'].squeeze().cpu().numpy()
        R_c2w = cam_pose[:3, :3]
        C = cam_pose[:3, 3]
    # Define frustum depth
    near_plane = 0.2
    far_plane = 1.5

    # Define frustum depth
    near_plane = 0.2
    far_plane = 1.5

    # Always use COLMAP image size for frustum if available
    if camera_params_path and image_name:
        H, W = img_height, img_width  # These are from COLMAP, original image size
    else:
        H, W = img_height, img_width  # fallback, may be feature size
    corners = np.array([
        [0, 0],
        [W-1, 0],
        [W-1, H-1],
        [0, H-1]
    ], dtype=np.float32)
    def pixel_to_cam(x, y, fx, fy, cx, cy, depth):
        X = (x - cx) * depth / fx
        Y = (y - cy) * depth / fy
        Z = depth
        return np.array([X, Y, Z])
    frustum_cam = [np.zeros(3)]
    for d in [near_plane, far_plane]:
        for x, y in corners:
            frustum_cam.append(pixel_to_cam(x, y, fx, fy, cx, cy, d))
    all_points_cam = np.stack(frustum_cam, axis=0)

    # Transform frustum points to world: X_world = R_c2w @ X_cam + C
    world_points = (R_c2w @ all_points_cam.T).T + C

    # Debug: Print camera pose info (after all variables are set)
    print("\n[DEBUG] Camera pose and frustum info:")
    print(f"fx, fy, cx, cy: {fx}, {fy}, {cx}, {cy}")
    if camera_params_path and image_name:
        print(f"Image size: {img_width}x{img_height} (COLMAP/original)")
    else:
        print(f"Image size: {img_width}x{img_height} (feature map)")
    print(f"near_plane: {near_plane}, far_plane: {far_plane}")
    print(f"Camera center (C): {C}")
    if camera_params_path and image_name:
        view_dir = R_c2w @ np.array([0,0,1])
        up_vec = R_c2w @ np.array([0,-1,0])
        print(f"View direction (world): {view_dir}")
        print(f"Up vector (world): {up_vec}")
    else:
        print("[DEBUG] Used fallback pose from tensor_data.pt")
    print("Frustum world points (first 5):\n", world_points[:5])

    # Define lines for the frustum
    # Lines from camera center to near plane corners, near plane rectangle,
    # far plane rectangle, and connecting lines.
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4], # Apex to near-plane corners
        [1, 2], [2, 3], [3, 4], [4, 1], # Near-plane rectangle
        [5, 6], [6, 7], [7, 8], [8, 5], # Far-plane rectangle
        [1, 5], [2, 6], [3, 7], [4, 8]  # Connecting near and far corners
    ]

    # Create a LineSet object
    frustum_lines = o3d.geometry.LineSet()
    frustum_lines.points = o3d.utility.Vector3dVector(world_points)
    frustum_lines.lines = o3d.utility.Vector2iVector(lines)
    frustum_lines.paint_uniform_color([1.0, 0.0, 0.0])  # Red

    # --- Combine and Save ---
    # To visualize both, we sample points along the lines of the frustum
    # and add them to the voxel point cloud.
    frustum_pcd = o3d.geometry.PointCloud()
    num_points_per_line = 20
    for line in lines:
        start_point = world_points[line[0]]
        end_point = world_points[line[1]]
        line_points = np.linspace(start_point, end_point, num_points_per_line)
        frustum_pcd.points.extend(o3d.utility.Vector3dVector(line_points))
    frustum_pcd.paint_uniform_color([1.0, 0.0, 0.0]) # Red

    print(f"Saving voxel grid and camera frustum visualization to {output_ply_path}")
    combined_pcd = voxel_pcd + frustum_pcd
    o3d.io.write_point_cloud(output_ply_path, combined_pcd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize camera frustum and voxel grid.")
    parser.add_argument("--tensor_data", required=True, help="Path to tensor_data.pt")
    parser.add_argument("--occupancy", required=True, help="Path to occupancy.pt")
    parser.add_argument("--output_ply", required=True, help="Path to save the output PLY file.")
    parser.add_argument("--camera_params", help="Path to camera_params.json (for COLMAP logic)")
    parser.add_argument("--image_name", help="Image name to visualize (for COLMAP logic)")
    args = parser.parse_args()

    visualize_frustum_and_voxels(args.tensor_data, args.occupancy, args.output_ply, args.camera_params, args.image_name)
