import open3d as o3d
import numpy as np
import json
import struct
from pathlib import Path

def read_points3d_binary(path_to_model_file):
    """
    see: https://colmap.github.io/format.html#points3d-bin
    """
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = struct.unpack('<Q', fid.read(8))[0]
        for _ in range(num_points):
            point3D_id = struct.unpack('<Q', fid.read(8))[0]
            xyz = np.fromfile(fid, np.float64, 3)
            rgb = np.fromfile(fid, np.uint8, 3)
            error = struct.unpack('<d', fid.read(8))[0]
            track_length = struct.unpack('<Q', fid.read(8))[0]
            track_elems = struct.unpack('<' + 'i' * 2 * track_length, fid.read(8 * track_length))
            points3D[point3D_id] = {'xyz': xyz, 'rgb': rgb, 'error': error, 'track_length': track_length, 'track_elems': track_elems}
    return points3D

def save_camera_visualization_ply(camera_params_file, colmap_points_file, output_ply_file, voxel_ply_file=None):
    # Load camera parameters
    with open(camera_params_file, 'r') as f:
        cameras = json.load(f)

    # Load COLMAP point cloud
    points3D = read_points3d_binary(colmap_points_file)
    points = np.array([p['xyz'] for p in points3D.values()])
    colors = np.array([p['rgb'] for p in points3D.values()]) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Create camera visualizations
    combined_mesh = o3d.geometry.TriangleMesh()
    for frame in cameras['images'].values():
        # Create a camera body as a sphere mesh
        cam_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.1) # Increased radius for visibility
        cam_mesh.paint_uniform_color([1.0, 0.0, 0.0]) # Red

        # Get rotation (quaternion) and translation from the frame data
        qvec = np.array(frame['qvec'])
        tvec = np.array(frame['tvec'])

        # Convert quaternion to a 3x3 rotation matrix.
        # Open3D expects quaternions in (w, x, y, z) format, which matches COLMAP.
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_quaternion(qvec)

        # Create a 4x4 transformation matrix (extrinsic matrix)
        transform_matrix = np.identity(4)
        transform_matrix[:3, :3] = rotation_matrix
        transform_matrix[:3, 3] = tvec
        
        # Apply transform to camera body
        cam_mesh.transform(transform_matrix)
        
        combined_mesh += cam_mesh

    # Sample points from the camera meshes to visualize them as points
    sampled_cam_pcd = combined_mesh.sample_points_uniformly(number_of_points=1000 * len(cameras['images']))
    
    # Combine the scene point cloud and the camera points
    combined_pcd = pcd + sampled_cam_pcd

    # If a voxel PLY file is provided, load it, color it, and add it
    if voxel_ply_file:
        voxel_pcd = o3d.io.read_point_cloud(voxel_ply_file)
        voxel_pcd.paint_uniform_color([0.0, 1.0, 0.0]) # Green
        combined_pcd += voxel_pcd

    o3d.io.write_point_cloud(output_ply_file, combined_pcd)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize COLMAP cameras and point cloud by saving to a PLY file.")
    parser.add_argument("--camera_params", required=True, help="Path to scaled_camera_params.json")
    parser.add_argument("--colmap_points", required=True, help="Path to points3D.bin from COLMAP sparse reconstruction")
    parser.add_argument("--voxel_ply", help="Path to the input voxel PLY file to visualize")
    parser.add_argument("--output_ply", required=True, help="Path to save the output PLY file")
    args = parser.parse_args()

    save_camera_visualization_ply(args.camera_params, args.colmap_points, args.output_ply, args.voxel_ply)
