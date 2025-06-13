import torch
import numpy as np
import open3d as o3d
import sys
import os
import csv
import glob
from scene.gaussian_model import GaussianModel

def load_gaussians_from_checkpoint(checkpoint_path, sh_degree=3):
    class DummyArgs:
        iterations = 1_000
        position_lr_init = 0.00016
        position_lr_final = 0.0000016
        position_lr_delay_mult = 0.01
        position_lr_max_steps = 30_000
        feature_lr = 0.0025
        opacity_lr = 0.05
        scaling_lr = 0.005
        rotation_lr = 0.001
        percent_dense = 0.01
        lambda_dssim = 0.2
        densification_interval = 100
        opacity_reset_interval = 3000
        densify_from_iter = 500
        densify_until_iter = 15_000
        densify_grad_threshold = 0.0002
        reg3d_interval = 2
        reg3d_k = 5
        reg3d_lambda_val = 2
        reg3d_max_points = 300000
        reg3d_sample_size = 1000

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_params = checkpoint[0] if isinstance(checkpoint, tuple) else checkpoint
    gaussians = GaussianModel(sh_degree)
    dummy_args = DummyArgs()
    gaussians.restore(model_params, dummy_args)
    return gaussians

def export_gaussians_to_ply(gaussians, output_ply):
    xyz = gaussians.get_xyz.detach().cpu().numpy()
    if hasattr(gaussians, "get_features_dc"):
        colors = gaussians.get_features_dc.detach().cpu().numpy()
        colors = np.clip(colors, 0, 1)
        if np.allclose(colors, 0):
            print("Warning: All DC colors are zero. Exported points will be gray.")
            colors = np.ones_like(xyz) * 0.5
    else:
        print("Warning: No DC color found. Exported points will be gray.")
        colors = np.ones_like(xyz) * 0.5
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(output_ply, pcd)
    print(f"Exported {xyz.shape[0]} Gaussians as points to {output_ply}")

def export_gaussian_ellipsoids_to_mesh(gaussians, output_mesh, max_ellipsoids=50000):
    xyz = gaussians.get_xyz.detach().cpu().numpy()
    scales = gaussians.get_scaling.detach().cpu().numpy()
    # Use DC color if available, else gray
    try:
        colors = gaussians._features_dc.detach().cpu().numpy()  # shape [N, 1, 3]
        if colors.shape[1] == 1 and colors.shape[2] == 3:
            colors = colors[:, 0, :]  # Now shape [N, 3]
        else:
            print("Warning: Unexpected DC color shape, using gray.")
            colors = np.ones_like(xyz) * 0.5
        colors = np.clip(colors, 0, 1)
    except Exception as e:
        print(f"Warning: Could not extract DC color ({e}). Exported ellipsoids will be gray.")
        colors = np.ones_like(xyz) * 0.5

    meshes = []
    N = len(xyz) if max_ellipsoids is None else min(len(xyz), max_ellipsoids)
    for i in range(N):
        center = xyz[i]
        scale = scales[i]
        color = colors[i] if colors is not None else [0.5, 0.5, 0.5]
        # Ensure scale is 3D
        if np.isscalar(scale):
            scale = np.array([scale, scale, scale])
        elif scale.shape == (1,):
            scale = np.repeat(scale, 3)
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=10)
        mesh.paint_uniform_color(color)
        mesh.scale(1.0, center=[0, 0, 0])
        mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices) * scale)
        mesh.translate(center)
        meshes.append(mesh)
    print(f"Exporting {N} ellipsoids as a mesh to {output_mesh}")
    mesh_out = meshes[0]
    for mesh in meshes[1:]:
        mesh_out += mesh
    o3d.io.write_triangle_mesh(output_mesh, mesh_out)
    print(f"Exported ellipsoids mesh to {output_mesh}")

def export_gaussian_sh_to_csv(gaussians, output_csv):
    # SH features: [N, n_coeffs, 3] (for RGB)
    if hasattr(gaussians, "features"):
        sh = gaussians.features.detach().cpu().numpy()
        N, n_coeffs, _ = sh.shape
        with open(output_csv, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            header = ["gaussian_idx"]
            for c in range(n_coeffs):
                for ch in ["R", "G", "B"]:
                    header.append(f"sh{c}_{ch}")
            writer.writerow(header)
            for i in range(N):
                row = [i]
                row.extend(sh[i].flatten())
                writer.writerow(row)
        print(f"Exported SH coefficients to {output_csv}")
    else:
        print("No SH features found in model.")

def export_ply_to_ply(input_ply, output_ply):
    pcd = o3d.io.read_point_cloud(input_ply)
    if len(pcd.colors) == 0:
        print("Warning: Input .ply has no color. Exporting as gray.")
        colors = np.ones((len(pcd.points), 3)) * 0.5
        pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(output_ply, pcd)
    print(f"Copied point cloud from {input_ply} to {output_ply}")

if __name__ == "__main__":
    # Define scenes to process
    scenes = ["ramen", "figurines", "teatime"]
    base_dir = "/home/neural_fields/Unified-Lift-Gabor/output/unifed_lift"
    
    for scene in scenes:
        chkpnt_folder = os.path.join(base_dir, scene, "chkpnts")
        output_folder = os.path.join(base_dir, scene, "gaussians")
        os.makedirs(output_folder, exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"Processing scene: {scene}")
        print(f"{'='*80}")
        
        chkpnt_files = sorted(glob.glob(os.path.join(chkpnt_folder, "*.pth")))
        
        if len(chkpnt_files) == 0:
            print(f"No .pth files found in {chkpnt_folder}")
            continue
            
        print(f"Found {len(chkpnt_files)} checkpoint files")
        
        for input_path in chkpnt_files:
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            gaussians = load_gaussians_from_checkpoint(input_path)
            xyz = gaussians.get_xyz.detach().cpu().numpy()
            num_points = xyz.shape[0]
            points_ply = os.path.join(output_folder, f"{scene}_{base_name}_{num_points}_pts.ply")
            ellipsoids_ply = os.path.join(output_folder, f"{scene}_{base_name}_{num_points}_ellipsoids.ply")
            print(f"\nProcessing {os.path.basename(input_path)}")
            print(f"Output files:")
            print(f"  Points: {points_ply}")
            print(f"  Ellipsoids: {ellipsoids_ply}")
            try:
                export_gaussians_to_ply(gaussians, points_ply)
                export_gaussian_ellipsoids_to_mesh(gaussians, ellipsoids_ply)
            except Exception as e:
                print(f"Error processing {input_path}: {e}")
                continue