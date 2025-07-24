def export_gaussian_centers_to_ply(gaussians, output_ply):
    xyz = gaussians._xyz.detach().cpu().numpy()
    # Use DC color if available, else gray
    try:
        colors = gaussians._features_dc.detach().cpu().numpy()  # shape [N, 3, 1] or [N, 1, 3]
        if colors.shape[1] == 3 and colors.shape[2] == 1:
            colors = colors[:, :, 0]
        elif colors.shape[1] == 1 and colors.shape[2] == 3:
            colors = colors[:, 0, :]
        else:
            colors = np.ones_like(xyz) * 0.5
        colors = np.clip(colors, 0, 1)
    except Exception as e:
        print(f"Warning: Could not extract DC color ({e}). Exported points will be gray.")
        colors = np.ones_like(xyz) * 0.5
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(output_ply, pcd)
    print(f"Exported {xyz.shape[0]} Gaussian centers as points to {output_ply}")

def export_gaussian_ellipsoids_to_ply(gaussians, output_ply, max_ellipsoids=None):
    xyz = gaussians._xyz.detach().cpu().numpy()
    scales = gaussians._scaling.detach().cpu().numpy()
    # Use DC color if available, else gray
    try:
        colors = gaussians._features_dc.detach().cpu().numpy()
        if colors.shape[1] == 3 and colors.shape[2] == 1:
            colors = colors[:, :, 0]
        elif colors.shape[1] == 1 and colors.shape[2] == 3:
            colors = colors[:, 0, :]
        else:
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
        # Use low-res sphere mesh (resolution=4 gives 12 vertices)
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=4)
        mesh.paint_uniform_color(color)
        mesh.scale(1.0, center=[0, 0, 0])
        mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices) * scale)
        mesh.translate(center)
        meshes.append(mesh)
    print(f"Exporting {N} ellipsoids as a mesh to {output_ply}")
    mesh_out = meshes[0]
    for mesh in meshes[1:]:
        mesh_out += mesh
    o3d.io.write_triangle_mesh(output_ply, mesh_out)
    print(f"Exported ellipsoids mesh to {output_ply}")
import torch
import numpy as np
import open3d as o3d
import sys
import os
import csv
import glob
import json
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

def save_gaussian_properties(gaussians, output_json):
    def tensor_to_list(obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif isinstance(obj, dict):
            return {k: tensor_to_list(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [tensor_to_list(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(tensor_to_list(v) for v in obj)
        elif isinstance(obj, set):
            return {tensor_to_list(v) for v in obj}
        else:
            return obj

    optimizer_state = gaussians.optimizer.state_dict() if gaussians.optimizer is not None else None
    optimizer_state_serializable = tensor_to_list(optimizer_state)
    props = {
        'active_sh_degree': gaussians.active_sh_degree,
        'max_sh_degree': gaussians.max_sh_degree,
        'num_objects': getattr(gaussians, 'num_objects', None),
        'percent_dense': getattr(gaussians, 'percent_dense', None),
        'spatial_lr_scale': getattr(gaussians, 'spatial_lr_scale', None),
        'xyz': gaussians._xyz.detach().cpu().numpy().tolist(),
        'features_dc': gaussians._features_dc.detach().cpu().numpy().tolist(),
        'features_rest': gaussians._features_rest.detach().cpu().numpy().tolist(),
        'scaling': gaussians._scaling.detach().cpu().numpy().tolist(),
        'rotation': gaussians._rotation.detach().cpu().numpy().tolist(),
        'opacity': gaussians._opacity.detach().cpu().numpy().tolist(),
        'objects_dc': gaussians._objects_dc.detach().cpu().numpy().tolist(),
        'max_radii2D': gaussians.max_radii2D.detach().cpu().numpy().tolist() if hasattr(gaussians.max_radii2D, 'detach') else str(gaussians.max_radii2D),
        'xyz_gradient_accum': gaussians.xyz_gradient_accum.detach().cpu().numpy().tolist() if hasattr(gaussians.xyz_gradient_accum, 'detach') else str(gaussians.xyz_gradient_accum),
        'denom': gaussians.denom.detach().cpu().numpy().tolist() if hasattr(gaussians.denom, 'detach') else str(gaussians.denom),
        'optimizer_state': optimizer_state_serializable
    }
    with open(output_json, 'w') as f:
        json.dump(props, f, indent=2)
    print(f"Saved Gaussian model properties to {output_json}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Export Gaussian ellipsoid mesh and center point cloud from a .ply file.")
    parser.add_argument("--input_ply", type=str, required=True, help="Input .ply file with Gaussian properties.")
    parser.add_argument("--output_prefix", type=str, required=False, default=None, help="Prefix for output files (default: input filename without extension)")
    parser.add_argument("--max_ellipsoids", type=int, default=5000, help="Maximum number of ellipsoids to export (default: 1000)")
    args = parser.parse_args()

    input_ply = args.input_ply
    if args.output_prefix is not None:
        output_prefix = args.output_prefix
    else:
        output_prefix = os.path.splitext(os.path.basename(input_ply))[0]
    output_dir = os.path.dirname(input_ply)

    print(f"\n{'='*80}")
    print(f"Processing input PLY: {input_ply}")
    print(f"{'='*80}")

    # Load Gaussian model from .ply
    gaussians = GaussianModel(sh_degree=3)
    gaussians.load_ply(input_ply)

    # Save center point cloud
    center_ply = os.path.join(output_dir, f"{output_prefix}_center.ply")
    export_gaussian_centers_to_ply(gaussians, center_ply)

    # Save ellipsoid mesh
    ellipsoids_ply = os.path.join(output_dir, f"{output_prefix}_ellipsoids.ply")
    export_gaussian_ellipsoids_to_ply(gaussians, ellipsoids_ply, max_ellipsoids=args.max_ellipsoids)

    print(f"Saved Gaussian center points to {center_ply}")
    print(f"Saved Gaussian ellipsoid mesh to {ellipsoids_ply}")
