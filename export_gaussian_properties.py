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
    chkpnt_folder = "/home/neural_fields/Unified-Lift-Gabor/data/scannetpp/officescene/chkpnts"
    output_folder = "/home/neural_fields/Unified-Lift-Gabor/data/scannetpp/officescene/gaussians"
    os.makedirs(output_folder, exist_ok=True)
    print(f"\n{'='*80}")
    print(f"Processing folder: {chkpnt_folder}")
    print(f"{'='*80}")
    chkpnt_files = sorted(glob.glob(os.path.join(chkpnt_folder, "*.pth")))
    if len(chkpnt_files) == 0:
        print(f"No .pth files found in {chkpnt_folder}")
    else:
        # Find the file with the largest number in its name (chkpntxxxxxx.pth)
        def extract_number(fname):
            import re
            m = re.search(r'chkpnt(\d+)', fname)
            return int(m.group(1)) if m else -1
        largest_file = max(chkpnt_files, key=lambda f: extract_number(os.path.basename(f)))
        base_name = os.path.splitext(os.path.basename(largest_file))[0]
        props_json = os.path.join(output_folder, f"officescene_{base_name}_gaussian_properties.json")
        print(f"\nProcessing {os.path.basename(largest_file)}")
        print(f"Output file: {props_json}")
        try:
            gaussians = load_gaussians_from_checkpoint(largest_file)
            save_gaussian_properties(gaussians, props_json)
            output_ply = os.path.join(output_folder, f"officescene_{base_name}_gaussians.ply")
            gaussians.save_ply(output_ply)
            print(f"Saved Gaussian model as PLY to {output_ply}")
        except Exception as e:
            print(f"Error processing {largest_file}: {e}")
