#
"""
Gaussian Color Rendering Script
Renders Gaussians with colors assigned from nearest voxel.
"""

# Add gaussian-splatting repo to Python path
import sys
sys.path.append('/home/neural_fields/gaussian-splatting')
sys.path.append('/home/neural_fields/gaussian-splatting/submodules/diff-gaussian-rasterization')
sys.path.append('/home/neural_fields/gaussian-splatting/utils')
sys.path.append('/home/neural_fields/gaussian-splatting/gaussian_renderer')
import torch
import numpy as np
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh, color_path):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    # --- Assign colors from npz ---
    color_data = np.load(color_path)
    colors = color_data["colors"].astype(np.float32) / 255.0  # shape (N,3)
    print(f"[DEBUG] Loaded assigned colors: {colors.shape}, dtype={colors.dtype}")
    # Overwrite only SH DC coefficients (first 3 channels) with custom colors
    # Keep higher-order SH coefficients from the trained model
    sh_dc = torch.from_numpy(colors).unsqueeze(1).cuda()  # shape (N, 1, 3)
    orig_dc = gaussians._features_dc.data
    if sh_dc.shape == orig_dc.shape:
        gaussians._features_dc.data[:] = sh_dc
        print(f"[DEBUG] Overwrote SH DC coefficients with custom colors: {sh_dc.shape}")
    else:
        print(f"[ERROR] Shape mismatch: custom colors {sh_dc.shape} vs original SH DC {orig_dc.shape}")
    # Zero out higher-order SH coefficients (keep shape)
    # orig_rest = gaussians._features_rest.data
    # gaussians._features_rest.data.zero_()
    # print(f"[DEBUG] Zeroed higher-order SH: {orig_rest.shape}")
    print(f"[DEBUG] Example assigned colors (first 5): {colors[:5]}")
    N = gaussians._features_dc.shape[0]

    # Change output folder for images
    render_path = os.path.join('/home/neural_fields/Unified-Lift-Gabor/voxel_to_gaussian/color', name, 'renders')
    gts_path = os.path.join('/home/neural_fields/Unified-Lift-Gabor/voxel_to_gaussian/color', name, 'gt')
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    # Use gaussians._features_dc and _features_rest for SH features
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # Use original camera intrinsics and image size (no reduction)
        # No changes to view.image_height, view.image_width, or view.K
        # No changes to pipeline.resolution, pipeline.image_height, pipeline.image_width
        print(f"[DEBUG] Rendering at resolution: view.image_height={getattr(view, 'image_height', None)}, view.image_width={getattr(view, 'image_width', None)}, pipeline.resolution={getattr(pipeline, 'resolution', None)}")
        print(f"[DEBUG] Number of Gaussians: {N}")
        try:
            buffer_size = int(getattr(view, 'image_height', 0)) * int(getattr(view, 'image_width', 0)) * N
            print(f"[DEBUG] Buffer size (image_height * image_width * N): {buffer_size}")
        except Exception as e:
            print(f"[DEBUG] Buffer size calculation error: {e}")
        rendering = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)["render"]
        gt = view.original_image[0:3, :, :]

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, separate_sh: bool, color_path: str):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh, color_path)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh, color_path)

if __name__ == "__main__":
    parser = ArgumentParser(description="Gaussian Color Rendering script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--color_path", type=str, required=False, default="/home/neural_fields/Unified-Lift-Gabor/voxel_to_gaussian/color/gaussian_colors.npz")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE, args.color_path)
