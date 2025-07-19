#
"""
Semantic Gaussian Rendering Script
Renamed from render.py to render_semantics.py
"""

# Add gaussian-splatting repo to Python path
import sys
sys.path.append('/home/neural_fields/gaussian-splatting')
sys.path.append('/home/neural_fields/gaussian-splatting/submodules/diff-gaussian-rasterization')
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


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)["render"]
        gt = view.original_image[0:3, :, :]

        if args.train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, separate_sh: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        # --- Semantic color assignment ---
        color_path = "/home/neural_fields/Unified-Lift-Gabor/voxel_to_gaussian/gaussian_labels.npz"
        color_data = np.load(color_path)
        colors = color_data["colors"].astype(np.float32) / 255.0  # shape (N,3)
        print(f"[DEBUG] Loaded semantic colors: {colors.shape}, dtype={colors.dtype}")

        # Check expected color format: renderer expects SH DC coefficients in gaussians._features_dc
        # Usually, RGB2SH converts (N,3) RGB in [0,1] to (N,3) SH DC
        # If unsure, check the shape and range of gaussians._features_dc after loading a normal scene
        try:
            from gaussian_renderer.utils import RGB2SH  # adjust import if needed
        except ImportError:
            def RGB2SH(rgb):
                # Fallback: identity mapping
                return rgb

        sh_dc = RGB2SH(torch.from_numpy(colors).cuda())  # shape (N,3)
        sh_dc = sh_dc.unsqueeze(-1)  # shape (N,3,1)
        gaussians._features_dc = torch.nn.Parameter(sh_dc.transpose(1,2).contiguous())
        print(f"[DEBUG] Assigned SH DC coefficients: {gaussians._features_dc.shape}, min={gaussians._features_dc.min().item()}, max={gaussians._features_dc.max().item()}")

        # Zero out higher-order SH coefficients
        N = gaussians._features_dc.shape[0]
        C = gaussians._features_dc.shape[1]
        H = (dataset.sh_degree + 1) ** 2 - 1
        gaussians._features_rest = torch.nn.Parameter(torch.zeros((N, C, H), device='cuda'))
        print(f"[DEBUG] Zeroed higher-order SH: {gaussians._features_rest.shape}")

        # Optional: print a few example colors
        print(f"[DEBUG] Example assigned colors (first 5): {colors[:5]}")

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE)