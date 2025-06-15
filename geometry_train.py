# geometry_train.py
# This version of the training script focuses only on geometric reconstruction using RGB images and COLMAP poses.

import os
import torch
import numpy as np
from random import randint
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from utils.loss_utils import l1_loss, ssim
from utils.general_utils import safe_state
from PIL import Image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import json
import uuid
from tqdm import tqdm

def prepare_output_and_logger(args):
    if not args.model_path:
        unique_str = os.getenv('OAR_JOB_ID', str(uuid.uuid4()))
        args.model_path = os.path.join("./output/", unique_str[0:10])

    print("Output folder:", args.model_path)
    os.makedirs(args.model_path, exist_ok=True)

    args.ckpt_dir = os.path.join(args.model_path, "chkpnts")
    os.makedirs(args.ckpt_dir, exist_ok=True)

    with open(os.path.join(args.model_path, "cfg_args"), 'w') as f:
        f.write(str(Namespace(**vars(args))))

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, use_wandb):
    first_iter = 0
    prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    if checkpoint:
        model_params, first_iter = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    progress_bar = tqdm(range(first_iter + 1, opt.iterations + 1), desc="Training progress")
    ema_loss = 0.0

    for iteration in range(first_iter + 1, opt.iterations + 1):
        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image = render_pkg["render"]

        gt_image = viewpoint_cam.original_image.to("cuda")
        loss = (1.0 - opt.lambda_dssim) * l1_loss(image, gt_image) + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        loss.backward()
        ema_loss = 0.4 * loss.item() + 0.6 * ema_loss

        if iteration % 10 == 0:
            progress_bar.set_postfix({"Loss": f"{ema_loss:.6f}"})
            progress_bar.update(10)

        if iteration in saving_iterations:
            print(f"\n[ITER {iteration}] Saving Gaussians")
            scene.save(iteration)

        if iteration in checkpoint_iterations:
            print(f"\n[ITER {iteration}] Saving Checkpoint")
            torch.save((gaussians.capture(), iteration), os.path.join(dataset.ckpt_dir, f"chkpnt{iteration}.pth"))

        if iteration == opt.iterations:
            print(f"\n[ITER {iteration}] Saving Final Checkpoint")
            torch.save((gaussians.capture(), iteration), os.path.join(dataset.ckpt_dir, f"chkpnt{iteration}.pth"))

        if iteration < opt.iterations:
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

    progress_bar.close()
    print("\nTraining complete.")

if __name__ == "__main__":
    parser = ArgumentParser(description="Unified-Lift Geometry-Only Training")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1000, 7000, 30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1000, 7000, 20000, 30000])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[7000])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--config_file", type=str, required=True, help="Path to config JSON")
    parser.add_argument("--use_wandb", action='store_true')
    parser.add_argument("--quiet", action="store_true", help="Suppress output if set")
    parser.add_argument("--mode", type=str, choices=["geometry", "unified_lift"], default="geometry", help="Which training mode to use")


    args = parser.parse_args()

    # Load JSON config
    with open(args.config_file, 'r') as f:
        config = json.load(f)
        for k, v in config.items():
            setattr(args, k, v)

    safe_state(args.quiet)
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.use_wandb)
