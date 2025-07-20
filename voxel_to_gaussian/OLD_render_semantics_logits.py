# Copy of render_semantics.py for logit-based semantic pipeline
# See original for full docstring and CLI

#
"""
Semantic Gaussian Rendering Script
Renamed from render.py to render_semantics.py
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


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh, logit_path):
    # Output folders (match color rendering)
    render_path = os.path.join('/home/neural_fields/Unified-Lift-Gabor/voxel_to_gaussian/semantics', name, 'renders')
    gts_path = os.path.join('/home/neural_fields/Unified-Lift-Gabor/voxel_to_gaussian/semantics', name, 'gt')
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    # --- Load per-Gaussian logits from npz and attach ---
    logit_data = np.load(logit_path)
    # --- Color blending code (commented out, for reference) ---
    # if "colors" in logit_data:
    #     colors = logit_data["colors"].astype(np.float32) / 255.0 * 2  # shape (N,3)
    #     print(f"[DEBUG] Loaded semantic colors: {colors.shape}, dtype={colors.dtype}")
    #     sh_dc = torch.from_numpy(colors).unsqueeze(1).cuda()  # shape (N, 1, 3)
    #     orig_dc = gaussians._features_dc.data
    #     if sh_dc.shape == orig_dc.shape:
    #         gaussians._features_dc.data[:] = sh_dc
    #         print(f"[DEBUG] Overwrote SH DC coefficients with semantic colors: {sh_dc.shape}")
    #     else:
    #         print(f"[ERROR] Shape mismatch: semantic colors {sh_dc.shape} vs original SH DC {orig_dc.shape}")
    #     # Optionally zero higher-order SH (commented out for lighting)
    #     orig_rest = gaussians._features_rest.data
    #     gaussians._features_rest.data.zero_()
    #     print(f"[DEBUG] Zeroed higher-order SH: {orig_rest.shape}")
    #     print(f"[DEBUG] Example assigned colors (first 5): {colors[:5]}")
    # --- End color blending code ---
    if "logits" in logit_data:
        logits = logit_data["logits"]  # shape (N, num_classes)
        gaussians.logits = torch.from_numpy(logits).float().cuda()
        print(f"[DEBUG] Loaded logits: {logits.shape}, dtype={logits.dtype}")
        assert gaussians.logits.shape[0] == gaussians._features_dc.shape[0], "Mismatch in number of Gaussians"
    else:
        print(f"[ERROR] No 'logits' array found in {logit_path}. Cannot attach per-Gaussian logits.")
    N = gaussians._features_dc.shape[0]

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        print(f"[DEBUG] Rendering at resolution: view.image_height={getattr(view, 'image_height', None)}, view.image_width={getattr(view, 'image_width', None)}, pipeline.resolution={getattr(pipeline, 'resolution', None)}")
        print(f"[DEBUG] Number of Gaussians: {N}")
        try:
            buffer_size = int(getattr(view, 'image_height', 0)) * int(getattr(view, 'image_width', 0)) * N
            print(f"[DEBUG] Buffer size (image_height * image_width * N): {buffer_size}")
        except Exception as e:
            print(f"[DEBUG] Buffer size calculation error: {e}")
        rendering = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)["render"]
        # --- Logit blending: rendering is expected to be [num_classes, H, W] ---
        if hasattr(gaussians, "logits") and rendering.dim() == 3:
            # Take argmax over classes for semantic mask
            semantic_mask = torch.argmax(rendering, dim=0).cpu().numpy().astype(np.uint8)  # shape [H, W]
            # Save semantic mask as PNG (raw indices)
            from PIL import Image
            mask_img = Image.fromarray(semantic_mask)
            mask_img.save(os.path.join(render_path, '{0:05d}_mask.png'.format(idx)))
            print(f"[DEBUG] Saved semantic mask: {os.path.join(render_path, '{0:05d}_mask.png'.format(idx))}")

            # --- Palette-based color mapping ---
            def get_palette(num_cls):
                n = num_cls
                palette = [0]*(n*3)
                for j in range(n):
                    lab = j
                    palette[j*3+0] = 0
                    palette[j*3+1] = 0
                    palette[j*3+2] = 0
                    i = 0
                    while lab > 0:
                        palette[j*3+0] |= (((lab >> 0) & 1) << (7-i))
                        palette[j*3+1] |= (((lab >> 1) & 1) << (7-i))
                        palette[j*3+2] |= (((lab >> 2) & 1) << (7-i))
                        i += 1
                        lab >>= 3
                return palette

            # Always use all classes from logits/prompts for legend and palette
            if "logits" in logit_data:
                num_classes = logit_data["logits"].shape[1]
            else:
                num_classes = 1
            palette = get_palette(num_classes)
            mask_img_color = Image.fromarray(semantic_mask)
            mask_img_color.putpalette(palette)
            mask_img_color.save(os.path.join(render_path, '{0:05d}_mask_color.png'.format(idx)))
            print(f"[DEBUG] Saved colored semantic mask: {os.path.join(render_path, '{0:05d}_mask_color.png'.format(idx))}")

            # --- Legend PNG generation ---
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            from PIL import Image as PILImage
            import io

            # Load label names from npz if present (as in analyze_gaussian_semantics.py)
            label_names = None
            if "prompts" in logit_data:
                label_names = [str(x) for x in logit_data["prompts"]]
            if label_names is None or len(label_names) != num_classes:
                label_names = [f"Label {i}" for i in range(num_classes)]
            # Count pixels for all classes (even if zero)
            counts = np.bincount(semantic_mask.flatten(), minlength=num_classes)
            fig, ax = plt.subplots(figsize=(5, 1 + 0.5*num_classes))
            patches = []
            for i in range(num_classes):
                color = tuple([v/255.0 for v in palette[3*i:3*i+3]])
                label = f"{label_names[i]} (Label {i}, count={counts[i]})"
                patch = mpatches.Patch(color=color, label=label)
                patches.append(patch)
            ax.legend(handles=patches, loc='center left', frameon=True)
            ax.axis('off')
            plt.tight_layout()
            # Save legend to buffer
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
            plt.close(fig)
            buf.seek(0)
            legend_img = PILImage.open(buf).convert('RGBA')
            buf.close()
            print(f"[DEBUG] Created legend image for idx={idx}")

            # --- Combine colored mask and legend side by side ---
            mask_img_rgb = mask_img_color.convert('RGB')
            # Resize legend to match mask height
            legend_img_resized = legend_img.resize((legend_img.width, mask_img_rgb.height), PILImage.LANCZOS)
            combined_width = mask_img_rgb.width + legend_img_resized.width
            combined_img = PILImage.new('RGB', (combined_width, mask_img_rgb.height), (255,255,255))
            combined_img.paste(mask_img_rgb, (0,0))
            combined_img.paste(legend_img_resized, (mask_img_rgb.width,0))
            combined_path = os.path.join(render_path, '{0:05d}_mask_with_legend.png'.format(idx))
            combined_img.save(combined_path)
            print(f"[DEBUG] Saved combined mask+legend PNG: {combined_path}")

            # Fallback: save color rendering (if not logit blending)
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}.png'.format(idx)))
            print(f"[DEBUG] Saved color rendering: {os.path.join(render_path, '{0:05d}.png'.format(idx))}")
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}.png'.format(idx)))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, separate_sh: bool, logit_path: str):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh, logit_path)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh, logit_path)

if __name__ == "__main__":
    parser = ArgumentParser(description="Semantic Gaussian Rendering script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--logit_path", type=str, required=False, default="/home/neural_fields/Unified-Lift-Gabor/voxel_to_gaussian/gaussian_labels.npz", help="Path to .npz file containing per-Gaussian logits.")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE, args.logit_path)