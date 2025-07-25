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
#sys.path.append('/home/neural_fields/gaussian-splatting/submodules/diff-gaussian-rasterization')
sys.path.append('/home/neural_fields/Unified-Lift-Gabor/submodules/diff-gaussian-rasterization')
sys.path.append('/home/neural_fields/gaussian-splatting/utils')
sys.path.append('/home/neural_fields/gaussian-splatting/gaussian_renderer')
import torch
import numpy as np
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gsplat import rasterization
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


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh, logit_path, first_only=False):
    # Output folders (match color rendering)
    base_path = '/home/neural_fields/Unified-Lift-Gabor/voxel_to_gaussian/semantics_87319_30000_NEW'
    render_path = os.path.join(base_path, name, 'renders')
    gts_path = os.path.join(base_path, name, 'gt')
    labels_path = os.path.join(base_path, name, 'labels')
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(labels_path, exist_ok=True)

    # --- Load per-Gaussian logits from npz and attach ---
    logit_data = np.load(logit_path)
    NUM_CHANNELS = 32  # Keep this in sync with config.h
    if "logits" in logit_data:
        logits = logit_data["logits"]  # shape (N, num_classes)
        # Pad or slice logits to NUM_CHANNELS
        if logits.shape[1] < NUM_CHANNELS:
            pad_width = NUM_CHANNELS - logits.shape[1]
            logits = np.pad(logits, ((0,0),(0,pad_width)), mode='constant')
        elif logits.shape[1] > NUM_CHANNELS:
            logits = logits[:, :NUM_CHANNELS]
        gaussians.logits = torch.from_numpy(logits).float().contiguous().cuda()
        print(f"[DEBUG] Loaded logits: {logits.shape}, dtype={logits.dtype}")
        assert gaussians.logits.shape[0] == gaussians._features_dc.shape[0], "Mismatch in number of Gaussians"
        print(f"[DEBUG] gaussians.logits shape: {gaussians.logits.shape}")
        if NUM_CHANNELS == 3:
            print("[WARNING] NUM_CHANNELS==3: logits used as colors_precomp, but SHs are expected for RGB rendering.")
    else:
        print(f"[ERROR] No 'logits' array found in {logit_path}. Cannot attach per-Gaussian logits.")
    N = gaussians._features_dc.shape[0]

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if first_only and idx > 0:
            print("[INFO] --first_only set, stopping after first image.")
            break
        print(f"[DEBUG] Rendering at resolution: view.image_height={getattr(view, 'image_height', None)}, view.image_width={getattr(view, 'image_width', None)}, pipeline.resolution={getattr(pipeline, 'resolution', None)}")
        print(f"[DEBUG] Number of Gaussians: {N}")
        # Prepare logit features for rasterization
        features = gaussians.logits[:, :NUM_CHANNELS].contiguous()
        # Prepare camera intrinsics
        import math
        tanfovx = math.tan(view.FoVx * 0.5)
        tanfovy = math.tan(view.FoVy * 0.5)
        focal_length_x = view.image_width / (2 * tanfovx)
        focal_length_y = view.image_height / (2 * tanfovy)
        K = torch.tensor(
            [
                [focal_length_x, 0, view.image_width / 2.0],
                [0, focal_length_y, view.image_height / 2.0],
                [0, 0, 1],
            ],
            device="cuda",
        )
        viewmat = view.world_view_transform.transpose(0, 1) # [4, 4]
        # Call gsplat rasterization directly
        bg_expanded = background.expand(int(view.image_height), int(view.image_width), background.shape[-1])
        render_colors, render_alphas, info = rasterization(
            means=gaussians.get_xyz,  # [N, 3]
            quats=gaussians.get_rotation,  # [N, 4]
            scales=gaussians.get_scaling,  # [N, 3]
            opacities=gaussians.get_opacity.squeeze(-1),  # [N,]
            colors=features,  # [N, NUM_CHANNELS]
            viewmats=viewmat[None],  # [1, 4, 4]
            Ks=K[None],  # [1, 3, 3]
            width=int(view.image_width),
            height=int(view.image_height),
            packed=False,
            sh_degree=None,  # Use features, not SH
            render_mode="RGB",  # For logits/features)
        )
        # [1, H, W, NUM_CHANNELS] -> [NUM_CHANNELS, H, W]
        rendering = render_colors[0].permute(2, 0, 1)
        # --- Semantic mask visualization and statistics ---
        if hasattr(gaussians, "logits") and rendering.dim() == 3:
            # Debug: print logit statistics and pixel logit vectors
            print(f"[DEBUG] Rendering logits tensor shape: {rendering.shape}, dtype={rendering.dtype}")
            print(f"[DEBUG] Logit min: {rendering.min().item()}, max: {rendering.max().item()}, mean: {rendering.mean().item()}")
            # Print logit vector for a few pixels
            h, w = rendering.shape[1], rendering.shape[2]
            for px, py in [(0,0), (h//2, w//2), (h-1, w-1)]:
                if px < h and py < w:
                    logit_vec = rendering[:, px, py].detach().cpu().numpy()
                    print(f"[DEBUG] Logit vector at pixel ({px},{py}): {logit_vec}")
            # Take argmax over classes for semantic mask
            semantic_mask = torch.argmax(rendering, dim=0)  # shape [H, W], torch tensor
            semantic_mask_np = semantic_mask.cpu().numpy().astype(np.uint8)
            from PIL import Image
            #mask_img = Image.fromarray(semantic_mask_np)
            #mask_img.save(os.path.join(render_path, '{0:05d}_mask.png'.format(idx)))
            #print(f"[DEBUG] Saved semantic mask: {os.path.join(render_path, '{0:05d}_mask.png'.format(idx))}")

            # --- Save per-pixel label index, name, and value ---
            H, W = semantic_mask.shape
            # label_indices: [H, W] (torch tensor)
            label_indices = semantic_mask.cpu()
            # label_values: [H, W] (torch tensor)
            label_values = rendering.detach().cpu()[label_indices, torch.arange(H)[:,None], torch.arange(W)]
            # label_names: [H, W] (numpy array of strings)
            if 'prompts' in logit_data:
                label_names_list = [str(x) for x in logit_data['prompts']]
            else:
                label_names_list = [f"Label {i}" for i in range(rendering.shape[0])]
            label_names = np.array([label_names_list[idx] for idx in label_indices.numpy().flatten()]).reshape(H, W)
            # Save all as a dict in a .pt file
            # Only save per-pixel label index
            torch.save({
                'label_indices': label_indices  # [H, W] torch.uint8
            }, os.path.join(labels_path, '{0:05d}_labels.pt'.format(idx)))
            print(f"[DEBUG] Saved per-pixel label indices: {os.path.join(labels_path, '{0:05d}_labels.pt'.format(idx))}")

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
            mask_img_color = Image.fromarray(semantic_mask.cpu().numpy().astype(np.uint8))
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
            counts = np.bincount(semantic_mask.cpu().numpy().flatten(), minlength=num_classes)
            # Increase legend height to fit up to 32 labels
            # Increase legend height to fit up to 32 labels, with more vertical space per label
            max_labels = max(num_classes, 32)
            # Use default figure size for natural legend alignment
            fig, ax = plt.subplots()
            patches = []
            for i in range(num_classes):
                color = tuple([v/255.0 for v in palette[3*i:3*i+3]])
                label = f"{label_names[i]} (Label {i}, count={counts[i]})"
                patch = mpatches.Patch(color=color, label=label)
                patches.append(patch)
            # Use vertical legend layout, one label per row
            ax.legend(handles=patches, frameon=True)
            ax.axis('off')
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

            # Fallback: save raw logits as .npy for debugging
            logits_npy_path = os.path.join(render_path, '{0:05d}_logits.npy'.format(idx))
            np.save(logits_npy_path, rendering.detach().cpu().numpy())
            print(f"[DEBUG] Saved raw logits: {logits_npy_path}")

            # --- Call logit_confidence_map.py to generate uncertainty map PNG ---
            import subprocess
            confidence_map_script = os.path.join(os.path.dirname(__file__), 'logit_confidence_map.py')
            confidence_map_png = logits_npy_path.replace('_logits.npy', '_confidence.png')
            try:
                subprocess.run([
                    'python', confidence_map_script,
                    '--logits', logits_npy_path,
                    '--out', confidence_map_png
                ], check=True)
                print(f"[DEBUG] Saved uncertainty/confidence map: {confidence_map_png}")
            except Exception as e:
                print(f"[WARNING] Could not generate confidence map for {logits_npy_path}: {e}")
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}.png'.format(idx)))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, separate_sh: bool, logit_path: str, first_only: bool = False):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

    NUM_CHANNELS = 32  # Keep this in sync with config.h and logits
    bg_color = [1.0] * NUM_CHANNELS if dataset.white_background else [0.0] * NUM_CHANNELS
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    if not skip_train:
        render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh, logit_path, first_only=first_only)

    if not skip_test:
        render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh, logit_path, first_only=first_only)

if __name__ == "__main__":
    parser = ArgumentParser(description="Semantic Gaussian Rendering script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--logit_path", type=str, required=False, help="Path to .npz file containing per-Gaussian logits.")
    parser.add_argument("--first_only", action="store_true", help="Render only the first image per run.")
    # parser.add_argument("--object_path", type=str, required=True, help="Path to the point cloud .ply file")
    args = get_combined_args(parser)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE, args.logit_path, first_only=args.first_only)
