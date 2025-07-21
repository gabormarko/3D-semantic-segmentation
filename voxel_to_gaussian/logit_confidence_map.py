import numpy as np
import argparse
import matplotlib.pyplot as plt
import os

def save_confidence_map(logits_path, out_path=None, vmin=None, vmax=None, cmap="viridis"):
    logits = np.load(logits_path)  # shape: [num_classes, H, W]
    # Softmax normalization over class axis (axis=0)
    logits_max = np.max(logits, axis=0, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    probs = exp_logits / np.sum(exp_logits, axis=0, keepdims=True)  # shape: [num_classes, H, W]
    # Compute confidence as top1 - top2 probability per pixel
    sorted_probs = np.sort(probs, axis=0)
    confidence = sorted_probs[-1] - sorted_probs[-2]  # shape: [H, W]
    if out_path is None:
        out_path = os.path.splitext(logits_path)[0] + "_confidence.png"
    plt.figure(figsize=(8, 6))
    im = plt.imshow(confidence, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, label="Confidence (top1 - top2 logit)")
    plt.title("Per-pixel Semantic Confidence Map")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Saved confidence map with colorbar to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save per-pixel confidence map (top1 - top2 logit) as PNG with colorbar.")
    parser.add_argument('--logits', type=str, required=True, help='Path to composited logits .npy file')
    parser.add_argument('--out', type=str, default=None, help='Output PNG path (default: <logits>_confidence.png)')
    parser.add_argument('--vmin', type=float, default=None, help='Min value for colorbar (default: auto)')
    parser.add_argument('--vmax', type=float, default=None, help='Max value for colorbar (default: auto)')
    parser.add_argument('--cmap', type=str, default='viridis', help='Colormap (default: viridis)')
    args = parser.parse_args()
    save_confidence_map(args.logits, args.out, args.vmin, args.vmax, args.cmap)
