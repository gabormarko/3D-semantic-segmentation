import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# ADE20K class labels (shortened for display, use full list if needed)
ADE20K_LABELS = [
    "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed", "windowpane", "grass",
    # ... (add all 150 class names if you want to display them)
]

def visualize_class_map(feature_path, output_path=None):
    features = np.load(feature_path)  # shape: (150, 480, 480)
    class_map = features.argmax(axis=0)  # shape: (480, 480)
    plt.figure(figsize=(8, 8))
    plt.imshow(class_map, cmap='tab20', vmin=0, vmax=149)
    plt.title(f"Predicted Class Map\n{os.path.basename(feature_path)}")
    plt.colorbar(label='Class Index')
    if output_path:
        plt.savefig(output_path)
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()
    plt.close()

def visualize_class_logits(feature_path, class_idx, output_path=None):
    features = np.load(feature_path)
    plt.figure(figsize=(8, 8))
    plt.imshow(features[class_idx], cmap='viridis')
    plt.title(f"Class {class_idx} Logits\n{ADE20K_LABELS[class_idx] if class_idx < len(ADE20K_LABELS) else ''}")
    plt.colorbar()
    if output_path:
        plt.savefig(output_path)
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize LSeg feature .npy files")
    parser.add_argument('--feature_path', type=str, required=True, help='Path to .npy feature file')
    parser.add_argument('--mode', type=str, choices=['classmap', 'logit'], default='classmap', help='Visualization mode')
    parser.add_argument('--class_idx', type=int, default=0, help='Class index for logit visualization')
    parser.add_argument('--output', type=str, default=None, help='Path to save visualization (optional)')
    args = parser.parse_args()

    if args.mode == 'classmap':
        visualize_class_map(args.feature_path, args.output)
    elif args.mode == 'logit':
        visualize_class_logits(args.feature_path, args.class_idx, args.output)

if __name__ == '__main__':
    main()
