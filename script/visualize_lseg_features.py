import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from PIL import Image
import matplotlib.patches as mpatches

# ADE20K class labels (shortened for display, use full list if needed)
"""
ADE20K_LABELS = [
    "wood", "chopstick"
    # "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed", "windowpane", "grass"
]
"""

ADE20K_LABELS = [
    "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed", "windowpane", "grass",
    "cabinet", "sidewalk", "person", "earth", "door", "table", "mountain", "plant", "curtain", "chair",
    "car", "water", "painting", "sofa", "shelf", "house", "sea", "mirror", "rug", "field", "armchair",
    "seat", "fence", "desk", "rock", "wardrobe", "lamp", "bathtub", "railing", "cushion", "base",
    "box", "column", "signboard", "chest", "counter", "sand", "sink", "skyscraper", "fireplace", "refrigerator",
    "grandstand", "path", "stairs", "runway", "case", "pool table", "pillow", "screen door", "stairway", "river",
    "bridge", "bookcase", "blind", "coffee table", "toilet", "flower", "book", "hill", "bench", "countertop",
    "stove", "palm", "kitchen island", "computer", "swivel chair", "boat", "bar", "arcade machine", "hovel", "bus",
    "towel", "light", "truck", "tower", "chandelier", "awning", "streetlight", "booth", "television", "airplane",
    "dirt track", "apparel", "pole", "land", "bannister", "escalator", "ottoman", "bottle", "buffet", "poster",
    "stage", "van", "ship", "fountain", "conveyer belt", "canopy", "washer", "plaything", "swimming pool", "stool",
    "barrel", "basket", "waterfall", "tent", "bag", "minibike", "cradle", "oven", "ball", "food", "step",
    "tank", "trade name", "microwave", "pot", "animal", "bicycle", "lake", "dishwasher", "screen", "blanket",
    "sculpture", "hood", "sconce", "vase", "traffic light", "tray", "ashcan", "fan", "pier", "crt screen",
    "plate", "monitor", "bulletin board", "shower", "radiator", "glass", "clock", "flag"
]

def visualize_class_map(feature_path, image_path=None, output_path=None):
    features = np.load(feature_path)  # shape: (num_classes, H, W)
    num_classes = min(features.shape[0], len(ADE20K_LABELS))
    class_map = features.argmax(axis=0)  # shape: (H, W)
    # Calculate softmax probabilities for each pixel
    probs = np.exp(features - np.max(features, axis=0, keepdims=True))
    probs = probs / np.sum(probs, axis=0, keepdims=True)
    # Use a color map with exactly as many colors as classes defined in labels
    cmap = plt.get_cmap('tab20', num_classes)
    plt.figure(figsize=(8, 8))
    if image_path and os.path.exists(image_path):
        img = Image.open(image_path).resize((features.shape[2], features.shape[1]))
        plt.imshow(img, alpha=0.5)
        plt.imshow(class_map, cmap=cmap, alpha=0.5, vmin=0, vmax=num_classes-1)
    else:
        plt.imshow(class_map, cmap=cmap, vmin=0, vmax=num_classes-1)
    plt.title(f"Predicted Class Map\n{os.path.basename(feature_path)}")
    #plt.colorbar(label='Class Index', ticks=range(0, num_classes, max(1, num_classes//10)))
    # Add legend with class names and mean probability (show up to 20 for readability)
    # Compute mean probability for each class (only where predicted)
    class_probs = []
    for i in range(num_classes):
        mask = (class_map == i)
        mean_prob = probs[i][mask].mean() if np.any(mask) else 0.0
        class_probs.append((i, mean_prob))
    # Sort by mean probability and select top 20
    class_probs = sorted(class_probs, key=lambda x: x[1], reverse=True)[:20]
    legend_patches = [
        mpatches.Patch(color=cmap(i), label=f"{i}: {ADE20K_LABELS[i] if i < len(ADE20K_LABELS) else ''} (p={mean_prob:.4f})")
        for i, mean_prob in class_probs if mean_prob > 0
    ]
    plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize='small')
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
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
        #print(f"Saved visualization to {output_path}")
    else:
        plt.show()
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize LSeg feature .npy files")
    parser.add_argument('--feature_path', type=str, required=True, help='Path to .npy feature file')
    parser.add_argument('--image_path', type=str, default=None, help='Path to original image (for overlay)')
    parser.add_argument('--mode', type=str, choices=['classmap', 'logit'], default='classmap', help='Visualization mode')
    parser.add_argument('--class_idx', type=int, default=0, help='Class index for logit visualization')
    parser.add_argument('--output', type=str, default=None, help='Path to save visualization (optional)')
    args = parser.parse_args()

    if args.mode == 'classmap':
        visualize_class_map(args.feature_path, args.image_path, args.output)
    elif args.mode == 'logit':
        visualize_class_logits(args.feature_path, args.class_idx, args.output)

if __name__ == '__main__':
    main()
