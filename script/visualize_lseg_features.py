import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# ADE20K class labels (shortened for display, use full list if needed)
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
