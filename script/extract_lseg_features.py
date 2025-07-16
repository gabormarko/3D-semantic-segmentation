# Script to extract LSeg features from all images in a directory
# Usage: python extract_lseg_features.py --input_dir /path/to/images --output_dir /path/to/features

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'lang-seg')))

import argparse
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

# print("LSeg feature extraction script loaded.")

# Import LSeg model from official repo (assumes installed via pip or local clone)
try:
    from modules.models.lseg_net import LSegNet
except ImportError:
    raise ImportError('Please install the official LSeg repo and make sure it is in your PYTHONPATH.')

# Full ADE20K 150-class label set, as used by default in LSeg - can be modified if needed
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
"""

# ScanNet++ labels
TOP100_PATH = "/home/neural_fields/Unified-Lift-Gabor/data/scannetpp/top100.txt"
with open(TOP100_PATH, "r") as f:
    LABELS = [line.strip() for line in f if line.strip()]
    
def load_lseg_model(device):
    # Load LSeg model (modify as needed for your setup)
    model = LSegNet(
        labels=LABELS,
        backbone='clip_vitl16_384',
        features=512,
        aux=False,
        arch_option=0,
        activation='relu',
        block_depth=2
    )
    # Download or load pretrained weights here if needed
    model.eval()
    model.to(device)
    return model

def extract_features(model, img_path, device):
    img = Image.open(img_path).convert('RGB')
    # Resize shorter side to 480, keep aspect ratio
    transform = transforms.Compose([
        transforms.Resize(360),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(img)
    _, h, w = tensor.shape
    # Pad to multiples of 32
    pad_h = (32 - h % 32) % 32
    pad_w = (32 - w % 32) % 32
    if pad_h > 0 or pad_w > 0:
        tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode='constant', value=0)
    input_tensor = tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)  # [1, C, H, W]
    # Remove padding from output if any
    if pad_h > 0 or pad_w > 0:
        output = output[:, :, :h, :w]
    return output.cpu().numpy().squeeze(0)  # [C, H, W]

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_lseg_model(device)
    img_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    img_files = sorted(img_files)#[:10]  # Only process the first 10 images
    for img_file in img_files:
        img_path = os.path.join(args.input_dir, img_file)
        features = extract_features(model, img_path, device)
        out_path = os.path.join(args.output_dir, img_file.replace('.jpg', '.npy').replace('.jpeg', '.npy').replace('.png', '.npy'))
        np.save(out_path, features.astype(np.float16))
        print(f'Saved features for {img_file} to {out_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract LSeg features from images')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory with input images')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save features')
    args = parser.parse_args()
    main(args)
