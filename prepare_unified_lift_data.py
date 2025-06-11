# prepare_unifiedlift_data.py
# This script restructures a raw ScanNet++ scene into the Unified-Lift folder format for geometry-only training.

import os
import shutil
import json
from pathlib import Path

def restructure_scannetpp_scene(scannet_scene_dir, output_scene_dir):
    """
    Reorganize ScanNet++ scene into Unified-Lift compatible geometry-only format.

    Args:
        scannet_scene_dir (str): Path to ScanNet++ scene (e.g. scannetpp_data/scene0207_01)
        output_scene_dir (str): Path where the converted scene should be stored (e.g. scenes/scene0207_01)
    """
    scannet_scene_dir = Path(scannet_scene_dir)
    output_scene_dir = Path(output_scene_dir)

    # Step 1: Create Unified-Lift folder structure
    (output_scene_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_scene_dir / "images_train").mkdir(parents=True, exist_ok=True)
    sparse_dir = output_scene_dir / "distorted" / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    # Step 2: Copy all RGB images into images/
    rgb_src_dir = scannet_scene_dir / "dslr" / "resized_undistorted_images"
    for img_path in rgb_src_dir.glob("*.jpg"):
        shutil.copy(img_path, output_scene_dir / "images" / img_path.name)

    # Step 3: Use train_test_lists.json to populate images_train/
    split_path = scannet_scene_dir / "dslr" / "train_test_lists.json"
    if split_path.exists():
        with open(split_path, 'r') as f:
            split = json.load(f)
            for img_name in split.get("train", []):
                src = output_scene_dir / "images" / img_name
                dst = output_scene_dir / "images_train" / img_name
                if src.exists():
                    os.symlink(src, dst)
                else:
                    print(f"Warning: training image {img_name} not found")
    else:
        print("Warning: train_test_lists.json not found — skipping image_train population")

    # Step 4: Copy COLMAP pose data into sparse folder
    colmap_dir = scannet_scene_dir / "dslr" / "colmap"
    for fname in ["cameras.txt", "images.txt", "points3D.txt"]:
        src_file = colmap_dir / fname
        if src_file.exists():
            shutil.copy(src_file, sparse_dir / fname)
        else:
            print(f"Warning: {fname} not found in colmap folder")

    print(f"✅ Scene {scannet_scene_dir.name} prepared at {output_scene_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to ScanNet++ scene folder")
    parser.add_argument("--output", required=True, help="Path to output Unified-Lift format folder")
    args = parser.parse_args()

    restructure_scannetpp_scene(args.input, args.output)
