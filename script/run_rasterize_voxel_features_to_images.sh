#!/bin/bash
# Script to rasterize 3D voxel features back to 2D images using COLMAP cameras
# Usage: bash run_rasterize_voxel_features_to_images.sh

set -e

# Activate your environment if needed (uncomment and edit below)
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate lang-seg

# Run the rasterization script
python3 script/rasterize_voxel_features_to_images.py
