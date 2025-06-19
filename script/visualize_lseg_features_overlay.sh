#!/bin/bash
# Visualize LSeg features with overlay and class names

FEATURE_PATH="data/lerf/teatime/lseg_features/frame_00001.npy"
IMAGE_PATH="data/lerf/teatime/images/frame_00001.jpg"
OUTPUT_PATH="data/lerf/teatime/lseg_features/frame_00001_classmap_overlay.png"

python script/visualize_lseg_features.py \
  --feature_path "$FEATURE_PATH" \
  --image_path "$IMAGE_PATH" \
  --mode classmap \
  --output "$OUTPUT_PATH"

echo "Overlay visualization saved to $OUTPUT_PATH"
