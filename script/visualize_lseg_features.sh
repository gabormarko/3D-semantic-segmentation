#!/bin/bash
# Visualize LSeg features for a given .npy file

FEATURE_PATH="data/lerf/figurines/lseg_features/frame_00001.npy"
OUTPUT_PATH="data/lerf/figurines/lseg_features/frame_00001_classmap.png"

# Visualize the predicted class map and save to file
python script/visualize_lseg_features.py \
  --feature_path "$FEATURE_PATH" \
  --mode classmap \
  --output "$OUTPUT_PATH"

echo "Visualization saved to $OUTPUT_PATH"
