#!/bin/bash
# Visualize LSeg features for a given .npy file

# LERF
#FEATURE_PATH="data/lerf/teatime/lseg_features/frame_00001.npy"
#OUTPUT_PATH="data/lerf/teatime/lseg_features/frame_00001_classmap.png"

# Visualize the predicted class map and save to file
# python script/visualize_lseg_features.py \
#  --feature_path "$FEATURE_PATH" \
#  --mode classmap \
#  --output "$OUTPUT_PATH"

#echo "Visualization saved to $OUTPUT_PATH"

# ADE20K
FEATURE_PATH="/home/neural_fields/Unified-Lift-Gabor/ade20k/lseg_features/ADE_train_00001860.npy"
OUTPUT_PATH="/home/neural_fields/Unified-Lift-Gabor/ade20k/lseg_features/ADE_train_00001860_classmap.png"
python script/visualize_lseg_features.py \
  --feature_path "$FEATURE_PATH" \
  --mode classmap \
  --output "$OUTPUT_PATH"

echo "Visualization saved to $OUTPUT_PATH"