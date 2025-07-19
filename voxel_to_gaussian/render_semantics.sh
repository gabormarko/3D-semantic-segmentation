#!/bin/bash
# Semantic Gaussian Rendering Pipeline
# Usage: bash render_semantics.sh

# --------- User Parameters ---------
MODEL_PATH="/home/neural_fields/gaussian-splatting/output/cc946f8b-f"   # Path to model directory
ITERATION=-1                           # Checkpoint iteration (-1 for latest)
GAUSSIAN_LABELS="/home/neural_fields/Unified-Lift-Gabor/voxel_to_gaussian/gaussian_semantics.npz"  # Path to semantic color npz
SKIP_TRAIN=false                       # Set to true to skip train rendering
SKIP_TEST=false                        # Set to true to skip test rendering
QUIET=false                            # Set to true for quiet mode

# --------- Command Construction ---------
CMD="python /home/neural_fields/Unified-Lift-Gabor/voxel_to_gaussian/render_semantics.py \
  --model_path $MODEL_PATH \
  --iteration $ITERATION \
  --color_path $GAUSSIAN_LABELS"

if [ "$QUIET" = true ]; then
  CMD+=" --quiet"
fi
if [ "$SKIP_TRAIN" = true ]; then
  CMD+=" --skip_train"
fi
if [ "$SKIP_TEST" = true ]; then
  CMD+=" --skip_test"
fi

# --------- Run ---------
echo "Running: $CMD"
eval $CMD
