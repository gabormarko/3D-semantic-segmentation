#!/bin/bash
# Semantic Gaussian Rendering Pipeline
# Usage: bash render_semantics.sh

# --------- User Parameters ---------
MODEL_PATH="/home/neural_fields/gaussian-splatting/output/cc946f8b-f"   # Path to model directory
#MODEL_PATH="/home/neural_fields/gaussian-splatting/output/237ab5b3-5"
#MODEL_PATH="/home/neural_fields/Unified-Lift-Gabor/betterGaussians/filtered_gaussians.ply"

ITERATION=-1                           # Checkpoint iteration (-1 for latest)
GAUSSIAN_LABELS="/home/neural_fields/Unified-Lift-Gabor/voxel_to_gaussian/gaussian_semantics_filt_gauss_45000.npz"  # Path to semantic logits npz
SKIP_TRAIN=false                       # Set to true to skip train rendering
SKIP_TEST=false                        # Set to true to skip test rendering
QUIET=false                            # Set to true for quiet mode
FIRST_ONLY=false                        # Set to true to render only the first image

# --------- Command Construction ---------
CMD="python /home/neural_fields/Unified-Lift-Gabor/voxel_to_gaussian/render_semantics_logits.py \
  --model_path $MODEL_PATH \
  --iteration $ITERATION \
  --logit_path $GAUSSIAN_LABELS"

if [ "$QUIET" = true ]; then
  CMD+=" --quiet"
fi
if [ "$SKIP_TRAIN" = true ]; then
  CMD+=" --skip_train"
fi
if [ "$SKIP_TEST" = true ]; then
  CMD+=" --skip_test"
fi
if [ "$FIRST_ONLY" = true ]; then
  CMD+=" --first_only"
fi

# --------- Run ---------
echo "Running: $CMD"
eval $CMD
