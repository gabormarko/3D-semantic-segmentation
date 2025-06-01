#!/bin/bash

# Model parameters
MODEL_PATH="output/unifed_lift/teatime"
SOURCE_PATH="data/lerf/teatime"
IMAGES_PATH=images
ITERATION=30000

# Surface detection parameters
OPACITY_THRESH=0.8
SCALE_THRESH=0.03
DENSITY_THRESH=0.3
K_NEIGHBORS=16

# Output parameters
OUTPUT_DIR="output/surface/teatime"
SAVE_PLY=true
EVAL=false
N_VIEWS=100
TRAIN_SPLIT=0.0

# Run surface detection
python script/detect_surface.py \
    --model_path $MODEL_PATH \
    --source_path $SOURCE_PATH \
    --images $IMAGES_PATH \
    --iteration $ITERATION \
    --opacity_threshold $OPACITY_THRESH \
    --scale_threshold $SCALE_THRESH \
    --density_threshold $DENSITY_THRESH \
    --k_neighbors $K_NEIGHBORS \
    --output_dir $OUTPUT_DIR \
    $([ "$SAVE_PLY" = true ] && echo "--save_ply") \
    $([ "$EVAL" = true ] && echo "--eval") \
    --n_views $N_VIEWS \
    --train_split $TRAIN_SPLIT 