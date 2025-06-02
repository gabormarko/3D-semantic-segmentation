#!/bin/bash

# Default parameters
MODEL_PATH="output/unifed_lift/ramen"  # Path to the trained model
SOURCE_PATH="data/lerf/ramen"  # Path to source data
IMAGES_PATH="images"  # Path to images (relative to SOURCE_PATH)
ITERATION=-1  # Use latest iteration
CELL_SIZE=0.05  # Size of voxel grid cells
OUTPUT_DIR="output/minkowski_grid"

# Run standalone Minkowski voxel grid generator
python script/minkowski_voxel_grid.py \
    --model_path $MODEL_PATH \
    --source_path $SOURCE_PATH \
    --images $IMAGES_PATH \
    --iteration $ITERATION \
    --cell_size $CELL_SIZE \
    --output_dir $OUTPUT_DIR
