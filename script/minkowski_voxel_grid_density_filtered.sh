#!/bin/bash

# Default parameters
MODEL_PATH="output/unifed_lift/officescene"  # Path to the trained model
SOURCE_PATH="data/scannetpp/officescene"  # Path to source data
IMAGES_PATH="images"  # Path to images (relative to SOURCE_PATH)
ITERATION=30000  # Use latest iteration
CELL_SIZE=0.05  # Size of voxel grid cells
OUTPUT_DIR="output/minkowski_grid/officescene_filtered"  # Output directory for voxel grid
DENSITY_EPS=0.05  # Epsilon radius for density filtering
DENSITY_MIN_NEIGHBORS=12  # Minimum neighbors for density filtering
OPACITY_THRESHOLD=0.9999 # Minimum opacity for a gaussian to be considered part of the surface

# Run standalone Minkowski voxel grid generator with density filtering
python script/minkowski_voxel_grid_density_filtered.py \
    --model_path $MODEL_PATH \
    --source_path $SOURCE_PATH \
    --images $IMAGES_PATH \
    --iteration $ITERATION \
    --cell_size $CELL_SIZE \
    --output_dir $OUTPUT_DIR \
    --density_eps $DENSITY_EPS \
    --density_min_neighbors $DENSITY_MIN_NEIGHBORS \
    --opacity_threshold $OPACITY_THRESHOLD
