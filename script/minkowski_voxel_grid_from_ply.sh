#!/bin/bash

# Default parameters
PLY_PATH="/home/neural_fields/gaussian-splatting/output/cc946f8b-f/point_cloud/iteration_45000/point_cloud.ply"  # Path to input .ply file
CELL_SIZE=0.05  # Size of voxel grid cells
OUTPUT_DIR="output/minkowski_grid/officescene_filtered_ply"  # Output directory for voxel grid
DENSITY_EPS=0.05  # Epsilon radius for density filtering
DENSITY_MIN_NEIGHBORS=6  # Minimum neighbors for density filtering
OPACITY_THRESHOLD=0.8 # Fraction of points to leave out (e.g. 0.8 keeps top 20% by opacity)

# Run Minkowski voxel grid generator from .ply file with density filtering
python script/minkowski_voxel_grid_from_ply.py \
    --ply $PLY_PATH \
    --cell_size $CELL_SIZE \
    --output_dir $OUTPUT_DIR \
    --density_eps $DENSITY_EPS \
    --density_min_neighbors $DENSITY_MIN_NEIGHBORS \
    --opacity_threshold $OPACITY_THRESHOLD
