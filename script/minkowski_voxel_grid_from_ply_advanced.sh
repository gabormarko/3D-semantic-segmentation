#!/bin/bash
# Usage: bash minkowski_voxel_grid_from_ply_advanced.sh <input_ply> <output_dir> [options]
set -e

INPUT_PLY="/home/neural_fields/gaussian-splatting/output/39ac5c9a-1/point_cloud/iteration_30000/point_cloud.ply"  # <-- set your .ply path here
OUTPUT_DIR="/home/neural_fields/Unified-Lift-Gabor/output/minkowski_grid/officescene_filtered_ply_adv"

# Set parameters here
CELL_SIZE=0.03  # Size of voxel grid cells
DENSITY_EPS=0.10    # Epsilon radius for density filtering
DENSITY_MIN_NEIGHBORS=4   # Minimum neighbors for density filtering
OPACITY_THRESHOLD=0.5 # Fraction of points to leave out (e.g. 0.8 keeps top 20% by opacity)
SCALE_THRESHOLD=0.5     # Maximum allowed scale for a Gaussian to be considered part of the surface
SPIKINESS_THRESHOLD=8.0 # Maximum allowed ratio of largest to smallest scale (spikiness filter)
ADAPTIVE_DENSITY="--adaptive_density"  # set to "" to disable
NORMAL_CONSISTENCY=1.0  # 1.0: DISABLED, Normal consistency threshold (0.0 to 1.0)
NORMAL_CONSISTENCY_EPS=0.05 # Epsilon radius for normal consistency
NORMAL_CONSISTENCY_MIN_NEIGHBORS=5  # Minimum neighbors for normal consistency

python3 script/minkowski_voxel_grid_from_ply_advanced.py \
    --ply "$INPUT_PLY" \
    --output_dir "$OUTPUT_DIR" \
    --cell_size "$CELL_SIZE" \
    --density_eps "$DENSITY_EPS" \
    --density_min_neighbors "$DENSITY_MIN_NEIGHBORS" \
    --opacity_threshold "$OPACITY_THRESHOLD" \
    --scale_threshold "$SCALE_THRESHOLD" \
    --spikiness_threshold "$SPIKINESS_THRESHOLD" \
    $ADAPTIVE_DENSITY \
    --normal_consistency "$NORMAL_CONSISTENCY" \
    --normal_consistency_eps "$NORMAL_CONSISTENCY_EPS" \
    --normal_consistency_min_neighbors "$NORMAL_CONSISTENCY_MIN_NEIGHBORS"
