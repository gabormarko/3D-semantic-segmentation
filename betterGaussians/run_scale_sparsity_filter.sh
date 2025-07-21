#!/bin/bash
# Usage: bash run_scale_sparsity_filter.sh <input_ply> <output_ply> [scale_min] [scale_max] [spikiness_threshold]
set -e

INPUT_PLY="/home/neural_fields/gaussian-splatting/output/cc946f8b-f/point_cloud/iteration_45000/point_cloud.ply"
OUTPUT_PLY="/home/neural_fields/Unified-Lift-Gabor/betterGaussians/output_scale_sparsity_filtered_gaussian.ply"
SCALE_MIN=0.0
SCALE_MAX=1.0
SPIKINESS_THRESHOLD=10.0

python3 scale_sparsity_filter.py \
    --gaussian_ply "$INPUT_PLY" \
    --scale_min "$SCALE_MIN" \
    --scale_max "$SCALE_MAX" \
    --spikiness_threshold "$SPIKINESS_THRESHOLD" \
    --out_ply "$OUTPUT_PLY"
