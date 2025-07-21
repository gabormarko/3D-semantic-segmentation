#!/bin/bash
# Filter Gaussians by distance to closest voxel
# Usage: bash run_dist_to_mesh_filter.sh

GAUSSIAN_PLY="/home/neural_fields/gaussian-splatting/output/cc946f8b-f/point_cloud/iteration_45000/point_cloud.ply"  # Path to Gaussian .ply file
VOXEL_PLY="/home/neural_fields/Unified-Lift-Gabor/output/minkowski_mesh_grid/mesh_minkowski_grid_vox27522.ply"  # Path to voxel grid .ply file
MAX_DIST=0.04  # Maximum allowed distance
OUT_PLY="/home/neural_fields/Unified-Lift-Gabor/betterGaussians/filtered_gaussians.ply"  # Output filtered .ply file

python /home/neural_fields/Unified-Lift-Gabor/betterGaussians/dist_to_mesh_filter.py \
    --gaussian_ply "$GAUSSIAN_PLY" \
    --voxel_ply "$VOXEL_PLY" \
    --max_dist $MAX_DIST \
    --out_ply "$OUT_PLY"
