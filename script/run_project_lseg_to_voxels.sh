#!/bin/bash
# Run 2D LSeg feature projection to 3D voxel grid
# Usage: bash run_project_lseg_to_voxels.sh [VOXEL_PLY] [FEATURE_DIR] [OUTPUT_FEATURES] [COLMAP_SPARSE_DIR]

# Default paths (edit as needed)
VOXEL_PLY="/home/neural_fields/Unified-Lift-Gabor/output/minkowski_grid/teatime_minkowski_68458vox_iter30000_grid.ply"
FEATURE_DIR="/home/neural_fields/Unified-Lift-Gabor/data/lerf/teatime/lseg_features"
OUTPUT_FEATURES="/home/neural_fields/Unified-Lift-Gabor/output/minkowski_grid/teatime_voxel_features.npy"
COLMAP_SPARSE_DIR="/home/neural_fields/Unified-Lift-Gabor/data/lerf/teatime/sparse/0"

# Allow overriding via command line
if [ ! -z "$1" ]; then VOXEL_PLY="$1"; fi
if [ ! -z "$2" ]; then FEATURE_DIR="$2"; fi
if [ ! -z "$3" ]; then OUTPUT_FEATURES="$3"; fi
if [ ! -z "$4" ]; then COLMAP_SPARSE_DIR="$4"; fi

python3 script/project_lseg_to_voxels.py \
    --voxel_ply "$VOXEL_PLY" \
    --feature_dir "$FEATURE_DIR" \
    --output_features "$OUTPUT_FEATURES" \
    --colmap_sparse_dir "$COLMAP_SPARSE_DIR"
