#!/bin/bash
# Usage: bash minkowski_mesh_voxel_grid.sh
# Creates a sparse voxel grid near the surface of a mesh geometry (PLY file)

MESH_FILE="/home/neural_fields/Unified-Lift-Gabor/data/scannetpp/officescene/scans/mesh_aligned_0.05.ply"
OUTPUT_DIR="output/minkowski_mesh_grid"
CELL_SIZE=0.05
DENSITY_EPS=0.05
DENSITY_MIN_NEIGHBORS=10

mkdir -p "$OUTPUT_DIR"

python3 /home/neural_fields/Unified-Lift-Gabor/script/minkowski_mesh_voxel_grid.py \
    --mesh_file "$MESH_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --cell_size $CELL_SIZE \
    --density_eps $DENSITY_EPS \
    --density_min_neighbors $DENSITY_MIN_NEIGHBORS

echo "Sparse voxel grid created in $OUTPUT_DIR"
