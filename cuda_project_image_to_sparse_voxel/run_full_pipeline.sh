#!/bin/bash
# Unified-Lift-Gabor full data pipeline script
# This script generates camera parameters, occupancy tensor, tensor data, and runs the CUDA projection kernel.
# Usage: bash run_full_pipeline.sh

set -e  # Exit on error

# Step 1: Generate scaled camera parameters
echo "[1/4] Generating scaled camera parameters..."
python generate_scaled_camera_params.py \
  --sparse_dir ../data/scannetpp/officescene/sparse/0 \
  --downscale_factor 4 \
  --output_dir camera_params

# Step 2: Build sparse occupancy tensor
echo "[2/4] Building sparse occupancy tensor..."
VOXEL_PLY="../output/minkowski_grid/officescene/officescene_minkowski_104515vox_iter50000_grid.ply"

# Extract voxel_size and grid_origin from the PLY header
echo "Extracting voxel_size and grid_origin from $VOXEL_PLY ..."
readarray -t ply_header < <(head -30 "$VOXEL_PLY")

# Extract voxel_size (assume a comment line like: comment voxel_size 0.05)
voxel_size=$(printf '%s\n' "${ply_header[@]}" | grep -Eo 'comment voxel_size [^ ]+' | awk '{print $3}')
# Extract grid_origin (assume a comment line like: comment grid_origin 0 0 0)
grid_origin=$(printf '%s\n' "${ply_header[@]}" | grep -Eo 'comment grid_origin [^\n]+' | cut -d' ' -f3-)

if [[ -z "$voxel_size" || -z "$grid_origin" ]]; then
  echo "ERROR: Could not extract voxel_size or grid_origin from $VOXEL_PLY. Check the PLY header comments!"
  exit 1
fi

echo "Extracted voxel_size: $voxel_size"
echo "Extracted grid_origin: $grid_origin"

python build_sparse_occupancy.py \
  --voxel_ply "$VOXEL_PLY" \
  --voxel_size "$voxel_size" \
  --grid_origin $grid_origin \
  --out_tensor occupancy.pt

# Step 3: Prepare tensor data
echo "[3/4] Preparing tensor data..."
python prepare_tensor_data.py \
  --lseg_dir ../data/scannetpp/officescene/lseg_features \
  --scaled_camera_params camera_params/scaled_camera_params.json \
  --occupancy occupancy.pt \
  --voxel_size "$voxel_size" \
  --grid_origin $grid_origin \
  --max_images 100 \
  --output tensor_data.pt

# Step 4: Run CUDA projection kernel
echo "[4/4] Running CUDA projection kernel..."
python debug_project_features.py \
  --tensor_data tensor_data.pt \
  --output proj_output.pt

echo "Pipeline complete. Outputs: camera_params/, occupancy.pt, tensor_data.pt, proj_output.pt"
