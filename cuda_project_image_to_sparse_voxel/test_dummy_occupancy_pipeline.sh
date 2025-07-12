#!/bin/bash
set -e

# Test CUDA pipeline with dummy dense occupancy grid
echo "Copying dummy_occupancy.pt to occupancy.pt..."
cp dummy_occupancy.pt occupancy.pt

echo "Preparing tensor_data.pt with dummy occupancy..."
python prepare_tensor_data.py \
  --lseg_dir ../data/scannetpp/officescene/lseg_features \
  --scaled_camera_params camera_params/camera_params.json \
  --occupancy occupancy.pt \
  --voxel_size 1.0 \
  --grid_origin 0 0 0 \
  --max_images 1 \
  --output tensor_data.pt

echo "Running CUDA debug kernel with dummy occupancy..."
python debug_project_features.py --tensor_data tensor_data.pt --output proj_output.pt

echo "Done. Check CUDA debug output for occ_idx hits in the dense region."
