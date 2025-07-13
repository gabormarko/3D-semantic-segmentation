#!/bin/bash
set -e

# Debug pipeline for a single image/view
# 1. Extract camera parameters
# 2. Build sparse occupancy tensor
# 3. Prepare tensor data for a single image
# 4. Visualize occupancy and features
# 5. Run CUDA projection kernel for a single view
# 6. Visualize projection output

IMG_NAME="DSC03423.JPG"
FEATURE_NAME="DSC03423.JPG.npy"
VOXEL_PLY="/home/neural_fields/Unified-Lift-Gabor/output/minkowski_grid/officescene/officescene_minkowski_9434vox_iter50000_grid.ply"

# Extract VOXEL_SIZE and GRID_ORIGIN from the PLY file
read VOXEL_SIZE GRID_ORIGIN <<< $(python3 -c "
with open('$VOXEL_PLY', 'rb') as f:
    voxel_size = None
    grid_origin = None
    for line in f:
        try:
            line = line.decode('ascii')
        except:
            break
        if 'comment voxel_size' in line:
            voxel_size = float(line.split()[-1])
        if 'comment grid_origin' in line:
            grid_origin = ' '.join(line.split()[-3:])
        if 'end_header' in line:
            break
    print(voxel_size, grid_origin)
")
echo "[INFO] Using VOXEL_SIZE=$VOXEL_SIZE, GRID_ORIGIN=$GRID_ORIGIN"

LSEG_DIR="../data/scannetpp/officescene/lseg_features"
# DEBUG: Using unscaled camera parameters for the entire pipeline to bypass scaling issues.
CAM_PARAMS="camera_params/camera_params.json"

# Set paths
# COLMAP_SPARSE_DIR="../data/scannetpp/officescene/colmap/sparse/0"
COLMAP_SPARSE_DIR="../data/scannetpp/officescene/sparse/0"
VOXEL_GRID_PATH="/home/neural_fields/Unified-Lift-Gabor/output/minkowski_grid/officescene/officescene_minkowski_9434vox_iter50000_grid.ply"

### DUMMY VOXEL GRID:
#VOXEL_PLY="/home/neural_fields/Unified-Lift-Gabor/cuda_project_image_to_sparse_voxel/dummy_occupancy_points_ascii.ply"
#VOXEL_SIZE=1
#GRID_ORIGIN="0 0 0"
#VOXEL_GRID_PATH="/home/neural_fields/Unified-Lift-Gabor/cuda_project_image_to_sparse_voxel/dummy_occupancy_points_ascii.ply"
###

CAMERA_PARAMS_DIR="camera_params"

# 1. Extract camera parameters
# Step 1: Extract camera parameters from COLMAP
echo "=== COLMAP Camera Parameters Extraction ==="
python ../cuda_project_clean/extract_colmap_cameras.py \
    --sparse_dir $COLMAP_SPARSE_DIR \
    --output_dir $CAMERA_PARAMS_DIR

if [ $? -ne 0 ]; then
    echo "COLMAP camera extraction failed."
    exit 1
fi

echo "--- Step 1: Camera parameter extraction complete ---"

# Step 1a visualization removed to proceed with pipeline debugging.

# 2. Build sparse occupancy tensor
python build_sparse_occupancy.py \
  --voxel_ply $VOXEL_PLY \
  --voxel_size $VOXEL_SIZE \
  --grid_origin $GRID_ORIGIN \
  --out_tensor occupancy.pt


# 3. Prepare tensor data for a single image (only the selected feature file)
TMP_LSEG_DIR=$(mktemp -d)
cp "$LSEG_DIR/$FEATURE_NAME" "$TMP_LSEG_DIR/"
python prepare_tensor_data.py \
  --lseg_dir "$TMP_LSEG_DIR" \
  --scaled_camera_params $CAM_PARAMS \
  --occupancy occupancy.pt \
  --voxel_size $VOXEL_SIZE \
  --grid_origin $GRID_ORIGIN \
  --max_images 1 \
  --output tensor_data.pt
rm -rf "$TMP_LSEG_DIR"

# 4. Visualize occupancy and features
# python visualize_projection.py --occupancy occupancy.pt --tensor_data tensor_data.pt --show

# 5. Run CUDA projection kernel for a single view
python debug_project_features.py --tensor_data tensor_data.pt --output proj_output.pt

# 6. Visualize camera frustum and voxel grid
echo "--- Step 6: Visualize camera frustum and voxel grid ---"
python visualize_frustum.py \
        --tensor_data "tensor_data.pt" \
        --occupancy "occupancy.pt" \
        --output_ply "combined_frustum_visualization.ply"

if [ $? -ne 0 ]; then
    echo "Frustum visualization failed."
    exit 1
fi

echo "--- Pipeline complete. Check combined_frustum_visualization.ply ---"
