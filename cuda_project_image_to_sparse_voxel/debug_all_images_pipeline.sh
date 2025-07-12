#!/bin/bash
set -e

# Debug pipeline for all images/views in the LSEG feature folder
# 1. Extract camera parameters
# 2. Build sparse occupancy tensor
# 3. Prepare tensor data for each image
# 4. Visualize occupancy and features
# 5. Run CUDA projection kernel for each view
# 6. Visualize projection output

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
    print(voxel_size, grid_origin)")
echo "[INFO] Using VOXEL_SIZE=$VOXEL_SIZE, GRID_ORIGIN=$GRID_ORIGIN"

LSEG_DIR="../data/scannetpp/officescene/lseg_features"
CAM_PARAMS="camera_params/camera_params.json"
COLMAP_SPARSE_DIR="../data/scannetpp/officescene/sparse/0"
VOXEL_GRID_PATH="/home/neural_fields/Unified-Lift-Gabor/output/minkowski_grid/officescene/officescene_minkowski_9434vox_iter50000_grid.ply"
CAMERA_PARAMS_DIR="camera_params"

# 1. Extract camera parameters
python ../cuda_project_clean/extract_colmap_cameras.py \
    --sparse_dir $COLMAP_SPARSE_DIR \
    --output_dir $CAMERA_PARAMS_DIR
if [ $? -ne 0 ]; then
    echo "COLMAP camera extraction failed."
    exit 1
fi
echo "--- Step 1: Camera parameter extraction complete ---"

# 2. Build sparse occupancy tensor
python build_sparse_occupancy.py \
  --voxel_ply $VOXEL_PLY \
  --voxel_size $VOXEL_SIZE \
  --grid_origin $GRID_ORIGIN \
  --out_tensor ALL_occupancy.pt

echo "[INFO] Processing all LSEG features in $LSEG_DIR ..."

# Ensure output directories exist
mkdir -p tensor_data proj_output vis

for FEATURE_PATH in "$LSEG_DIR"/*.npy; do
    IMG_NAME=$(basename "${FEATURE_PATH%.npy}")
    echo "[INFO] Processing $IMG_NAME ..."
    # 3. Prepare tensor data for this image (process one at a time)
    python prepare_tensor_data.py \
      --lseg_dir $LSEG_DIR \
      --scaled_camera_params $CAM_PARAMS \
      --occupancy ALL_occupancy.pt \
      --voxel_size $VOXEL_SIZE \
      --grid_origin $GRID_ORIGIN \
      --max_images 1 \
      --output "tensor_data/tensor_data_${IMG_NAME}.pt"

    # 4. Visualize occupancy and features
    python visualize_projection.py --occupancy ALL_occupancy.pt --tensor_data "tensor_data/tensor_data_${IMG_NAME}.pt" --show

    # 5. Run CUDA projection kernel for this view
    python debug_project_features.py --tensor_data "tensor_data/tensor_data_${IMG_NAME}.pt" --output "proj_output/proj_output_${IMG_NAME}.pt"

    # remove large tensor data file to save space after output is created
    rm "tensor_data/tensor_data_${IMG_NAME}.pt"
done

echo "--- Step 6: Visualize camera frustum and voxel grid (last image) ---"
python visualize_frustum.py \
    --tensor_data "tensor_data/tensor_data_${IMG_NAME}.pt" \
    --occupancy "ALL_occupancy.pt" \
    --output_ply "vis/combined_frustum_visualization_${IMG_NAME}.ply"
if [ $? -ne 0 ]; then
    echo "Frustum visualization failed."
    exit 1
fi
echo "--- Pipeline complete for all images. Check vis/combined_frustum_visualization_${IMG_NAME}.ply ---"
