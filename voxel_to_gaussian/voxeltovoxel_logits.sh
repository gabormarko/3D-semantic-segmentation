#!/bin/bash
set -e
# Run Open-Vocabulary 3D Labeller pipeline for voxel-level segmentation labels and coloring

# Set input file paths and prompts here
VOXEL_FILE="/home/neural_fields/Unified-Lift-Gabor/cuda_project_image_to_sparse_voxel/voxel_feature_checkpoints_vox59293/checkpoint_features_150.pt"         # Change to your voxel feature file
OUT_DIR="/home/neural_fields/Unified-Lift-Gabor/voxel_to_gaussian/semantics_voxel_gauss" # Directory for output files
LABELS_OUT="$OUT_DIR/voxel_semantics.npz" # Output file for per-voxel labels
PROMPTS=("chair" "table" "wall" "door" "laptop" "floor" "ceiling")         # List your prompts here
DEVICE="cuda"                            # "cuda" or "cpu"

mkdir -p "$OUT_DIR"

echo "[Step 1] Query (assign semantic logits and color voxels)"
python3 /home/neural_fields/Unified-Lift-Gabor/voxel_to_gaussian/voxeltovoxel_logits.py query \
    --vox "$VOXEL_FILE" \
    --prompt "${PROMPTS[@]}" \
    --out "$LABELS_OUT" \
    --device "$DEVICE"

PLY_OUT="${LABELS_OUT/.npz/_colored_voxels.ply}"
echo "Pipeline complete. Output: $LABELS_OUT (contains per-voxel semantic labels and logits), $PLY_OUT (colored voxel .ply)"
