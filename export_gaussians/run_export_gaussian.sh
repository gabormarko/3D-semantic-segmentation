#!/bin/bash
# Usage: ./run_export_gaussian.sh <scene> <base_dir> <opacity_threshold> [checkpoint_file]

SCENE=${1:-officescene}
BASE_DIR=${2:-/home/neural_fields/Unified-Lift-Gabor/output/unifed_lift}
OPACITY_THRESHOLD=${3:-0.9}
CHKPT_FILE=${4}

SCRIPT_DIR="$(dirname "$0")"
PYTHON_SCRIPT="$SCRIPT_DIR/export_gaussian.py"

if [ -n "$CHKPT_FILE" ]; then
    INPUT_CHKPT="$CHKPT_FILE"
    if [ ! -f "$INPUT_CHKPT" ]; then
        echo "Specified checkpoint file does not exist: $INPUT_CHKPT"
        exit 1
    fi
    echo "Processing specified checkpoint: $INPUT_CHKPT"
else
    # Find the latest checkpoint file in the checkpoint folder
    CHKPT_DIR="$BASE_DIR/$SCENE/chkpnts"
    INPUT_CHKPT=$(ls -1v "$CHKPT_DIR"/*.pth 2>/dev/null | tail -n 1)
    if [ -z "$INPUT_CHKPT" ]; then
        echo "No checkpoint files found in $CHKPT_DIR"
        exit 1
    fi
    echo "Processing latest checkpoint: $INPUT_CHKPT"
fi

python3 "$PYTHON_SCRIPT" \
    --scene "$SCENE" \
    --base_dir "$BASE_DIR" \
    --opacity_threshold "$OPACITY_THRESHOLD" \
    --input_checkpoint "$INPUT_CHKPT"
