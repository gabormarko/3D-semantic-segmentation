#!/bin/bash
# Automated extraction of LSeg features for all three LERF datasets (first 10 images each)

export PYTHONPATH=$PYTHONPATH:$(realpath lang-seg)

# LERF
#declare -a DATASETS=(figurines teatime ramen)

#for DATASET in "${DATASETS[@]}"; do
#  INPUT_DIR="data/lerf/${DATASET}/images"
#  OUTPUT_DIR="data/lerf/${DATASET}/lseg_features"
#  mkdir -p "$OUTPUT_DIR"
#  echo "Extracting LSeg features for $DATASET..."
#  python script/extract_lseg_features.py \
#    --input_dir "$INPUT_DIR" \
#    --output_dir "$OUTPUT_DIR"
#done

# ADE20K
INPUT_DIR="/home/neural_fields/Unified-Lift-Gabor/ade20k/test"
OUTPUT_DIR="/home/neural_fields/Unified-Lift-Gabor/ade20k/lseg_features"
mkdir -p "$OUTPUT_DIR"
echo "Extracting LSeg features for ADE20K..."
python script/extract_lseg_features.py \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR"  