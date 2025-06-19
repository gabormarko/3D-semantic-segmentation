#!/bin/bash
# Automated extraction of LSeg features for all three LERF datasets (first 10 images each)

export PYTHONPATH=$PYTHONPATH:$(realpath lang-seg)

declare -a DATASETS=(figurines teatime ramen)

for DATASET in "${DATASETS[@]}"; do
  INPUT_DIR="data/lerf/${DATASET}/images"
  OUTPUT_DIR="data/lerf/${DATASET}/lseg_features"
  mkdir -p "$OUTPUT_DIR"
  echo "Extracting LSeg features for $DATASET..."
  python script/extract_lseg_features.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR"
done
