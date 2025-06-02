#!/bin/bash

# Default parameters
MODEL_PATH="output/unifed_lift/ramen"  # Path to the trained model
SOURCE_PATH="data/lerf/ramen"  # Path to source data
IMAGES_PATH="images"  # Path to images (relative to SOURCE_PATH)
ITERATION=-1  # Use latest iteration
CELL_SIZE=0.05  # Size of hash grid cells
HASH_SIZE=1048576  # 2^20
MAX_POINTS_PER_CELL=32
OUTPUT_DIR="output/hash_grid"
SAVE_PLY=true
TEST_QUERIES=1000
EVAL=true  # Evaluation mode
N_VIEWS=0  # Use all views
TRAIN_SPLIT=0.8

# Run hash grid test
python script/test_hash_grid.py \
    --model_path $MODEL_PATH \
    --source_path $SOURCE_PATH \
    --images $IMAGES_PATH \
    --iteration $ITERATION \
    --cell_size $CELL_SIZE \
    --hash_size $HASH_SIZE \
    --max_points_per_cell $MAX_POINTS_PER_CELL \
    --output_dir $OUTPUT_DIR \
    --test_queries $TEST_QUERIES \
    --n_views $N_VIEWS \
    --train_split $TRAIN_SPLIT \
    $([ "$EVAL" = true ] && echo "--eval") \
    $([ "$SAVE_PLY" = true ] && echo "--save_ply") 