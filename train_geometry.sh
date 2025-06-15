# Increase open file limit
ulimit -n 65536
#!/bin/bash

# Create output directory if it doesn't exist
output_dir="./output/unified_lift"
mkdir -p "${output_dir}"

# Define your scenes
scenes=("classroomscene" "officescene")

# Loop over scenes
for index in "${scenes[@]}"; do
    echo "Processing scene: $index"
    python geometry_train.py -s ./data/${index} -m ${output_dir}/${index} \
    --config_file config/gaussian_dataset/train.json \
    --resolution 1
done

