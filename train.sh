mkdir output
output_dir="./output/unified_lift/"
mkdir ${output_dir}

scenes=("ramen", "teatime")
for index in "${!scenes[@]}"; do
    python train_unified_lift.py -s ./data/${scenes[$index]} -m ${output_dir}/${scenes[$index]}   --config_file config/gaussian_dataset/train.json --weight_loss 1e-0 --train_split --use_wandb
done