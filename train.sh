mkdir output
output_dir="./output/unifed_lift/"
mkdir ${output_dir}

#scenes=("figurines" "ramen" "teatime")
scenes=("teatime")
for index in "${!scenes[@]}"; do
    python train_unified_lift.py -s ./data/lerf/${scenes[$index]} -m ${output_dir}/${scenes[$index]}  --iterations 15000 --config_file config/gaussian_dataset/train.json --weight_loss 1e-0 --train_split --use_wandb --checkpoint_iterations 1 10 100 1000 5000 10000 15000 20000 25000 30000
done