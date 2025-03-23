scenes=("figurines" "ramen" "teatime")

for index in "${!scenes[@]}"; do
        
    python render_lerf_mask_unified_lift.py  -m ./Released_checkpoint/${scenes[$index]}  --skip_train --iteration 30000
    
done


python script/eval_lerf_mask_unified_lift.py --excel_name ./results/Unifed_Lift --pred_path ./Released_checkpoint/