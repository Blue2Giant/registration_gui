@echo off
cd /d "D:\hand_craft_registration\SRIF-master\python_registration_gui\algorithms\pytorch\matching_anything"

call conda activate "D:\hand_craft_registration\SRIF-master\python_registration_gui\.conda_envs\matching_anything"

set "pairs_dir=D:\hand_craft_registration\WSSF-main\WSSF-main\ht_eval_for_own_affine"
set "roma=D:\hand_craft_registration\SRIF-master\matching_anything\weights\matchanything_roma.ckpt"

python "D:\hand_craft_registration\SRIF-master\python_registration_gui\algorithms\pytorch\matching_anything\matching_batch_demo.py" ^
  "D:\hand_craft_registration\SRIF-master\python_registration_gui\algorithms\pytorch\matching_anything\config.py" ^
  --method matchanything_roma@-@ransac_affine ^
  --ckpt_path "%roma%" ^
  --pairs_dir "%pairs_dir%" ^
  --imgresize 832 ^
  --output_dir "D:\hand_craft_registration\SRIF-master\python_registration_gui\outputs\demo_output_batch_matching"
