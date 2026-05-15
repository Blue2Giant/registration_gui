@echo off
cd /d "D:\hand_craft_registration\SRIF-master\python_registration_gui\algorithms\pytorch\matching_anything"

call conda activate "D:\hand_craft_registration\SRIF-master\python_registration_gui\.conda_envs\matching_anything"

set "roma=D:\hand_craft_registration\SRIF-master\matching_anything\weights\matchanything_roma.ckpt"
set "pairs_dir=D:\hand_craft_registration\WSSF-main\WSSF-main\ht_eval_for_own_origin"

python "D:\hand_craft_registration\SRIF-master\python_registration_gui\algorithms\pytorch\matching_anything\registration_batch_nogt_demo.py" ^
  "D:\hand_craft_registration\SRIF-master\python_registration_gui\algorithms\pytorch\matching_anything\config.py" ^
  --method matchanything_roma@-@ransac_affine ^
  --ckpt_path "%roma%" ^
  --pairs_dir "%pairs_dir%" ^
  --imgresize 832 ^
  --warp_dir "demo_output_batch_nogt/warped" ^
  --chessboard_dir "demo_output_batch_nogt/chessboard" ^
  --chessboard_tile 128 ^
  --save_chessboard ^
  --output_dir "demo_output_batch_nogt"
