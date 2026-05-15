@echo off
cd /d "D:\hand_craft_registration\SRIF-master\python_registration_gui\algorithms\pytorch\matching_anything"

call conda activate "D:\hand_craft_registration\SRIF-master\python_registration_gui\.conda_envs\matching_anything"

set "roma=D:\hand_craft_registration\SRIF-master\matching_anything\weights\matchanything_roma.ckpt"
set "pairs_dir=D:\hand_craft_registration\WSSF-main\WSSF-main\pairs_random_affine_match"
set "pairs_dir=D:\hand_craft_registration\WSSF-main\WSSF-main\OSdataset_16"
python "D:\hand_craft_registration\SRIF-master\python_registration_gui\algorithms\pytorch\matching_anything\registration_batch_demo.py" ^
  "D:\hand_craft_registration\SRIF-master\python_registration_gui\algorithms\pytorch\matching_anything\config.py" ^
  --method matchanything_roma@-@ransac_affine ^
  --ckpt_path "%roma%" ^
  --pairs_dir "%pairs_dir%" ^
  --imgresize 832 ^
  --plot_matches ^
  --chessboard_tile 128 ^
  --save_chessboard ^
  --gt_direction 1to2 ^
  --match_err_thr 5 ^
  --max_pairs 0 ^
  --output_dir "D:\hand_craft_registration\SRIF-master\python_registration_gui\outputs\output_batch_matchinganything"

