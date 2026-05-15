@echo off
cd /d "D:\hand_craft_registration\SRIF-master\python_registration_gui\algorithms\pytorch\MapGlue"

call conda activate base

set "pairs_dir=D:\hand_craft_registration\WSSF-main\WSSF-main\pairs_random_affine_match"
set "weights=D:\hand_craft_registration\SRIF-master\python_registration_gui\algorithms\pytorch\MapGlue\weights\fastmapglue_model.pt"
set "pairs_dir=D:\hand_craft_registration\WSSF-main\WSSF-main\OSdataset_16"

python "D:\hand_craft_registration\SRIF-master\python_registration_gui\algorithms\pytorch\MapGlue\map_glue_batch_demo.py" ^
  --pairs_dir "%pairs_dir%" ^
  --weights "%weights%" ^
  --device cuda ^
  --plot_matches ^
  --save_chessboard ^
  --chessboard_tile 128 ^
  --output_dir "D:\hand_craft_registration\SRIF-master\python_registration_gui\outputs\demo_output_batch_mapglue"
