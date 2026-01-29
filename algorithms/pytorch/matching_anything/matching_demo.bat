@echo off
cd /d "D:\hand_craft_registration\SRIF-master\python_registration_gui\algorithms\pytorch\matching_anything"

set "roma=D:\hand_craft_registration\SRIF-master\matching_anything\weights\matchanything_roma.ckpt"
set "eloftr=D:\hand_craft_registration\SRIF-master\matching_anything\weights\matchanything_eloftr.ckpt"

python "D:\hand_craft_registration\SRIF-master\matching_anything\matching_infer_demo.py" ^
  "D:\hand_craft_registration\SRIF-master\matching_anything\config.py" ^
  --method matchanything_roma@-@ransac_affine ^
  --ckpt_path "%roma%" ^
  --img0 "D:\hand_craft_registration\SRIF-master\dataset\HT\pair1_1.jpg" ^
  --img1 "D:\hand_craft_registration\SRIF-master\dataset\HT\pair1_2.jpg" ^
  --imgresize 832 ^
  --plot_matches ^
  --output_dir "demo_output"
