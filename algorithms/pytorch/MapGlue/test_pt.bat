@echo off
python "D:\hand_craft_registration\SRIF-master\python_registration_gui\scripts\check_torchscript_pt.py" ^
  "D:\hand_craft_registration\SRIF-master\python_registration_gui\algorithms\pytorch\MapGlue\weights\fastmapglue_model.pt" ^
  --device cpu
