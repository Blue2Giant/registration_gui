python D:\hand_craft_registration\SRIF-master\python_registration_gui\algorithms\pytorch\MapGlue\map_glue_demo.py ^
    "D:\hand_craft_registration\WSSF-main\WSSF-main\ht_eval_for_own_affine\pair1_1.jpg" ^
    "D:\hand_craft_registration\WSSF-main\WSSF-main\ht_eval_for_own_affine\pair1_2.jpg" ^
    "D:\hand_craft_registration\SRIF-master\python_registration_gui\outputs\map_glue_demo\matches.txt" ^
    --device cuda ^
    --weights "D:\hand_craft_registration\SRIF-master\python_registration_gui\algorithms\pytorch\MapGlue\weights\fastmapglue_model.pt" ^
    --save_chessboard ^
    --chessboard_tile 128 ^
    --chessboard_out "D:\hand_craft_registration\SRIF-master\python_registration_gui\outputs\map_glue_demo\chessboard.png"
