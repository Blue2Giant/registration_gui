@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

call conda activate base

set "MAIN_CFG=D:\hand_craft_registration\SRIF-master\matching_anything\config.py"
set "ROMA_CKPT=D:\hand_craft_registration\SRIF-master\matching_anything\weights\matchanything_roma.ckpt"
set "OUTPUT_ROOT=D:\hand_craft_registration\SRIF-master\python_registration_gui\outputs\matching_anything_batch_3datasets"
set "GT_DIRECTION=1to2"
set "IMG_RESIZE=832"
set "SUCCESS_THR=4"

if not exist "%MAIN_CFG%" (
  echo [ERROR] MAIN_CFG not found: %MAIN_CFG%
  exit /b 1
)

if not exist "%ROMA_CKPT%" (
  echo [ERROR] ROMA_CKPT not found: %ROMA_CKPT%
  exit /b 1
)

call :run_dataset "OSdataset_16" "D:\hand_craft_registration\WSSF-main\WSSF-main\OSdataset_16"
if errorlevel 1 exit /b 1

call :run_dataset "HT_random_sar_aug_20260510" "D:\hand_craft_registration\organized_pairs_for_eval\HT_random_sar_aug_20260510"
if errorlevel 1 exit /b 1

call :run_dataset "SAR2OPT" "D:\hand_craft_registration\organized_pairs_for_eval\SAR2OPT"
if errorlevel 1 exit /b 1

echo [DONE] matching_anything 3 个数据目录已全部跑完。
exit /b 0

:run_dataset
set "DATASET_NAME=%~1"
set "PAIRS_DIR=%~2"
set "OUTPUT_DIR=%OUTPUT_ROOT%\%DATASET_NAME%"

echo.
echo ============================================================
echo [RUN] Dataset: %DATASET_NAME%
echo [RUN] Pairs Dir: %PAIRS_DIR%
echo [RUN] Output Dir: %OUTPUT_DIR%
echo ============================================================

python "%SCRIPT_DIR%registration_batch_demo.py" ^
  "%MAIN_CFG%" ^
  --method matchanything_roma@-@ransac_affine ^
  --ckpt_path "%ROMA_CKPT%" ^
  --pairs_dir "%PAIRS_DIR%" ^
  --output_dir "%OUTPUT_DIR%" ^
  --imgresize %IMG_RESIZE% ^
  --gt_direction %GT_DIRECTION% ^
  --success_match_threshold %SUCCESS_THR% ^
  --plot_matches ^
  --save_chessboard ^
  --chessboard_tile 128

if errorlevel 1 (
  echo [ERROR] Dataset failed: %DATASET_NAME%
  exit /b 1
)
exit /b 0
